import argparse, os, shutil
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score, hamming_loss, multilabel_confusion_matrix
from util import *
import time 
from model import * 
import torch.nn.functional as F

#bestmodel_fish_ 是从头开始用不平均的0.1训的
#bestmodel_fish_balanced 是用平均训完以后用1的不平均轮一遍的
#bestmodel_pretraining 是用平均训的


parser = argparse.ArgumentParser(description='OURS')
parser.add_argument('--num_epochs', default=50, type=int, help='train epoch number')
parser.add_argument('--start_epoch', default=0, type=int, help='start epoch number')
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate. Default=2e-4")

########################################################
########################################################
#parser.add_argument("--pretrained", default='checkpoint/OURS_bestmodel_fish.pt', type=str, help="")
parser.add_argument("--pretrained", default=None, type=str, help="path to pretrained model (default: none)") 
########################################################
########################################################

parser.add_argument('--in_len', default=10000, type=int, help='input sequence length')
parser.add_argument('--in_dim', default=3, type=int, help='input sequence dimension')
parser.add_argument('--out_dim', default=14, type=int, help='output sequence dimension')
parser.add_argument("--print_freq", default=1, type=int, help="the freq of print during training")
parser.add_argument("--save_freq", default=1, type=int, help="the freq of save checkpoint")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--weight_decay", default=0.0001, type=int, help="weight_decay")
parser.add_argument("--p_threshold", default=0.5, type=int, help="")

def entropy(labels):
    label_counts = Counter(labels)
    total_count = len(labels)
    entropy = -sum((count/total_count) * np.log2(count/total_count) for count in label_counts.values())
    return entropy

def seg_point(x):
    result = []
    for i in range(x.shape[0]):
        sublist_result = []
        prev = None
        for j in x[i]:
            if x[i,j] != prev:
                sublist_result.append(1)
                prev = x[i,j]
            else:
                sublist_result.append(0)
        result.append(sublist_result)
    return result


# def tmse(seq):
#     mse = nn.MSELoss(reduction='none')
    
#     out = torch.mean(
#         torch.clamp(
#             mse(
#                 F.log_softmax(seq[:, :, 1:], dim=1), 
#                 F.log_softmax(seq.detach()[:, :, :-1], dim=1)
#             ), 
#             min=0, 
#             max=100
#         )
#     )
#     return out

def stmse(seq):
    mse = nn.MSELoss(reduction='sum')
    
    num_chunks = seq.shape[-1] // 200   #10000/50 = 200
    chunks = torch.chunk(seq, num_chunks, dim=-1)
    chunk_tmse = []
    for i, chunk in enumerate(chunks):
        chunk_tmse.append(
            mse(
                F.log_softmax(chunk[:, :, 1:], dim=1), 
                F.log_softmax(chunk.detach()[:, :, :-1], dim=1)
            )*1e4)

    return max(chunk_tmse)

        
def pnorm(weights):
    normB = torch.norm(weights, 2, 1)
    ws = weights.clone()
    for i in range(weights.size(0)):
        ws[i] = ws[i] / torch.pow(normB[i], 0.2)
    return ws


def train(model, data_loader, criterion, criterion2, optimizer, epoch, num_epochs, start_epoch, in_dim, out_dim):
        model.train()
        data_bar = tqdm(data_loader)
        softmax = nn.Softmax(dim=-1)
        for data_list,label in data_bar:
            pl = 0
            nl = 0
            l_seg = 0
            t_mse = 0
            label = Variable(label, requires_grad=False)
            label = label.type(torch.LongTensor) 
            label = label.to(device)
            label_seg = seg_point(label)
            label_neg = (label.squeeze().unsqueeze(-1).repeat(1, 1) + torch.LongTensor(len(label), 1).random_(1, out_dim-1).to(device)) % out_dim
            label_neg = Variable(label_neg.unsqueeze(0).to(device))   #[5000, 1]
            pred_list = []
            seg_list = []
            pred_neg_list = []
            for j in range(len(data_list)):
                data = Variable(data_list[j], requires_grad=True)  # [batch size, src dim, src len]
                data = torch.nn.functional.normalize(data, dim = -1)
                data = data.to(device)
                pred_, seg_ = model(data.float())
                pred_neg_ = torch.log(torch.clamp(1.-F.softmax(pred_, -1), min=1e-5, max=1.))
                pred_list.append(pred_)
                seg_list.append(seg_)
                pred_neg_list.append(pred_neg_)
                
            pred = torch.cat(pred_list)
            seg = torch.cat(seg_list)
            pred_neg = torch.cat(pred_neg_list)

            pred = torch.mean(pred, 0).unsqueeze(0)
            seg = torch.mean(seg, 0).unsqueeze(0)
            pred_neg = torch.mean(pred_neg, 0).unsqueeze(0)
            
            
            for i in range(label.shape[0]):
                l_seg += criterion(seg[i], torch.tensor(label_seg[i]).to(device))
                pred_,label_ = torch.max(softmax(pred[i]), -1)
                pred_max = torch.mean(pred_.float())
                if pred_max > 0: #0.15
                    pl += criterion(pred[i], label[i].squeeze())
                else:
                    nl += criterion2(pred_neg[i], label_neg[i].squeeze())
            e = entropy(label_) 
            t_mse = stmse(seg.permute(0,2,1).float())*(1/e)
            loss = pl + nl + l_seg + t_mse
            model.zero_grad()
            loss.backward()
            optimizer.step()

            data_bar.set_description(desc='[%d/%d] total loss: %.3f CE_seg: %.3f TMSE: %.3f' % (
                epoch, num_epochs + start_epoch, loss, l_seg.item(), t_mse.item()))
        torch.cuda.empty_cache()

        
def test(model, data_loader, criterion, epoch, num_epochs, start_epoch):
    model.eval()
    with torch.no_grad():
        valing_results = {'loss': 0}
        data_bar = tqdm(data_loader)
        for val_data, val_label in data_bar:
            batch_size = val_data.shape[0]
            
            val_data = torch.nn.functional.normalize(val_data, dim = -1)
            val_data = val_data.to(device)
            val_label = val_label.type(torch.LongTensor) 
            val_label = val_label.to(device)
            val_pred, val_seg = model(val_data.float())
            
            loss = 0
            for i in range(val_label.shape[0]):
                l = criterion(val_pred[i], val_label[i].squeeze())
                loss += l
            valing_results['loss'] += loss.item()/batch_size
            data_bar.set_description(desc='[%d/%d] loss: %.3f' % (
            epoch, num_epochs + start_epoch, loss.item()/batch_size))
           
    return loss, valing_results
                
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    args = parser.parse_args()
    print(args)
    
    gname = "OURS"
    print('model name: ', gname)
    # =============================
    
    net = Model(in_dim = args.in_dim, num_classes=args.out_dim, in_len = args.in_len).to(device)
    
    model = net.to(device)
   
    args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    print('# {} parameters:'.format(gname), sum(param.numel() for param in model.parameters()))  
    

    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.NLLLoss()
    criterion.to(device)
    criterion2.to(device)
    model.to(device)

    optimizer = optim.Adam(model.parameters(),lr=args.lr)

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            args.start_epoch = checkpoint["epoch"]
            best_loss = checkpoint["best_loss"]
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print("=> no model found at '{}'".format(args.pretrained))
    else:
        best_loss = float('inf')

        
    
    print('Training started ...') 
    all_loss = [[],[]]
    
    
    for epoch in range(args.start_epoch + 1, args.num_epochs + args.start_epoch + 1):  
        train_dataset = fish_dataloader(data='/data/processed_data/dma_src_train_fish', label='/data/processed_data/dma_trg_train_fish', batch_size=args.batch_size,num_workers=5)
        val_dataset = fish_dataloader(data = '/data/processed_data/dma_src_val_fish', label = '/data/processed_data/dma_trg_val_fish', batch_size=args.batch_size,num_workers=5)

        train_loader = train_dataset.run('eval_train') 
        val_loader = val_dataset.run('test') 
        
        ############################
        #    Training networks:    #
        ###########################        
        print('=== Training ===')
        train(model, train_loader, criterion, criterion2, optimizer, epoch, args.num_epochs, args.start_epoch, args.in_dim, args.out_dim)

        
        ############################
        #          Eval:          #
        ###########################
        print('=== Evaluation ===')
        loss, valing_results = test(model, val_loader, criterion, epoch, args.num_epochs, args.start_epoch)
        
        ############################
        #      Save result:       #
        ###########################
        print("epoch: {}\tloss:{} ".format(epoch, valing_results['loss'] / len(val_loader)))
        vec_loss = valing_results['loss']/len(val_loader)

        if vec_loss  < best_loss:
            best_loss = vec_loss 
            is_best = True
        else:
            is_best = False
        save_checkpoint({
            "epoch": epoch,
            "best_loss": best_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, is_best, gname)

def save_checkpoint(state, is_best, name):
    filename = "checkpoint/%s_checkpoint_epoch_%d.pt" % (name, state["epoch"])
    if is_best:
        print("saving the epoch {} as best model".format(state["epoch"]))
        torch.save(state, filename)
        shutil.copyfile(filename, 'checkpoint/%s_bestmodel_fish_tmse_pl.pt'%(name))
    print("Checkpoint saved to {}".format(filename))

if __name__ == "__main__":
    main()