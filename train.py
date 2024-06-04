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

parser = argparse.ArgumentParser(description='OURS')
parser.add_argument('--num_epochs', default=50, type=int, help='train epoch number')
parser.add_argument('--start_epoch', default=0, type=int, help='start epoch number')
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=2e-4")

########################################################
########################################################
#parser.add_argument("--pretrained", default='checkpoint/OURS_bestmodel_fish.pt', type=str, help="")
parser.add_argument("--pretrained", default=None, type=str, help="path to pretrained model (default: none)") 
########################################################
########################################################

parser.add_argument('--in_len', default=10000, type=int, help='input sequence length')
parser.add_argument('--in_dim', default=3, type=int, help='input sequence dimension')
parser.add_argument('--out_dim', default=15, type=int, help='output sequence dimension')
parser.add_argument("--print_freq", default=1, type=int, help="the freq of print during training")
parser.add_argument("--save_freq", default=1, type=int, help="the freq of save checkpoint")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--weight_decay", default=0.001, type=int, help="weight_decay")
parser.add_argument("--p_threshold", default=0.5, type=int, help="")

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

def tmse(seq):
    mse = nn.MSELoss(reduction='none')
    
    out = torch.mean(
        torch.clamp(
            mse(
                F.log_softmax(seq[:, :, 1:], dim=1), 
                F.log_softmax(seq.detach()[:, :, :-1], dim=1)
            ), 
            min=0, 
            max=100
        )
    )
    return out

        
def pnorm(weights):
    normB = torch.norm(weights, 2, 1)
    ws = weights.clone()
    for i in range(weights.size(0)):
        ws[i] = ws[i] / torch.pow(normB[i], 0.2)
    return ws


def train(model, model2, data_loader, criterion, criterion2, optimizer, epoch, num_epochs, start_epoch, in_dim):
        model.train()
        model2.eval()
        data_bar = tqdm(data_loader)
        for data,label in data_bar:
            data, label = Variable(data, requires_grad=True), Variable(label, requires_grad=False)  # [batch size, src dim, src len]

            data = torch.nn.functional.normalize(data, dim = -1)
            data = data.to(device)
            label = label.type(torch.LongTensor) 
            label = label.to(device)
            label_seg = seg_point(label)
            pred2, seg2, mad2 = model2(data.float())
            pred, seg, mad = model(data.float(), mad2)
            _, pred_tmp = torch.max(pred, -1) 
            
            l = 0
            l_seg = 0
            t_mse = 0
            
            t_mse = tmse(seg.permute(0,2,1).float())
            t_mse += tmse(pred.permute(0,2,1).float())
            for i in range(label.shape[0]):
                l += criterion(pred[i], label[i].squeeze())
                l_seg += criterion2(seg[i], torch.tensor(label_seg[i]).to(device))
            
            loss = l + l_seg + t_mse
            model.zero_grad()
            loss.backward()
            optimizer.step()
           
            data_bar.set_description(desc='[%d/%d] CE: %.3f CE_seg: %.3f TMSE: %.3f' % (
                epoch, num_epochs + start_epoch, l.item(), l_seg.item(), t_mse.item()))
        torch.cuda.empty_cache()

        
def test(model, model2, data_loader, criterion, criterion2, epoch, num_epochs, start_epoch):
    model.eval()
    model2.eval()
    with torch.no_grad():
        valing_results = {'loss': 0}
        data_bar = tqdm(data_loader)
        for val_data, val_label in data_bar:
            batch_size = val_data.shape[0]
            
            val_data = torch.nn.functional.normalize(val_data, dim = -1)
            val_data = val_data.to(device)
            val_label = val_label.type(torch.LongTensor) 
            val_label = val_label.to(device)
            val_pred, val_seg, val_mad = model(val_data.float())
            
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
    args = parser.parse_args()
    print(args)
    
    gname = "OURS"
    print('model name: ', gname)
    # =============================
    
    net1 = Model(in_dim = args.in_dim, num_classes=args.out_dim, in_len = args.in_len).to(device)
    net2 = Model(in_dim = args.in_dim, num_classes=args.out_dim, in_len = args.in_len).to(device)
    
    model1 = net1.to(device)
    model2 = net2.to(device)
   
    args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    print('# {} parameters:'.format(gname), sum(param.numel() for param in model1.parameters()))
    print('# {} parameters:'.format(gname), sum(param.numel() for param in model2.parameters()))   
    
    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    criterion.to(device)
    model1.to(device)
    model2.to(device)

    optimizer1 = optim.Adam(model1.parameters(),lr=args.lr)
    optimizer2 = optim.Adam(model2.parameters(),lr=args.lr)

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            args.start_epoch = checkpoint["epoch"]
            best_loss = checkpoint["best_loss"]
            model1.load_state_dict(checkpoint["model1"])
            model2.load_state_dict(checkpoint["model2"])
            optimizer1.load_state_dict(checkpoint["optimizer1"])
            optimizer2.load_state_dict(checkpoint["optimizer2"])
        else:
            print("=> no model found at '{}'".format(args.pretrained))
    else:
        best_loss = float('inf')

        
    
    print('Training started ...') 
    all_loss = [[],[]]
    
    
    for epoch in range(args.start_epoch + 1, args.num_epochs + args.start_epoch + 1):  
        train_dataset = fish_dataloader(data='data/dma_src_train_fish', label='data/dma_trg_train_fish', batch_size=args.batch_size,num_workers=5)
        val_dataset = fish_dataloader(data = 'data/dma_src_val_fish', label = 'data/dma_trg_val_fish', batch_size=args.batch_size,num_workers=5)

        train_loader1 = train_dataset.run('eval_train') 
        train_loader2 = train_dataset.run('eval_train') 
        val_loader = val_dataset.run('test') 
        
        ############################
        #    Training networks:    #
        ###########################        
        print('=== Training Net 1 ===')
        train(model1, model2, train_loader1, criterion, criterion2, optimizer1, epoch, args.num_epochs, args.start_epoch, args.in_dim)
        
        print('=== Training Net 2 ===')
        train(model2, model1, train_loader2, criterion, criterion2, optimizer2, epoch, args.num_epochs, args.start_epoch, args.in_dim)
        
        ############################
        #          Eval:          #
        ###########################
        print('=== Evaluation ===')
        loss, valing_results = test(model1, model2, val_loader, criterion, criterion2, epoch, args.num_epochs, args.start_epoch)
        
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
            "model1": model1.state_dict(),
            "optimizer1": optimizer1.state_dict(),
            "model2": model2.state_dict(),
            "optimizer2": optimizer2.state_dict()
        }, is_best, gname)

def save_checkpoint(state, is_best, name):
    filename = "checkpoint/%s_checkpoint_epoch_%d.pt" % (name, state["epoch"])
    if is_best:
        print("saving the epoch {} as best model".format(state["epoch"]))
        torch.save(state, filename)
        shutil.copyfile(filename, 'checkpoint/%s_bestmodel_fish_.pt'%(name))
    print("Checkpoint saved to {}".format(filename))

if __name__ == "__main__":
    main()