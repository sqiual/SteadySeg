import argparse, os, shutil
import torch
import random, math
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.transforms as T
from util import *
import time 

from model import *

from sklearn.metrics import f1_score
import torch.nn.functional as F
from dtaidistance import dtw
from sklearn.metrics import mean_squared_error

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score

from collections import Counter
#================================
#================================
gname = 'DSN'
in_dim = 3
in_len = 10000
out_dim = 15
plot = False
batch_size = 1
#================================
#================================
def count_stages(lst):
    stages = 1  # Initialize the count of stages to 1 as we always start with the first stage
    for i in range(1, len(lst)):
        if lst[i] != lst[i-1]:
            stages += 1
    return stages

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

def main(plot = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    test_dataset = fish_dataloader(data = 'data/dma_src_test_fish', label = 'data/dma_trg_test_fish', batch_size=batch_size,num_workers=5)
    # set up model
    test_loader = test_dataset.run('test')
    net = Model(in_dim = in_dim, num_classes=out_dim, in_len = in_len).to(device)
    net2 = Model(in_dim = in_dim, num_classes=out_dim, in_len = in_len).to(device)
    
    model = net.to(device)
    model2 = net2.to(device)
    criterion = nn.CrossEntropyLoss() # (reduction="sum")
    criterion.to(device)
    
    path = 'checkpoint/OURS_bestmodel_fish.pt'
    sname = gname
        
    print("=> loading checkpoint '{}'".format(path))
    print('Processing testing data files ...')
    
    checkpoint = torch.load(path)
    start_epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]
    model.load_state_dict(checkpoint["model1"])
    model2.load_state_dict(checkpoint["model2"])
    print("the model of epoch %d" % start_epoch)

    if torch.cuda.is_available():
        model.to(device)
        model2.to(device)
    else:
        model.cpu()
        model2.cpu()
        
    model.eval()
    model2.eval()
    with torch.no_grad():
        testing_results = {'f1_weighted':0, 'f1_micro': 0, 'f1_macro': 0}
        test_bar = tqdm(test_loader)
        k = 0
        t_mse = 0
        
        for test_data, test_label in test_bar:
            test_data = torch.nn.functional.normalize(test_data, dim = 1)
            test_data = test_data.to(device)
            test_label = test_label.type(torch.LongTensor) 
            test_label = test_label.to(device)
            test_label = test_label.squeeze(dim = 1)
            test_label_seg = seg_point(test_label)
            
            test_pred, test_seg, test_mad = model(test_data.float())
            softmax = nn.Softmax(dim = -1)
            
            F1_weighted, F1_micro, F1_macro = 0,0,0
            for i in range(test_label.shape[0]):
                test_predlabel = np.argmax(softmax(test_pred[i].float()).cpu().numpy(), axis = -1)
                f1_weighted = f1_score(test_predlabel, test_label[i].float().cpu().numpy(), average = 'weighted')
                f1_micro = f1_score(test_predlabel, test_label[i].float().cpu().numpy(), average = 'micro')
                f1_macro = f1_score(test_predlabel, test_label[i].float().cpu().numpy(), average = 'macro')
                F1_weighted += f1_weighted
                F1_micro += f1_micro
                F1_macro += f1_macro

            testing_results['f1_weighted'] += (F1_weighted)/batch_size   
            testing_results['f1_micro'] += (F1_micro)/batch_size   
            testing_results['f1_macro'] += (F1_macro)/batch_size   

            test_bar.set_description(desc='F1 weighted: %.3f  F1 micro: %.3f  F1 macro: %.3f' % ((F1_weighted)/batch_size, (F1_micro)/batch_size, (F1_macro)/batch_size))

    print('========= Testing Result =========')
    print('F1 weighted: {}'.format(testing_results['f1_weighted']/len(test_bar)))
    print('F1 micro: {}'.format(testing_results['f1_micro']/len(test_bar)))
    print('F1 macro: {}'.format(testing_results['f1_macro']/len(test_bar)))
    return



if __name__ == "__main__":
    main()