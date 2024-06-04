from os import listdir
from os.path import join
import torch
import numpy as np
import json
import random
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import math


def random_transformation(data,target):
    # Randomly choose one of the augmentation methods
    augmentation_method = np.random.choice([no_augmentation, axis_shifting, random_masking, jittering])
    # Apply the chosen augmentation method to the sequence
    if augmentation_method == axis_shifting:
        augmented_sequence, augmented_target = augmentation_method(data,target)
    else:
        augmented_sequence, augmented_target = augmentation_method(data,target)
    return augmented_sequence, augmented_target

def no_augmentation(data,target):
    return data,target

def axis_shifting(data,target, max_shift=5):
    # Randomly shift the sequence along the time axis
    sequence = np.concatenate((np.squeeze(data, axis = -1),np.expand_dims(target, 0)), axis = 0)
    shift_amount = np.random.randint(-max_shift-5, max_shift + 5)
    augmented_whole_sequence = np.roll(sequence, shift_amount, axis = -1)
    
    augmented_sequence = augmented_whole_sequence[:-1, :]
    augmented_target = augmented_whole_sequence[-1,:]
    return np.expand_dims(augmented_sequence, -1), augmented_target

def random_masking(data,target, mask_prob=0.05, mask_value=-1):
    mask = np.random.rand(*data.shape) < mask_prob
    masked_sequence = np.where(mask, mask_value, data)
    return masked_sequence, target

def jittering(data,target, noise_factor=0.05):
    noise = noise_factor * np.random.randn(*data.shape)
    jittered_sequence = data + noise
    return jittered_sequence, target




class Fish(Dataset):       
    def __init__(self, data, label, mode, pred=[], probability=[]):
        super(Fish, self).__init__()
        data_filenames = [join(data, x) for x in listdir(data)]
        label_filenames = [join(label, x) for x in listdir(label)]
        data_value = [torch.load(data_filenames[x]) for x in range(len(data_filenames))]
        data_value = np.stack(data_value)
        targets = [torch.load(label_filenames[x]) for x in range(len(label_filenames))]
        self.mode = mode
        self.data = data_value
        self.targets = targets        

            
    def __getitem__(self, index):
        out_data = self.data[index, :, :]
        out_dim = out_data[:3, :]
        out_label = self.targets[index]
        
        if self.mode=='original':
            return out_dim, out_label
        
        elif self.mode=='trans':
            out_dim = np.expand_dims(out_dim, -1)#.transpose(1,2,0)
            out_dim,out_label = random_transformation(out_dim,out_label)
            out_dim = np.squeeze(out_dim, -1)

            return out_dim.astype(np.float32),out_label.astype(np.float32)
    
        
    def __len__(self):
        return len(self.data)
    
class fish_dataloader():  
    def __init__(self, data, label, batch_size, num_workers):
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def run(self,mode,pred=[],prob=[]):      
        if mode=='eval_train':
            eval_dataset = Fish(self.data, self.label, mode='trans')      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)          
            return eval_loader 
        if mode=='test':
            eval_dataset = Fish(self.data, self.label, mode='original')      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)          
            return eval_loader    
