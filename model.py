import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import math
from scipy.stats import entropy
from collections import Counter

def crop_conc(x1, x2):
    crop_x2 = x2[:,:,:x1.size()[2]]
    return torch.cat((x1, crop_x2), 1)


def manually_pad(x, dim):
    tmp = torch.zeros(x.size()[0], x.size()[1], dim, device=x.device)
    tmp[:, :, :x.size()[2]] = x[:, :, :x.size()[2]]
    return tmp

def mean_absolute_deviation_from_mode(data):
    counter = Counter(data)
    mode = counter.most_common(1)[0][0]
    
    mad = sum(abs(x - mode) for x in data) / len(data)
    return mad

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5, bidirectional=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        return out

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return moving_mean, res

    
class Model(nn.Module):
    def __init__(self, in_dim = 3, num_classes=10, in_len = 10000):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.in_dim = in_dim
        self.softmax = nn.Softmax(dim = -1)
        self.in_planes = 4
        self.in_len = in_len
        self.linear_x = nn.Linear(1, num_classes)
        self.linear_s = nn.Linear(1, 2)
        self.projection = nn.Linear(num_classes, num_classes)
        kernel_size = 25
        self.decomposition = series_decomp(kernel_size)
        
        self.enc_1 = nn.Sequential(
            nn.Conv1d(in_dim, 16, 5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Conv1d(16, 16, 5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.enc_2 = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.enc_3 = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.enc_4 = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.enc_5 = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Conv1d(256, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.decoder_b1 = nn.Sequential(
            nn.Upsample(625),
            nn.Conv1d(256, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.decoder_b2 = nn.Sequential(
            nn.Conv1d(256, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(2500),
            nn.Conv1d(128, 64, 6, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.decoder_b3 = nn.Sequential(
            nn.Conv1d(128, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Upsample(5000),
            nn.Conv1d(64, 32, 8, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.decoder_b4 = nn.Sequential(
            nn.Conv1d(64, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Upsample(10000),
            nn.Conv1d(32, 16, 10, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.decoder_b5 = nn.Sequential(
            nn.Conv1d(32, 16, 5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Conv1d(16, 16, 5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.enc_1s = nn.Sequential(
            nn.Conv1d(in_dim, 16, 5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Conv1d(16, 16, 5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.enc_2s = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.enc_3s = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.enc_4s = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.enc_5s = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Conv1d(256, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.decoder_b1s = nn.Sequential(
            nn.Upsample(625),
            nn.Conv1d(256, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.decoder_b2s = nn.Sequential(
            nn.Conv1d(256, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(2500),
            nn.Conv1d(128, 64, 6, padding=2), 
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.decoder_b3s = nn.Sequential(
            nn.Conv1d(128, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Upsample(5000),
            nn.Conv1d(64, 32, 8, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.decoder_b4s = nn.Sequential(
            nn.Conv1d(64, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Upsample(10000),
            nn.Conv1d(32, 16, 10, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.decoder_b5s = nn.Sequential(
            nn.Conv1d(32, 16, 5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Conv1d(16, 16, 5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        
        self.segment_classifier = nn.Sequential(
            nn.Conv1d(16, num_classes, 1),
            nn.BatchNorm1d(num_classes),
            nn.Tanh(),

            nn.ConstantPad1d(0, 0), 
        )
        
        self.segment_classifiers = nn.Sequential(
            nn.Conv1d(16, 2, 1),
            nn.BatchNorm1d(2),
            nn.Tanh(),

            nn.ConstantPad1d(0, 0),
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv1d(num_classes, num_classes, 1),
        )
        self.final_convs = nn.Sequential(
            nn.Conv1d(2, 2, 1),
        )
        
        self.features_t = nn.Sequential(
            nn.Conv1d(in_dim, 64, 400, 50),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(),
            nn.Conv1d(64, 128, 8),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 8),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 8),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.features_p = nn.Sequential(
            nn.Conv1d(in_dim, 64, 400, 50),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(),
            nn.Conv1d(64, 128, 8),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 8),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 8),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        self.lstm_t = nn.Sequential(
            BiLSTM(3328, 128, 2),
        )
        #         self.lstm_p = nn.Sequential(
        #             BiLSTM(12288, 512, 2),
        #         )
        self.out_x = nn.Linear(256, in_len)
        self.out_s = nn.Linear(1664, in_len)
        self.out = nn.Linear(num_classes, num_classes)

    
    def forward(self, x, mad_diver = []):
        x, s = self.decomposition(x)
        enc_1 = self.enc_1(x)
        enc_2 = self.enc_2(enc_1)
        enc_3 = self.enc_3(enc_2)
        enc_4 = self.enc_4(enc_3)
        enc_5 = self.enc_5(enc_4)

        x = self.decoder_b1(enc_5)
        x = crop_conc(x, enc_4) 
        x = self.decoder_b2(x)
        x = manually_pad(x, 2500)
        x = crop_conc(x, enc_3)
        x = self.decoder_b3(x)
        x = manually_pad(x, 5000)
        x = crop_conc(x, enc_2)
        x = self.decoder_b4(x)
        x = manually_pad(x, 10000)
        x = crop_conc(x, enc_1)
        x = self.decoder_b5(x)

        enc_1s = self.enc_1s(s)
        enc_2s = self.enc_2s(enc_1s)
        enc_3s = self.enc_3s(enc_2s)
        enc_4s = self.enc_4s(enc_3s)
        enc_5s = self.enc_5s(enc_4s)

        s = self.decoder_b1s(enc_5s)
        s = crop_conc(s, enc_4s) 
        s = self.decoder_b2s(s)
        s = manually_pad(s, 2500)
        s = crop_conc(s, enc_3s)
        s = self.decoder_b3s(s)
        s = manually_pad(s, 5000)
        s = crop_conc(s, enc_2s)
        s = self.decoder_b4s(s)
        s = manually_pad(s, 10000)
        s = crop_conc(s, enc_1s)
        s = self.decoder_b5s(s)
        
        x = self.segment_classifier(x)
        s = self.segment_classifiers(s)
        
        out_x = self.final_conv(x)
        out_s = self.final_convs(s)
        
        unimulti_label = torch.argmax(self.softmax(out_x), dim = -1) 
        mad = []
        for i in range(out_x.shape[0]):
            label_counts = {label: np.sum(unimulti_label[i].cpu().numpy() == label) for label in np.unique(unimulti_label[i].cpu().numpy())}
            mad_ = mean_absolute_deviation_from_mode(label_counts)
            mad.append(mad_)
        
        if len(mad_diver) != 0:
            use_mad = mad_diver
        else: 
            use_mad = mad

        if  use_mad[0] > 5: 
            _, seg_ = torch.max(out_s, 1) 
            out = self.projection((out_x + seg_.unsqueeze(dim = 1)).permute(0,2,1))#.permute(0,2,1)
        else:
            out = self.projection(out_x.permute(0,2,1))
        out_s = out_s.permute(0,2,1)
        return out, out_s, mad

    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')             