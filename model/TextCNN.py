import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from sklearn.decomposition import PCA

class block(nn.Module):
    """
    卷积(kernel_size * embedding_len) -> 激活(ReLU) -> 采样(MaxPooling)
    Args:
        kernel_size  : conv kernel size
        emb_len      : embedding length
        max_len      : maximum length of sentenses
        hidden_num   : the number of channel
    """
    def __init__(self, kernel_size, emb_len, max_len, hidden_num):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels=1, out_channels=hidden_num, kernel_size=(kernel_size, emb_len))
        self.act = nn.ReLU()
        self.mxp = nn.MaxPool1d(kernel_size=max_len - kernel_size + 1, stride=1)
        
    def forward(self, batch_emb):
        res_layer1 = self.cnn.forward(batch_emb)
        res_layer2 = self.act.forward(res_layer1)
        res_layer2 = res_layer2.squeeze(-1)
        res_layer3 = self.mxp.forward(res_layer2)
        res_layer3 = res_layer3.squeeze(-1) 
        return res_layer3

class Feature_Extractor(nn.Module):
    def __init__(self, emb_matrix, max_len, hidden_num):
         super().__init__()
        
         self.emb_matrix = emb_matrix
         self.emb_len = emb_matrix.weight.shape[1]
         
         self.block1 = block(2, self.emb_len, max_len, hidden_num)
         #self.block2 = block(3, self.emb_len, max_len, hidden_num)
         self.block3 = block(4, self.emb_len, max_len, hidden_num)
         #self.block4 = block(5, self.emb_len, max_len, hidden_num)
         self.block5 = block(6, self.emb_len, max_len, hidden_num)

    def forward(self, batch_idx):
        batch_emb = self.emb_matrix(batch_idx)
        
        res_b1 = self.block1(batch_emb)
        #res_b2 = self.block2(batch_emb)
        res_b3 = self.block3(batch_emb)
        #res_b4 = self.block4(batch_emb)
        res_b5 = self.block5(batch_emb)
        
        feature_vec = torch.cat((res_b1,  res_b3,  res_b5), dim=-1)
        
        return feature_vec
class classifier(nn.module):
    def __init__(self, hidden_num, channels, class_num, drop_prob=0.5):
        super().__init__()
        
        self.input_dim = hidden_num * channels
        self.output_dim = class_num
        
        self.fc = nn.Linear(self.input_dim, self.output_dim)
        self.dropout = nn.Dropout(drop_prob)
        
    def forward(self, feature_vec):
        predict = self.fc(feature_vec)
        predict = self.dropout(predict)
        return predict
    
class TextCNN(nn.Module):
    """
        针对情感极性分类的textCNN:暂定为5种size的kernel
    Args:
        emb_matrix: embedding matrix 
        max_len   : maximum length of sentenses
        class_num : the number of classes 
        hidden_num: the number of channels of convolution layer
    """
    
    def __init__(self, feature_extractor, classifier):
        super().__init__()
        
        self.feature_learn = feature_extractor
        self.classifier = classifier
        
        
    def forward(self, batch_idx, pre_train=True):
        
        feature_vec = self.feature_learn(batch_idx)
        
        predict = self.classifier(feature_vec)
        
        if pre_train:
            predict = f.softmax(predict)
        
        return predict
        

     
        