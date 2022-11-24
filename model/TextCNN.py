import os
import numpy as np
import torch
import torch.nn as nn


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


class TextCNN(nn.Module):
    """
        针对情感极性分类的textCNN:暂定为5种size的kernel
    Args:
        emb_matrix: embedding matrix 
        max_len   : maximum length of sentenses
        class_num : the number of classes 
        hidden_num: the number of channels of convolution layer
    """
    
    def __init__(self, emb_matrix, max_len, class_num, hidden_num, drop_porb=0.5):
        super().__init__()
        
        self.emb_matrix = emb_matrix
        self.emb_len = emb_matrix.weight.shape[1]
        
        self.block1 = block(2, self.emb_len, max_len, hidden_num)
        self.block2 = block(3, self.emb_len, max_len, hidden_num)
        self.block3 = block(4, self.emb_len, max_len, hidden_num)
        self.block4 = block(5, self.emb_len, max_len, hidden_num)
        self.block5 = block(6, self.emb_len, max_len, hidden_num)
        
        self.fc = nn.Linear(hidden_num * 5, class_num)
        
        self.dropout = nn.Dropout(drop_porb)
        
        self.loss = nn.CrossEntropyLoss()
        
        
    def forward(self, batch_idx, batch_label=None):
        batch_emb = self.emb_matrix(batch_idx)
        
        res_b1 = self.block1(batch_emb)
        res_b2 = self.block2(batch_emb)
        res_b3 = self.block3(batch_emb)
        res_b4 = self.block4(batch_emb)
        res_b5 = self.block5(batch_emb)
        
        feature_vec = torch.cat((res_b1, res_b2, res_b3, res_b4, res_b5), dim=-1)
        
        pre = self.fc(feature_vec)
        pre = self.dropout(pre)
        
        if batch_label is not None:
            loss_val = self.loss(pre, batch_label)
            return loss_val
            
        else:
            predict = torch.argmax(pre, dim=-1)
            return predict
        
        
        