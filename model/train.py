from TextCNN import TextCNN
import torch 
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# 构建词库
def build_curpus(emb_len):
    pdata = pd.read_csv("../data/online_shopping_10_cats.csv")
    train_text = pdata['review']    
    
    word_to_index = {'<PAD>': 0, '<UNK>': 1}
    for text in train_text:
        if type(text) == float:
            continue
        for word in text:
            word_to_index[word] = word_to_index.get(word, len(word_to_index))
    
    emb_layer = nn.Embedding(len(word_to_index), emb_len)
    
    return word_to_index, emb_layer


class TextDataset(Dataset):
    def __init__(self, all_text, all_label, word_to_index, max_len):
        self.all_text = all_text
        self.all_label = all_label
        self.word_to_index = word_to_index
        self.max_len = max_len
        
        
    def __getitem__(self, index):
        if type(self.all_text[index]) == float:
            text_data = "@"
        else:
            text_data = self.all_text[index][:self.max_len]
        text_label = int(self.all_label[index])
        
        text_index = [self.word_to_index.get(i, 1) for i in text_data]
        # padding:
        text_index = text_index + [0] * (self.max_len - len(text_data))
       
        text_index = torch.tensor(text_index).unsqueeze(dim=0) 
        return text_index, text_label
    
    def __len__(self):
        return len(self.all_text)
    


if __name__ == '__main__':
    train_data = np.load("../data/计算机X.npy", allow_pickle=True)
    train_label = np.load("../data/计算机Y.npy", allow_pickle=True)
    
    max_len = 250
    hidden_num = 100
    emb_len = 100
    class_num = len(set(train_label))
    epoch = 100
    batch_size = 200
    lr = 0.001
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    word_2_index, emb_index = build_curpus(emb_len)
    train_dataset = TextDataset(train_data, train_label, word_2_index, max_len)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)
    
    model = TextCNN(emb_index, max_len, class_num, hidden_num).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    print("/***************************START TRAINING**************************/")
    
    for e in range(epoch):
        for batch_idx, batch_label in train_dataloader:
            batch_idx = batch_idx.to(device)
            batch_label = batch_label.to(device)
            
            loss = model.forward(batch_idx, batch_label)
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            print(f"loss : {loss:.3f} on epoch {e}") 
   
   
    print("/****************************START TESTING*************************/") 
    env = dict()
    env = {
        0:"书籍", 1:"计算机", 2:"酒店", 3:"蒙牛", 4:"平板", 5:"热水器", 6:"手机", 7:"水果", 8:"洗发水", 9:"衣服"
    }
    
    for e in range(10):
        print(f"starting test on env{e} : {env[e]}")
        test_data = np.load("../data/" + env[e] + "X.npy", allow_pickle=True)
        test_label = np.load("../data/" + env[e] + "Y.npy", allow_pickle=True)
        
        test_dataset = TextDataset(test_data, test_label, word_2_index, max_len)
        test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)
        
        right_num = 0
        for batch_idx, batch_label in test_dataloader:
            batch_idx = batch_idx.to(device)
            batch_label = batch_label.to(device)
            
            pre = model.forward(batch_idx)
            right_num += int(torch.sum(pre == batch_label))
            
        print(f"acc = {right_num/len(test_dataset)*100:.2f}%")
        
        
    print("Over")