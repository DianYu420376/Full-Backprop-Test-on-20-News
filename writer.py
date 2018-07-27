
# coding: utf-8

# In[ ]:

# Date:2018.07.21
# Author: Runyu Zhang


# In[13]:

from matplotlib import pyplot as plt
import torch


# In[21]:

class Writer:
    def __init__(self):
        self.scalar_dict = {}
        self.tensor_dict = {}
        
    def add_scalar(self, name, scalar):
        if self.scalar_dict.get(name) is None:
            self.scalar_dict[name] = [scalar]
        else:
            self.scalar_dict[name].append(scalar)
            
    def add_tensor(self, name, tensor):
        if self.tensor_dict.get(name) is None:
            self.tensor_dict[name] = [tensor]
        else:
            self.tensor_dict[name].append(tensor)
            
    def get(self, name):
        scalar =  self.scalar_dict.get(name)
        if scalar is not None:
            return scalar
        else:
            tensor = self.tensor_dict.get(name)
            if tensor is not None:
                return tensor
            else:
                print('No variable with name:', name, 'in the dictionary')
                
    def plot_scalar(self, name):
        lst = self.scalar_dict.get(name)
        plt.plot(lst)
        plt.show()
        
    def plot_tensor(self, name, idx_lst):
        tensor_lst = self.tensor_dict.get(name)
        for i in idx_lst:
            if i < len(tensor_lst):
                tensor = tensor_lst[i]
                fig = plt.figure(figsize = (15,105))
                plt.imshow(tensor.t())
                plt.show()
                
    def cat_lst(self, name, dim=0):
        data_lst =  self.scalar_dict.get(name)
        if data_lst is not None:
            cat_data = torch.cat(data_lst, dim)
            return cat_data
        else:
            data_lst = self.tensor_dict.get(name)
            if data_lst is not None:
                cat_data = torch.cat(data_lst, dim)
                return cat_data
            else:
                print('No variable with name:', name, 'in the dictionary')
        


# In[33]:

import numpy as np
S_lst = np.load('20_news_group_S_lst.npy')


# In[50]:

S_lst = [S_lst[i][0] for i in range(len(S_lst))]
history = Writer()
history.tensor_dict['output'] = S_lst


# In[52]:

S = history.cat_lst('output')


# In[53]:

S.shape


# In[ ]:



