
# coding: utf-8

# In[163]:


# How I get the 20 news groups dataset
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from matplotlib import pyplot as plt
import Ipynb_importer
from scipy.sparse import save_npz

data = fetch_20newsgroups_vectorized(subset = 'all')


target = data.target
target_names = data.target_names
data = data.data

save_npz('20news', data)


# In[164]:


# import package
import torch
from torch.autograd import Variable
import Ipynb_importer
from deep_nmf import Deep_NMF, Energy_Loss_Func
from writer import Writer


# In[165]:


import torch.utils.data
import torch


# In[166]:


# The matrix is tooo large if transformed into a torch tensor, so I have to do that in a stochastic way.
# Here I define a dataloader that will take the subset of the sparse numpy matrix and then only transform this subset into
# a torch Tensor, which will save memories

# class sparsedata(torch.utils.data.Dataset):
#     def __init__(self, data, label,transform = None):
#         self.data = data
#         self.label = label
#         self.transform = transform
#         self.len = data.shape[0]
#         assert(self.len == len(label))
#     def __getitem__(self, index):
#         inputs = self.data[index,:]
#         if self.transform is not None:
#             inputs = self.transform(inputs)
#         target = self.label[index]
#         inputs = torch.Tensor(inputs.todense()).double()
#         if type(index) == int:
#             target = torch.Tensor([target])
#         else:
#             target = torch.Tensor(target)
#         return inputs, target
#     def __len__(self):
#         return self.len


# In[167]:


# This is another dataset class for loading the semisupervised version for crossentropy criterion
class sparsedata_cr_entr(torch.utils.data.Dataset):
    def __init__(self, data, label, l = None, transform = None):
        self.data = data
        self.label = label
        self.transform = transform
        self.len = data.shape[0]
        self.l = l
        assert(self.len == len(label))
    def __getitem__(self, index):
        inputs = self.data[index,:]
        if self.transform is not None:
            inputs = self.transform(inputs)
        target = self.label[index]
        inputs = torch.Tensor(inputs.todense()).double()
        if type(index) == int:
            target = torch.Tensor([target]).long()
        else:
            target = torch.Tensor(target).long()
        if self.l is None:
            return inputs, target
        else:
            l = self.l[index,:]
            if type(index) == int:
                l = torch.Tensor([l]).double()
            else:
                l = torch.Tensor(l)
            return inputs, target, l.double()
    def __len__(self):
        return self.len


# In[173]:


# This is the dataset class for loading the semisupervised version for L2 criterion
class sparsedata_L2(torch.utils.data.Dataset):
    def __init__(self, data, Y, L = None, transform = None):
        self.data = data
        self.Y = Y
        self.transform = transform
        self.len = data.shape[0]
        self.L = L
        if L is not None:
            assert(self.len == L.shape[0])
        assert(self.len == Y.shape[0])
    def __getitem__(self, index):
        inputs = self.data[index,:]
        if self.transform is not None:
            inputs = self.transform(inputs)
        target = self.Y[index,:]
        inputs = torch.Tensor(inputs.todense()).double()
        if type(index) == int:
            target = torch.Tensor([target]).double()
        else:
            target = torch.Tensor(target).double()
        if self.L is None:
            return inputs, target
        else:
            L = self.L[index]
            if type(index) == int:
                L = torch.Tensor([L]).double()
            else:
                L = torch.Tensor(L).double()
            return inputs, target, L
    def __len__(self):
        return self.len


# In[169]:


# testing the dataset class
import scipy.io as sio
directory ='known_labels'
L = sio.loadmat(directory)
L20 = L.get('L20').T
L50 = L.get('L50').T
L90 = L.get('L90').T
l = L20[:,0]
directory_Y = 'Y'
Y = sio.loadmat(directory_Y)
Y = Y.get('Y').T


# In[170]:


# Defining the network structure
m = data.shape[1]
k1 = 200
k2 = 20
net = Deep_NMF([m,k2],20)
loss_func = Energy_Loss_Func(lambd = 1, classification_type = 'L2')


# In[159]:


# # Training process!

# # setting training parameters
# epoch = 5
# lr = 10
# loss_lst = []
# # train!
# for epo in range(epoch):
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size = 5, shuffle = True)
#     total_loss = 0
#     for (i, (inputs, label,l_batch)) in enumerate(dataloader):
#         t1 = time.time()
#         inputs = inputs.view([inputs.shape[0], inputs.shape[2]])
#         inputs, label = Variable(inputs), Variable(label)
#         S_lst,pred = net(inputs)
#         loss = loss_func(net, inputs, S_lst,pred,label.view([label.shape[0], -1]),l_batch.view([l_batch.shape[0], -1]))
#         loss.backward()
#         loss_lst.append(loss.data)
#         t2 = time.time()
#         total_loss += loss.data
#         print(loss.data)
#         for A in net.parameters():
#             A.data = A.data.sub_(lr*A.grad.data)
#             A.data = A.data.clamp(min = 0)
#     print('epoch = ', epo, '\n', total_loss)


# In[162]:


def get_whole_output(net, dataset, param_lst = None):
    history = Writer()
    # initialize the network with certain initial value
    if param_lst is not None:
        for (i,param) in enumerate(net.parameters()):
            param.data = param_lst[i]
    # start to forward propagate, 100 at a time
    n = len(dataset)
    if n%100 == 0:
        batch_num = n/100
    else:
        batch_num = n//100 + 1
    print('batch_num = ', batch_num, '\n')
    for i in range(batch_num):
        print('current at batch:', i)
        try:
            (inputs, label) = dataset[i*100:(i+1)*100]
        except:
            (inputs, label) = dataset[i*100:]
        history.add_tensor('label', label)
        output = net(inputs)
        history.add_tensor('output', output)
    return history

