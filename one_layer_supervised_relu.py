
# coding: utf-8

# In[1]:

'''
Date: 2018.07.30
Author: Cathy Zhang

Dataset: 20newsgroup with heading footing and quotes
Property: Fully supervised (should look into the 50% semi-supervised later on)
Network Structure: [m, 20] + linear + relu + linear
'''
# define global variables
save_PATH = 'saved_data/'
save_filename = 'one_layer_supervised_relu'


# In[2]:

# import package
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from deep_nmf import Deep_NMF, Energy_Loss_Func, Fro_Norm
from writer import Writer
from matplotlib import pyplot as plt
import numpy as np
from auxillary_functions import *
# load the dataset for twenty news
from twenty_news_group_data_loading import data, Y, target,L20, L50, L90, sparsedata_cr_entr, sparsedata_L2


# In[17]:

# defining a new nn module for this network structure
class Deep_NMF_Classifier(nn.Module):
    def __init__(self, depth_info, c1, c2):
        super(Deep_NMF_Classifier, self).__init__()
        self.deep_nmf = Deep_NMF(depth_info)
        self.classifier = Classifier(depth_info[-1], c1, c2)
    def forward(self, X):
        S_lst = self.deep_nmf(X)
        pred = self.classifier(S_lst)
        return S_lst, pred
    
class Classifier(nn.Module):
    def __init__(self, k, c1, c2):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(k, c1).double()
        self.linear2 = nn.Linear(c1, c2).double()
        
    def forward(self, S_lst):
        S = S_lst[-1]
        S1 = self.linear1(S)
        S1 = F.relu(S1)
        pred = self.linear2(S1)
        return pred


# In[20]:

m = data.shape[1]
depth_info = [m,20]
net = Deep_NMF_Classifier(depth_info, 100, 20)
lambd = 300
loss_func = Energy_Loss_Func(lambd = lambd, classification_type = 'L2')
data_input = data*1000
dataset = sparsedata_L2(data_input, Y)
criterion = Fro_Norm()


# In[ ]:

# training process
# setting training parameters
batchsize = 100
epoch = 4
lr_nmf = 7000
lr_cl = 10
loss_lst = []
# train!
for epo in range(epoch):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchsize, shuffle = True)
    total_loss = 0
    for (i, (inputs, label)) in enumerate(dataloader):
        inputs = inputs.view([inputs.shape[0], inputs.shape[2]])
        label = label.view([label.shape[0], -1])
        inputs, label = Variable(inputs), Variable(label)
        # train the linear classifier
        S_lst, pred = net(inputs)
        print('training the classifier')
        for k in range(100):
            net.zero_grad()
            S_lst_data = [S_lst[i].data for i in range(len(S_lst))]
            pred = net.classifier(S_lst_data)
            loss = criterion(pred, label)
            loss.backward()
            #print(loss.data)
            for A in net.classifier.parameters():
                A.data = A.data.sub_(lr_cl*A.grad.data)
        # train the lsqnonneg layers
        S_lst,pred = net(inputs)
        loss = loss_func(net, inputs, S_lst,pred,label)
        loss.backward()
        loss_lst.append(loss.data)
        total_loss += loss.data
        print('training the nmf layer')
        print(loss.data)
        for A in net.lsqnonneglst.parameters():
            A.data = A.data.sub_(lr_nmf*A.grad.data)
            A.data = A.data.clamp(min = 0)
    print('epoch = ', epo, '\n', total_loss)


# In[ ]:

# save the data
np.savez(save_PATH + save_filename,
         param_lst = list(net.parameters()), loss_lst = loss_lst)


# In[ ]:

# plot the loss curve
plt.plot(loss_lst)
plt.show()
# Get the whole output of the whole dataset (running forward propagation on the whole dataset)
S, pred = get_whole_output(net, dataset)
# Get the accuracy
accuracy = torch.sum(torch.argmax(pred, 1) 
                     == torch.argmax(torch.from_numpy(Y),1))/len(dataset)
print(accuracy)
# Get the reconstruction error
A_np = net.lsqnonneglst[0].A.data.numpy()
S_np = S.data.numpy()
fro_error, fro_X = calc_reconstruction_error(data_input, A_np, S_np)
print(fro_error/fro_X)


# In[ ]:

# save the data
np.savez(save_PATH + save_filename, S = S, pred = pred,
         param_lst = list(net.parameters()), loss_lst = loss_lst)

