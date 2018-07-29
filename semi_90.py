
# coding: utf-8

# In[3]:


'''
Training a semi supervised one layer NMF on 20 news group dataset, with 90% of observing data
'''


# In[3]:


# import package
import torch
from torch.autograd import Variable
from deep_nmf import Deep_NMF, Energy_Loss_Func
from writer import Writer
from matplotlib import pyplot as plt
import numpy as np


# In[4]:


# load the dataset for twenty news
from twenty_news_group_data_loading import data, Y, L20, L50, L90, sparsedata_cr_entr, sparsedata_L2#, get_whole_output


# In[5]:


# Define the network
m = data.shape[1]
k = 20
c = 20
lambd = 100
net = Deep_NMF([m, k], c)
loss_func = Energy_Loss_Func(lambd = lambd, classification_type = 'L2')
dataset = sparsedata_L2(data*1000, Y, L90)


# In[17]:


# Training process!

# setting training parameters
batchsize = 100
epoch = 7
lr_nmf = 5000
lr_cl = 0.1
loss_lst = []
total_loss_lst = []
# train!
for epo in range(epoch):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchsize, shuffle = True)
    total_loss = 0
    for (i, (inputs, label,l_batch)) in enumerate(dataloader):
        # train the lsqnonneg layers
        inputs = inputs.view([inputs.shape[0], inputs.shape[2]])
        label = label.view([label.shape[0], -1])
        l_batch = l_batch.view([l_batch.shape[0],-1])
        inputs, label = Variable(inputs), Variable(label)
        S_lst,pred = net(inputs)
        loss = loss_func(net, inputs, S_lst,pred,label,l_batch)
        loss.backward()
        loss_lst.append(loss.data)
        total_loss += loss.data
        print('training the nmf layer')
        print(loss.data)
        for A in net.lsqnonneglst.parameters():
            A.data = A.data.sub_(lr_nmf*A.grad.data)
            A.data = A.data.clamp(min = 0)
            A.requires_grad = False
        # train the linear classifier
        print('training the classifier')
        for k in range(10):
            net.zero_grad()
            S_lst,pred = net(inputs)
            loss = loss_func(net, inputs, S_lst,pred,label,l_batch)
            loss.backward()
            print(loss.data)
            for A in net.linear.parameters():
                A.data = A.data.sub_(lr_cl*A.grad.data)
        for A in net.lsqnonneglst.parameters():
            A.requires_grad = True
    print('epoch = ', epo, '\n', total_loss)
    total_loss_lst.append(total_loss)


# In[1]:


plt.plot(total_loss_lst)
plt.show()


# In[6]:


# SAVE YOUR RESULT!!!!!
A = net.lsqnonneglst[0].A.data
B = net.linear.weight.data
A_np = A.numpy()
B_np = B.numpy()
np.save('semi_90_A', A_np)
np.save('semi_90_B', B_np)


# In[20]:


# Doing forward propagation on the whole dataset, remember to SAVE S and prod!
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
            (inputs, label, l_batch) = dataset[i*100:(i+1)*100]
        except:
            (inputs, label, l_batch) = dataset[i*100:]
        history.add_tensor('label', label)
        output, pred = net(inputs)
        history.add_tensor('output', output)
        history.add_tensor('pred', pred)
    return history


