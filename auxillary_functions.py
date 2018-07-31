import torch
from writer import Writer
import numpy as np

def get_whole_output(net, dataset, param_lst = None):
    '''
    get the whole output for the huge dataset
    
    ---- Inputs:
    net: the net
    dataset: should be an object in the torch.utils.data.Dataset
    param_lst = None: specifies the initial value for net
    ---- Outputs:
    S
    pred (optional)
    '''
    for A in net.parameters():
        A.requires_grad = False
    history = Writer()
    # initialize the network with certain initial value
    if param_lst is not None:
        for (i,param) in enumerate(net.parameters()):
            param.data = param_lst[i]
    # start to forward propagate, 100 at a time
    n = len(dataset)
    if n%100 == 0:
        batch_num = n//100
    else:
        batch_num = n//100 + 1
    print('batch_num = ', batch_num, '\n')
    for i in range(batch_num):
        print('current at batch:', i)
        try:
            total_data = dataset[i*100:(i+1)*100]
            inputs = total_data[0]
        except:
            total_data = dataset[i*100:]
            inputs = total_data[0]
        total_output = net(inputs)
        if type(total_output) is tuple:
            output, pred = total_output
            history.add_tensor('output', output[0].data)
            history.add_tensor('pred', pred.data)
        else:
            output = total_output
            history.add_tensor('output', output.data)
    S_lst = history.get('output')
    S = torch.cat(S_lst,0)
    pred_lst = history.get('pred')
    for A in net.parameters():
        A.requires_grad = True
    if pred_lst is not None:
        pred_lst = history.get('pred')
        pred = torch.cat(pred_lst, 0)
        return S, pred
    else:
        return S


# In[6]:


def calc_reconstruction_error(X,A,S):
    '''
    Compute the relative reconstruction error
    
    ---- Inputs:
    X: sparse numpy matrix
    A: numpy array
    S: numpy array
    ---- Outputs:
    fro_error
    fro_X
    '''
    fro_error = 0
    fro_X = 0
    n = X.shape[0]
    if n%100 == 0:
        batch_num = n//100
    else:
        batch_num = n//100 + 1
    for i in range(batch_num):
        try:
            X_ = X[i*100:(i+1)*100].todense()      
            temp = np.linalg.norm(X_ - S[i*100:(i+1)*100]@A)
            temp_X = np.linalg.norm(X_)
        except:
            X_ = X[i*100:].todense()  
            temp = np.linalg.norm(X_ - S[i*100:]@A)
            temp_X = np.linalg.norm(X_)
        fro_error += temp**2
        fro_X += temp_X**2
    return fro_error, fro_X
