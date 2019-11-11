CUDA_LAUNCH_BLOCKING="1"

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import OneHotEncoder, Normalizer, LabelEncoder
from sklearn.compose import ColumnTransformer,make_column_transformer 

import torchvision
from torchvision import transforms, datasets, models

from torch import Tensor
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

print("\n\n\n\n\nCuda available:",torch.cuda.is_available(),"\n\n\n\n\n")
use_cuda = True

################Dataset####################################################
data = pd.read_csv("/home/jui/Desktop/intrusion/kddcup.data_10_percent_corrected", names=['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
                'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root',
                'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
                'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
                'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
                'dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
                'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label'])

X = data.iloc[:,0:41]
y = data.iloc[:,41]
print('Shape of x:', X.shape)
data.head()
#####################################################################################





###############Data_Preprocessing###################################3       
categorical_features = ['protocol_type','service','flag','land','logged_in',\
                                        'is_host_login','is_guest_login']

tmp = np.setdiff1d(list(data.columns),categorical_features,[])
numerical_features = list(np.setdiff1d(tmp,['label']))

preprocess = make_column_transformer(\
    (numerical_features, Normalizer()),\
    (categorical_features, OneHotEncoder()))

X_new = preprocess.fit_transform(X)
y_new = np.array(np.multiply(y =='normal.',1))

print(X_new.shape)
print(y_new.shape,(y_new).mean())
######################################################################################




#############################Data_Splitting#####################################
xtrain, xtest, ytrain, ytest = tts(X_new, y_new, test_size = 0.2)
print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)
#####################################################################################


#######################Pytorch_DataLoader######################
class cyber_data(Dataset):

    def __init__(self,v1,v2):
        self.X = list(v1)
        self.y = list(v2)

    def __getitem__(self, id):
      # print('k')
      tmp1 = self.X[id]
      tmp2 = self.y[id]
      return tmp1, tmp2

    def __len__(self):
        return len(self.X)

traindata = cyber_data(xtrain, ytrain)
train_dataloader = DataLoader(dataset = traindata, batch_size= 128, shuffle=True)
######################################################################



#####################################DNN Class##########################
class dnn_class(nn.Module):
    
    def weight_init(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.out.weight)
    
        nn.init.xavier_uniform_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.bias)
        nn.init.xavier_uniform_(self.fc4.bias)
        nn.init.xavier_uniform_(self.out.bias)

    def __init__(self, din, dout, w1, w2, w3,w4):
        
        super(dnn_class, self).__init__()
        
        self.fc1= nn.Linear(din, w1)
        self.fc2= nn.Linear(w1, w2)
        self.fc3= nn.Linear(w2, w3)
        self.fc4= nn.Linear(w3, w4)
        self.out= nn.Linear(w4, dout)
    
    
    def forward(self, y):
        
        y = F.relu(self.fc1(y.float()))
        y = F.relu(self.fc2(y.float()))
        y = F.relu(self.fc3(y.float()))
        y = F.relu(self.fc4(y.float()))
        y = F.sigmoid(self.out(y))
        return y
###############################################################

  
dnn = dnn_class(121,1,512,256,128,64)
if use_cuda and torch.cuda.is_available():
    dnn.cuda()
learning_rate = 0.0001
bce = nn.BCELoss()
n_epochs = 10
optimizer_dnn = torch.optim.Adam(dnn.parameters(), lr=learning_rate, betas=(0.5, 0.999))

#####################Training########################
def training(data_loader, n_epochs):
    dnn.train()
    
    for en,(a,b) in enumerate(data_loader):

        
        a = Variable(a)
        b = Variable(b)

        if use_cuda and torch.cuda.is_available():
            a = a.cuda()
            b = b.cuda()

        
        b = b.type(torch.float)

        optimizer_dnn.zero_grad()
       
        preds = dnn(a)
        loss = bce(preds,b)
        loss.backward()
        optimizer_dnn.step()

        if(en%1000==0):
          print ("[Epoch: %d] [Iter: %d] [Loss: %f]" % (ep+1,en+1,loss.cpu().detach().numpy()))

for ep in range(n_epochs):
    
    training(train_dataloader, ep+1)
    torch.save(dnn, "dnn_{}.pth".format(ep+1))
#######################################################################



#########################Testing##################################
   
DNN = torch.load("dnn_9.pth")

a = np.array(xtest)
b = np.asfarray(ytest)
a = Variable(torch.from_numpy(a), volatile = True)

if use_cuda and torch.cuda.is_available():
    a = a.cuda()

Preds = DNN(a)
_ , Predictions = torch.max(Preds, 0)

print(abs(Predictions.cpu().numpy()-b).mean()*100)
#############################################################
