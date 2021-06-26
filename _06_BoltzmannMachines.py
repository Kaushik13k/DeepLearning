#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


# Import the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header =None, engine='python', encoding= 'latin-1') 
# this dosent contain header So we specify header =None
# engine='python' --> dataset to work efficient


movies.head()


users =  pd.read_csv('ml-1m/users.dat', sep = '::', header =None, engine='python', encoding= 'latin-1')
users.head()
# id, gender, age

ratings =  pd.read_csv('ml-1m/ratings.dat', sep = '::', header =None, engine='python', encoding= 'latin-1')
ratings.head() 
# users, movie_id, ratings


# Prepare train and test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t') # has 80000 ratings


training_set.head()
# users, movie_id, ratings


# convert training set to array
training_set = np.array(training_set, dtype='int')


# Prepare train and test set
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')
test_set


# Getting the number of movies and users
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0]))) # max of max because to ge the max of both test, training!
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))
print(nb_users)
print(nb_movies)


# Converting the data into an array with users in lines and movies in column
def convert(data):
    new_data = [] #we do list of list, 1st list for 1st user, 2nd list for 2nd user etc
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:,0] == id_users] # movies indexs of all rated movies
        id_ratings = data[:, 2][data[:,0] == id_users]# 1st list --> ratings of the 1st user and so on
        ratings = np.zeros(nb_movies) # list of nb_movies list
        ratings[id_movies - 1] = id_ratings   # Replace 0 by 1 , '-1' -->  because we have index starting from 0, 'id_ratings' --> update with real ratings
        new_data.append(list(ratings)) # for all the users
    return new_data

training_set = convert(training_set)

test_set = convert(test_set)


# Converting the data into torch tenosrs
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Convert the ratings into binary rating 1 (liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1


# creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh): #nv --> number of visible nodes, nh --> number of hidden nodes
        self.W = torch.randn(nh, nv)   #W --> weights
        self.a = torch.randn(1, nh ) #bias for hidden nodes, '1, nh, ' --> to make 2D
        self.a = torch.randn(1, nv ) #bias for visible nodes, '1, nh, ' --> to make 2D
    
    def sample_h(self, x): # probability for hidden node got like or no
        wx = torch.mm(x, self.W.t()) ## mm -> product, t--> transpose
        activation = wx + self.a.expand_as(wx) # expand_as -> bias is applied to each batch
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y): # probability for visible node got like or no
        wy = torch.mm(y, self.W) ## mm -> product, t--> transpose
        activation = wy + self.b.expand_as(wy) # expand_as -> bias is applied to each batch
        p_v_given_v = torch.sigmoid(activation)
        return p_v_given_v, torch.bernoulli(p_v_given_v)
    
    # k-step Contrastive divergence
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
        
nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)


"""## Training the RBM"""

nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
  train_loss = 0
  s = 0.
  for id_user in range(0, nb_users - batch_size, batch_size):
    vk = training_set[id_user : id_user + batch_size]
    v0 = training_set[id_user : id_user + batch_size]
    ph0,_ = rbm.sample_h(v0)
    for k in range(10):
      _,hk = rbm.sample_h(vk)
      _,vk = rbm.sample_v(hk)
      vk[v0<0] = v0[v0<0]
    phk,_ = rbm.sample_h(vk)
    rbm.train(v0, vk, ph0, phk)
    train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
    s += 1.
  print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

"""## Testing the RBM"""

test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))