# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 12:09:03 2021

@author: 13kau
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Import the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header =None, engine='python', encoding= 'latin-1') 
# this dosent contain header So we specify header =None
# engine='python' --> dataset to work efficient


users =  pd.read_csv('ml-1m/users.dat', sep = '::', header =None, engine='python', encoding= 'latin-1')
# id, gender, age

ratings =  pd.read_csv('ml-1m/ratings.dat', sep = '::', header =None, engine='python', encoding= 'latin-1')
# users, movie_id, ratings

# Preparing the training set and the rest set!
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t') # has 80000 ratings
# users, movie_id, ratings

# convert training set to array
training_set = np.array(training_set, dtype='int')


# Prepare train and test set
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

# Getting the number of movies and users
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0]))) # max of max because to ge the max of both test, training!
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

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

# Converting the data into torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of neural Network
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        # Starts withe nb_movies and ends with nb movies... Because the input and output should have same number!
        self.fc1 = nn.Linear(nb_movies, 20) #20 neuron in 1nd hidden layer # Encode
        self.fc2 = nn.Linear(20, 10) #10 neuron in 2nd hidden layer # Encode
        self.fc3 = nn.Linear(10, 20) #10 neuron in 3nd hidden layer # Decode
        self.fc4 = nn.Linear(20, nb_movies) #10 neuron in 4nd hidden layer # Decode
        self.activation = nn.Sigmoid() # Activation Function
    
    # Action that takes place in Network.. ie. Encoding and decoding with applying activation
    def forward(self, x):
        x = self.activation(self.fc1(x)) # first encoding, ass activation on fc1 fcl(x) --> left of fc1
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x)) # first decoding, ass activation on fc1 fc3(x) --> left of fc3
        x = self.fc4(x)
        return x

sae = SAE()
criteronn = nn.MSELoss() # Loss
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5) # lr --> learning rate

# Training the SAE
nb_epochs = 200
for eopchs in range(1, nb_epochs + 1): #loop epochs
    train_loss = 0
    s = 0. # user rated atleast 1 movie '0.' because to calculate the root mean squared error
    # loop users
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0) # Variable().unsqueeze() --> Additional dimention for batch
        target = input.clone() # copy of the inputs
        # to save as much memor as possible.. ie. if there is user with 0 ratings then it is excluded
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criteronn(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data*mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: '+str(eopchs)+ 'loss:' +str(train_loss/s))

# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
  input = Variable(training_set[id_user]).unsqueeze(0)
  target = Variable(test_set[id_user]).unsqueeze(0)
  if torch.sum(target.data > 0) > 0:
    output = sae(input)
    target.require_grad = False
    output[target == 0] = 0
    loss = criteronn(output, target)
    mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
    test_loss += np.sqrt(loss.data*mean_corrector)
    s += 1.
print('test loss: '+str(test_loss/s))
        