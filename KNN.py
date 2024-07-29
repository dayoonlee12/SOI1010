# -*- coding: utf-8 -*-

import numpy as np
import torch
from torchvision import datasets
trainset = datasets.MNIST ( root ="./ data", train =True ,
download = True )
testset = datasets.MNIST ( root ="./ data", train =False ,
download = True )

# Indices for train /val splits : train_idx , valid_idx
np. random . seed (0)
val_ratio = 0.1 #10% of training data will be splited for validation data
train_size = len( trainset ) #100
indices = list ( range ( train_size )) #list of idx that represents traning examples
split_idx = int(np. floor ( val_ratio * train_size )) #number of validation data->10
np. random . shuffle ( indices )
train_idx , val_idx = indices [ split_idx :], indices [: split_idx ]
train_data = trainset . data [ train_idx ]. float ()/255. #normalizing image data of training set
#to be between 0 and 1
train_labels = trainset . targets [ train_idx ] #corresponding label for train_data
val_data = trainset . data [ val_idx ]. float ()/255.
val_labels = trainset . targets [ val_idx ]
test_data = testset . data . float ()/255.
test_labels = testset . targets

"""# **a. Implement an iterative method (using for loop) to classify a single new example. Write down your observations.**"""

def predict_iterative(k,x):
    distances = torch.zeros(len(train_data))
    for i in range(len(train_data)):
        distances[i] = torch.sqrt(torch.sum((train_data[i]-x) ** 2))
    k_indices = torch.argsort(distances)[:k] #indices of KNN
    k_nearest_labels = train_labels[k_indices]
    pred = torch.bincount(k_nearest_labels).argmax().item()
    return pred

import matplotlib.pyplot as plt
new_example = test_data[3]
plt.title(test_labels[3].item())
plt.imshow(new_example.squeeze(),cmap = "gray")

import time
start = time.time()
prediction1 = predict_iterative(5,new_example)
end = time.time()
print(f"predicted label:{prediction1}")
print(f"iterative method took {end-start} sec.")

"""# **b. Use the broadcasting concept you learned in the laboratory session to classify a single new example. Compare against the result from (a).**




"""

def predict_new(k,x):
    #x:a single new example
    distances = torch.sqrt(torch.sum((train_data-x) ** 2,dim=(1,2)))
    #train_data.shape->torch.Size([54000,28,28])
    #a single new example x->torch.Size([28,28])->braodcasting used here
    k_indices = torch.argsort(distances)[:k] #indices of KNN
    k_nearest_labels = train_labels[k_indices]
    pred = torch.bincount(k_nearest_labels).argmax().item()
    return pred

import matplotlib.pyplot as plt
new_example = test_data[3]
plt.title(test_labels[3].item())
plt.imshow(new_example.squeeze(),cmap = "gray")

import time
start = time.time()
prediction2 = predict_new(5,new_example)
end = time.time()
print(f"predicted label:{prediction2}")
print(f"With broadcasting,it took {end-start} sec.")

"""# **(c) Now, implement a k-NN algorithm (starting with k=5) and its training/validation/evaluation code to perform multiclass classification over all digits, using the implementation from (b). Write down your observations.**"""

def euclidean_distance(train_data,x):
  distances = torch.sqrt(torch.sum((train_data-x)**2,dim=(1,2)))
  return distances

def manhattan_distance(train_data,x):
  distances = torch.sqrt(torch.sum(torch.abs(train_data-x),dim=(1,2)))
  return distances

def cosine_similarity(train_data,x):
  numerator = torch.sum(train_data*x,dim=(1,2))#분자 #(54000)
  norm_traindata = torch.sqrt(torch.sum(train_data**2,dim=(1,2)))
  norm_x = torch.sqrt(torch.sum(x**2))
  denominator = norm_traindata*norm_x #(54000)
  similarity = numerator/denominator #element-wise division->cosine_similarity for x and each train_data
  return similarity

def minkowsky(train_data,x,p):
  distances = torch.sum((train_data-x)**p,dim=(1,2))**(1/p)
  return distances

class KNearestNeighbors:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self,X_test):
        pred = torch.zeros(len(X_test))
        for i,x in enumerate(X_test):
          distances = torch.sqrt(torch.sum((self.X_train-x)**2,dim=(1,2)))
          k_indices = torch.argsort(distances)[:self.k] #indices of KNN
          k_nearest_labels = train_labels[k_indices]
          pred_single = torch.bincount(k_nearest_labels).argmax().item()
          pred[i] = pred_single
          print(i) #to check process
        return pred

    def predict_Manhattan(self,X_test):
        pred = torch.zeros(len(X_test))
        for i,x in enumerate(X_test):
          distances = manhattan_distance(self.X_train,x)
          k_indices = torch.argsort(distances)[:self.k] #indices of KNN
          k_nearest_labels = train_labels[k_indices]
          pred_single = torch.bincount(k_nearest_labels).argmax().item()
          pred[i] = pred_single
          print(i) #to check process
        return pred

    def predict_cosine(self,X_test):
        pred = torch.zeros(len(X_test))
        for i,x in enumerate(X_test):
          distances = cosine_similarity(self.X_train,x)
          k_indices = torch.argsort(distances,descending = True)[:self.k] #indices of KNN
          k_nearest_labels = train_labels[k_indices]
          pred_single = torch.bincount(k_nearest_labels).argmax().item()
          pred[i] = pred_single
          print(i) #to check process
        return pred

    def predict_Minkowsky(self,X_test,p):
        pred = torch.zeros(len(X_test))
        for i,x in enumerate(X_test):
          distances = minkowsky(self.X_train,x,p)
          k_indices = torch.argsort(distances)[:self.k] #indices of KNN
          k_nearest_labels = train_labels[k_indices]
          pred_single = torch.bincount(k_nearest_labels).argmax().item()
          pred[i] = pred_single
          print(i) #to check process
        return pred



    def get_accuracy(self,prediction,y_test):
          correct = torch.sum(prediction==y_test).item()
          accuracy = correct/len(y_test)
          print(f"{correct}/{len(y_test)} -> accuracy:{accuracy}")
          return accuracy

    def change_k(self,k):
          self.k = k

#k = 5, accuracy on val_data
knn = KNearestNeighbors(5)
knn.fit(train_data, train_labels)
predicted_val = knn.predict(val_data)
evaluated_score = knn.get_accuracy(predicted_val,val_labels)

#k = 5, error rate
wrong = []
for i in range(len(val_data)):
  prediction = predicted_val[i].item()
  answer = val_labels[i]
  if(prediction != answer):
    wrong.append(i)
    print(f"val_data[{i}] -> prediction: {prediction} but answer:{answer} ")
num_of_wrong = len(wrong)
error_rate = num_of_wrong/len(val_data)
print(f"error rate: {error_rate}")

# Validation: to find optimal K->k that has the highest accuracy(lowest error rate) on val_data
l = []
knn = KNearestNeighbors()
knn.fit(train_data, train_labels)
optimal_k = 0
best_accuracy = 0
for k in range(7,16,2):#k->7,9,11,13,15
  knn.change_k(k)
  predicted_val = knn.predict(val_data) #validation
  accuracy = knn.get_accuracy(predicted_val,val_labels) #evaluation
  l.append((k,accuracy))
  if(accuracy > best_accuracy):
      best_accuracy = accuracy
      optimal_k = k
print(l)
print("--")
print(f"best accuracy:{best_accuracy} when k = {optimal_k}")

new_tuple = (5,5836/6000)
l.insert(0, new_tuple)
# Print the updated list
print(l)

import matplotlib.pyplot as plt

x = [item[0] for item in l]
y = [item[1] for item in l]

# Label the axes
plt.xlabel("k")
plt.ylabel("Accuracy")

plt.plot(x,y,color = 'green', linestyle = 'dashed', lw = 3,marker = 'o', markerfacecolor = 'red', markersize = 9)


# Show the plot
plt.show()

"""# **(d) Improve the algorithm from (c) [Hint: Try to find the desirable distance function, which can be found by googling or going through PyTorch document].**"""

knn = KNearestNeighbors(5)
knn.fit(train_data, train_labels)
predicted_val = knn.predict_cosine(val_data) #validation
evaluated_score = knn.get_accuracy(predicted_val,val_labels) #evaluation

"""# **(f) Try at least two other options for each hyperparameter. Report the performance foreach option.**"""

#k = 7
knn = KNearestNeighbors(7)
knn.fit(train_data, train_labels)
predicted_val = knn.predict_cosine(val_data) #validation
evaluated_score = knn.get_accuracy(predicted_val,val_labels) #evaluation

#k = 13
knn = KNearestNeighbors(13)
knn.fit(train_data, train_labels)
predicted_val = knn.predict_cosine(val_data) #validation
evaluated_score = knn.get_accuracy(predicted_val,val_labels) #evaluation

#Manhanttan distance
knn = KNearestNeighbors(5)
knn.fit(train_data, train_labels)
predicted_val = knn.predict_Manhattan(val_data) #validation
evaluated_score = knn.get_accuracy(predicted_val,val_labels) #evaluation

#minkwosky distance with p = 3
knn = KNearestNeighbors(5)
knn.fit(train_data, train_labels)
predicted_val = knn.predict_Minkowsky(val_data,3)
evaluated_score = knn.get_accuracy(predicted_val,val_labels) #evaluation

"""# **(g) You can try more options if you want. What is the final test accuracy?**"""

#1. Euclidean: k = 5 was chosen as valid hyperparameter in (c)
knn = KNearestNeighbors(5)
knn.fit(train_data, train_labels)
predicted_val = knn.predict(test_data)
evaluated_score = knn.get_accuracy(predicted_val,test_labels)

# To set optimal K->k that has the highest accuracy(lowest error rate) on val_data
l = []
knn = KNearestNeighbors()
knn.fit(train_data, train_labels)
optimal_k = 0
best_accuracy = 0
for k in range(7,16,2):#k->7,9,11,13,15
  knn.change_k(k)
  predicted_val = knn.predict_cosine(val_data)
  accuracy = knn.get_accuracy(predicted_val,val_labels)
  l.append((k,accuracy))
  if(accuracy > best_accuracy):
      best_accuracy = accuracy
      optimal_k = k
print(l)
print("--")
print(f"best accuracy:{best_accuracy} when k = {optimal_k}")

#cosine similarity
new_tuple = (5,5881/6000)
l.insert(0, new_tuple)

x = [item[0] for item in l]
y = [item[1] for item in l]

# Label the axes
plt.xlabel("k")
plt.ylabel("Accuracy")

plt.plot(x,y,color = 'green', linestyle = 'dashed', lw = 3,marker = 'o', markerfacecolor = 'red', markersize = 9)


# Show the plot
plt.show()

knn = KNearestNeighbors(5)
knn.fit(train_data, train_labels)
predicted_val = knn.predict_cosine(test_data)
evaluated_score = knn.get_accuracy(predicted_val,test_labels)