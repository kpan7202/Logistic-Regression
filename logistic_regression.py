# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 10:56:55 2017

@author: Kurniawan
"""

import pandas as pd
import numpy as np
import scipy.optimize as opt
import collections as col
import time
import random
from datetime import datetime

#load data into dataframe and preprocess to sort and set label number
def readTrainingData():
    X = pd.read_csv('training_data.csv', header = None)
    Y = pd.read_csv('training_labels.csv', header = None)
    #sort the data as the records and labels are not ordered 
    X = X.sort_values(0)
    Y = Y.sort_values(0)
    labels = col.Counter(Y[1]) #count the labels
    #add number for each label 
    Y.loc[:,2] = np.zeros(Y.shape[0],int)  
    i = 0
    labelList = []
    for label in labels:
        Y.iloc[np.where(Y[1] == label)[0],2] = i #add number index for each matched label name 
        labelList.append(label)
        i += 1
        
    return X, Y, labelList #X: DataFrame(r,c), Y: DataFrame(r,3), labelList: list - integer

#reduce dimentionality of features
def pca(X, threshold = 0.99):
    r, c = X.shape
    X = np.matrix(X)
    sigma = np.dot(X.T, X) / r #calculate covariance matrix
    U, s, V = np.linalg.svd(sigma) #use svd to get eigen values and eigen vectors
    #find k to get new dimension reduction number that still retain variances within threshold
    sumS = np.sum(s)
    k = len(s) - 1
    while k > 0:
        sumK = np.sum(s[:k])
        if (sumK / sumS) < threshold:
            break
        k -= 1
    #generate new features based on k
    Ur = U[:,:k]
    Z = np.dot(X, Ur)
    return Z, Ur

def sigmoid(z): #z: matrix(r, 1)-float
    return 1 / (1 + np.exp(-z))

def gradFunc(w, X, y, l = 0): #w: vector(c)-float, X: 2darray(r,c)-float , y: 2darray(r,1)-integer, l: float-lambda for regularization
    W = w.reshape(len(w),1) #make sure w is matrix(c,1) 
    h = sigmoid(np.dot(X, W))
    grad = np.dot(X.T, (h - y)) / len(y)
    #regularization
    W[0:0] = 0
    reg = l / len(y) * W
    return (grad + reg).T #transposed to matrix(1,c) so optimize function won't throw error
    
def costFunc(w, X, y, l = 0): #w: vector(c)-float, X: 2darray(r,c)-float , y: 2darray(r,1)-integer, l: float-lambda for regularization
    W = w.reshape(len(w),1) #make sure w is (c,1) matrix
    h = sigmoid(np.dot(X, W))
#    cost = np.mean(np.multiply(-y, np.log(h)) - np.multiply((1 - y), np.log(1 - h)))
    cost = float(np.dot(-y.T, np.ma.log(h)) - np.dot((1 - y.T), np.ma.log(1 - h))) / len(y)
    #regularization
    W[0:0] = 0
    reg = l / (2 * len(y)) * np.sum(np.power(W,2))
    return cost + reg

def gradientDescent(w, X, y, l = 0, alpha = 0.01, maxIter = 1000, e = 0.00001): #w: vector(c)-float, X: 2darray(r,c)-float , y: 2darray(r,1)-integer, l: float-lambda for regularization, alpha: learning rate, maxIter: maximum iteration, e: cost threshold
    i = 0
    minCost = []
    cont = True
    while(i < maxIter and cont):
       dW = gradFunc(w, X, y, l)
       cost = costFunc(w, X, y, l)
       minCost.append(cost)
       w += - alpha * dW.T
       if i > 0 and abs(minCost[i-1] - cost) < e:
           cont = False
       i += 1
       if i%20 == 0:
           print("Iteration %i: %f" % (i, cost))
           
    return w.T, minCost

def training(X, y, l = 0, n = 2): # X: 2darray(r,c)-float , y: 2darray(r,1)-integer from 0, l: float-lambda for regularization, n: integer-num of labels
    print("Start time: " + time.strftime("%d-%m-%Y %H:%M:%S", time.localtime()))
    n = 1 if n <=2 else n
    r, c = X.shape
    #convert data into matrix
    X = np.matrix(X)
    y = np.matrix(y)
    
    #X = np.insert(X, 0, 1, axis=1) #add 1 to index 0 of X
    X = np.hstack((np.ones((r,1)),X))
    wList = np.zeros((n,c + 1)) #store weights  
    for i in range(n):
        initW = np.zeros((c + 1,1)) # init weight     
        #result = gradientDescent(initW, X, (y == i) * 1, l, alpha = 0.1, maxIter = 200)
        #result = opt.fmin_tnc(costFunc, initW, fprime=gradFunc, args=(X, (y == i) * 1, l))
        result = opt.fmin_l_bfgs_b(costFunc, initW,maxiter=50,fprime=gradFunc, args=(X, (y == i) * 1, l))       
        #print("iteration {}: {}".format(i,result[2]['nit']))
        wList[i,:] = result[0]

    print("End time: " + time.strftime("%d-%m-%Y %H:%M:%S", time.localtime()))
    
    return wList

def predict(wList, X): #wList: array(f,c)-float, X: 2darray(r,c)-float
    r, c = X.shape
    # convert data into matrix
    W = np.matrix(wList)
    X = np.matrix(X)
    X = np.insert(X, 0, 1, axis=1)
    scores = np.array(np.dot(X, W.T)) #calculate the score
    y = np.array([np.argmax(scores, axis = 1), scores.max(1)]).T  #predict the score y - array of [label number, score]
    return y 

def measure(y, y1, n = 2): #y:array(r,1)-integer, #y1: array(r,1)-integer, n: integer - number of labels
    #convert into matrix
    y = np.matrix(y)
    y1 = np.matrix(y1)
    #construct confusion matrix
    confMatrix = np.zeros((n,n))
    for i in range(len(y)):
        actual = y[i,0]
        predicted = y1[i, 0]
        confMatrix[actual, predicted] += 1
    #calculate True Positive for each label       
    TP = []
    for i in range(n):
        TP.append(confMatrix[i,i])
    #calculate the performance
    accuracy = np.sum(TP) / np.sum(confMatrix)
    precision = np.mean(np.nan_to_num(np.array(TP) / np.sum(confMatrix, axis = 0)))
    recall = np.mean(np.nan_to_num(np.array(TP) / np.sum(confMatrix, axis = 1)))
    Fscore = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, Fscore, confMatrix

def crossValidation(X, y, l = 0, n = 2, folds = 1): #X: 2darray(r,c)-float , y: 2darray(r,1)-integer from 0, l: float-lambda for regularization, n: integer-num of labels, folds: integer
    r, c = X.shape
    if folds > 1:
        slices = int(r / folds) #find the size of test set
    else:
        slices = r
    
    performances = []
    indices = list(range(r))
    random.shuffle(indices) #randomize the order
    for i in range(folds):        
        if (slices != r): # decide to use cv or whole data
            testIdx = indices[i*slices:(i+1)*slices]
            trainIdx = np.delete(np.array(indices),range(i*slices,(i+1)*slices)).tolist()
            XTrain = X[trainIdx,:]
            yTrain = y[trainIdx,:]
            XTest = X[testIdx,:]
            yTest = y[testIdx,:]
        else:
            XTrain = X
            yTrain = y
            XTest = X
            yTest = y
            
        a1 = datetime.now()    
        W = training(XTrain, yTrain, l, n) # train training set to get weights
        b1 = datetime.now()
        c1=b1-a1
        print('Time taken for the fold '+ str(i) + ' is:' + str(c1.total_seconds()))
        
        yPredict = predict(W, XTest) #use weights to predict test set labels
        accuracy, precision, recall, Fscore, conf_matrix = measure(yTest, yPredict[:,:1].astype(int), n)
        df=pd.DataFrame(conf_matrix)
        mat_name='conf_matrix'+str(i)+'.csv'
        df.to_csv(mat_name)
        #print(conf_matrix)
        #compute the performance of prediction
        performances.append([accuracy, precision, recall, Fscore])
        print(performances)
    # average the performances    
    performances = np.mean(np.array(performances), axis = 0)
    return performances   
        
               
def test(W, labels): #W:2darray(class size, c)-float, labels: list of labels
    #read test data
    Xtest = pd.read_csv('test_data.csv', header = None)
    r, c  = Xtest.shape 
    yTest = predict(W, Xtest.iloc[:,1:]) #predict and calculate score
    #set the label and save the output
    output = pd.DataFrame()
    output.loc[:,0] = Xtest[0]
    for i in range(r):
        idx = int(yTest[i,0])
        output.loc[i,1] = labels[idx]
    output.to_csv("predicted_labels.csv", header = None, index = None)
    
if __name__ == "__main__":
    random.seed(1)
    X, y, labelList = readTrainingData()
#    ref = X[0]
#    X = X.iloc[:,1:]
#    Z = pca(X)

    # do cross validation
    l = [0, 0.01, 0.03, 0.1, 0.3] #try different lambda values
    performances = {}
    maxAcc = 0
    for i in l:
        performances[str(i)] = crossValidation(X.iloc[:,1:].values, y.iloc[:,2:].values, i, len(labelList), 10)
        print("Cross Validation Performance with regularization lambda: " + str(i))
        print("Accuracy: " + str(performances[str(i)][0]))
        print("Precision: " + str(performances[str(i)][1]))
        print("Recall: " + str(performances[str(i)][2]))
        print("Fscore: " + str(performances[str(i)][3]))
        if maxAcc < performances[str(i)][0]:
            performances["bestL"] = i
            maxAcc = performances[str(i)][0]
            
    # fit & predict using best lambda & all training data   
    W = training(X.iloc[:,1:].values, y.iloc[:,2:].values, performances["bestL"], len(labelList))
    y1 = predict(W, X.iloc[:,1:].values)
    accuracy, precision, recall, Fscore = measure(y.iloc[:,2:].values, y1[:,:1].astype(int), len(labelList))
    test(W, labelList)

    