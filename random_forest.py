# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 12:31:40 2021

@author: Hardik
"""
import idx2numpy 
from sklearn.ensemble import RandomForestClassifier

train_x_data = 'dataset/train-images.idx3-ubyte'
train_y_data = 'dataset/train-labels.idx1-ubyte'
test_x_data = 'dataset/t10k-images.idx3-ubyte'
test_y_data = 'dataset/t10k-labels.idx1-ubyte'

x_train=idx2numpy.convert_from_file(train_x_data)
x_test=idx2numpy.convert_from_file(test_x_data)
y_train=idx2numpy.convert_from_file(train_y_data)
y_test=idx2numpy.convert_from_file(test_y_data)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

nsamples = x_train.shape
x_train1 = x_train.reshape((nsamples[0], nsamples[1]*nsamples[2]))

ksamples = x_test.shape
x_test1 = x_test.reshape((ksamples[0], ksamples[1]*ksamples[2]))

rf=RandomForestClassifier(n_estimators=100, random_state = 10)
rf.fit(x_train1, y_train)

pred_1 = rf.predict(x_test1)

print(rf.score(x_test1, y_test))
