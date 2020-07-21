import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from skimage.color import gray2rgb

def load_data_mnnist():
    path="/media/hadi/laban/data_sets/keras/mnist/"
    
    xtrain=np.load(path+"x_train.npy")
    # xtrain=xtrain.reshape(xtrain.shape[0],xtrain.shape[1],xtrain.shape[2],1)
    xtrain=gray2rgb(xtrain)
    xtrain=np.asarray(xtrain,dtype=np.float64)

    xtest=np.load(path+"x_test.npy")
    # xtest=xtest.reshape(xtest.shape[0],xtest.shape[1],xtest.shape[2],1)
    xtest=gray2rgb(xtest)
    xtest=np.asarray(xtest,dtype=np.float64)
    
    ytrain=np.load(path+"y_train.npy")
    ytest=np.load(path+"y_test.npy")

    xtrain/=255
    xtest/=255
    return xtrain,ytrain,xtest,ytest

def transform_labels(y_train,y_test):
  y_train_test = np.concatenate((y_train,y_test),axis =0)
  encoder = LabelEncoder()
  new_y_train_test = encoder.fit_transform(y_train_test) 
  encoder = OneHotEncoder()
  new_y_train_test = encoder.fit_transform(new_y_train_test.reshape(-1,1))
  new_y_train = new_y_train_test[0:len(y_train)]
  new_y_test = new_y_train_test[len(y_train):]
  return new_y_train, new_y_test
