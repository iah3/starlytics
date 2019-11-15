import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import json
import numpy as np
from scipy import sparse
train_or_test = 'test'
filestart = 0
filename = 'gvf-protoss-'+str(filestart)+'-1000-'+train_or_test+'.npy'
y_test, x_test = np.split(np.load(filename),[1],axis=1)
y_test = np.ravel(y_test)

filename = 'model.sav'
clf = joblib.load(filename)

accuracy = sum(sum([clf.predict(x_test) == y_test])) / len(y_test)
print ('testing accuracy: ' +str(accuracy))