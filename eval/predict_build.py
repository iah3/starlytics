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

x_test1, y_test, x_test = np.split(np.load('gvf-protoss-xy-s1000-train.npy'),[1,2],axis=1)
y_test = np.ravel(y_test)

print('loaded data')

filename = 'model_build.sav'
clf = joblib.load(filename)

print('loaded model')

accuracy = sum(sum([clf.predict(x_test) == y_test])) / len(y_test)
print ('testing accuracy: ' + str(accuracy))
