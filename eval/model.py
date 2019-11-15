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
train_or_test = 'train'
filestart = 0
filename = 'gvf-protoss-'+str(filestart)+'-1000-'+train_or_test+'.npy'
y_train, x_train = np.split(np.load(filename),[1],axis=1)
y_train = np.ravel(y_train)

clf = RandomForestClassifier(n_estimators=50,random_state=0)
clf.fit(x_train, y_train)

filename = 'model.sav'
joblib.dump(clf,filename)

accuracy = sum(sum([clf.predict(x_train) == y_train])) / len(y_train)
print ('training accuracy: ' +str(accuracy))
