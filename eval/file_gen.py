import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
import json
import numpy as np
from scipy import sparse

train_or_test = 'test'
filestart = 0
size = 100000
JSON_FOLDER = '.\\parsed_replays\\GlobalFeatureVector\\train_val_test\\Protoss_vs_Protoss\\'
JSON_FILE = train_or_test+'.json'
with open(JSON_FOLDER+JSON_FILE) as json_data:
    json_data_all = json.load(json_data)

PATH_ALL=[]

for i in range(0,len(json_data_all)):
    try:
        temp1 = json_data_all[i]['Protoss']
        for j in range(0,len(temp1)):
            PATH_ALL.append(temp1[j]['global_path'])
    except:
        pass
del json_data_all
temp1 = np.asarray( sparse.load_npz(PATH_ALL[filestart]).todense() )
temp2 = np.array([range(0,temp1.shape[0])])
temp = np.hstack((temp1, temp2.T ))
gvf = temp
for i in range (filestart+1, min(filestart+size,len(PATH_ALL)) ):
    try:
        print(i)
        temp1 = np.asarray( sparse.load_npz(PATH_ALL[i]).todense() )
        temp2 = np.array([range(0,temp1.shape[0])])
        temp = np.hstack((temp1, temp2.T ))
        gvf = np.append(gvf, temp ,  axis = 0)
    except:
        pass
filename = 'gvf-protoss-'+str(filestart)+'-1000-'+train_or_test+'.npy'
np.save(filename, gvf)