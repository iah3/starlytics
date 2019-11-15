import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import json
import numpy as np
from scipy import sparse
from sklearn.externals import joblib

vs_type='Terran_vs_Terran'
train_or_test = 'train'
filestart = 0
size = 1000
JSON_FOLDER = '.\\parsed_replays\\GlobalFeatureVector\\train_val_test\\'+str(vs_type)+'\\'
JSON_FILE = train_or_test+'.json'
json_data_all=[]
with open(JSON_FOLDER+JSON_FILE) as json_data:
    json_data_all = json.load(json_data)

PATH_ALL=[]

for i in range(0,len(json_data_all)):
    try:
        temp_key = list(json_data_all[i].keys())
        temp1 = json_data_all[i][temp_key[0]]
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
filename = 'gvf-'+vs_type+'-1000-'+train_or_test+'.npy'
np.save(filename, gvf)
del gvf
del PATH_ALL
##############################################################################
train_or_test = 'train'
filename = 'gvf-'+vs_type+'-1000-'+train_or_test+'.npy'
y_train, x_train = np.split(np.load(filename),[1],axis=1)
y_train = np.ravel(y_train)

clf = RandomForestClassifier(n_estimators=50,random_state=0)
clf.fit(x_train, y_train)

filename = 'model_'+vs_type+'.sav'
joblib.dump(clf,filename)

accuracy = sum(sum([clf.predict(x_train) == y_train])) / len(y_train)
print ('training accuracy: ' +str(accuracy))
del x_train
del y_train
del clf
#####################################################################################
train_or_test = 'test'
size = 1000
JSON_FOLDER = '.\\parsed_replays\\GlobalFeatureVector\\train_val_test\\'+str(vs_type)+'\\'
JSON_FILE = train_or_test+'.json'
with open(JSON_FOLDER+JSON_FILE) as json_data:
    json_data_all = json.load(json_data)

PATH_ALL=[]

for i in range(0,len(json_data_all)):
    try:
        temp_key = list(json_data_all[i].keys())
        temp1 = json_data_all[i][temp_key[0]]
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
filename = 'gvf-'+vs_type+'-1000-'+train_or_test+'.npy'
np.save(filename, gvf)
del gvf
del PATH_ALL
####################################################################################
filename = 'gvf-'+vs_type+'-1000-'+train_or_test+'.npy'
y_test, x_test = np.split(np.load(filename),[1],axis=1)
y_test = np.ravel(y_test)

filename = 'model_'+vs_type+'.sav'
clf = joblib.load(filename)

accuracy = sum(sum([clf.predict(x_test) == y_test])) / len(y_test)
print ('testing accuracy: '+str(vs_type)+ '  '  +str(accuracy))