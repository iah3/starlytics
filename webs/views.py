from django.shortcuts import render

# Create your views here.

import sys
from django.http import HttpResponse
import matplotlib as mpl
mpl.use('Agg') # Required to redirect locally
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand
from django.template import RequestContext
import io
import numpy as np
from scipy import sparse
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import os
from os import listdir
from os.path import isfile, join
def get_image(request):

    # GENERATE MAP ##################################
    # PATH = FOLDER_S + FILE_S[0]
    times = 200
    filenumber = 0
    flag_upload = False
    temp_flag = 0
    if request.method == 'POST':
        temp = 0
        try:
            # print ('yes')
            temp = request.POST.get('cur',0)
            temp_file = request.POST.get('file', 0)
            temp_flag = request.POST.get('flagupload', 0)

            filenumber = int(temp_file)

            for count, x in enumerate(request.FILES.getlist("files")):
                def process(f):
                    with open('.\\parsed_replays\\Uploads\\replay_' + str(count)+'.npz', 'wb+') as destination:
                        for chunk in f.chunks():
                            destination.write(chunk)
                process(x)
                flag_upload = True
        except:
            pass
        times = temp
    else:
        times=0
    times = int(times)
    
    if flag_upload:
        temp_flag = 1
    temp_flag = int(temp_flag)

    if temp_flag == 1:
        flag_upload = True

    filenumber = max(filenumber, 0)
    # print (flag_upload)
    replay_duration = generate_image(filenumber,times,flag_upload)

    path_file,flag_upload = generate_path(filenumber,'global',flag_upload)

    results = get_accuracy(path_file,times,flag_upload)
    return render(request, 'home.html', 
        {'duration': replay_duration, 'current': times, 
        'prediction': results['prediction'], 'actual': results['actual'],
        'overall': results['overall'], 'filenumber': filenumber , 'flagupload': temp_flag})

def generate_path(file_no, file_type,flag_upload):
    # # GET FILE NAMES ######################
    # JSON_FOLDER = '.\\parsed_replays\\GlobalFeatureVector\\train_val_test\\Protoss_vs_Protoss\\'
    # JSON_FILE = 'train.json'
    # with open(JSON_FOLDER+JSON_FILE) as json_data:
    #     json_data_all = json.load(json_data)


    # GENERATE PATH TO FILES ########################
    FOLDER_S = '.\\parsed_replays\\SpatialFeatureTensor\\Protoss_vs_Protoss\\Protoss\\'
    FOLDER_G = '.\\parsed_replays\\GlobalFeatureVector\\Protoss_vs_Protoss\\Protoss\\'
    
    FILE_G = [f for f in listdir(FOLDER_G) if isfile(join(FOLDER_G, f))]
    FILE_S=[f for f in listdir(FOLDER_S) if isfile(join(FOLDER_S, f))]

    # for i in range(0,len(json_data_all)):
    #     try:
    #         temp1 = json_data_all[i]['Protoss']
    #         for j in range(0,len(temp1)):
    #             FILE_G.append(temp1[j]['global_path'])
    #             FILE_S.append(temp1[j]['spatial_path_S'])
    #     except:
    #         pass
    # del json_data_all

    file_no = min(file_no, len(FILE_G)-1)
    
    filename = FOLDER_G + FILE_G[file_no]

    if file_type == "global":
        filename = FOLDER_G + FILE_G[file_no]
    else:
        try:
            filename = FOLDER_S + FILE_S[file_no]
        except:
            flag_upload = True
            pass
    return filename,flag_upload

def get_accuracy(filename,time_cur,flag_upload):
    if flag_upload:
        filename = '.\\parsed_replays\\Uploads\\'
        filename = filename + 'replay_1.npz'
    y_test, x_test = np.split(np.asarray( sparse.load_npz(filename).todense() ),[1],axis=1)
    y_test = np.ravel(y_test)

    modelname = 'model.sav'
    clf = joblib.load(modelname)
    y_predict = clf.predict(x_test)

    accuracy = sum(sum([clf.predict(x_test) == y_test])) / len(y_test)
    return {'prediction': int(y_predict[time_cur]+1), 'actual': int(y_test[time_cur]+1), 'overall': str(round(accuracy,4))}

def generate_image(filenumber,times,flag_upload):
    if flag_upload:
        PATH = '.\\parsed_replays\\Uploads\\'
        PATH = PATH + 'replay_0.npz'
    else:
        PATH, flag_upload = generate_path(filenumber,"spatial",flag_upload)

    data = np.asarray(sparse.load_npz(PATH).todense()).reshape([-1, 13, 64, 64])
    x=[]
    y=[]
    z1=[]
    z2=[]
    for i in range(0,64):
        for j in range (0,64):
            x.append(i)
            y.append(j)
            z1.append(data[times,11,i,j])
            z2.append(data[times,8,i,j])
    plt.scatter(x, y, c=z2, marker='s', linewidth=0)
    if (z1):
        plt.scatter(x, y, c=z1, alpha=0.45, marker='s', linewidth=0)
    plt.axis('off')
    path_img = '.\\starlytics\\static\\images\\'
    plt.savefig(path_img+'graph.png',dpi=300,bbox_inches='tight',pad_inches=0,transparent=True)
    return data.shape[0]-1