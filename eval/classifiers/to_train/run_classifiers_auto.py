import os

import numpy as np
import pandas as pd
import time 

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

######################################### Reading and Splitting the Data ###############################################
cwd = os.getcwd()
matchup = 'PvP' #Change the matchup here
s = 'lr, mlp, rfc, rfcgrid, best, best_score, recall, specificity\n'
classifiers = ['lr', 'mlp', 'rfc', 'rfcgrid']
target_names = ['0', '1']
for fich in os.listdir(cwd):
    score = []
    if fich[:3] == matchup:
        print fich
        recall = []
        specificity = []

        # Read in all the data.
        data = pd.read_csv(fich)
        s_data = train_test_split(data, shuffle=True, random_state=100, test_size=0.30)

        # Separate out the x_data and y_data.
        x_train = s_data[0].loc[:, data.columns != "y"]
        y_train = s_data[0].loc[:, "y"]

        x_test = s_data[1].loc[:, data.columns != "y"]
        y_test = s_data[1].loc[:, "y"]

        


        # ############################################### Linear Regression ###################################################
        lr = LinearRegression()
        lr.fit(x_train, y_train)

        y_ptrain = lr.predict(x_train)
        y_ptrain = y_ptrain.round()

        y_pred = lr.predict(x_test)
        y_pred = y_pred.round()
        
        score.append(accuracy_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred, average='macro'))
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        spec = float(tn)/(tn+fp)
        specificity.append(spec)


        # ############################################### Multi Layer Perceptron #################################################
        mlp = MLPClassifier(hidden_layer_sizes=(178, 178, 178))
        mlp.fit(x_train, y_train)

        y_ptrain = mlp.predict(x_train)
        y_ptrain = y_ptrain.round()

        y_pred = mlp.predict(x_test)
        y_pred = y_pred.round()
        score.append(accuracy_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred, average='macro'))
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        spec = float(tn)/(tn+fp)
        specificity.append(spec)

        # ############################################### Random Forest Classifier ##############################################
        rfc = RandomForestClassifier()
        rfc.fit(x_train, y_train)

        y_ptrain = rfc.predict(x_train)
        y_ptrain = y_ptrain.round()

        y_pred = rfc.predict(x_test)
        y_pred = y_pred.round()
        score.append(accuracy_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred, average='macro'))
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        spec = float(tn)/(tn+fp)
        specificity.append(spec)
        
        rfc = RandomForestClassifier()
        parameters = {'n_estimators' : [1, 10, 20, 30], 'max_depth' : [2, 5, 10, 20]}
        clf = GridSearchCV(rfc, parameters, cv = 10)
        clf.fit(x_train, y_train)

        score.append(accuracy_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred, average='macro'))
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        spec = float(tn)/(tn+fp)
        specificity.append(spec)
        
        maxi = np.argmax(score)
        score.append(classifiers[np.argmax(score)])
        score.append(score[maxi])
        score.append(recall[maxi])
        score.append(specificity[maxi])
        

        for val in score:
            s+=str(val)
            s+= ','
        s = s[:-1]
        s+= '\n'
fich = open(matchup+'.csv', 'w')
fich.write(s)
fich.close()
#recall = sensitivity, specificity //


