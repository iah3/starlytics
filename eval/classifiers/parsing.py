import os

import numpy as np
from scipy import sparse

cwd = os.getcwd()
dic = {}
dic['ZvZ'] = ('Zerg_vs_Zerg', 'Zerg')
dic['PvP'] = ('Protoss_vs_Protoss', 'Protoss')
dic['PvT'] = ('Protoss_vs_Terran', 'Protoss', 'Terran')
dic['PvZ'] = ('Protoss_vs_Zerg', 'Protoss', 'Zerg')
dic['TvZ'] = ('Terran_vs_Zerg', 'Terran', 'Zerg')
dic['TvT'] = ('Terran_vs_Terran', 'Terran')
matchup = 'PvP' #Change the matchup here
matchup2 = dic[matchup][0]
matchup3 = dic[matchup][1]
os.chdir(cwd+'\\GlobalFeatureVector\\'+matchup2+'\\'+matchup3)

print len(os.listdir(os.getcwd()))

def get_len_shortest():
    l = []
    for rep in os.listdir(os.getcwd()):
        PATH = os.getcwd()+'\\'+str(rep)
        F = np.asarray(sparse.load_npz(PATH).todense())
        l.append(len(F))
    print min(l)

def get_len_longest():
    l = []
    for rep in os.listdir(os.getcwd()):
        PATH = os.getcwd()+'\\'+str(rep)
        F = np.asarray(sparse.load_npz(PATH).todense())
        l.append(len(F))
    print max(l)

def get_vectrep(time=50):
    s = 'minerals,vespene,food_cap,food_used,food_army,food_workers,idle_worker_count,army_count,warp_gate_count,larva_count,y\n'
    for rep in os.listdir(os.getcwd()):
        PATH = os.getcwd()+'\\'+str(rep)
        F = np.asarray(sparse.load_npz(PATH).todense())
        if len(F)>time+1:
            subF = F[time,16:26]
            for i in subF:
                s+=str(i)+','
            s+=str(int(F[0,0]))+'\n'
    return s

#Change the desired frames used below
for i in ['050', '075', '100', '150', '300', '350', '400', '450', '500', '550', '600']:
    print i
    name = matchup + '_frame'+i
    s = get_vectrep(int(i))
    fich = open(cwd+'\\to_train\\'+name+'.csv', 'w')
    fich.write(s)
    fich.close()
