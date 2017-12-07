__author__ = 'jcajandi_0809'

import pandas as pd
import numpy as np
import glob
import pickle
import os

from sklearn.utils import shuffle
from sklearn import preprocessing

#put all waves data into a single array, for each channel
folder_path = r'data'
materials = [r'\acrylic',r'\glass',r'\particle',r'\plywood',r'\whiteboard']

waves1 = []
waves2 = []
waves3 = []
waves4 = []

for material in materials:
    filenames = glob.glob(folder_path + material + '/*.pkl')
    for filename in filenames:
        with open(filename, 'rb') as fp:
            ain1, ain2, ain3, ain4 = pickle.load(fp)
            fp.close()
        waves1.append(ain1)
        waves2.append(ain2)
        waves3.append(ain3)
        waves4.append(ain4)

data_waves1 = np.array(waves1)
data_waves2 = np.array(waves2)
data_waves3 = np.array(waves3)
data_waves4 = np.array(waves4)

print(data_waves1.shape)
print(data_waves2.shape)
print(data_waves3.shape)
print(data_waves4.shape)

#get labels
label_file = r'data/labels.csv'
labels = pd.read_csv(label_file, header=None)

le = preprocessing.LabelEncoder()

data_pd1 = pd.DataFrame(data_waves1, index= None)
data_pd2 = pd.DataFrame(data_waves2, index= None)
data_pd3 = pd.DataFrame(data_waves3, index= None)
data_pd4 = pd.DataFrame(data_waves4, index= None)

data_pd1['labels'] = labels
data_pd2['labels'] = labels
data_pd3['labels'] = labels
data_pd4['labels'] = labels

data_pd1.to_csv(r'data\data_chan1.csv', index= False)
data_pd2.to_csv(r'data\data_chan2.csv', index= False)
data_pd3.to_csv(r'data\data_chan3.csv', index= False)
data_pd4.to_csv(r'data\data_chan4.csv', index= False)

print(data_pd1.shape)
print(data_pd2.shape)
print(data_pd3.shape)
print(data_pd4.shape)

data_pd1 = shuffle(data_pd1)
data_pd2 = shuffle(data_pd2)
data_pd3 = shuffle(data_pd3)
data_pd4 = shuffle(data_pd4)

labels1_raw = data_pd1['labels'].values
chan1_data = data_pd1.drop(['labels'],axis = 1)
labels2_raw = data_pd2['labels'].values
chan2_data = data_pd2.drop(['labels'],axis = 1)
labels3_raw = data_pd3['labels'].values
chan3_data = data_pd3.drop(['labels'],axis = 1)
labels4_raw = data_pd4['labels'].values
chan4_data = data_pd4.drop(['labels'],axis = 1)

le.fit(labels1_raw)
chan1_labels = le.transform(labels1_raw)
le.fit(labels1_raw)
chan2_labels = le.transform(labels2_raw)
le.fit(labels1_raw)
chan3_labels = le.transform(labels3_raw)
le.fit(labels1_raw)
chan4_labels = le.transform(labels4_raw)


with open(r'data\channel1.pkl', 'wb') as fp:
    pickle.dump([chan1_labels, chan1_data], fp, -1)
    fp.close()

with open(r'data\channel2.pkl', 'wb') as fp:
    pickle.dump([chan2_labels, chan2_data], fp, -1)
    fp.close()

with open(r'data\channel3.pkl', 'wb') as fp:
    pickle.dump([chan3_labels, chan3_data], fp, -1)
    fp.close()

with open(r'data\channel4.pkl', 'wb') as fp:
    pickle.dump([chan4_labels, chan4_data], fp, -1)
    fp.close()