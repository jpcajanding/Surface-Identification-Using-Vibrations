import numpy as np
import pickle
from sklearn import svm
import serial

train_file = r'data\channel4.pkl'
test_file = r'data\channel1.pkl'

test = 81
with open(train_file, 'rb') as fp:
    train_labels, train_data = pickle.load(fp)
    fp.close()

n_train, n_input = np.shape(train_data)

max_val = max(train_data)
norm_data = train_data/max_val
train_data = np.array(norm_data)
train_labels = np.array(train_labels)
test_data = train_data[test:,:]
test_labels = train_labels[test:]

training_data = train_data[:test-1,:]
training_labels = train_labels[:test-1]

print('Sample Size:', n_input, n_train)

clf = svm.SVC(degree=2, kernel = 'linear',tol=0.00001)

clf.fit(training_data,training_labels)

print(clf.score(test_data,test_labels))

ser = serial.Serial('COM9',baudrate=250000)
print(ser.name)

x = 1
while True:
    ain1 = np.zeros(202)
    ain2 = np.zeros(202)
    ain3 = np.zeros(202)
    ain4 = np.zeros(202)

    for i in range(202):
        data= ser.readline()
        ain1[i],ain2[i],ain3[i],ain4[i] = data.split()

    print(clf.predict(ain4))
