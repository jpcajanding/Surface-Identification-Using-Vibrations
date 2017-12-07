__author__ = 'jcajandi_0809'

import serial
import numpy as np
import matplotlib.pyplot as plt
import pickle

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

    plt.figure(1)
    plt.subplot(411)
    plt.plot(ain1,'r')
    plt.subplot(412)
    plt.plot(ain2,'r')
    plt.subplot(413)
    plt.plot(ain3,'r')
    plt.subplot(414)
    plt.plot(ain4,'r')
    plt.show()

    datafile = 'glass_scissor_' + str(x) +'.pkl'
    with open(datafile,'wb') as fp:
        pickle.dump([ain1,ain2,ain3,ain4],fp,-1)
        fp.close()

    x = x + 1