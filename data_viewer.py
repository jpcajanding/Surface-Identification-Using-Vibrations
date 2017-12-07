__author__ = 'jcajandi_0809'

import numpy as np
import matplotlib.pyplot as plt
import pickle
i=5
# for i in range (1,16):
datafile = r'data\whiteboard_scissor_' + str(i) + '.pkl'

with open(datafile, 'rb') as fp:
    ain1, ain2, ain3, ain4 = pickle.load(fp)
    fp.close()

print(ain1)
# plt.figure(1)
# plt.subplot(411)
# plt.plot(ain1,'r')
# plt.subplot(412)
# plt.plot(ain2,'r')
# plt.subplot(413)
# plt.plot(ain3,'r')
# plt.subplot(414)
# plt.plot(ain4,'r')
# plt.show()