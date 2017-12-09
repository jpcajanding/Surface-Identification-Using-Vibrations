import numpy as np
import pickle
from sklearn import svm
import pandas

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

# with open(test_file, 'rb') as fp:
#     test_labels, test_data = pickle.load(fp)
#     fp.close()

print(clf.score(test_data,test_labels))
print(clf.predict(test_data))