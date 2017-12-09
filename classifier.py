import numpy as np
import pickle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical

train_file = r'data\channel4.pkl'

with open(train_file, 'rb') as fp:
    train_labels, train_data = pickle.load(fp)
    fp.close()

train_labels_cat = to_categorical(train_labels, num_classes=5)

n_train, n_input = np.shape(train_data)
n_hidden = int(n_input * 2)
learning_rate = 0.001
dropout = 0.9
batch_size = 32

print('Sample Size:', n_input, n_train)

print('Building model...')

# model is a 2 layer ANN
model = Sequential()
model.add(Dense(n_hidden, input_dim=n_input))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(n_hidden))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(5))
model.add(Activation('sigmoid'))

# optimizer
sgd = SGD(lr=learning_rate)
adam = Adam(lr=learning_rate)
# compile model
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

max_val = max(train_data)
norm_data = train_data/max_val

print('Model built.')
print('Training...')
model.fit(norm_data, train_labels_cat, epochs=1000, batch_size=batch_size, shuffle=True)
print('Training finished.')