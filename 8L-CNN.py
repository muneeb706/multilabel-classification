import numpy as np # linear algebra
# version of panda should be 0.25.1 because of .ix function
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# pip install opencv-python
import cv2

# pip install tqdm
from tqdm import tqdm

from sklearn.model_selection import KFold
import time

# Pre-processing the train and test data

x_train = []
x_test = []
y_train = []

df_train = pd.read_csv('./dataset/train_v2.csv')
df_test = pd.read_csv('./dataset/sample_submission_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

labels = ['blow_down',
          'bare_ground',
          'conventional_mine',
          'blooming',
          'cultivation',
          'artisinal_mine',
          'haze',
          'primary',
          'slash_burn',
          'habitation',
          'clear',
          'road',
          'selective_logging',
          'partly_cloudy',
          'agriculture',
          'water',
          'cloudy']

label_map = {'agriculture': 14,
             'artisinal_mine': 5,
             'bare_ground': 1,
             'blooming': 3,
             'blow_down': 0,
             'clear': 10,
             'cloudy': 16,
             'conventional_mine': 2,
             'cultivation': 4,
             'habitation': 9,
             'haze': 6,
             'partly_cloudy': 13,
             'primary': 7,
             'road': 11,
             'selective_logging': 12,
             'slash_burn': 8,
             'water': 15}

for f, tags in tqdm(df_train.values[:18000], miniters=1000):
    img = cv2.imread('./dataset/train-jpg/{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_train.append(cv2.resize(img, (128, 128)))
    y_train.append(targets)


y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float32) / 255.

print(x_train.shape)
print(y_train.shape)

###############################

# Create n-folds cross-validation

import numpy as np
from sklearn.metrics import fbeta_score

from tensorflow.keras.layers import BatchNormalization

nfolds = 3

num_fold = 0
sum_score = 0

yfull_test = []
yfull_train = []

kf = KFold(n_splits=nfolds, shuffle=True, random_state=1)

for train_index, test_index in kf.split(y_train):
    start_time_model_fitting = time.time()

    X_train = x_train[train_index]
    Y_train = y_train[train_index]
    X_valid = x_train[test_index]
    Y_valid = y_train[test_index]

    num_fold += 1
    print('Start KFold number {} from {}'.format(num_fold, nfolds))
    print('Split train: ', len(X_train), len(Y_train))
    print('Split valid: ', len(X_valid), len(Y_valid))

    kfold_weights_path = os.path.join('', 'weights_kfold_' + str(num_fold) + '.h5')

    model = Sequential()
    model.add(Conv2D(16, 2, 2, activation='sigmoid'))
    model.add(BatchNormalization())

    model.add(Conv2D(26, 2, 2, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, 1, activation='sigmoid'))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(Conv2D(64, 3, 1, activation='sigmoid'))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, 1, activation='sigmoid'))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(Conv2D(64, 3, 1, activation='sigmoid'))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(1024, activation='sigmoid'))
    model.add(Dense(17, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=0),
        ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]

    model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid),
              batch_size=128, verbose=1, nb_epoch=10, callbacks=callbacks,
              shuffle=True)

    print(model.summary())

    if os.path.isfile(kfold_weights_path):
        model.load_weights(kfold_weights_path)

    p_valid = model.predict(X_valid, batch_size=128, verbose=2)
    print("F beta score: " + str(fbeta_score(Y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')))