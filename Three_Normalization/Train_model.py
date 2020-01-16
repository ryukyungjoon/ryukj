from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from keras import layers, models

from tensorflow.keras.utils import to_categorical
from collections import Counter

from matplotlib import pyplot as plt

import pandas as pd
import numpy as np


class Train_model:

    def train_model(train_X, train_Y, classes_y, test_X, test_Y):
        print('model training...')

        print('MLP Classifier')

        print('string type [train_Y] : {}'.format(train_Y))
        use_model = MLPClassifier(hidden_layer_sizes=(30, 10), max_iter=10, random_state=42)

        mini_batch_size = 10000
        batch_size = len(train_Y)
        total_epoch = int(batch_size / mini_batch_size)
        current_batch = 0

        for i in range(1, total_epoch):
            end_batch = i * mini_batch_size
            use_model._partial_fit(train_X[current_batch:end_batch], train_Y[current_batch:end_batch], classes=classes_y)
            current_batch = end_batch
        use_model._partial_fit(train_X[current_batch:batch_size], train_Y[current_batch:batch_size], classes=classes_y)
        pred = use_model.predict(test_X)

        '''
        use_model_score = cross_val_score(use_model, test_X, test_Y, scoring='accuracy', cv=10, n_jobs=8)
        use_model_cross_val = cross_val_predict(use_model, test_X, test_Y, cv=10, n_jobs=8)
        '''

        use_model_confusion_matrix = confusion_matrix(test_Y, pred)

        return use_model_confusion_matrix

Nin = 49                # 입력 노드의 개수
Nh_l = [1000, 500, 100] # 히든 레이어 개수
number_of_class = 15    # 분류 클래스 개수
Nout = number_of_class  # 출력 노드의 개수

class DNN(models.Sequential):

    def __init__(self, Nin, Nh_l, Nout):
        super().__init__()
        self.add(layers.Dense(Nh_l[0], activation='relu', input_shape=(Nin,), name='Hidden-1'))
        self.add(layers.Dense(Nh_l[1], activation='relu', name='Hidden-2'))
        self.add(layers.Dense(Nh_l[2], activation='relu', name='Hidden-3'))
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def dnn_model(train_X, train_Y, test_X, test_Y):
        print('model training...')
        print('DNN Classifier')

        ## train_X reshape

        train_X = np.array(train_X)
        test_X = np.array(test_X)

        print(train_X.shape)        # (samples, 49)
        print(test_X.shape)         # (samples, 49)

        a, b = train_X.shape     # 2D shape를 가진다.
        a1, b1 = test_X.shape

        train_X = train_X.reshape(-1, b)
        test_X = test_X.reshape(-1, b1)

        train_X = np.array(train_X)
        test_X = np.array(test_X)

        ## Count Y_Label
        train_Y_cnt = Counter(train_Y)
        test_Y_cnt = Counter(test_Y)

        output_size = len(train_Y_cnt.keys())
        output_size2 = len(test_Y_cnt.keys())
        print('train Attack Sample Count : {}'.format(dict(train_Y_cnt)))
        print('test Attack Sample Count : {}'.format(dict(test_Y_cnt)))
        print('Total Output Size : {}'.format(output_size))
        print('Total Output Size : {}'.format(output_size2))

        ## String 2 Float
        l_encoder = LabelEncoder()
        y_train = l_encoder.fit_transform(train_Y)
        y_test = l_encoder.fit_transform(test_Y)

        key = ['BENIGN', 'DDoS', 'DoS GoldenEye', 'Heartbleed', 'PortScan', 'Bot', 'FTP-Patator', 'DoS Hulk', 'Web Attack XSS', 'DoS slowloris', 'Web Attack Sql Injection', 'Web Attack Brute Force', 'DoS Slowhttptest', 'Infiltration', 'SSH-Patator']
        value = l_encoder.transform(key)
        attack_mapping = dict(zip(key, value))

        print('Attack Mapping : {}'.format(attack_mapping))

        ## One-Hot
        train_Y2 = to_categorical(y_train, num_classes=output_size)
        test_Y2 = to_categorical(y_test, num_classes=output_size)
        print('Original Data : {}'.format(train_Y))
        print('Original Data : {}'.format(test_Y))
        print('\nOne-Hot Result from Y_Train : \n{}'.format(train_Y2))
        print('\nOne-Hot Result from Y_Test : \n{}'.format(test_Y2))

        use_model = DNN(Nin, Nh_l, Nout)
        history = use_model.fit(train_X, train_Y2, epochs=100, batch_size=10000,
                                validation_split=0.2, verbose=1)
        performance_test = use_model.evaluate(test_X, test_Y2, batch_size=10000)

        return performance_test, history