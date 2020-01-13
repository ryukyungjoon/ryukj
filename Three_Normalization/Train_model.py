from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

import keras
import numpy as np
from keras import layers, models
from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from collections import Counter


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
        use_model_score = cross_val_score(use_model, test_X, test_Y, scoring='accuracy', cv=10, n_jobs=8)
        use_model_cross_val = cross_val_predict(use_model, test_X, test_Y, cv=10, n_jobs=8)
        use_model_confusion_matrix = confusion_matrix(test_Y, use_model_cross_val)

        return use_model_cross_val, use_model_score, use_model_confusion_matrix

Nin = 49                # 입력 노드의 개수
Nh_l = [100, 50]        # 히든 레이어 개수
number_of_class = 15    # 분류 클래스 개수
Nout = number_of_class  # 출력 노드의 개수

class DNN(models.Sequential):

    def __init__(self, Nin, Nh_l, Nout):
        super().__init__()
        self.add(layers.Dense(Nh_l[0], activation='relu', input_shape=(Nin,), name='Hidden-1'))
        self.add(layers.Dense(Nh_l[1], activation='relu', name='Hidden-2'))
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def dnn_model(train_X, train_Y, test_X, test_Y):
        print('model training...')
        print('DNN Classifier')

        print(train_Y)
        train_cnt_Y = Counter(train_Y)
        test_cnt_Y = Counter(test_Y)
        classes_y = len(train_cnt_Y.keys())
        print('Attack Type Count : {}'.format(dict(train_cnt_Y)))

        # Label(Type : String -> Float)
        l_encoder = LabelEncoder()
        train_Y = l_encoder.fit(train_Y)
        print('float type {}'.format(train_Y))

        key = ['BENIGN', 'DDoS', 'DoS GoldenEye', 'Heartbleed', 'PortScan', 'Bot', 'FTP-Patator', 'DoS Hulk', 'Web Attack XSS', 'DoS slowloris', 'Web Attack Sql Injection', 'Web Attack Brute Force', 'DoS Slowhttptest', 'Infiltration', 'SSH-Patator']
        value = l_encoder.transform(key)
        attack_maping = dict(zip(key, value))
        print('Attack Mapping : {}'.format(attack_maping))

        train_Y = to_categorical(train_Y, num_classes=classes_y)
        test_Y = to_categorical(test_Y, num_classes=classes_y)
        print('one-hot encoding : \n train_Y {} \n test_Y {}'.format(train_Y, test_Y))

        use_model = DNN(Nin, Nh_l, Nout)
        history = use_model.fit(train_X, train_Y, epochs=100, batch_size=10000, validation_split=0.2)
        performance_test = use_model.evaluate(test_X, test_Y, batch_size=10000)

        return performance_test