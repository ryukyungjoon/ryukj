from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, cross_val_predict

from keras import layers, models, Input

from tensorflow.keras.utils import to_categorical

import numpy as np

import time

from matplotlib import pyplot as plt

class Train_model:
    def train_model(train_X, train_Y, classes_y, test_X, test_Y):
        print('model training...')
        print('MLP Classifier')
        use_model = MLPClassifier(hidden_layer_sizes=(1000, 500, 100), max_iter=100, random_state=42)

        print('string type [train_Y]:', train_Y)

        mini_batch_size = 10000
        batch_size = len(train_Y)
        total_epoch = int(batch_size / mini_batch_size)
        current_batch = 0
        start = time.time()
        for i in range(1, total_epoch):
            end_batch = i * mini_batch_size
            use_model.partial_fit(train_X[current_batch:end_batch], train_Y[current_batch:end_batch], classes=classes_y)
            current_batch = end_batch
        use_model.partial_fit(train_X[current_batch:batch_size], train_Y[current_batch:batch_size], classes=classes_y)
        end = time.time()
        print(f"MLP training time{end-start}s")

        # use_model_score = cross_val_score(use_model, test_X, test_Y, scoring='accuracy', cv=10, n_jobs=8)
        # print(use_model_score)
        # train_Y_pred = cross_val_predict(use_model, train_X, train_Y, cv=10, n_jobs=8)
        test_Y_pred = use_model.predict(test_X)
        print(test_Y_pred)
        print('Accuracy Performance :', metrics.accuracy_score(test_Y, test_Y_pred))
        use_model_confusion_matrix = confusion_matrix(test_Y, test_Y_pred)

        return use_model_confusion_matrix, test_Y_pred

Nin = 43
Nh_l = [1000, 500, 100] # 히든 레이어 개수
number_of_class = 5    # 분류 클래스 개수
Nout = number_of_class  # 출력 노드의 개수

class DNN(models.Sequential):

    def __init__(self, Nin, Nh_l, Nout):        # 모델 구조 정의
        super().__init__()
        self.add(layers.Dense(Nh_l[0], activation='relu', input_shape=(Nin,), name='Hidden-1'))
        self.add(layers.Dense(Nh_l[1], activation='relu', name='Hidden-2'))
        self.add(layers.Dense(Nh_l[2], activation='relu', name='Hidden-3'))
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def dnn_model(train_X, train_Y, test_X, test_Y, norm_type):
        print('DNN Classifier')

        print('model training...')

        train_x = np.array(train_X)
        test_X = np.array(test_X)

        print(train_x.shape)        # (samples, 49)
        print(test_X.shape)         # (samples, 49)

        a, b = train_x.shape     # 2D shape를 가진다.(samples_num, features_num)
        a1, b1 = test_X.shape

        train_x = train_x.reshape(-1, b)
        test_X = test_X.reshape(-1, b1)

        train_x = np.array(train_x)
        test_X = np.array(test_X)

        ## String 2 Float
        l_encoder = LabelEncoder()
        y_train = l_encoder.fit_transform(train_Y)
        y_test = l_encoder.fit_transform(test_Y)
        print(np.unique(y_train))
        # key = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0', '11.0', '12.0', '13.0', '14.0']
        # key = ['BENIGN', 'DDoS', 'DoS GoldenEye', 'Heartbleed', 'PortScan', 'Bot', 'FTP-Patator', 'DoS Hulk', 'Web Attack XSS', 'DoS slowloris', 'Web Attack Sql Injection', 'Web Attack Brute Force', 'DoS Slowhttptest', 'Infiltration', 'SSH-Patator']
        # key = ['BENIGN', 'DDoS', 'Heartbleed', 'PortScan', 'Bot', 'FTP-Patator', 'Web Attack', 'DoS', 'Infiltration', 'SSH-Patator']
        # key = ['BENIGN', 'DDoS', 'DoS GoldenEye', 'PortScan', 'Bot', 'FTP-Patator', 'DoS Hulk', 'Web Attack XSS', 'DoS slowloris', 'Web Attack Brute Force', 'DoS Slowhttptest', 'SSH-Patator']
        # key = ['DoS', P]

        # value = l_encoder.transform(key)
        # attack_mapping = dict(zip(key, value))

        # print('Attack Mapping : {}'.format(attack_mapping))

        ## One-Hot
        Onehot_train_Y2 = to_categorical(y_train, num_classes=Nout)
        Onehot_test_Y2 = to_categorical(y_test, num_classes=Nout)
        print('Original Data : {}'.format(train_Y))
        print('Original Data : {}'.format(test_Y))
        print('\nOne-Hot Result from Y_Train : \n{}'.format(Onehot_train_Y2))
        print('\nOne-Hot Result from Y_Test : \n{}'.format(Onehot_test_Y2))

        # Model Instance 호출
        use_model = DNN(Nin, Nh_l, Nout)

        # Learning
        training_start = time.time()
        history = use_model.fit(train_x, Onehot_train_Y2, epochs=100, batch_size=64,
                                validation_split=0.2, verbose=1)
        training_end = time.time()
        print(f"DNN Training Time : {training_end-training_start}")

        # Model Evaluate
        performance_test = use_model.evaluate(test_X, Onehot_test_Y2, batch_size=10000)
        print('Test Loss and Accuracy ->', performance_test)

        pred = use_model.predict(test_X)
        cm = confusion_matrix(np.argmax(Onehot_test_Y2, axis=1), np.argmax(pred, axis=1))
        print(cm)

        use_model.save_weights('NSL-KDD_full_model_weights_qnt')
        use_model_json = use_model.to_json()
        with open('NSL-KDD_full_model_qnt.json', 'w') as json_file:
            json_file.write(use_model_json)

        # Model Save & Load
        use_model.save_weights('h5 File/NSL-KDD_DNN Classifier['+norm_type+'].h5')

        return cm, history, pred, y_test