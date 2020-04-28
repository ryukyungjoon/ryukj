from Data_cleaning import Cleaning as cl
from Data_split import Data_split as ds
from Semi_balancing import Semi_balancing as sb
from Data_Normalization import Data_Normalization as dn
from Drawing import Drawing as dw

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 5

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


encoding_h = [1000, 500, 250, 125, 60]
encoding_out = 30
decoding_h = [60, 125, 250, 500, 1000]
decoding_output = 78
class Autoencoder_Test():
    def __init__(self):
        self.input_dim = 78
        self.latent_dim = 30
        self.num_classes = 15
        self.batch_size = 128
        self.epochs = 20

        input_img = Input(shape=(self.input_dim))
        latent_img = Input(shape=(self.latent_dim))

        encode_output = self.build_encoder(input_img)
        decode_output = self.build_decoder(latent_img)

        # autoencoder 모델 구성
        self.autoencoder_model = Model(input_img, self.build_decoder(self.build_encoder(input_img)))
        self.autoencoder_model.compile(loss='mean_squared_error', optimizer='adam')
        self.autoencoder_model.summary()

        # Full Freeze 모델 구성
        self.full_model = Model(input_img, self.build_fc(encode_output))
        self.full_model.summary()

        for i, layer in enumerate(self.full_model.layers[0:7]):                     # Weights Transfer & Freeze
            layer.set_weights(self.autoencoder_model.layers[i].get_weights())
            layer.trainable = False
        self.full_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        self.full_model.summary()

    def build_encoder(self, input_img):
        enc_1 = Dense(encoding_h[0], activation='relu', name='encoder-1')(input_img)
        enc_2 = Dense(encoding_h[1], activation='relu', name='encoder-2')(enc_1)
        enc_3 = Dense(encoding_h[2], activation='relu', name='encoder-3')(enc_2)
        enc_4 = Dense(encoding_h[3], activation='relu', name='encoder-4')(enc_3)
        enc_5 = Dense(encoding_h[4], activation='relu', name='encoder-5')(enc_4)
        enc_output = Dense(encoding_out, name='encoder-output')(enc_5)

        return enc_output

    def build_decoder(self, latent_img):
        dec_1 = Dense(decoding_h[0], activation='relu', name='decoder-1')(latent_img)
        dec_2 = Dense(decoding_h[1], activation='relu', name='decoder-2')(dec_1)
        dec_3 = Dense(decoding_h[2], activation='relu', name='decoder-3')(dec_2)
        dec_4 = Dense(decoding_h[3], activation='relu', name='decoder-4')(dec_3)
        dec_5 = Dense(decoding_h[4], activation='relu', name='decoder-5')(dec_4)
        dec_output = Dense(decoding_output, name='decoder-output')(dec_5)

        return dec_output

    def build_fc(self, enco):
        den = Dense(self.input_dim, activation='relu')(enco)
        out = Dense(self.num_classes, activation='softmax')(den)
        return out

    def main(self):
        data_loc = "../dataset/fin_dataset/"
        data_file = "5. combine_dataset"
        data_format = ".csv"

        features = ['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label']

        print("Data Loading...")
        data = pd.read_csv(data_loc+data_file+data_format, sep=',', dtype='unicode')
        x, y = cl.data_cleaning(data)                              # data cleaning
        bal_data = pd.concat([x, y], axis=1)
        # print(x, y)
        # bal_data = sb.sampling_1(x, y, features, data_file)        # data balancing
        # print(bal_data)                                            # sample : 832373
        a, b = bal_data.shape
        norm_set = dn.normalizations(bal_data, 'qnt', subsample=a)
        # norm_set = pd.read_csv("../dataset/fin_dataset/cicids_ae_qnt.csv")
        norm_set = norm_set.dropna()
        print(norm_set)

        train_X, train_Y, test_X, test_Y = ds.ae_split(norm_set)
        print(np.unique([test_Y]))
        # norm_train = pd.read_csv('../dataset/fin_dataset/qnt_train(AE).csv', sep=',', dtype='unicode')
        # norm_test = pd.read_csv('../dataset/fin_dataset/qnt_test(AE).csv', sep=',', dtype='unicode')

        l_encoder = LabelEncoder()
        y_train = l_encoder.fit_transform(train_Y)
        y_test = l_encoder.fit_transform(test_Y)
        Onehot_train_Y2 = to_categorical(y_train, num_classes=self.num_classes)
        Onehot_test_Y2 = to_categorical(y_test, num_classes=self.num_classes)
        print('Original Data : {}'.format(train_Y))
        print('Original Data : {}'.format(test_Y))
        print('\nOne-Hot Result from Y_Train : \n{}'.format(Onehot_train_Y2))
        print('\nOne-Hot Result from Y_Test : \n{}'.format(Onehot_test_Y2))

        autoencoder_train = self.autoencoder_model.fit(train_X, train_X, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2)

        dw.loss_graph(autoencoder_train)

        print('== Full Model Start ==')
        print(train_X.shape, Onehot_train_Y2.shape)

        classify_train = self.full_model.fit(train_X, Onehot_train_Y2, batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_split=0.2)

        dw.loss_acc_graph(classify_train)

        # UnFreeze model
        for layer in self.full_model.layers[0:7]:
            layer.trainable = True
        self.full_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        self.full_model.summary()

        import time
        start = time.time()
        f_classify_train = self.full_model.fit(train_X, Onehot_train_Y2, batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_split=0.2)
        end = time.time()
        print(f'full_model Training Time{end-start}')

        # Save Weights & Model
        self.full_model.save_weights('CICIDS_full_model_weights')
        full_model_json = self.full_model.to_json()
        with open('CICIDS_full_model.json', 'w') as json_file:
            json_file.write(full_model_json)

        dw.loss_acc_graph(f_classify_train)

        print('======= test evaluation ========')

        test_eval = self.full_model.evaluate(test_X, Onehot_test_Y2, verbose=1)
        print('Test loss and Accuracy :', test_eval)
        pred = self.full_model.predict(test_X)
        pred = np.argmax(np.round(pred), axis=1)
        Onehot_test_Y2 = np.argmax(np.round(Onehot_test_Y2), axis=1)
        print('prediction : ', pred)
        print('test_Y :', Onehot_test_Y2)

        cm = confusion_matrix(y_test, pred)
        print(cm)
        classes_y = np.unique([train_Y])
        print(classes_y)

        dw.print_confusion_matrix(cm, class_names=classes_y)

if __name__ == '__main__':
    ae_test = Autoencoder_Test()
    ae_test.main()