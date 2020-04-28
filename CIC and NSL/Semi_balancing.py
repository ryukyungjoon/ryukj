from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import pandas as pd
import numpy as np

class Semi_balancing:
    def sampling_1(x, y, features, data_file):
        print("sampling strategy-1")
        down_dic = {
            'BENIGN': 250000,
            'Bot': 1956,
            'DDoS': 128025,
            'DoS GoldenEye': 10293,
            'DoS Hulk': 230124,
            'DoS Slowhttptest': 5499,
            'DoS slowloris': 5796,
            'FTP-Patator': 7935,
            'Heartbleed': 11,
            'Infiltration': 36,
            'SSH-Patator': 5897,
            'PortScan': 158804,
            'Web Attack Brute Force': 1507,
            'Web Attack Sql Injection': 21,
            'Web Attack XSS': 652
        }
        up_dic = {
            'BENIGN': 250000,
            'Bot': 5000,
            'DDoS': 128025,
            'DoS GoldenEye': 10293,
            'DoS Hulk': 230124,
            'DoS Slowhttptest': 5499,
            'DoS slowloris': 5796,
            'FTP-Patator': 7935,
            'Heartbleed': 5000,
            'Infiltration': 5000,
            'SSH-Patator': 5897,
            'PortScan': 158804,
            'Web Attack Brute Force': 5000,
            'Web Attack Sql Injection': 5000,
            'Web Attack XSS': 5000
        }

        rus = RandomUnderSampler(sampling_strategy=down_dic, random_state=0)
        sm = SMOTE(kind='regular', sampling_strategy=up_dic, random_state=0)

        print("Data Resampling...")
        x_resampled, y_resampled = rus.fit_sample(x, y)
        x_resampled_1, y_resampled_1 = sm.fit_sample(x_resampled, y_resampled)
        x_resampled_2 = pd.DataFrame(x_resampled_1)
        y_resampled_2 = pd.DataFrame(y_resampled_1)

        data_resampled = pd.concat([x_resampled_2, y_resampled_2], axis=1)
        data_resampled = pd.DataFrame(data_resampled)
        print(data_resampled)

        print("After OverSampling, the shape of data:{}\n".format(data_resampled.shape))

        data_resampled.to_csv("../dataset/fin_dataset/"+data_file+"_bal.csv", header=features, index=False)
        data_resampled = pd.read_csv("../dataset/fin_dataset/"+data_file+"_bal.csv", sep=',', dtype='unicode')

        return data_resampled

    def sampling_2(x, y, features):
        print("sampling strategy-2")
        down_dic = {
            'BENIGN': 150000,
            'Bot': 1956,
            'DDoS': 128025,
            'DoS GoldenEye': 10293,
            'DoS Hulk': 150000,
            'DoS Slowhttptest': 5499,
            'DoS slowloris': 5796,
            'FTP-Patator': 7935,
            'Heartbleed': 11,
            'Infiltration': 36,
            'SSH-Patator': 5897,
            'PortScan': 150000,
            'Web Attack Brute Force': 1507,
            'Web Attack Sql Injection': 21,
            'Web Attack XSS': 652
        }
        up_dic = {
            'BENIGN': 150000,
            'Bot': 150000,
            'DDoS': 150000,
            'DoS GoldenEye': 150000,
            'DoS Hulk': 150000,
            'DoS Slowhttptest': 150000,
            'DoS slowloris': 150000,
            'FTP-Patator': 150000,
            'Heartbleed': 150000,
            'Infiltration': 150000,
            'SSH-Patator': 150000,
            'PortScan': 150000,
            'Web Attack Brute Force': 150000,
            'Web Attack Sql Injection': 150000,
            'Web Attack XSS': 150000
        }

        rus = RandomUnderSampler(sampling_strategy=down_dic, random_state=0)
        sm = SMOTE(kind='regular', sampling_strategy=up_dic, random_state=0)

        print("Data Resampling...")
        x_resampled, y_resampled = rus.fit_sample(x, y)
        x_resampled_1, y_resampled_1 = sm.fit_sample(x_resampled, y_resampled)
        x_resampled_2 = pd.DataFrame(x_resampled_1)
        y_resampled_2 = pd.DataFrame(y_resampled_1)

        data_resampled = pd.concat([x_resampled_2, y_resampled_2], axis=1)
        data_resampled = pd.DataFrame(data_resampled)
        print(data_resampled)

        print("After OverSampling, the shape of train_x:{}".format(data_resampled.shape))
        print("After OverSampling, the shape of train_x:{} \n".format(data_resampled.shape))

        # data_resampled.to_csv("../dataset/fin_dataset/(논문구현_dilated2).csv", header=features, index=False)
        # data_resampled = pd.read_csv("../dataset/fin_dataset/(논문구현_dilated2).csv", sep=',', dtype='unicode')

        return data_resampled

    def nsl_sampling_1(x, y, features, sampling_strategy):
        train_up_dic = {
            'normal': 67343,
            'DoS': 45927,
            'Probe': 11656,
            'R2L': 3000,
            'U2R': 1500
        }

        test_up_dic = {
            'normal': 9711,
            'DoS': 7458,
            'Probe': 2421,
            'R2L': 3000,
            'U2R': 1500
        }

        print("Data Resampling...")
        if sampling_strategy == 'test':
            print("in")
            test_sm = SMOTE(kind='borderline1', sampling_strategy=test_up_dic, random_state=0)
            a, b = test_sm.fit_sample(x, y)

        elif sampling_strategy == 'train':
            train_sm = SMOTE(kind='borderline1', sampling_strategy=train_up_dic, random_state=0)
            a, b = train_sm.fit_sample(x, y)

        label = ['outcome']
        sampled_x = pd.DataFrame(a, columns=list(features))
        sampled_y = pd.DataFrame(b, columns=list(label))

        data_resampled = pd.concat([sampled_x, sampled_y], axis=1)

        print(data_resampled)

        print("After OverSampling, the shape of train_x:{}".format(data_resampled.shape))
        print("After OverSampling, the shape of train_x:{} \n".format(data_resampled.shape))

        data_resampled.to_csv('../dataset/fin_dataset/NSL-KDD_semi_dataset1['+sampling_strategy+'].txt', mode='w', index=False)

        return data_resampled