from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import pandas as pd

class Semi_balancing:
    def sampling_1(x, y, features):
        print("semi-balancing")
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

        print("After OverSampling, the shape of train_x:{}".format(data_resampled.shape))
        print("After OverSampling, the shape of train_x:{} \n".format(data_resampled.shape))

        data_resampled.to_csv("../dataset/fin_dataset/(논문구현_dilated).csv", header=features, index=False)
        data_resampled = pd.read_csv("../dataset/fin_dataset/(논문구현_dilated).csv", sep=',', dtype='unicode')

        return data_resampled


