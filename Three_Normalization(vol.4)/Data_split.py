from sklearn.model_selection import train_test_split

import numpy as np

class Data_split:
    def data_split(split_data):
        print("[train_set] <=> [test_set]")

        train_set, test_set = train_test_split(split_data, test_size=0.3)

        train_Y, train_X = train_set['Label'], train_set.drop('Label', 1)
        test_Y, test_X = test_set['Label'], test_set.drop('Label', 1)
        raw_encoded, raw_cat = test_set["Label"].factorize()

        return train_X, train_Y, test_X, test_Y, raw_encoded, raw_cat

    def _load_data(path):
        load_data = np.load(path)

        train_data, test_data = train_test_split(load_data, test_size=0.3, random_state=13)

        train_x = train_data[:, :-1].astype(float)
        train_y = train_data[::, -1]

        test_x = test_data[:, :-1].astype(float)
        test_y = test_data[::, -1]

        return train_x, test_x, train_y, test_y