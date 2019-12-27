from sklearn.model_selection import train_test_split

class Data_split:
    def data_split(split_data):
        print("[train_set] <=> [test_set]")

        train_set, test_set = train_test_split(split_data, test_size=0.3)

        train_Y, train_X = train_set['Label'], train_set.drop('Label', 1)
        test_Y, test_X = test_set['Label'], test_set.drop('Label', 1)
        raw_encoded, raw_cat = test_set["Label"].factorize()

        return train_X, train_Y, test_X, test_Y, raw_encoded, raw_cat