from sklearn.preprocessing import QuantileTransformer

import pandas as pd
import numpy as np

class Data_Normalization_AE:
    def normalizations_ae(train_X, test_X, train_Y, test_Y, normalization_type, subsample):
        print('Data Normalizing...')
        train_y = train_Y
        test_y = test_Y

        train_x = pd.DataFrame(train_X)
        data = pd.concat([train_x, train_Y], axis=1)

        train_x.dropna(axis=0)
        test_X.dropna(axis=0)
        train_x = train_x[(train_x.T != 0).any()]
        test_X = test_X[(test_X.T != 0).any()]
        remain_features = data.head(n=0)
        print("remain_features:", remain_features)

        if normalization_type == 'qnt':
            qnt_train = QuantileTransformer(n_quantiles=15, subsample=subsample[0])
            qnt_test = QuantileTransformer(n_quantiles=15, subsample=subsample[1])

            x_train = qnt_train.fit_transform(train_X)
            x_test = qnt_test.fit_transform(test_X)
            print("qnt: ")
            qnt_train = pd.DataFrame(x_train)
            qnt_test = pd.DataFrame(x_test)
            norm_train = qnt_train[(qnt_train.T != 0).any()]  # Remove Zero records
            norm_test = qnt_test[(qnt_test.T != 0).any()]  # Remove Zero records
            norm_train = pd.concat([norm_train, train_Y], axis=1)
            norm_test = pd.concat([norm_test, train_Y], axis=1)

        print("qnt train:", norm_train)
        print("qnt test:", norm_test)

        norm_train.dropna(axis=1)
        norm_train.dropna(axis=0)
        norm_test.dropna(axis=1)
        norm_test.dropna(axis=0)

        norm_train.to_csv('../dataset/fin_dataset/qnt_train(AE).csv', header=list(remain_features), index=False)
        norm_test.to_csv('../dataset/fin_dataset/qnt_test(AE).csv', header=list(remain_features), index=False)

        return norm_train, norm_test, train_y, test_y