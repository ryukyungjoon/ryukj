from sklearn.preprocessing import minmax_scale, StandardScaler, quantile_transform

import pandas as pd

class Data_Normalization:
    def normalizations(fe_data, normalization_type):
        print('Data Normalizing...')
        fe_data1 = pd.DataFrame(fe_data)
        # fe_data1 = fe_data1.rename(columns={'Fwd Header Length.1': 'Fwd Header Length'})
        y, x = fe_data1['Label'], fe_data1.drop('Label', 1)
        x.dropna(axis=0)
        x = x[(x.T != 0).any()]
        remain_features = fe_data1.head(n=0)

        if normalization_type == 'mms':
            mms = minmax_scale(x, feature_range=(0, 255))
            print("mms: ")
            mms = pd.DataFrame(mms)
            y = pd.DataFrame(y)
            norm_set = mms[(mms.T != 0).any()]  # Remove Zero records
            norm_set = pd.concat([norm_set, y], axis=1)

        if normalization_type == 'std':
            std = StandardScaler()
            print("std: ")
            x_scale = std.fit_transform(x)
            x_scale = pd.DataFrame(x_scale)
            y = pd.DataFrame(y)
            norm_set = x_scale[(x_scale.T != 0).any()]  # Remove Zero records
            norm_set = pd.concat([norm_set, y], axis=1)

        if normalization_type == 'qnt':
            qnt = quantile_transform(x, n_quantiles=15, subsample=832373)
            print("qnt: ")
            y = pd.DataFrame(y)
            qnt = pd.DataFrame(qnt)
            norm_set = qnt[(qnt.T != 0).any()]  # Remove Zero records
            print(type(norm_set))
            norm_set = pd.concat([norm_set, y], axis=1)

        print(norm_set)

        norm_set.dropna(axis=1)
        norm_set.to_csv("../dataset/fin_dataset/"+normalization_type+"4.csv", header=list(remain_features), index=False)

        return norm_set