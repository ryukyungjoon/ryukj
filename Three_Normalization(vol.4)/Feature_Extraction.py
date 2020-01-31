from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Feature_Extraction:
    def feature_extraction_1(x, y, features):
        RF = RandomForestClassifier(random_state=0, n_jobs=-1)
        RFmodel = RF.fit(x, y)
        Importances = RFmodel.feature_importances_
        print(Importances)
        std = np.std([tree.feature_importances_ for tree in RFmodel.estimators_],
                     axis=0)
        indices = np.argsort(Importances)[::-1]
        print('Feature Ranking:')
        for f in range(x.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], Importances[indices[f]]))

        # Drawing Feature Importance Graph
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(x.shape[1]), Importances[indices],
                color="b", yerr=std[indices])
        plt.xticks(range(x.shape[1]), indices, rotation=90)
        plt.xlim([-1, x.shape[1]])
        plt.show()

        xx = pd.DataFrame(x)

        #Feature Importance
        print("indices:", indices)
        for k in range(len(Importances)):
            if Importances[indices[k]] < 0.001:
                f = indices[k]
                xx = xx.drop(features[f], axis=1)

        # Feature Correlation
        corr = xx.rank().corr(method="spearman")
        corr = pd.DataFrame(corr)
        columns = np.full((corr.shape[0],), True, dtype=bool)

        for i in range(corr.shape[0]):
            for j in range(i + 1, corr.shape[0]):
                if corr.iloc[i, j] >= 0.95:
                    if columns[j]:
                        columns[j] = False

        selected_columns = xx.columns[columns]
        xx = xx[selected_columns]
        print("XX: {}".format(xx))
        fe_data = pd.concat([xx, y], axis=1)
        remain_features = fe_data.head(0)
        fe_data = pd.DataFrame(fe_data)
        fe_data.to_csv("../dataset/fin_dataset/Feature_Extraction_1.csv", header=list(remain_features), index=False)

        return fe_data