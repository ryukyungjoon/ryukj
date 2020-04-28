import numpy as np
import pandas as pd


from NSLKDD_Feature_Extraction import Feature_Extraction

from Data_split import Data_split as ds
from Train_model import DNN as dnn
from Drawing import Drawing as dw
from Semi_balancing import Semi_balancing as sb

from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import Counter

class Main:
    if __name__ == "__main__":
        data_loc = "../dataset/NSL-KDD/"
        data_format = ".csv"
        train_data_file = "qnt_KDDTrain_category"
        test_data_file = "qnt_KDDTest_category"
        data_format_txt = ".txt"

        print("Data Loading...")
        train_X, train_Y = ds._load_data_txt(data_loc + train_data_file + data_format_txt)
        test_X, test_Y = ds._load_data_txt(data_loc + test_data_file + data_format_txt)
        print(Counter(test_Y))
        print(Counter(train_Y))
        features = train_X.head(0)

        fe = Feature_Extraction()
        train_data, remain_features = fe.feature_extraction(train_X, train_Y, features)
        train_Y, train_X = train_data['outcome'], train_data.drop('outcome', 1)

        drop_features = Counter(features) - Counter(remain_features)
        drop_features = list(drop_features)
        print(features)
        print(remain_features)
        print(drop_features)


        train_X = np.array(train_X)
        test_X = np.array(test_X)

        classes_y = np.unique([train_Y])
        print(classes_y)
        features = list(features)
        test_X = pd.DataFrame(test_X, columns=features)
        print(test_X)

        # Test dataset's drop features
        for i in range(len(drop_features)):
            test_X = test_X.drop(drop_features[i], axis=1)
        print(test_X)

        test_X = np.array(test_X)

        # Deep Neural Network Learning
        use_model_confusion_matrix, history, pred, y_test = dnn.dnn_model(train_X, train_Y, test_X, test_Y, norm_type='qnt')
        std = StandardScaler()
        confusion_matrixs = std.fit_transform(use_model_confusion_matrix)

        pred = np.argmax(np.round(pred), axis=1)
        print(pred)
        dw.print_confusion_matrix(confusion_matrixs, classes_y)

        raw_encoded, raw_cat = test_Y.factorize()

        y_class = np.unique([raw_encoded])
        acc = accuracy_score(y_test, pred)
        print(acc)
        report = classification_report(y_test, pred, labels=y_class, target_names=classes_y)
        print(str(report))