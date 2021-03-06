import pandas as pd
import numpy as np

from Three_Normalization.Data_cleaning import Cleaning as cl
from Three_Normalization.Semi_balancing import Semi_balancing as sb
from Three_Normalization.Feature_Extraction import Feature_Extraction as fe
from Three_Normalization.Data_Normalization import Data_Normalization as dn
from Three_Normalization.Data_split import Data_split as ds
from Three_Normalization.Train_model import Train_model as tm
from Three_Normalization.Train_model import DNN as dnn
from Three_Normalization.Drawing import Drawing as dw

class Main:
    if __name__ == "__main__":
        features = ['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label']
        data_loc = "../dataset/fin_dataset/"
        data_format = ".csv"
        data_file = "5. combine_dataset"

        print("Data Loading...")
        data = pd.read_csv(data_loc + data_file + data_format, sep=',', dtype='unicode')

        data, data_Label = cl.data_cleaning(data)
        classes_y = np.unique(data_Label)

        sampling_strategy = ['1', '2']

        for i in range(1, len(sampling_strategy)):
            if i == 1:
                new_data = sb.sampling_1(data, data_Label, features)
            elif i == 2:
                new_data = sb.sampling_2(data, data_Label, features)

            new_data = pd.DataFrame(new_data)
            print(new_data)
            bal_data_Label, bal_data = new_data['Label'], new_data.drop('Label', 1)

            fe = fe.feature_extraction_1(bal_data, bal_data_Label, features)

            normalization_type = ['mms', 'std', 'qnt']
            for norm in range(0, len(normalization_type)):
                norm_data = dn.normalizations(fe, normalization_type[norm])
                train_X, train_Y, test_X, test_Y, raw_encoded, raw_cat = ds.data_split(norm_data)
                use_model_confusion_matrix = tm.train_model(train_X, train_Y, classes_y, test_X, test_Y)
                # report = classification_report(test_Y, use_model_cross_val, target_names=raw_cat)

                # 그래프 그리기(confusion matrix, histogram)
                dw.print_confusion_matrix(use_model_confusion_matrix, classes_y)
                # dw.print_score_graph(use_model_score, index=classes_y, columns=classes_y)

                # print(str(report))

            for norm in range(0, len(normalization_type)):
                norm_data = dn.normalizations(fe, normalization_type[norm])
                train_X, train_Y, test_X, test_Y, raw_encoded, raw_cat = ds.data_split(norm_data)
                performance_test, history = dnn.dnn_model(train_X, train_Y, test_X, test_Y)
                print('Test Loss and Accuracy ->', performance_test)

                # dw.drawplot(history)