from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, f1_score

class Train_model:
    def train_model(train_X, train_Y, classes_y, test_X, test_Y):
        print('model training...')

        print('MLP Classifier')
        use_model = MLPClassifier(hidden_layer_sizes=(30, 10), max_iter=10, random_state=42)

        mini_batch_size = 10000
        batch_size = len(train_Y)
        total_epoch = int(batch_size / mini_batch_size)
        current_batch = 0

        for i in range(1, total_epoch):
            end_batch = i * mini_batch_size
            use_model._partial_fit(train_X[current_batch:end_batch], train_Y[current_batch:end_batch], classes=classes_y)
            current_batch = end_batch

        use_model._partial_fit(train_X[current_batch:batch_size], train_Y[current_batch:batch_size], classes=classes_y)
        use_model_score = cross_val_score(use_model, test_X, test_Y, scoring='accuracy', cv=10, n_jobs=8)
        use_model_cross_val = cross_val_predict(use_model, test_X, test_Y, cv=10, n_jobs=8)
        use_model_confusion_matrix = confusion_matrix(test_Y, use_model_cross_val)

        return use_model_cross_val, use_model_score, use_model_confusion_matrix