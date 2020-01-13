import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Drawing:
    def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 10), fontsize=14):
        df_cm = pd.DataFrame(
            confusion_matrix, index=class_names, columns=class_names,)
        plt.figure(figsize=figsize)
        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def print_score_graph(model_score, index, columns, figsize=(10, 10)):
        df_sg = pd.DataFrame(
            model_score, index=index, columns=columns,)
        fig, ax = plt.subplot(1, 2)
        df_sg.plot(kind='bar', ax=ax[0])
        plt.figure(figsize=figsize)
        plt.ylabel("score")
        plt.xlabel("")
        plt.show()
