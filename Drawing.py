import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Drawing:
    def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
        df_cm = pd.DataFrame(
            confusion_matrix, index=class_names, columns=class_names,)
        fig = plt.figure(figsize=figsize)
        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
