import seaborn as sns
import matplotlib.pyplot as plt


def plot_boxplot(column,target_column,dataset,title = " "):
    axes = sns.boxplot(x = target_column , y = column , palette=["c","r"],hue=target_column,data=dataset).set_title(title,fontsize=15)
    plt.show()


def data_analysis(dataset_complete):
    target_column = 'diagnosis'
    for column in dataset_complete.columns:
        if column == target_column:
            continue
        if dataset_complete[column].dtype == 'int64' or dataset_complete[column].dtype == 'float64':
                plot_boxplot(column,'diagnosis',dataset_complete)        
        else:
            None
