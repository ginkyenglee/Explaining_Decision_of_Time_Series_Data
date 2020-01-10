import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import numpy as np
import os
def plot_train_history(train_df, valid_df,save_path):

    #Loss graph
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1,2,1)
    ax.plot(train_df['loss'].tolist(), label='train loss')
    ax.plot(train_df['loss'].tolist(), label='valid loss', color='Red')

    plt.xlim(0,len(train_df))
    plt.legend(fontsize=12, loc='upper right')
    plt.title('Loss graph', fontsize=15)
    plt.xlabel('epoch', fontsize=13)
    plt.ylabel('loss', fontsize=13)

    plt.savefig(os.path.join(save_path,  'loss_graph.png'))
    print("save {}".format(os.path.join(save_path,  'loss_graph.png')))
    
    #Acc graph
    ax = fig.add_subplot(1,2,2)
    ax.plot(train_df['acc'].tolist(), label='train acc')
    ax.plot(train_df['acc'].tolist(), label='valid acc', color='Red')

    plt.xlim(0,None)
    plt.legend(fontsize=12, loc='upper right')
    plt.title('Acc graph', fontsize=15)
    plt.xlabel('epoch', fontsize=13)
    plt.ylabel('acc', fontsize=13)


    plt.savefig(os.path.join(save_path, 'acc_graph.png'))

    print("save {}".format(os.path.join(save_path, 'acc_graph.png')))
            

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig=plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt =  'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
    plt.subplot(122)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.tight_layout()
    plt.show()