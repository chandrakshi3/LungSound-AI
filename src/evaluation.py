from sklearn.metrics import f1_score,precision_score,recall_score,multilabel_confusion_matrix
def evaluate_predictions(y_true,y_pred):
    out={'macro_f1':float(f1_score(y_true,y_pred,average='macro',zero_division=0)),'labels':{}}
    for i,label in enumerate(['crackles','wheezes']): out['labels'][label]={'precision':float(precision_score(y_true[:,i],y_pred[:,i],zero_division=0)),'recall':float(recall_score(y_true[:,i],y_pred[:,i],zero_division=0)),'f1':float(f1_score(y_true[:,i],y_pred[:,i],zero_division=0))}
    out['multilabel_confusion_matrices']=multilabel_confusion_matrix(y_true,y_pred).tolist(); return out
