from sklearn.metrics import f1_score, roc_auc_score, recall_score
import numpy as np

def compute_metrics(labels, probs, preds):
    auc= roc_auc_score(labels, probs)
    accuracy= np.mean(np.array(preds)==np.array(labels))
    f1= f1_score(labels, preds)
    recall= recall_score(labels, preds)
    specificity= recall_score(labels, preds, pos_label= 0)

    return {
        "auc": auc,
        "accuracy": accuracy,
        "f1": f1,
        "recall": recall,
        "specificity": specificity
    }

