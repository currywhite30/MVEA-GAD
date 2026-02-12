from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score
)

class Evaluator:
    def __init__(self) -> None:
        pass

    def eval_roc_auc(self, label, score):
        roc_auc = roc_auc_score(y_true=label, y_score=score)
        return roc_auc
    
    def eval_recall_at_k(self, label, score, k=None):
        if k is None:
            k = sum(label)
        recall_at_k = sum(label[score.topk(k).indices]) / sum(label)
        return recall_at_k
    
    def eval_precision_at_k(self, label, score, k=None):
        if k is None:
            k = sum(label)
        precision_at_k = sum(label[score.topk(k).indices]) / k
        return precision_at_k

    def eval_average_precision(self, label, score):
        average_precision = average_precision_score(y_true=label, y_score=score)
        return average_precision
    
    def eval_f1(self, label, pred):
        f1 = f1_score(y_true=label, y_pred=pred)
        return f1
    
