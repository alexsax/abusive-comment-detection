from numpy import mean
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score

class Evaluator(object):
  def __init__(self):
    self.predicted = []
    self.gold = []
    self.probs = []
    self.loss = []

  def report(self):
    self.roc = roc_auc_score(self.gold, self.probs, average='micro')
    self.ce = mean(self.loss)
    self.f1 = f1_score(self.gold, self.predicted)
    print "\tROC:", self.roc 
    print "\tCE:", self.ce
    print "\tF1:", self.f1
    print classification_report(self.gold, self.predicted, target_names=["clean", "insulting"])
    self.precision = precision_score(self.gold, self.predicted)
    self.recall = recall_score(self.gold, self.predicted)

  def summary_dict(self):
    confusion = confusion_matrix(self.gold, self.predicted, labels=["clean", "insulting"])
    del self.predicted
    del self.gold
    del self.loss
    del self.probs
    return {"roc": self.roc,
            "ce": self.ce,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall, 
            "confusion": confusion
    }