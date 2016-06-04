from utils import *
from random import shuffle

def buildDataset(reader):
    labels = []
    feat_dicts = []
    raw_examples = []
    for comment, label in reader():
        labels.append(label)
        raw_examples.append(comment)
    res = zip(raw_examples, labels)
    shuffle(res)
    return res



for c, l in buildDataset(trainReader)[:50]:
	print c, "::", l