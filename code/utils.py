''' 
Author: Sasha Sax
Date: Spring 2016

buildDataset and experiment taken from CS224u (Spring 2016)
'''

MAX_COMMENT_LENGTH = 200
NUM_CLASSES = 2
GLOVE_DIMS = 300

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


TRAIN_FILENAME = "../data/train.csv"
DEV_FILENAME = "../data/dev.csv"
TEST_FILENAME = "../data/test.csv"
TRAIN_PLUS_DEV_FILENAME = "../data/train_plus_dev.csv"


# TRAIN_FILENAME = "../data/train_parsed.csv"
# DEV_FILENAME = "../data/dev_parsed.csv"
# TEST_FILENAME = "../data/test_parsed.csv"
# TRAIN_PLUS_DEV_FILENAME = "../data/train_plus_dev_parsed.csv"

import csv 
import codecs
import unicodedata
import re
import pickle
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from collections import Counter, defaultdict
import cs224utils as utils
import pylab as pl
import numpy as np
import pandas as pd
import statsmodels.api as sm
from random import shuffle, sample
from math import ceil



''' Pick a tokenizer '''
# import nltk
# tkzr = nltk.data.load('tokenizers/punkt/english.pickle')

# from nltk.tokenize.stanford import StanfordTokenizer
# tkzr = StanfordTokenizer("../assets/stanford-postagger.jar")

from nltk.tokenize import TweetTokenizer
tkzr = TweetTokenizer()


''' This is to filter URLs and USERS '''
from nlpparser import NLPParser as NLPParser
t_prsr = NLPParser()


def dataReader(filename, tokenize=False):
  ''' Returns an iterator that yields (comment, label) tuples '''
  reader = csv.DictReader(codecs.open(filename, 'rU', 'utf-8'), delimiter=',', quotechar='"')
  for line in reader:
    cleaned = cleanComment(line['comment'])
    if len(cleaned) > MAX_COMMENT_LENGTH - 2:
        continue
    yield (cleaned, int(line['insult']))

def trainReader():
  ''' Convenience function to return the training data set '''
  return dataReader(TRAIN_FILENAME)

def devReader():
  ''' Convenience function to return the dev data set '''
  return dataReader(DEV_FILENAME)

def trainPlusDevReader():
    ''' Convenience function to return the merged training + dev data set '''
    return dataReader(TRAIN_PLUS_DEV_FILENAME)

def testReader():
    ''' Convenience function to return the test data set '''
    return dataReader(TEST_FILENAME)

def cleanComment(comment):
  ''' Returns a cleaned version of the comment '''
  comment = comment.replace('\\\\xc2', ' ')
  comment = comment.replace('\\\\xa0', ' ')
  comment = comment.replace('\\\\n', ' ')
  comment = comment.replace('\\xc2', ' ')
  comment = comment.replace('\\xa0', ' ')
  comment = comment.replace('\\n', ' ')
  comment = comment.replace('\\\\\'', ' ') # yikes
  tkzd = tkzr.tokenize(comment)
  comment = " ".join(tkzd)
  comment = t_prsr.parse(comment).html
  return comment

def load_glove_vectors():
  """Loads in the glove vectors from data/glove.6B """

  glove_home = '../assets/glove/'
  src_filename = os.path.join(glove_home, 'glove.6B.50d.txt')
  reader = csv.reader(open(src_filename), delimiter=' ', quoting=csv.QUOTE_NONE) 
  
  word_to_index = {}
  embedding_mat = []

  counter = 0
  for line in reader:
    word_to_index[line[0]] = counter
    vec = np.array(list(map(float, line[1: ])))
    embedding_mat.append(vec)
    counter += 1

  return word_to_index, embedding_mat

def unigramsPhi(comment):
  ''' The basis of a unigrams feature function
  inputs: 
    comment : a string that contains the comment
  
  outputs:
    a Counter containing the count of each unigram
  '''

  return Counter(comment.split(" "))

def fitNBClassifier(X, y):
    nb = MultinomialNB()
    #     cv = 5
    #     param_grid = {'alpha': [0., 0.5, 1., 2.],
    #                   'fit_prior': [True, False]}
    #     return fit_classifier_with_crossvalidation(X, y, nb, cv, param_grid)
    return nb.fit(X, y)

def fitMaxentClassifier(X, y):    
    """Wrapper for `sklearn.linear.model.LogisticRegression`. This is also 
    called a Maximum Entropy (MaxEnt) Classifier, which is more fitting 
    for the multiclass case.
    
    Parameters
    ----------
    X : 2d np.array
        The matrix of features, one example per row.
        
    y : list
        The list of labels for rows in `X`.
    
    Returns
    -------
    sklearn.linear.model.LogisticRegression
        A trained `LogisticRegression` instance.
    
    """
    mod = LogisticRegression(fit_intercept=True)
    mod.fit(X, y)
    return mod

def fitSVCClassifier(X, y):
    mod = SVC(probability=True)
    mod.fit(X, y)
    return mod


def fitClassifierWithCrossvalidation(X, y, basemod, cv, param_grid, scoring='f1_macro'): 
    """Fit a classifier with hyperparmaters set via cross-validation.

    Parameters
    ----------
    X : 2d np.array
        The matrix of features, one example per row.
        
    y : list
        The list of labels for rows in `X`.  
    
    basemod : an sklearn model class instance
        This is the basic model-type we'll be optimizing.
    
    cv : int
        Number of cross-validation folds.
        
    param_grid : dict
        A dict whose keys name appropriate parameters for `basemod` and 
        whose values are lists of values to try.
        
    scoring : value to optimize for (default: f1_macro)
        Other options include 'accuracy' and 'f1_micro'. See
        http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            
    Prints
    ------
    To standard output:
        The best parameters found.
        The best macro F1 score obtained.
        
    Returns
    -------
    An instance of the same class as `basemod`.
        A trained model instance, the best model found.
        
    """    
    # Find the best model within param_grid:
    crossvalidator = GridSearchCV(basemod, param_grid, cv=cv, scoring=scoring)
    crossvalidator.fit(X, y)
    # Report some information:
    print "Best params", crossvalidator.best_params_
    print "Best score: %0.03f" % crossvalidator.best_score_
    # Return the best model found:
    return crossvalidator.best_estimator_

def fitMaxentWithCrossvalidation(X, y):
    """A MaxEnt model of dataset with hyperparameter 
    cross-validation. Some notes:
        
    * 'fit_intecept': whether to include the class bias feature.
    * 'C': weight for the regularization term (smaller is more regularized).
    * 'penalty': type of regularization -- roughly, 'l1' ecourages small 
      sparse models, and 'l2' encourages the weights to conform to a 
      gaussian prior distribution.
    
    Other arguments can be cross-validated; see 
    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    
    Parameters
    ----------
    X : 2d np.array
        The matrix of features, one example per row.
        
    y : list
        The list of labels for rows in `X`.   
    
    Returns
    -------
    sklearn.linear_model.LogisticRegression
        A trained model instance, the best model found.
    
    """    
    basemod = LogisticRegression()
    cv = 5
    param_grid = {'fit_intercept': [True, False], 
                  'C': [0.4, 0.6, 0.8, 1.0, 2.0, 3.0],
                  'penalty': ['l1','l2'],
                  'scoring': [roc_auc_score]}    
    return fitClassifierWithCrossvalidation(X, y, basemod, cv, param_grid)

def buildDataset(reader, phi, class_func=lambda x: x, vectorizer=None, tfidf=False):
    """Core general function for building experimental datasets.
    
    Parameters
    ----------
    reader : iterator
       Should be `train_reader`, `dev_reader`, or another function
       defined in those terms. This is the dataset we'll be 
       featurizing.
       
    phi : feature function
       Any function that takes an `nltk.Tree` instance as input 
       and returns a bool/int/float-valued dict as output.
       
    class_func : function on the SST labels
       Any function like `binary_class_func` or `ternary_class_func`. 
       This modifies the SST labels based on the experimental 
       design. If `class_func` returns None for a label, then that 
       item is ignored.
       
    vectorizer : sklearn.feature_extraction.DictVectorizer    
       If this is None, then a new `DictVectorizer` is created and
       used to turn the list of dicts created by `phi` into a 
       feature matrix. This happens when we are training.
              
       If this is not None, then it's assumed to be a `DictVectorizer` 
       and used to transform the list of dicts. This happens in 
       assessment, when we take in new instances and need to 
       featurize them as we did in training.
       
    Returns
    -------
    dict
        A dict with keys 'X' (the feature matrix), 'y' (the list of
        labels), 'vectorizer' (the `DictVectorizer`), and 
        'raw_examples' (the `nltk.Tree` objects, for error analysis).
    
    """    
    labels = []
    feat_dicts = []
    raw_examples = []
    for comment, label in reader():
        cls = class_func(label)
        # None values are ignored -- these are instances we've
        # decided not to include.
        if cls != None:
            labels.append(cls)
            feat_dicts.append(phi(comment))
            raw_examples.append(comment)
    feat_matrix = None
    # In training, we want a new vectorizer:    
    if vectorizer == None:
        vectorizer = DictVectorizer(sparse=True)
        if tfidf:
            vectorizer = TfidfVectorizer(lowercase=False) #DictVectorizer(sparse=True) #TfidfVectorizer() # Or use DictVectorizer(sparse=True)
        feat_matrix = vectorizer.fit_transform(feat_dicts)
    # In assessment, we featurize using the existing vectorizer:
    else:
        feat_matrix = vectorizer.transform(feat_dicts)
    return {'X': feat_matrix, 
            'y': labels, 
            'vectorizer': vectorizer, 
            'raw_examples': raw_examples}

def experiment(
    train_reader=trainReader, 
    assess_reader=None, 
    train_size=0.8,
    phi=unigramsPhi, 
    class_func=lambda x: x,
    train_func=fitMaxentClassifier,
    score_func=utils.safe_macro_f1,
    verbose=True,
    tfidf=False,
    return_model=[]):
    """Generic experimental framework for SST. Either assesses with a 
    random train/test split of `train_reader` or with `assess_reader` if 
    it is given.
    
    Parameters
    ----------
    train_reader : SST iterator (default: `train_reader`)
        Iterator for training data.
       
    assess_reader : iterator or None (default: None)
        If None, then the data from `train_reader` are split into 
        a random train/test split, with the the train percentage 
        determined by `train_size`. If not None, then this should 
        be an iterator for assessment data (e.g., `dev_reader`).
        
    train_size : float (default: 0.7)
        If `assess_reader` is None, then this is the percentage of
        `train_reader` devoted to training. If `assess_reader` is
        not None, then this value is ignored.
       
    phi : feature function (default: `unigrams_phi`)
        Any function that takes an `nltk.Tree` instance as input 
        and returns a bool/int/float-valued dict as output.
       
    class_func : function on the SST labels
        Any function like `binary_class_func` or `ternary_class_func`. 
        This modifies the SST labels based on the experimental 
        design. If `class_func` returns None for a label, then that 
        item is ignored.
       
    train_func : model wrapper (default: `fit_maxent_classifier`)
        Any function that takes a feature matrix and a label list
        as its values and returns a fitted model with a `predict`
        function that operates on feature matrices.
    
    score_metric : function name (default: `utils.safe_macro_f1`)
        This should be an `sklearn.metrics` scoring function. The 
        default is weighted average F1 (macro-averaged F1). For 
        comparison with the SST literature, `accuracy_score` might
        be used instead. For micro-averaged F1, use
        
        (lambda y, y_pred : f1_score(y, y_pred, average='micro', pos_label=None))
                
        For other metrics that can be used here, see
        see http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
        
    verbose : bool (default: True)
        Whether to print out the model assessment to standard output.
        Set to False for statistical testing via repeated runs.
       
    Prints
    -------    
    To standard output, if `verbose=True`
        Model accuracy and a model precision/recall/F1 report. Accuracy is 
        reported because many SST papers report that figure, but the 
        precision/recall/F1 is better given the class imbalances and the 
        fact that performance across the classes can be highly variable.
        
    Returns
    -------
    float
        The overall scoring metric as determined by `score_metric`.
    
    """        
    # Train dataset:
    train = buildDataset(train_reader, phi, class_func, vectorizer=None, tfidf=tfidf)
    # Manage the assessment set-up:
    X_train = train['X']
    y_train = train['y']
    X_assess = None 
    y_assess = None
    if assess_reader == None:
         X_train, X_assess, y_train, y_assess = train_test_split(
                X_train, y_train, train_size=train_size)
    else:
        # Assessment dataset using the training vectorizer:
        assess = buildDataset(
            assess_reader, 
            phi, 
            class_func, 
            vectorizer=train['vectorizer'])
        X_assess, y_assess = assess['X'], assess['y']
    # Train:      
    mod = train_func(X_train, y_train)    
    return_model.append(mod) # Optionally return the model
    return_model.append(train['vectorizer'])

    # Predictions:
    threshold = 0.5
    predict_probs = mod.predict_proba(X_assess)[:,1]
    predictions = np.copy(predict_probs)
    predictions[predictions > threshold] = 1
    predictions[predictions <= threshold] = 0

    # Report:
    if verbose:
        print('Percent insults: %0.03f' % (1.*sum(y_assess)/len(y_assess)))
        print(classification_report(y_assess, predictions, digits=3))
        print('Accuracy: %0.03f' % accuracy_score(y_assess, predictions))
        print 'AUC:     ', roc_auc_score(y_assess, predict_probs, average='micro') # micro for comparison to kaggle
        print 'AUC:     ', roc_auc_score(y_assess, predict_probs) # micro for comparison to kaggle
    # Return the overall score:
    return score_func(y_assess, predictions)



class Vocab(object):
  def __init__(self):
    self.word_to_index = {}
    self.index_to_word = {}
    self.word_freq = defaultdict(int)
    self.total_words = 0
    self.unknown = '@'
    self.add_word(self.unknown, count=0)
    self.start = '<GO>'
    self.end = '<STOP>'
    self.add_word(self.start, count=0)
    self.add_word(self.end, count=0)

  def add_word(self, word, count=1):
    if word not in self.word_to_index:
      index = len(self.word_to_index)
      self.word_to_index[word] = index
      self.index_to_word[index] = word
    self.word_freq[word] += count

  def construct(self, dataset):
    for comment, label in dataset: 
        for word in comment:
          self.add_word(word)
    self.total_words = float(sum(self.word_freq.values()))
    print '{} total chars with {} uniques'.format(self.total_words, len(self.word_freq))

  def encode(self, word):
    if word not in self.word_to_index:
      word = self.unknown
    return self.word_to_index[word]

  def decode(self, index):
    return self.index_to_word[index]

  def __len__(self):
    return len(self.word_freq)

def char_iterator(raw_data, batch_size, downsample=True, char_dropout=None):
    """
    Takes in a list of (EncodedComment, label) tuples where 
        EncodedComment is a list of indices.
    returns data, label, data_length
    """
    comments, labels = zip(*raw_data)

    def shortEnough(idx):
        return (len(raw_data[idx][0]) <= MAX_COMMENT_LENGTH)
    
    def isInsult(idx):
        return (raw_data[idx][1] == 1)

    # Because there are so many more noninsults, we have to downsample the insults
    if downsample:
        insults = [i for i in range(len(raw_data)) if shortEnough(i) and isInsult(i)]  # We'll be down-sampling non-insults
        nonInsults = [i for i in range(len(raw_data)) if shortEnough(i) and not isInsult(i)]
        idxs = sample(nonInsults, len(insults)) + insults
        downsampled = [raw_data[idx] for idx in idxs]
        size_sorted_data = sorted(downsampled, key=lambda x: len(x[0]), reverse=True)
    else:
        size_sorted_data = sorted(raw_data, key=lambda x: len(x[0]), reverse=True)

    epoch_size = len(size_sorted_data) // batch_size
    randomized_batch_idxs = range(epoch_size) # Only select the batches we can use
    shuffle(randomized_batch_idxs)


    # put data into a large matrix
    data = np.zeros([len(size_sorted_data), MAX_COMMENT_LENGTH])
    labels = np.zeros([len(size_sorted_data), NUM_CLASSES])
    seq_lengths = np.zeros([len(size_sorted_data), ])
    for i, (comment, label) in enumerate(size_sorted_data):
        if char_dropout is not None:
            n_keep=int(ceil(char_dropout * len(comment)))
            sampled = sample(range(1, len(comment)-1), n_keep - 2)
            kept_idxs = [0] + sorted( sampled ) + [len(comment) - 1] # Select the columns to keep
            comment = comment[np.array(kept_idxs)]
        seq_lengths[i] = len(comment)
        data[i,:] += np.pad(comment, (0, MAX_COMMENT_LENGTH - len(comment)), mode='constant', constant_values=0)
        labels[i, label] = 1.

    def batch(i):
        """
        Returns the required batch in the following form:
            x : [seq_length x batch_size] array which contains sequences
            y : [batch_size] vector of labels
            seq_length : [batch_size] vector of the length of each comment in the batch
        """
        batch_data = data[i * batch_size : (i + 1) * batch_size]
        batch_seq_lengths = seq_lengths[i * batch_size : (i + 1) * batch_size]
        batch_labels = labels[i * batch_size : (i + 1) * batch_size]
        # print "Seq lengths for batch:", batch_seq_lengths
        return batch_data.T, batch_labels, batch_seq_lengths

    for i in randomized_batch_idxs:
        yield batch(i)

def quickPickle(obj, filename):
    with open(filename, 'w') as f:
        pickle.dump(obj, f, protocol=0)  # Human readabled

def quickUnpickle(filename):
    with open(filename) as f:
        return pickle.load(f)

def parse_data(filename):
    file_toks = filename.split(".")
    file_toks[-2] = file_toks[-2] + "_parsed"
    newfilename = ".".join(file_toks)
    reader = csv.DictReader(codecs.open(filename, 'rU', 'utf-8'), delimiter=',', quotechar='"')
    with open(newfilename, "wb") as writef:
        writer = csv.DictWriter(writef, delimiter=',',  quotechar='"', fieldnames=["comment", "insult"])
        writer.writeheader()
        for line in reader:
            line['comment'] = cleanComment(line['comment'])[1:-1] # Clip the quotes
            writer.writerow({"comment": line['comment'], "insult": line["insult"]})

if __name__ == "__main__":
    pass
