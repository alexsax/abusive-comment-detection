'''
A LogisticRegression baseline for the abusive comment detection task.

Implements a LogisticRegression classifier using tfidf-weighted unigram+bigram 
  features, with some regularizing (lowering, stemming, and URL replacement). 

This places it in 13th/50 place on the Kaggle task here: 
  https://www.kaggle.com/c/detecting-insults-in-social-commentary/leaderboard/private
Naive Bayes was tried with similar but lower scores than maxent

Things that didn't help: 
  stopwords/punctuation removal
'''

from utils import *
from nltk.util import bigrams
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import os
stemmer = LancasterStemmer()

from nltk.corpus import stopwords
from model import interactive_version
from nltk.tokenize import TweetTokenizer
tkzr = TweetTokenizer()
from nlpparser import NLPParser as NLPParser
t_prsr = NLPParser()

stop_words = set(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']) # remove it if you need punctuation 
word_to_index = {}
embedding_mat = []



def newUnigramsPhi(comment): # Overrides the one from utils
  ''' The basis of a unigrams feature function
  inputs: 
    comment : a string that contains the comment
  
  outputs:
    a Counter containing the count of each unigram
  '''
  lowered = [tok for tok in comment.split()] # Stemming + punc
  return Counter(lowered)

def bigramsPhi(comment):
    """The basis for a bigrams feature function.
    """
    sent = [stemmer.stem(tok) for tok in comment.split()] # Stemming + punc
    unis = Counter()
    sent = ["<<START>>"] + sent + ["<<END>>"]
    unis.update(bigrams(sent))                             # Bigrams
    return unis

def biasPhi(comment):
  ''' Returns an always-on feature.
  When used with LogisticRegression, 
  will become a most-frequent-class estimator
  '''
  return Counter(["bias"])

def justPassCommentPhi(comment):
  sent = " ".join([stemmer.stem(tok) for tok in comment.split()])
  return sent

def encode(word):
  return embedding_mat[word_to_index[word]]
  # if word in word_to_index:
  #   return embedding_mat[word_to_index[word]]
  # else:
    # return np.zeros(GLOVE_DIMS)

def vectorAvgPhi(comment):

  vectors = [encode(word) for word in comment.split() if word in word_to_index]
  vectors.append( np.zeros((GLOVE_DIMS)) )
  vector_avg = np.sum(vectors, axis=0) 
  # if len(vectors) > 1:
  #   vector_avg /= (len(vectors) - 1)
  features = {}
  for i, val in enumerate(vector_avg):
    features["<DIM" + str(i) + ">"] = val
  return features


def mixture_of_experts(comment):
  tkzd = tkzr.tokenize(comment)
  fixed = " ".join(tkzd)
  fixed = t_prsr.parse(fixed).html
  feats = newUnigramsPhi(fixed)
  feats.update(vectorAvgPhi(fixed))
  probs, pred, probs_over_time = charRNN(comment)
  feats["<CHAR-RNN>"] = probs[0]
  return feats


def load_glove_vectors():
  """Loads in the glove vectors from data/glove.6B """

  glove_home = '../../../CS224u/glove.6B/'
  src_filename = os.path.join(glove_home, 'glove.6B.' + str(GLOVE_DIMS) + 'd.txt')
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

if __name__ == "__main__":
  print("MOST FREQUENT CLASS")
  experiment(train_reader=trainPlusDevReader, 
            assess_reader=testReader,
            phi=biasPhi)

  print("\n\nLOGISTIC MAXENT CLASSIFIER - UNIGRAMS")
  experiment(train_reader=trainPlusDevReader,
            assess_reader=testReader,
            phi=newUnigramsPhi)

  print("\n\nLOGISTIC MAXENT CLASSIFIER - UNI+BIGRAMS")
  experiment(train_reader=trainPlusDevReader, 
            assess_reader=testReader,
            phi=bigramsPhi)


  print("\n\nLOGISTIC MAXENT CLASSIFIER - TFIDF-UNIGRAMS")
  experiment(train_reader=trainPlusDevReader, 
            assess_reader=testReader,
            phi=justPassCommentPhi,
            tfidf=True)

  word_to_index, embedding_mat =  load_glove_vectors()
  print("\n\nLOGISTIC MAXENT CLASSIFIER - VECTOR-BOW")
  experiment(train_reader=trainPlusDevReader, 
            assess_reader=testReader,
            phi=vectorAvgPhi)

  print("\n\nLOGISTIC SVC CLASSIFIER - UNI+BIGRAMS")
  experiment(train_reader=trainPlusDevReader, 
            assess_reader=testReader,
            train_func=fitSVCClassifier,
            phi=bigramsPhi)

  # # Set PRESENT to True in model.py and make sure that the correct model is
  # #   specified in that file. 
  # charRNN, model, session = interactive_version()
  # print("\n\nLOGISTIC CLASSIFIER - MIXTURE OF EXPERTS")
  # experiment(train_reader=trainPlusDevReader, 
  #           assess_reader=testReader,
  #           phi=mixture_of_experts)