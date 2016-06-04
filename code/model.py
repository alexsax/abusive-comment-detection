""" 
Author: Sasha Sax 
Spring 2016

Implements GRU, LSTM, and RNN for the Impermium Insult detection dataset. 
The dataset can be parsed with 'process_data.py' and then use utils to point
the datareaders to the new data set
"""

from copy import deepcopy
import time
import getpass
import sys

import numpy as np
import os.path
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
from tensorflow.models.rnn import seq2seq
from tensorflow.python.framework import ops
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import utils
from utils import char_iterator, Vocab, quickPickle, quickUnpickle, \
                  trainReader, testReader, devReader, trainPlusDevReader
from evaluate import Evaluator
import pickle
import subprocess

import argparse
import json
from subprocess import call

TEST_ONLY = False
DYNAMIC = False
PRESENT = True


class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.

  Present using : lstm__h_250__lr_0.00142501119989__l2_8.71232610211e-05__dp_0.887847599744__cdp_0.724880349503__bs_32__includec_False
  """
  def __init__(self):
    # Learning parameters
    self.lr = 0.0001#0.00142501119989
    self.early_stopping = None #3
    self.max_epochs = 150

    # Sizes    
    self.rnn_size = 300
    self.embed_size = 300
    self.batch_size = 125 
    self.num_labels = utils.NUM_CLASSES # 0 or 1
    self.seq_length = utils.MAX_COMMENT_LENGTH

    # Model types
    self.num_layers = 1
    self.model = 'rnn' #lstm
    self.bidirectional = False

    # Regularization
    self.dropout = 1.0 #0.887847599744
    self.char_dropout = 1.0 #0.724880349503
    self.l2 = 1e-06 #8.71232610211e-05
    self.debug = False


    if TEST_ONLY:
      # ex save file: lstm__h_50__lr_0.002__l2_1e-06__dp_0.5__cdp_0.7__bs_125__embed_50__best_f1.weights
      self.batch_size = 32
      self.num_labels = utils.NUM_CLASSES # 0 or 1
      self.seq_length = utils.MAX_COMMENT_LENGTH
      self.model = 'lstm'
      self.bidirectional = False
      self.num_layers = 1
      self.rnn_size = 300
      self.embed_size = 300
      self.max_epochs = 50
      self.early_stopping = None #3
      self.dropout = 0.5
      self.char_dropout = 0.7
      self.lr = 0.002
      self.debug = False
      self.l2 = 3e-05
      pass

    if PRESENT:
      # If using interactive_version()
      self.batch_size = 32 # Batch not implemented, yet
      self.num_labels = 2 # 0 or 1
      self.seq_length = utils.MAX_COMMENT_LENGTH
      self.model = 'lstm'
      self.bidirectional = False
      self.num_layers = 1
      self.rnn_size = 300
      self.embed_size = 300
      self.max_epochs = 150
      self.early_stopping = None #3
      self.dropout = 0.5
      self.char_dropout = 0.7
      self.lr = 0.002
      self.debug = False
      self.l2 = 3e-05

  def cfgFromNamespace(self, ns):
    """ Sets all the properties of a config object from the passed in namespace. 
      This is used for setting the config from the command line.
    """
    props = ['batch_size', 'model', 'bidirectional', 'num_layers', 'rnn_size', 'max_epochs', 'early_stopping', 'dropout',
              'char_dropout', 'lr', 'l2', 'embed_size']
    for prop in props:
      if hasattr(ns, prop) and getattr(ns, prop):
        setattr(self, prop, getattr(ns, prop))
        print "setting", prop


def cfgStr(c):
  """ Should really be config.toString() """
  return "__".join(
        [c.model, "h_"+str(c.rnn_size), "lr_"+str(c.lr), "l2_"+str(c.l2), 
          "dp_"+str(c.dropout), "cdp_"+str(c.char_dropout), "bs_" + str(c.batch_size),
          "embed_" + str(c.embed_size), "bd_" + str(c.bidirectional)
        ])



class RNNClassifier():
    def __init__(self, configs, infer=False):
        ''' Initializes the model '''
        self.cells = []
        self.config = configs
        self.load_data(debug=configs.debug)
        self.add_placeholders()
        self.inputs = self.add_embedding()
        self.final_state = self.add_model(self.inputs)
        self.output = self.add_projection(self.final_state)
        self.add_prediction(self.output)
        self.calculate_loss = self.add_loss_op(self.output)
        self.train_step = self.add_train_op(self.calculate_loss)
        self.no_op = tf.no_op()


    ##########################################################################################
    #          DATA AND CHECKPOINTING
    ##########################################################################################
    def encode_data(self, reader, name="DATA"):
      """
      Encodes data into a numpy array which can then be passed to char_iterator. 
      """
      data = []
      max_length = 0
      number_in_threshold = 0
      number_insults = 0
      num_total = 0
      for comment, label in reader(): # Read each comment
        if label == 1: number_insults += 1 # Count the number which are insults
        num_total += 1
        max_length = max(max_length, len(comment))  # Record the longest comment

        encoded = [self.vocab.encode(self.vocab.start)] + [self.vocab.encode(char) for char in comment] + [self.vocab.encode(self.vocab.end)]

        if len(encoded) <= utils.MAX_COMMENT_LENGTH: 
          number_in_threshold += 1 # Count the number which are short enough for inclusion 
        else:
          continue
        data.append((np.array(encoded), label)) # Add the comment into the data
      print "---------LOADED " + name + "------------"
      print "Total comments in the dataset:", len(data)
      print "Max comment length:", number_in_threshold
      print "Percent under threshold (", utils.MAX_COMMENT_LENGTH, "):", 100.*number_in_threshold/num_total
      print "Percent insults:", 100.*number_insults/len(data)
      print "--------------------------------"
      return data

    def load_data(self, debug=False):
        """Loads the dataset into memory, and saves it into a train/dev/test split"""
        self.vocab = Vocab()
        self.vocab.construct(trainPlusDevReader())
        self.encoded_train = self.encode_data(trainPlusDevReader, "TRAIN")
        self.encoded_valid = self.encode_data(testReader, "DEV")[:750]
        self.encoded_test = self.encode_data(testReader, "TEST")

        if debug:
          num_debug = 30
          self.encoded_train = self.encoded_train[:num_debug]
          self.encoded_valid = self.encoded_valid[:num_debug]
          self.encoded_test = self.encoded_test[:num_debug]
        print "Loaded data"

    def saveCheckpointInfo(self, train_evaluator=None, validation_evaluator=None, test_evaluator=None):
      """
      Saves important information from each epoch.
      Will save to a file:
          config : the configuration of the model
          summaries : a list of evalutator.summary_dict() for each evaluator passed in
      If a test_evaluator is passed in, saves to the _result.pkl.
      If no test_evaluator is passed in, saves results in _checkpoint.pkl
      """

      epoch_summary = {}
      # Create a summary from the passed in evaluators
      if train_evaluator:
        epoch_summary["train"] = train_evaluator.summary_dict()
      if validation_evaluator:
        epoch_summary["validation"] = validation_evaluator.summary_dict()
      if test_evaluator:
        epoch_summary["test"] = test_evaluator.summary_dict()

      # Append/create the epoch_summaries property
      try: 
        self.epoch_summaries.append(epoch_summary)
      except AttributeError:
        self.epoch_summaries = [epoch_summary]

      # Create a dict from the config and summaries and pickle it
      obj = {"config": self.config,
              "summaries": self.epoch_summaries}
      if test_evaluator:
        quickPickle(obj, "../output/" + cfgStr(self.config) + "__result.pkl")
      else:
        quickPickle(obj, "../output/" + cfgStr(self.config) + "__checkpoint.pkl")

    def loadCheckpointInfo(self):
      """ Load a checkpoint into memory
        *_checkpoint.pkl must match with the corresponding saved *.weights file

        The pickle has the following
        "config" : a Config object describing the model
        "summaries" : [epoch0, epoch1, ...]
          epoch : A dict of {dataset: summary} where dataset \in ["train", "validation", "test"]
            summary : Stores the following results from Evaluator
              "f1" : F1 score for the model on the dataset in an epoch
              "recall" 
              "precision"
              "roc" : roc auc for the model on the dataset in an epoch
              "ce" : Cross entropy loss for an epoch

        Note that if dropout (or char_dropout) is used, then the "train" summary will test on 
          "train" with dropout set to 1.0. The summary just uses the post-dropout scores. 
      """
      obj = quickUnpickle("../output/" + cfgStr(self.config) + "__checkpoint.pkl")
      self.epoch_summaries = obj["summaries"]
      me = self.config.max_epochs # This is the only thing that could change
      self.config = obj["config"]
      self.config.max_epochs = me

      # Load best values so that we don't obliterate good results
      best_val_roc = -float('inf')
      best_val_ce = float('inf')
      best_val_f1 = -float('inf')
      for epoch_num, results in enumerate(self.epoch_summaries):
        best_val_roc = max(results["validation"]["roc"], best_val_roc)
        best_val_ce = min(results["validation"]["ce"], best_val_ce)
        best_val_f1 = max(results["validation"]["f1"], best_val_f1)
        best_val_epoch = epoch_num

      return len(self.epoch_summaries), best_val_roc, best_val_ce, best_val_f1, best_val_epoch # N epochs


    ##########################################################################################
    #          PLACEHOLDERS, EMBEDDING, PROJECTION, PREDICTION
    ##########################################################################################
    def add_placeholders(self):
      """ Adds placeholders to the graph """
      self.input_placeholder = tf.placeholder(tf.int32, [self.config.seq_length, self.config.batch_size], name="inputPlaceholder")
      self.sequence_lengths = tf.placeholder(tf.int32, [self.config.batch_size,], name="sequenceLengthsPlaceholder")
      self.labels_placeholder = tf.placeholder(tf.float32, [self.config.batch_size, self.config.num_labels], name="labelsPlaceholder")
      self.dropout_placeholder = tf.placeholder(tf.float32, name="dropoutPlaceholder")
      print "Added placeholders:", self.config.rnn_size, "hidden dims |", self.config.num_layers, "layers"

    def add_embedding(self):
      """ Adds the character embedding matrix to the graph """
      embedding = tf.get_variable("embedding", [len(self.vocab), self.config.embed_size], 
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
      inputs = tf.nn.embedding_lookup(embedding, self.input_placeholder)

      if not DYNAMIC:
        inputs = tf.unpack(inputs)
          # inputs = [tf.squeeze(_input, [1]) for _input in inputs]
      print "Added character embeddings:"
      return inputs

    def add_projection(self, rnn_output):
      """ Adds a projection layer to the graph """
      with tf.variable_scope('projection'):
        W_width = self.config.rnn_size
        if self.config.bidirectional: # Bidirectional models also require twice the params
          W_width *= 2

        W = tf.get_variable("weights", [W_width, self.config.num_labels], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("bias", [self.config.num_labels], initializer=tf.constant_initializer(0))
      output = tf.matmul(rnn_output, W) + b
      return output

    def add_prediction(self, output):
      """ Adds prediction elements to the graph
        probs : returns the probabilities over the classes
        prediction : the most likely class 
      """
      self.probs = tf.nn.softmax(output, name="PredictedProb")[:,1]
      if self.config.batch_size == 1: # If we're testing, we may want to see intermediate probs
        with tf.variable_scope('projection', reuse=True):
          W = tf.get_variable("weights")
          b = tf.get_variable("bias")
          packed = tf.squeeze(tf.pack(self.all_outputs))
          self.probs_over_time = tf.nn.softmax(tf.matmul(packed, W[:self.config.rnn_size,:]) + b, name="PredictedProbOverTime")

      self.prediction = tf.argmax(output, 1)
      print "Added prediction"


    ##########################################################################################
    #         LOSS + TRAIN OPS
    ##########################################################################################
    def add_loss_op(self, output):
        ce = tf.nn.softmax_cross_entropy_with_logits(output, self.labels_placeholder, name="LossFn")
        loss = tf.reduce_mean(ce)
        if self.config.model == "lstm":
          for v in tf.all_variables():
            if "LSTMCell/W_0" in v.name:
              print "\tL2 regularization on", v.name
              loss += self.config.l2 * tf.nn.l2_loss(v)
        elif self.config.model == "gru":
          with tf.variable_scope("RNN/MultiRNNCell/Cell0/GRUCell/Gates/Linear", reuse=True) as vs:
            loss += self.config.l2 * tf.nn.l2_loss(tf.get_variable("Matrix"))
          with tf.variable_scope("RNN/MultiRNNCell/Cell0/GRUCell/Candidate/Linear", reuse=True) as vs:
            loss += self.config.l2 * tf.nn.l2_loss(tf.get_variable("Matrix"))
          with tf.variable_scope("projection", reuse=True):
            loss += self.config.l2 * tf.nn.l2_loss(tf.get_variable("weights"))
        print "Added loss op"
        return loss

    def add_train_op(self, loss):
        print "Added train op"
        return tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)


    ##########################################################################################
    #          RNN + CELL
    ##########################################################################################
    def new_cell_model(self):
      """ Creates a RNNCell for use in the model. 
          Adds dropout.
      """
      # Create the cell used in the model, set layers
      # Consulted with https://github.com/sherjilozair/char-rnn-tensorflow/blob/master/model.py

      self.W = None
      if self.config.model == 'rnn':
          cell = rnn_cell.BasicRNNCell(self.config.rnn_size)
      elif self.config.model == 'gru':
          cell = rnn_cell.GRUCell(self.config.rnn_size)
      elif self.config.model == 'lstm':
          cell = rnn_cell.LSTMCell(self.config.rnn_size)
      else:
          raise Exception("model type not supported: {}".format(self.config.model))
      cell = rnn_cell.DropoutWrapper(cell, 
                                    input_keep_prob=self.dropout_placeholder, 
                                    output_keep_prob=self.dropout_placeholder)
      multi_cell = rnn_cell.MultiRNNCell([cell] * self.config.num_layers)
      self.cells.append(multi_cell)
      return multi_cell

    def extract_last_relevant(self, outputs, length):
      # From user erickrf at https://stackoverflow.com/questions/35835989/how-to-pick-the-last-valid-output-values-from-tensorflow-rnn
      """
      Args:
          outputs: [Tensor(batch_size, output_neurons)]: A list containing the output
              activations of each in the batch for each time step as returned by
              tensorflow.models.rnn.rnn.
          length: Tensor(batch_size): The used sequence length of each example in the
              batch with all later time steps being zeros. Should be of type tf.int32.

      Returns:
          Tensor(batch_size, output_neurons): The last relevant output activation for
              each example in the batch.
      """
      output = tf.transpose(tf.pack(outputs), perm=[1, 0, 2])
      # Query shape.
      batch_size = tf.shape(output)[0]
      max_length = int(output.get_shape()[1])
      num_neurons = int(output.get_shape()[2])
      # Index into flattened array as a workaround.
      index = tf.range(0, batch_size) * max_length + (length - 1)
      flat = tf.reshape(output, [-1, num_neurons])
      relevant = tf.gather(flat, index)
      return relevant

    def add_model(self, inputs):
      """ Adds the model to the graph. 
        Has different models based on the config parameters 
      """
      print "Setting up model"
      # Single or bidirectional
      last_cell_start_idx = (self.config.num_layers - 1)*self.config.rnn_size
      if self.config.bidirectional:
          buildfn = rnn.bidirectional_rnn
          # initial_state_fw and initial_state_bw default to zeros
          all_outputs, final_fw, final_bw = rnn.bidirectional_rnn(cell_fw=self.new_cell_model(), 
                                                      cell_bw=self.new_cell_model(), 
                                                      sequence_length=self.sequence_lengths,
                                                      inputs=inputs, dtype=tf.float32)
          # rnn_output = tf.concat(concat_dim=1, values=[
          #                           final_fw[:, last_cell_start_idx:], 
          #                           final_bw[:, last_cell_start_idx:]
          #                       ]) # We want just the output of the top layer
          rnn_output = all_outputs[-1]
      else:
        # Choose the single-directional RNN. If DYNAMIC is set, use the dynamic_rnn
        #   for faster loading of model. It remains to be seen we can use DYNAMIC for the 
        #   presentation
        if not DYNAMIC:
          all_outputs, rnn_output = rnn.rnn(
            self.new_cell_model(), 
            inputs=inputs, 
            sequence_length=self.sequence_lengths,
            dtype=tf.float32)
        else:
          all_outputs, rnn_output = rnn.dynamic_rnn(
            self.new_cell_model(), 
            inputs=inputs, 
            sequence_length=self.sequence_lengths,
            time_major=True,
            dtype=tf.float32)
          all_outputs = tf.unpack(all_outputs)
        rnn_output = rnn_output[:, last_cell_start_idx:] # Take the top layer
      self.all_outputs = all_outputs
      self.final_state = self.extract_last_relevant(all_outputs, self.sequence_lengths)
      print "Added model:", cfgStr(self.config)
      return self.final_state


    ##########################################################################################
    #          RUNTIME METHODS
    ##########################################################################################
    def run_epoch(self, session, data, train_op=None, verbose=5):
      """ Runs an epoch of the model. 
        Downsamples non-insults, then runs a session.
        If no train_op is passed in, then the model will not train but will instead go into test mode and set 
          dropout to 1.0 for best performance. 
        Saves epoch information into model_checkpoint.pkl via the Evaluators.
      """
      downsample = False
      if train_op:
        downsample=True
        char_dp = self.config.char_dropout
      else:
        char_dp = 1.0
      total_steps = sum(1 for x in char_iterator(data, self.config.batch_size, downsample))
      evaluator = Evaluator()


      for step, (x, y, sequence_lengths) in enumerate(
          char_iterator(data, self.config.batch_size, downsample, char_dropout=char_dp)):
          # The data will be in the following form:
          #     x : [seq_length x batch_size] array which contains sequences
          #     y : [batch_size] vector of labels
          #     seq_length : [batch_size] vector of the length of each comment in the batch
          # We need to pass in the initial state and retrieve the final state to give
          # the RNN proper history
          self.run_step(session, x, y, sequence_lengths, step, total_steps, evaluator, train_op=train_op, verbose=verbose)

          if verbose and step % verbose == 0:
              sys.stdout.write('\r{} / {} : ce = {}'.format(
                  step, total_steps, np.mean(evaluator.loss)))
              sys.stdout.flush()
      if verbose:
        sys.stdout.write('\r                                               ')
        sys.stdout.write('\r')
        evaluator.report()
      return evaluator

    def run_step(self, session, x, y, sequence_lengths, step, total_steps, evaluator, train_op=None, verbose=10):
      """ Runs one batch in the epoch 
        Pass in an evaluator to save the results of the batch. The evaluator will later be run to determine
          summary statistics for the epoch. 
        If no train_op is passed in, then the model will not train but will instead go into test mode and set 
          dropout to 1.0 for best performance. 
      """
      # Check if testing/training
      if not train_op:
          train_op = self.no_op
          dp = 1
      else:
          dp = self.config.dropout

      feed = {self.input_placeholder: x,
              self.labels_placeholder: y,
              self.sequence_lengths: sequence_lengths,
              self.dropout_placeholder: dp}

      probs, pred, loss, _ = session.run([self.probs, self.prediction, self.calculate_loss, train_op], feed_dict=feed)
      evaluator.loss.append(loss)
      [evaluator.probs.append(p) for p in probs]
      [evaluator.predicted.append(p) for p in pred]
      [evaluator.gold.append(np.argmax(ex)) for ex in y]





##########################################################################################
#   META-METHODS
##########################################################################################

def test_RNNLM(config=Config()):
  """ Main function """
  # Initialize the model
  model = RNNClassifier(config)
  init = tf.initialize_all_variables()
  saver = tf.train.Saver()

  with tf.Session() as session:
    session.run(init)

    # We will checkpoint whenever model achieves a 'personal best' on 
    #   any one of these metrics
    best_val_roc = -float('inf')
    best_val_ce = float('inf')
    best_val_f1 = -float('inf')
    best_val_epoch = 0
  
    
    # If we are training the model, then continue where we left off
    if not TEST_ONLY:
      startEpoch = 0

      # Load the checkpoint from memory, if it exists
      if os.path.isfile('../models/'+ cfgStr(config) + '.weights'):
        saver.restore(session, '../models/'+ cfgStr(config) + '.weights')
        startEpoch, best_val_roc, best_val_ce, best_val_f1, best_val_epoch = model.loadCheckpointInfo()
        print "Resuming from epoch", startEpoch
        print "Best AUC:", best_val_roc
        print "Best CE:", best_val_ce
        print "Best F1:", best_val_f1
      else:
        print "Didnt find", '../models/'+ cfgStr(config) + '.weights'

      # This is the main training loop
      for epoch in xrange(config.max_epochs):
        # Skip to the current epoch if restoring from checkpoint
        if epoch < startEpoch: 
          continue

        print 'Epoch {}'.format(epoch)
        start = time.time()

        print "TRAIN Results:"
        train_evaluator = model.run_epoch(
            session, model.encoded_train,
            train_op=model.train_step) # Train

        print "MODEL:", cfgStr(model.config)
        validation_evaluator = model.run_epoch(session, model.encoded_valid) # Validation
        model.saveCheckpointInfo(
          train_evaluator=train_evaluator,
          validation_evaluator=validation_evaluator)


        # Save intermediate results if they're good enough
        if validation_evaluator.f1 > best_val_f1 and validation_evaluator.f1 > 0.72:
          best_val_f1 = validation_evaluator.f1
          saver.save(session, '../models/'+ cfgStr(config) + '__best_f1.weights')
          print "Best f1"
        if validation_evaluator.roc > best_val_roc and validation_evaluator.roc > 0.80:
          best_val_roc = validation_evaluator.roc
          saver.save(session, '../models/'+ cfgStr(config) + '__best_roc.weights')
          print "Best roc"

        # Early stopping
        if config.early_stopping and epoch - best_val_epoch > config.early_stopping:
          break

        # Print epoch time info
        print 'Total time: {}'.format(time.time() - start)
        print "\n"
      saver.save(session, '../models/'+ cfgStr(config) + '.weights') # Save every time

    # Test time! Load the best model and check performance. 
    #   Saves into a __results.pkl          
    saver.restore(session, '../models/'+ cfgStr(config) + '__best_f1.weights')
    print '=-=' * 5
    train_evaluator = model.run_epoch(session, model.encoded_train)
    validation_evaluator = model.run_epoch(session, model.encoded_valid)
    test_evaluator = model.run_epoch(session, model.encoded_test)
    model.saveCheckpointInfo(train_evaluator=train_evaluator,
                            validation_evaluator=validation_evaluator,
                            test_evaluator=test_evaluator)
    print '=-=' * 5


def interactive_version():
  """ Returns a predict method where you pass in the comment, and get back
    prob : np.array(probability)
    pred : np.array(predicted class)
    probs_over_time : np.array([probability_time1, ...])

    Loads the model from "cfgStr(default config).weights"
  """
  c = Config()
  old_config = Config()
  c.batch_size = 1
  model = RNNClassifier(c)
  init = tf.initialize_all_variables()
  saver = tf.train.Saver()
  session = tf.Session()
  saver.restore(session, '../models/'+ cfgStr(old_config) + '__best_f1.weights')

  def predict(comment):
    encoded = [model.vocab.encode(model.vocab.start)] + [model.vocab.encode(char) for char in comment] + [model.vocab.encode(model.vocab.end)]
    seq_length = len(encoded)
    comment = np.array(encoded)
    encoded = np.pad(comment, (0, utils.MAX_COMMENT_LENGTH - len(comment)), mode='constant', constant_values=0)
    encoded = np.reshape(encoded, (1, -1)).T
    feed = {model.input_placeholder: encoded,
            model.labels_placeholder: np.zeros([1,2]),
            model.sequence_lengths: [seq_length],
            model.dropout_placeholder: 1.0}
    probs, pred, probs_over_time = session.run([model.probs, model.prediction, model.probs_over_time], feed_dict=feed)
    return probs, pred, probs_over_time

  return predict, model, session

def parse_arguments():
  """ Parses model parameters from the command line """
  parser = argparse.ArgumentParser(description='Run the model for CS224d')
  parser.add_argument('-batch_size', nargs='?', help="batch size to use", type=int)
  parser.add_argument('-max_epochs', nargs='?', help="maximum number of epochs to run", type=int)
  parser.add_argument('-char_dropout', nargs='?', help="proportion of characters to keep", type=float)
  parser.add_argument('-rnn_size', nargs='?', help="hidden dimension size", type=int)
  parser.add_argument('-dropout', nargs='?', help="proportion of lstm units to keep", type=float)
  parser.add_argument('-lr', nargs='?', help="learning rate", type=float)
  parser.add_argument('-l2', nargs='?', help="l2 regularization", type=float)
  parser.add_argument('-embed_size', nargs='?', help="embed_size", type=int)
  return parser.parse_args()


def main():
  args = parse_arguments()
  c = Config()
  c.cfgFromNamespace(args)
  test_RNNLM(c)

if __name__ == "__main__":
  main()
