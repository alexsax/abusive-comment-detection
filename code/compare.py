"""
Creates graphs for evaluating training performance. Reads in from checkpoint.pkl files.
"""
import matplotlib
matplotlib.use('Agg')
from utils import quickUnpickle, quickPickle
from evaluate import Evaluator
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
from model import Config, cfgStr
import numpy as np

convolution_window = 5

OUTPUT_DIR = "../output/"

char_dropout = [
  "final_stuff/lstm__h_300__lr_0.002__l2_3e-05__dp_1.0__cdp_0.5__bs_32__embed_300__bd_False__checkpoint.pkl",
  "final_stuff/lstm__h_300__lr_0.002__l2_3e-05__dp_1.0__cdp_0.6__bs_32__embed_300__bd_False__checkpoint.pkl",
  "final_stuff/lstm__h_300__lr_0.002__l2_3e-05__dp_1.0__cdp_0.7__bs_32__embed_300__bd_False__checkpoint.pkl",
  "final_stuff/lstm__h_300__lr_0.002__l2_3e-05__dp_1.0__cdp_0.8__bs_32__embed_300__bd_False__checkpoint.pkl",
  "final_stuff/lstm__h_300__lr_0.002__l2_3e-05__dp_1.0__cdp_0.9__bs_32__embed_300__bd_False__checkpoint.pkl"
]

lr = [
  "final_stuff/lstm__h_300__lr_0.1__l2_3e-05__dp_1.0__cdp_1.0__bs_32__embed_300__bd_False__checkpoint.pkl",
  "final_stuff/lstm__h_300__lr_0.01__l2_3e-05__dp_1.0__cdp_1.0__bs_32__embed_300__bd_False__checkpoint.pkl",
  "final_stuff/lstm__h_300__lr_0.001__l2_3e-05__dp_1.0__cdp_1.0__bs_32__embed_300__bd_False__checkpoint.pkl",
  "final_stuff/lstm__h_300__lr_0.0001__l2_3e-05__dp_1.0__cdp_1.0__bs_32__embed_300__bd_False__checkpoint.pkl",
  "final_stuff/lstm__h_300__lr_1e-05__l2_3e-05__dp_1.0__cdp_1.0__bs_32__embed_300__bd_False__checkpoint.pkl"
]

l2 = [
  "final_stuff/lstm__h_300__lr_0.002__l2_0.1__dp_1.0__cdp_0.7__bs_32__embed_300__bd_False__checkpoint.pkl", 
  "final_stuff/lstm__h_300__lr_0.002__l2_0.01__dp_1.0__cdp_0.7__bs_32__embed_300__bd_False__checkpoint.pkl", 
  "final_stuff/lstm__h_300__lr_0.002__l2_0.001__dp_1.0__cdp_0.7__bs_32__embed_300__bd_False__checkpoint.pkl", 
  # "final_stuff/lstm__h_300__lr_0.002__l2_0.0001__dp_1.0__cdp_0.7__bs_32__embed_300__bd_False__checkpoint.pkl", 
  "final_stuff/lstm__h_300__lr_0.002__l2_1e-05__dp_1.0__cdp_0.7__bs_32__embed_300__bd_False__checkpoint.pkl", 
  "final_stuff/lstm__h_300__lr_0.002__l2_1e-06__dp_1.0__cdp_0.7__bs_32__embed_300__bd_False__checkpoint.pkl", 
  "final_stuff/lstm__h_300__lr_0.002__l2_1e-07__dp_1.0__cdp_0.7__bs_32__embed_300__bd_False__checkpoint.pkl"
]

dropout = [
  "final_stuff/lstm__h_300__lr_0.002__l2_3e-05__dp_0.5__cdp_0.7__bs_32__embed_300__bd_False__checkpoint.pkl",
  "final_stuff/lstm__h_300__lr_0.002__l2_3e-05__dp_0.6__cdp_0.7__bs_32__embed_300__bd_False__checkpoint.pkl",
  "final_stuff/lstm__h_300__lr_0.002__l2_3e-05__dp_0.7__cdp_0.7__bs_32__embed_300__bd_False__checkpoint.pkl",
  "final_stuff/lstm__h_300__lr_0.002__l2_3e-05__dp_0.8__cdp_0.7__bs_32__embed_300__bd_False__checkpoint.pkl",
  "final_stuff/lstm__h_300__lr_0.002__l2_3e-05__dp_0.9__cdp_0.7__bs_32__embed_300__bd_False__checkpoint.pkl"
]

rnn_size = [
  # "final_stuff/lstm__h_100__lr_0.002__l2_1e-06__dp_0.5__cdp_0.7__bs_125__embed_100__bd_False__short__checkpoint.pkl",
  "final_stuff/lstm__h_25__lr_0.002__l2_1e-06__dp_0.5__cdp_0.7__bs_125__embed_25__bd_False__checkpoint.pkl",
  "final_stuff/lstm__h_50__lr_0.002__l2_1e-06__dp_0.5__cdp_0.7__bs_125__embed_50__bd_False__checkpoint.pkl",
  "final_stuff/lstm__h_100__lr_0.002__l2_1e-06__dp_0.5__cdp_0.7__bs_125__embed_100__bd_False__checkpoint.pkl",
  "final_stuff/lstm__h_200__lr_0.002__l2_1e-06__dp_0.5__cdp_0.7__bs_125__embed_200__bd_False__checkpoint.pkl",
  "final_stuff/lstm__h_300__lr_0.002__l2_1e-06__dp_0.5__cdp_0.7__bs_125__embed_300__bd_False__checkpoint.pkl",
  "final_stuff/lstm__h_400__lr_0.002__l2_1e-06__dp_0.5__cdp_0.7__bs_125__embed_400__bd_False__checkpoint.pkl",
  "final_stuff/lstm__h_500__lr_0.002__l2_1e-06__dp_0.5__cdp_0.7__bs_125__embed_500__bd_False__checkpoint.pkl", 
]

embed_size = [
  "final_stuff/lstm__h_300__lr_0.002__l2_3e-05__dp_1.0__cdp_1.0__bs_32__embed_35__bd_False__checkpoint.pkl",
  "final_stuff/lstm__h_300__lr_0.002__l2_3e-05__dp_1.0__cdp_1.0__bs_32__embed_50__bd_False__checkpoint.pkl",
  "final_stuff/lstm__h_300__lr_0.002__l2_3e-05__dp_1.0__cdp_1.0__bs_32__embed_100__bd_False__short__checkpoint.pkl",
  "final_stuff/lstm__h_300__lr_0.002__l2_3e-05__dp_1.0__cdp_1.0__bs_32__embed_200__bd_False__checkpoint.pkl",
  "final_stuff/lstm__h_300__lr_0.002__l2_3e-05__dp_1.0__cdp_1.0__bs_32__embed_300__bd_False__checkpoint.pkl", 
]

regularized_model = "final_stuff/lstm__h_300__lr_0.002__l2_1e-06__dp_0.5__cdp_0.7__bs_125__embed_300__bd_False__checkpoint.pkl"
unregularized_model = "final_stuff/lstm__h_300__lr_0.002__l2_0__dp_1.0__cdp_1.0__bs_125__embed_300__bd_False__checkpoint.pkl"
superregularized_model = "final_stuff/lstm__h_300__lr_0.002__l2_0.01__dp_1.0__cdp_0.7__bs_32__embed_300__bd_False__checkpoint.pkl"



def processResults(filename, smoothed=True):
  """ Turns the checkpoint.pkl files into a format acceptable to matplotlib """
  all_results = quickUnpickle(filename)
  config = all_results["config"]
  all_results = all_results["summaries"]
  intermediate_summaries = {
    "train": [],
    "validation": [],
  }

  overall = {}

  for epoch_result in all_results:
    for dataset in intermediate_summaries:
      if "test" in epoch_result:
        overall[dataset] = epoch_result[dataset]
      else:
        try:
          intermediate_summaries[dataset].append(epoch_result[dataset])
        except:
          pass

  summaries = {
    "train": defaultdict(lambda: []),
    "validation": defaultdict(lambda: []),
  }

  for dataset in intermediate_summaries:
    for epoch in intermediate_summaries[dataset]:
      for key, value in epoch.items():
        summaries[dataset][key].append(value)

  if smoothed:
    for dataset in intermediate_summaries:
      for key in ["f1", "ce", "roc"]:
        # print np.asarray(summaries[dataset][key]).shape
        summaries[dataset][key] = np.convolve(np.asarray(summaries[dataset][key]), np.ones(convolution_window)/(1.0*convolution_window))

  return config, summaries
model_results = {}



def plot(dataset, metric, models_to_compare, propertyfn, name, labels=None, styles=None):
  """ 
    Line plot of ``metric'' over each epoch during training for the given dataset.
    Includes a line for each model in models_to_compare, and the labels can be set in a list
  """
  for model_num, model in enumerate(models_to_compare):
    config, model_results[model] = processResults(OUTPUT_DIR + model)
    if labels:
      config.model_name = labels[model_num]
    if styles:
      linestyle = styles[model_num]
    else:
      linestyle = '-'
    plt.plot(model_results[model][dataset][metric][:-5], label = propertyfn(config), linestyle=linestyle)
  metric_to_legend = {"f1": (0.8, 0.3), "roc": (0.8, 0.3), "ce": (0.8, 1) }
  if metric == "ce" and dataset == "validation":
    metric_to_legend[metric] = (0.8, 0.3)
  plt.legend(bbox_to_anchor=metric_to_legend[metric],
              ncol=3, title=name)
  plt.title(metric +' history (' + dataset + ')')
  plt.xlabel('Iteration')
  plt.ylabel(metric)
  plt.savefig("../analysis/" + metric + "_history_" + dataset + "_" + "_".join(name.split()) + ".png")
  plt.close()
  plt.show()

def plot_with_train_validation(metric, reg_model, unreg_model, superregularized_model, name, labels=None):
  """ Line plot of the metric score at each epoch for the three levels of regularization. """
  metric_to_legend = {"f1": (0.8, 0.3), "roc": (0.8, 0.3), "ce": (0.8, 1) }
  reg_config, model_results[reg_model] = processResults(OUTPUT_DIR + reg_model)
  unreg_config, model_results[unreg_model] = processResults(OUTPUT_DIR + unreg_model)
  unreg_config, model_results[superregularized_model] = processResults(OUTPUT_DIR + superregularized_model)

  steps = min(reg_config.max_epochs, unreg_config.max_epochs)
  fig = plt.figure()
  ax = plt.subplot(111)
  
  ax.plot(model_results[reg_model]["train"][metric][:steps-convolution_window], label = "Regularized Training", linestyle="-", color='green')
  ax.plot(model_results[unreg_model]["train"][metric][:steps-convolution_window], label = "Unregularized Training", linestyle="-", color='blue')
  ax.plot(model_results[superregularized_model]["train"][metric][:steps-convolution_window], label = "Superregularized Training", linestyle="-", color='red')

  if metric == "ce":  metric_to_legend[metric] = (0.8, 0.3)
  ax.plot(model_results[reg_model]["validation"][metric][:steps-convolution_window], label = "Regularized Validation", linestyle='-.', color='green')
  ax.plot(model_results[unreg_model]["validation"][metric][:steps-convolution_window], label = "Unregularized Validation", linestyle='-.', color='blue')
  ax.plot(model_results[superregularized_model]["validation"][metric][:steps-convolution_window], label = "Superregularized Validation", linestyle='-.', color='red')
  # plt.legend(bbox_to_anchor=metric_to_legend[metric],
  #             ncol=2, title=name)


  # Shrink current axis's height by 10% on the bottom
  box = ax.get_position()
  ax.set_position([box.x0, box.y0 + box.height * 0.15,
                   box.width, box.height * 0.85])

  # Put a legend below current axis
  ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, ncol=2)

  plt.title(name + ' (' + metric + ') ')
  plt.xlabel('Iteration') 
  plt.ylabel(metric)
  plt.savefig("../analysis/" + metric + "_train_v_valid_" + "_".join(name.split()) + ".png")
  plt.close()
  plt.show()



if __name__ == '__main__':
  for metric in ["f1", "ce", "roc"]:
    for dataset in ["train", "validation"]:
      plot(dataset, metric, char_dropout, lambda c: c.char_dropout, "Character Dropout")
      plot(dataset, metric, lr, lambda c: c.lr, "Learning Rate")
      # plot(dataset, metric, embed_size, lambda c: c.embed_size, "embed_size")
      plot(dataset, metric, l2, lambda c: c.l2, "L2 Regularization")
      plot(dataset, metric, dropout, lambda c: c.dropout, "Dropout")
      plot(dataset, metric, rnn_size, lambda c: c.rnn_size, "Hidden Size")
    plot_with_train_validation(metric, regularized_model, unregularized_model, superregularized_model, "Effect of Regularization")

