#/usr/bin/python
# Author: Arushi Raghuvanshi
# Edited: Sasha Sax
# File: run_baseline.py
# ---------------------
# Parallelize runs of a program across multiple Stanford corn machines.
# This script generates parameter combinations and hands off to
# run_baseline.exp.
#
# This script should be run from within screen on a corn server.
# > ssh SUNetID@corn.stanford.edu
# > cd path/to/repo/scripts
# > screen
# > python run_baseline.py
# > # You can press "ctrl-a d" to detach from screen and "ctrl-a r" to re-attach.

import os
import time
import numpy as np
import random

servers =  [17, 18, 20, 21, 22] + [10, 11, 12, 13, 14, 15, 16] + range(18, 40)

def get_server_number(counter):
  return '%02d' % (servers[counter%len(servers)])

# Keeps track of which corn server to use.
counter = 0

# Generate random parameters in range
max_epochs = 200

lrs = [2e-03] # np.random.uniform(7e-4,3e-3,3)            
l2s = [1e-05] # np.random.uniform(5e-6,2e-5,1)            
dropout_rates = [1.0] # np.random.uniform(0.75,0.8,1)     
char_dp_rates = [0.7] # np.random.uniform(0.70,0.8,2)     
sizes = [300] # random.sample(xrange(300,301), 1)         
batch_sizes = [ 32 ]                                        
embed_sizes = [50, 100, 200, 300]                         

for rnn_size in sizes:
  for char_dropout in char_dp_rates:
    for dropout in dropout_rates:
      for l2 in l2s:
        for lr in lrs:
          for batch_size in batch_sizes:
            for embed_size in embed_sizes:
              # These parameters will be passed to model.py.
              parameters = " ".join([
                "-batch_size", str(batch_size),
                "-max_epochs", str(max_epochs),
                "-char_dropout", str(char_dropout),
                "-rnn_size", str(rnn_size),
                "-l2", str(l2),
                "-lr", str(lr),
                "-dropout", str(dropout),
                "-embed_size", str(embed_size)
                ])
              command = "/usr/bin/expect -f run_baseline.exp %s '%s' %s %s &" \
                % (get_server_number(counter), parameters, "USER", "PASSWORD")
              print 'Executing command:', command
              os.system(command)
              counter += 1
              time.sleep(5)
