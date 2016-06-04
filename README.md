# abusive-comment-detection

Implements a charLSTM (with GRU, RNN, bidirectionality, multiple hidden layer) options to detect insults in Impermium's Kaggle dataset.

See the paper in reports/final_paper for the full writeup.

The python notebook in code/ contains a demo of the trained network vs. the bigram baseline

# Results
|Model   	|F1   	|AUC   	|
|---	|---	|---	|
|Unigrams   	                  |0.663   	|0.794   	|
|Uni + Bigrams   	              |0.640   	|0.794   	|
|Stemming + Unigrams          	|0.671   	|0.796   	|
|Stemming + TFIDF + Unigrams   	|0.603   	|0.767   	|
|Stemming + Uni + Bigrams     	|0.665   	|**0.816**   	|
|Stem + 1,2grams SVM          	|0.580   	|0.736   	|
|GloVe Vector-Avg 100d        	|0.507   	|0.694   	|
|GloVe Vector-Avg 200d         	|0.568   	|0.733   	|
|GloVe Vector-Avg 300d        	|0.615   	|0.743   	|
|charLSTM 50d   	              |0.704   	|0.769   	|
|charLSTM 350d                  |**0.721**    |0.795    |
|bd charLSTM 300d   	          |0.702   	|0.756   	|
|2-layer charLSTM 300d      	  |0.667   	|0.510   	|
|GRU 300d   	                  |0.694   	|0.756   	|
