""" Run data-level preprocessing so we don't have to do it every epoch """
import utils
from collections import Counter
import csv 
import codecs

if __name__ == '__main__':
	for filename in [utils.TRAIN_FILENAME, utils.TRAIN_PLUS_DEV_FILENAME, utils.DEV_FILENAME, utils.TEST_FILENAME]:
		utils.parse_data(filename)
