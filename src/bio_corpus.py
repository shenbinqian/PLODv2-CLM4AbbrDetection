#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: shenbinqian
"""

from flair.data import Corpus
from flair.datasets import ColumnCorpus


def get_bio_corpus(folder, train_file='train_bio.conll', test_file='test_bio.conll', dev_file='val_bio.conll'):
    # define columns
    columns = {0: 'text', 1: 'pos', 2: 'ner'}
        
    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(folder, columns,
            train_file=train_file,
            test_file=test_file,
            dev_file=dev_file)
    return corpus
