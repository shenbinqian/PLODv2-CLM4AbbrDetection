#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: shenbinqian
"""

from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from flair.embeddings import FlairEmbeddings
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--continue_pretrain", type=str, default="False", help="continue pretrained model")
    parser.add_argument("--corpus_path", type=str, default="./corpus", help="path to corpus")
    parser.add_argument("--is_forward", type=str, default="True", help="is forward")
    parser.add_argument("--plm_name", type=str, default="news-forward", help="name of pre-trained language model if continue_pretrain is True")
    parser.add_argument("--out", type=str, default="plm", help="output name")
    parser.add_argument("--hidden_size", type=int, default=2048, help="hidden size")
    parser.add_argument("--sequence_length", type=int, default=256, help="sequence length")
    parser.add_argument("--learning_rate", type=float, default=20.0, help="learning rate")
    parser.add_argument("--mini_batch_size", type=int, default=100, help="mini batch size")
    parser.add_argument("--max_epochs", type=int, default=300, help="max epochs")
    args = parser.parse_args()
    return args


def pretrain(corpus_folder: str = "./corpus", is_forward: str = "True", out: str = "plm", hidden_size: int = 2048, sequence_length: int = 256, mini_batch_size: int = 100, learning_rate:float = 0.1, max_epochs: int = 300):
    
    # load the default character dictionary
    dictionary: Dictionary = Dictionary.load('chars')
    is_forward = eval(is_forward)
    # get your corpus, process forward and at the character level
    corpus = TextCorpus(corpus_folder,
                        dictionary,
                        is_forward,
                        character_level=True)
    
    # instantiate your language model, set hidden size and number of layers
    language_model = LanguageModel(dictionary,
                                   is_forward,
                                   hidden_size=hidden_size,
                                   nlayers=1)
    
    # train your language model
    trainer = LanguageModelTrainer(language_model, corpus)
    
    if is_forward:
        out_path = out + "-forward"
    else:
        out_path = out + "-backward"

    trainer.train(out_path,
                  sequence_length=sequence_length,
                  mini_batch_size=mini_batch_size,
                  learning_rate=learning_rate,
                  max_epochs=max_epochs)


def continue_pretrain(corpus_folder: str = "./corpus", plm_name: str = "news-forward", out: str = "cplm", sequence_length: int = 256, mini_batch_size: int = 100, learning_rate:float = 0.1, max_epochs: int = 300):
    # instantiate an existing LM, such as one from the FlairEmbeddings
    # "news-forward or backward"
    language_model = FlairEmbeddings(plm_name).lm
    
    # are you fine-tuning a forward or backward LM?
    is_forward_lm = language_model.is_forward_lm
    
    # get the dictionary from the existing language model
    dictionary: Dictionary = language_model.dictionary
    
    # get your corpus, process forward and at the character level
    corpus = TextCorpus(corpus_folder,
                        dictionary,
                        is_forward_lm,
                        character_level=True)
    
    # use the model trainer to fine-tune this model on your corpus
    trainer = LanguageModelTrainer(language_model, corpus)

    if is_forward_lm:
        out_path = out + "-forward"
    else:
        out_path = out + "-backward"
    
    trainer.train(out_path,
                  sequence_length=sequence_length,
                  mini_batch_size=mini_batch_size,
                  learning_rate=learning_rate,
                  max_epochs=max_epochs,
                  patience=10,
                  checkpoint=True)
    

def main():
    args = parse_args()
    if args.continue_pretrain== "True":
        continue_pretrain(args.corpus_path, args.plm_name, args.out, int(args.sequence_length), int(args.mini_batch_size), float(args.learning_rate), int(args.max_epochs))
    else:
        pretrain(args.corpus_path, args.is_forward, args.out, int(args.hidden_size), int(args.sequence_length), int(args.mini_batch_size), float(args.learning_rate), int(args.max_epochs))

if __name__ == "__main__":
    main()
