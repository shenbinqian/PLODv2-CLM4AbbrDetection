#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: shenbinqian
"""

from src.bio_corpus import get_bio_corpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import argparse
import ast


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bio_folder", type=str, default="./PLODv2/filtered_data", help="folder to BIO data")
    parser.add_argument("--embed_model", type=str, default='("glove", "news-forward", "news-backward")', help="What model embeddings to stack, can be 2, 3, or 4 model embeddings. The order of stacked embeddings SHOULD be WordEmbeddings (optional), Transformer embeddings (optional), FlairEmbeddings (required for both forward and backward models).")
    parser.add_argument("--save_folder", type=str, default="./stacked_glove_news", help="the folder name to save the model to")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="learning rate")
    parser.add_argument("--mini_batch_size", type=int, default=32, help="mini batch size")
    parser.add_argument("--max_epochs", type=int, default=150, help="max epochs")
    parser.add_argument("--use_transformer", type=str, default="False", help="whether to use Transformer embeddings while stacking 3 or 4 model embeddings")
    args = parser.parse_args()
    return args

def finetune_ner(bio_folder: str, embed_model: iter, save_folder: str, learning_rate: float=0.1, mini_batch_size: int=32, max_epochs: int=150, use_transformer: str="False"):

    # 1. get the corpus
    corpus = get_bio_corpus(bio_folder)
    
    # 2. what label do we want to predict?
    label_type = 'ner'
    
    # 3. make the label dictionary from the corpus
    label_dict = corpus.make_label_dictionary(label_type=label_type)

    # 4. get embed model
    '''The order of stacked embeddings SHOULD be WordEmbeddings (optional), Transformer embeddings (optional), FlairEmbeddings (required for both forward and backward models). 
    For example, ('news-forward', 'news-backward'), ('glove', 'news-forward', 'news-backward') or ('glove', 'bert-base-uncased', 'news-forward', 'news-backward').'''

    if type(embed_model) == str:
        embed_model = ast.literal_eval(embed_model)

    if len(embed_model) == 2:
    #while only stacking two embeddings, it has to be FlairEmbeddings
        flair_forward, flair_backward = embed_model
        embedding_types = [
            FlairEmbeddings(flair_forward),
            FlairEmbeddings(flair_backward),
        ]
    elif len(embed_model) == 3:
        if use_transformer == "False":
            word_embed, flair_forward, flair_backward = embed_model
            embedding_types = [
                WordEmbeddings(word_embed),
                FlairEmbeddings(flair_forward),
                FlairEmbeddings(flair_backward),
            ]
        else:
            transformer_embed_name, flair_forward, flair_backward = embed_model
            transformer_embed = TransformerWordEmbeddings(model=transformer_embed_name,
                                                          layers="-1",
                                                          subtoken_pooling="first",
                                                          fine_tune=True,
                                                          use_context=True,)
            embedding_types = [
                transformer_embed,
                FlairEmbeddings(flair_forward),
                FlairEmbeddings(flair_backward),
            ]
    elif len(embed_model) == 4:
        if use_transformer == "False":
            word_embed, word_embed2, flair_forward, flair_backward = embed_model
            embedding_types = [
                WordEmbeddings(word_embed),
                WordEmbeddings(word_embed2),
                FlairEmbeddings(flair_forward),
                FlairEmbeddings(flair_backward),
            ]
        else:
            word_embed, transformer_embed_name, flair_forward, flair_backward = embed_model
            transformer_embed = TransformerWordEmbeddings(model=transformer_embed_name,
                                                            layers="-1",
                                                            subtoken_pooling="first",
                                                            fine_tune=True,
                                                            use_context=True,)
            embedding_types = [
                WordEmbeddings(word_embed),
                transformer_embed,
                FlairEmbeddings(flair_forward),
                FlairEmbeddings(flair_backward),
            ]
    else:
        raise ValueError("We only support embed_model to be 2, 3, or 4, i.e. stacking four different embeddings.")

    
    # 5. initialize embeddings
    embeddings = StackedEmbeddings(embeddings=embedding_types)
    
    # 6. initialize sequence tagger
    tagger = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type=label_type,
                            use_crf=True)
    
    # 7. initialize trainer
    trainer = ModelTrainer(tagger, corpus)
    
    # 8. start training
    trainer.train(save_folder,
                  learning_rate=learning_rate,
                  mini_batch_size=mini_batch_size,
                  max_epochs=max_epochs,
                  embeddings_storage_mode="none")

def main():
    args = parse_args()
    finetune_ner(args.bio_folder, args.embed_model, args.save_folder, float(args.learning_rate), int(args.mini_batch_size), int(args.max_epochs), args.use_transformer)

if __name__=="__main__":
    main()
