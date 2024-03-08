#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: shenbinqian
"""

from src.bio_corpus import get_bio_corpus
from flair.models import SequenceTagger
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bio_folder", type=str, default="./PLODv2/filtered_data", help="folder to BIO data")
    parser.add_argument("--model_path", type=str, default="surrey-nlp/flair-abbr-pubmed-filtered", help="path to trained model")
    parser.add_argument("--pred_file", type=str, default="./predictions.tsv", help="name for the prediction file")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="mini batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="num workers")
    args = parser.parse_args()
    return args


def predict(bio_folder="./PLODv2/filtered_data",
            model_path="surrey-nlp/flair-abbr-pubmed-filtered",
            pred_file="./predictions.tsv",
            mini_batch_size=8,
            num_workers=8):

    '''bio_folder: str = folder to the BIO data
       model_path: str = path to the trained NER model. It can be a local path (for example "./ner_pubmed_filtered/best-model.pt") or a Huggingface model name (for example "surrey-nlp/flair-abbr-pubmed-filtered")
       pred_file: str = path to the prediction file'''

    corpus = get_bio_corpus(bio_folder)
    model = SequenceTagger.load(model_path)


    test_result = model.evaluate(
        corpus.test,
        gold_label_type=model.label_type,
        mini_batch_size=mini_batch_size,
        num_workers=num_workers,
        out_path=pred_file,
        embedding_storage_mode="none",
        main_evaluation_metric=("micro avg", "f1-score"),
        gold_label_dictionary= None,
        exclude_labels=[],
        return_loss=False,
    )

    print(test_result.detailed_results)

def main():
    args = parse_args()
    predict(args.bio_folder, args.model_path, args.pred_file, int(args.mini_batch_size), int(args.num_workers))

if __name__ == "__main__":
    main()
