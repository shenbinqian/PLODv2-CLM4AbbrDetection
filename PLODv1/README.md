# The PLOD v1 dataset

The PLOD v1 dataset is from our paper [PLOD: An Abbreviation Detection Dataset for Scientific Documents](https://aclanthology.org/2022.lrec-1.71/).


The files under the two folders are the same from our previous GitHub repository: https://github.com/surrey-nlp/PLOD-AbbreviationDetection. Files that end with ".conll" contain BIO annotated data. Files that start with "PLOS" are the PLOS dataset that contains text segments and the corresponding indices of abbreviations and long forms in the text segments. They can also be accessed via our HuggingFace [unfiltered](https://huggingface.co/datasets/surrey-nlp/PLOD-unfiltered) and [filtered](https://huggingface.co/datasets/surrey-nlp/PLOD-filtered) datasets.

As pointed out in our latest paper, the BIO data, i.e., the ".conll" files contain misannotated information, so we re-annotated it using the [POS_BIO_annotation.ipynb](https://github.com/shenbinqian/PLODv2-CLM4AbbrDetection/blob/main/notebooks/POS_BIO_annotation.ipynb). The new BIO data is in the [PLODv2](https://github.com/shenbinqian/PLODv2-CLM4AbbrDetection/tree/main/PLODv2) folder. We strongly recommend using the files in PLODv2 for training/fine-tuning systems for abbreviation detection.