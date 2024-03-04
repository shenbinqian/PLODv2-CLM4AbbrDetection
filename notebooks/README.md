# Jupyter notebooks for BIO annotation & Fine-tuning

The "POS_BIO_annotation.ipynb" notebook is used to convert the PLOS dataset (for example "PLODv1/filtered_data/PLOS-test15-filtered") into BIO-annotated data for NER system training. 

The "fine_tuning_abbr_det.ipynb" notebook contains the code for fine-tuning pre-trained Transformer models, such as the RoBERTa-large model as our baseline. The majority of the notebook comes from our LREC 2022 paper [PLOD: An Abbreviation Detection Dataset for Scientific Documents](https://github.com/surrey-nlp/PLOD-AbbreviationDetection). We added a section "Compare RoBERTa-large model predictions with flair models for Disagreement Analysis" to compare the predictions of different models for the disagreement analysis in our LREC-COLING 2024 paper.