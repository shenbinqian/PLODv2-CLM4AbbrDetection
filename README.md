# PLODv2: Character-level Language Models for Abbreviation and Long-form Detection
This repository contains the code and PLODv2 dataset to train character-level language models (CLMs) for abbreviation and long-form detection released with our LREC-COLING 2024 publication (coming soon).


## Installation

```
git clone https://github.com/shenbinqian/PLODv2-CLM4AbbrDetection.git
conda create -n abbrdet python=3.9
conda activate abbrdet
cd PLODv2-CLM4AbbrDetection
pip install -r requirements.txt
```

## Train abbreviation detection systems

Use our [PLODv2 dataset](https://github.com/shenbinqian/PLODv2-CLM4AbbrDetection/tree/main/PLODv2) for training abbreviation detection systems:

```
CUDA_VISIBLE_DEVICES=0 python -m src.train_ner \
    --bio_folder ./PLODv2/filtered_data \
    --embed_model '("glove", "news-forward", "news-backward")' \
    --save_folder ./stacked_glove_news_filtered \
    --learning_rate 0.01 \ 
    --mini_batch_size 32 \
    --max_epochs 150 \
    --use_transformer False
```

## Train character-level language models

Train character-level language models from scratch or for continued pre-training:

```
CUDA_VISIBLE_DEVICES=0 python -m src.flair_clm \
    --continue_pretrain False \
    --corpus_path ./corpus \
    --is_forward True \
    --out plm \
    --learning_rate 20.0 \ 
    --mini_batch_size 100 \
    --max_epochs 300 \
    --sequence_length 256 \
    --hidden_size 2048
```


## Our models for abbreviation detection

Our fine-tuned models for abbreviation and long form detection can be seen as follows:


No. | PLODv2-Unfiltered                                       | PLODv2-Filtered                                       |
|:--:|:-------------------------------------------------------:|:-----------------------------------------------------:|
| 1 | [surrey-nlp/roberta-large-finetuned-abbr-unfiltered-plod](https://huggingface.co/surrey-nlp/roberta-large-finetuned-abbr-unfiltered-plod) | [surrey-nlp/roberta-large-finetuned-abbr-filtered-plod](https://huggingface.co/surrey-nlp/roberta-large-finetuned-abbr-filtered-plod) |
| 2 | [surrey-nlp/flair-abbr-pubmed-unfiltered](https://huggingface.co/surrey-nlp/flair-abbr-pubmed-unfiltered)  | [surrey-nlp/flair-abbr-pubmed-filtered](https://huggingface.co/surrey-nlp/flair-abbr-pubmed-filtered)  |
| 3 | [surrey-nlp/flair-abbr-roberta-pubmed-plos-unfiltered](https://huggingface.co/surrey-nlp/flair-abbr-roberta-pubmed-plos-unfiltered) | [surrey-nlp/flair-abbr-roberta-pubmed-plos-filtered](https://huggingface.co/surrey-nlp/flair-abbr-roberta-pubmed-plos-filtered)  | 


In the table, No. 1 series models are finetuned on PLODv2 dataset using RoBERTa-large. No. 2 series models are finetuned on PLODv2 by stacking of character-level [PubMed models](https://github.com/flairNLP/flair/blob/master/resources/docs/HUNFLAIR.md). No. 3 series models are fine-tuned on PLODv2 by stacking of RoBERTa-large and our [continued pretrained character-level language models](https://github.com/shenbinqian/PLODv2-CLM4AbbrDetection/tree/main/clm/cplm-pubmed-plos) on [PLOS](https://github.com/shenbinqian/PLODv2-CLM4AbbrDetection/tree/main/PLODv1) based on PubMed models.


## Inference

To run (or fine-tune) Transformer models such as RoBERTa large, check our [jupyter notebooks](https://github.com/shenbinqian/PLODv2-CLM4AbbrDetection/tree/main/notebooks). Inference using our fine-tuned stacked-embedding models via flair:

```
CUDA_VISIBLE_DEVICES=0 python -m src.predict \
    --bio_folder ./PLODv2/filtered_data \
    --model_path surrey-nlp/flair-abbr-pubmed-filtered \
    --pred_file ./predictions.tsv \
    --mini_batch_size 8 \
```

## Citation

Zilio, L, Qian, S., Kanojia, D. and Orasan, C., 2024. Utilizing Character-level Models for Efficient Abbreviation and Long-form Detection. Accepted by the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024). 

Zilio, L., Saadany, H., Sharma, P., Kanojia, D. and Orasan, C., 2022. PLOD: An Abbreviation Detection Dataset for Scientific Documents. In *Proceedings of the Thirteenth Language Resources and Evaluation Conference*. 