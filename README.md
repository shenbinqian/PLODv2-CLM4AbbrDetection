# PLODv2: Character-level Language Models for Abbreviation and Long-form Detection
This repository contains the code and PLODv2 dataset to train character-level language models (CLMs) for abbreviation and long-form detection released with our paper [Using character-level models for efficient abbreviation and long-form detection](https://aclanthology.org/2024.lrec-main.270/) at LREC-COLING 2024.


## Installation

```
git clone https://github.com/surrey-nlp/PLODv2-CLM4AbbrDetection.git
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

Zilio, L, Qian, S., Kanojia, D. and Orasan, C., 2024. Utilizing Character-level Models for Efficient Abbreviation and Long-form Detection. In *Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)*. 

Zilio, L., Saadany, H., Sharma, P., Kanojia, D. and Orasan, C., 2022. PLOD: An Abbreviation Detection Dataset for Scientific Documents. In *Proceedings of the Thirteenth Language Resources and Evaluation Conference*. 

## BibTex Citation

Please use the following citation while citing this work:

```
@inproceedings{zilio-etal-2024-character-level,
    title = "Character-level Language Models for Abbreviation and Long-form Detection",
    author = "Zilio, Leonardo  and
      Qian, Shenbin  and
      Kanojia, Diptesh  and
      Orasan, Constantin",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.270",
    pages = "3028--3037",
    abstract = "Abbreviations and their associated long forms are important textual elements that are present in almost every scientific communication, and having information about these forms can help improve several NLP tasks. In this paper, our aim is to fine-tune language models for automatically identifying abbreviations and long forms. We used existing datasets which are annotated with abbreviations and long forms to train and test several language models, including transformer models, character-level language models, stacking of different embeddings, and ensemble methods. Our experiments showed that it was possible to achieve state-of-the-art results by stacking RoBERTa embeddings with domain-specific embeddings. However, the analysis of our first run showed that one of the datasets had issues in the BIO annotation, which led us to propose a revised dataset. After re-training selected models on the revised dataset, results show that character-level models achieve comparable results, especially when detecting abbreviations, but both RoBERTa large and the stacking of embeddings presented better results on biomedical data. When tested on a different subdomain (segments extracted from computer science texts), an ensemble method proved to yield the best results for the detection of long forms, and a character-level model had the best performance in detecting abbreviations.",
}
```

```
@inproceedings{zilio-etal-2022-plod,
    title = "{PLOD}: An Abbreviation Detection Dataset for Scientific Documents",
    author = "Zilio, Leonardo  and
      Saadany, Hadeel  and
      Sharma, Prashant  and
      Kanojia, Diptesh  and
      Or{\u{a}}san, Constantin",
    editor = "Calzolari, Nicoletta  and
      B{\'e}chet, Fr{\'e}d{\'e}ric  and
      Blache, Philippe  and
      Choukri, Khalid  and
      Cieri, Christopher  and
      Declerck, Thierry  and
      Goggi, Sara  and
      Isahara, Hitoshi  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Mazo, H{\'e}l{\`e}ne  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.71",
    pages = "680--688",
    abstract = "The detection and extraction of abbreviations from unstructured texts can help to improve the performance of Natural Language Processing tasks, such as machine translation and information retrieval. However, in terms of publicly available datasets, there is not enough data for training deep-neural-networks-based models to the point of generalising well over data. This paper presents PLOD, a large-scale dataset for abbreviation detection and extraction that contains 160k+ segments automatically annotated with abbreviations and their long forms. We performed manual validation over a set of instances and a complete automatic validation for this dataset. We then used it to generate several baseline models for detecting abbreviations and long forms. The best models achieved an F1-score of 0.92 for abbreviations and 0.89 for detecting their corresponding long forms. We release this dataset along with our code and all the models publicly at \url{https://github.com/surrey-nlp/PLOD-AbbreviationDetection}",
}
```