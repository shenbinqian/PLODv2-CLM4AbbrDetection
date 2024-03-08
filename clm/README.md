# Pre-trained Character-level language models

This folder contains pre-trained (i.e., plm-plos) and continued pre-trained (i.e., cplm-pubmed-plos) character-level language models we trained on the PLOS dataset from stratch or using the PubMed model in Flair. Each model contains a forward and backward model.


These models are not used directly for NER or abbreviation detection tasks, but they can be fine-tuned for NER tasks via the [src/train_ner.py](https://github.com/shenbinqian/PLODv2-CLM4AbbrDetection/blob/main/src/train_ner.py) script. To do this, just pass an arugment using "--embed_model", for example, **--embed_model "('glove', 'clm/cplm-pubmed-plos/forward/best-lm.pt', 'clm/cplm-pubmed-plos/backward/best-lm.pt')"** as shown in the example in src/train_ner.py.
