# Static Word Embeddings

We support the stacking of static word embeddings including [GloVe](https://aclanthology.org/D14-1162/), [fastText](https://arxiv.org/abs/1607.04606) and pubmed model embeddings.

To use GloVe embeddings, just pass a string "glove" as an argument of "--embed_model" while calling train_ner.py. To use fastText or pubmed embeddings, first unzip the two zip files under the "static_embeddings" folder and pass the path (the gensim file) as an argument of "--embed_model". Encoder-based transformer embeddings can also be used for embedding stacking simply by passing a string of the pre-trained transformer model name.