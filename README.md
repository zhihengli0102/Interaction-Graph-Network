# IGN

This repo contains the source code for the following paper:
-   Zhiheng Li, Zhihao Yang, Lei Wang, Yin Zhang, Hongfei Lin, Jian Wang. Lexicon Knowledge Boosted Interaction Graph Network for Adverse Drug Reaction Recognition from Social Media. 

## Dependency package

IGN uses the following dependencies:

-   [Python 3.6](https://www.python.org/)
-   [Pytorch 1.0.1](https://pytorch.org/get-started/locally/)
-   [NLTK](http://www.nltk.org/)

## Content
-   adj_generation
    - chunk_phrase: Get noun phrases using nltk
    - termkb_adj.py: Get adjacency matrices
-   corpus
    -   [ADR lexicon](http://diego.asu.edu/downloads/publications/ADRMine/ADR_lexicon.tsv): The lexicon used in IGN
    - file_format.txt: Example of the input of IGN
-   code
    - config
    - dataset: Load the dataset
    - elmo_model: ELMo layer
    - emb_dict: Load pre-trained word embeddings
    - layer_crf: CRF layer
    - layer_gat: GAT layer
    - layer_word: Word representation layer
    - model: IGN aechitecture
    - tagger_test: Evaluate results
    - train: Train an IGN model
- models
- preprocess
    - config
    - emb_dict: Build vocabulary
    - gen_phrase_emb: Get phrase embeddings
    - get_feature: Generate features from texts
- processed_file
- results

## Train IGN
Using termkb_adj.py to get adjacency matrices:
```
python termkb_adj.py --adr_lexicon filename --dataset_bio filename
```
Using get_features.py to get the input of IGN:
```
python get_features.py --word_emb filename --dataset_bio filename
```
To train an IGN model, prepare the fea_file.pkl and util_file and then run the train.py script:
```
python train.py
```
To test an IGN model, you need to provide the model and parameter files and then run the tagger_test.py script:
```
python tagger_test.py
```

