import argparse


class Config:
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    # gen_phrase_emb
    parser.add_argument('--word_emb', type=str, default='../corpus/wiki-100.vec')  #ori
    parser.add_argument('--adr_lexicon', type=str, default='../corpus/ADR_lexicon.txt')  # ori
    # get_features
    parser.add_argument('--dataset_bio', type=str, default='../corpus/cadec.bio')  #
    opt = parser.parse_args()

    phrase_emb = '../processed_file/phrase_100-avg.vec'
    adj_file = '../processed_file/adj_bcs.pkl'

    fea_file = '../processed_file/fea_file.pkl'
    util_file = '../processed_file/util_file.pkl'

