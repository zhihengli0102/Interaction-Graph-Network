import numpy as np
import pickle as pkl
from config import Config


unk = 'UNKNOWN-1'


def get_wemb_dic(w_file):
    word_dict = {}
    word_emb = []

    emb_rf = open(w_file, 'r', encoding='UTF-8')

    vec_size = int(emb_rf.readline().strip().split(' ')[1])
    word_dict['padding-0'] = 0
    word_emb.append([0.0] * vec_size)  # padding-0
    word_dict[unk] = 1
    word_emb.append([1.0] * vec_size)  # UNKNOWN-1

    line = emb_rf.readline()
    while line:
        word = line.split()[0]
        vec = line[len(word) + 1:-1].strip().split()
        word_dict[word] = len(word_dict)
        word_emb.append([float(v) for v in vec])
        line = emb_rf.readline()
    emb_rf.close()
    return word_emb, word_dict


def get_ph_dic(p_file):
    p_list = []
    with open(p_file, 'rb') as handle:
        adj_dict = pkl.load(handle)

    for key_id in adj_dict.keys():
        nodes = adj_dict[key_id][0]
        p_list += nodes
    return list(set(p_list))


def match_pemb(word_emb, word_dict, p_list):
    ph_emb = {}
    for ph in p_list:
        word_list = ph.split()
        emb = np.array([0.0] * len(word_emb[0]))
        for w in word_list:
            if w not in word_dict:
                w = unk
            emb += np.array(word_emb[word_dict[w]])

        ph_emb[ph] = (emb / float(len(word_list))).tolist()
    return ph_emb  # without padding or unk

def get_phrase_emb():
    p_list = get_ph_dic(Config.adj_file)
    word_emb, word_dict = get_wemb_dic(Config.opt.word_emb)

    ph_emb = match_pemb(word_emb, word_dict, p_list)

    wr = open(Config.phrase_emb, 'w')
    wr.write(str(len(ph_emb)) + ' ' + str(len(word_emb[0])) + '\n')
    for key in ph_emb.keys():
        wr.write('_'.join(key.split()) + ' ' + ' '.join([str(v) for v in ph_emb[key]]) + '\n')
    wr.close()

if __name__ == '__main__':
    get_phrase_emb()
    print('Done...')