from config import Config
import emb_dict
import pickle as pkl
from gen_phrase_emb import get_phrase_emb

unknow_word = 'UNKNOWN-1'
tag_dict = {"pading-0": 0}
word_dict = {"pading-0": 0, unknow_word: 1}
kb_dict = {'padding-0': 0, 'B-ADR': 1, 'I-ADR': 2, 'E-ADR': 3, 'S-ADR': 4, 'O': 5}
phrase_dict = {"pading-0": 0, unknow_word: 1}


def get_fea(inses, adj_dict):
    # fea_list = ['tag', 'word', 'sen_len', 'adr_t', 'add_node_idx_seq', 'adj_w', 'adj_b', 'adj_s']
    ins_seq = [[], [], [], [], [], [], [], []]
    for ins in inses:
        # [token ori_word start end adr tag]
        ins_id = ins.split('\n')[0]
        get_part = lambda x: [sample.strip().split('\t')[x] for sample in ins.split('\n')[1:]]
        tokens = get_part(0)
        adr_t = get_part(4)
        tag_t = get_part(5)

        ins_seq[0].append(tag2idx(tag_t, tag_dict))  # target tags
        ins_seq[1].append(fea2idx(tokens, word_dict))  # word_idx
        ins_seq[2].append(len(tokens))  # sen_len
        ins_seq[3].append(fea2idx(bio2bioes(adr_t, 'ADR'), kb_dict))  # BIO or BIOES

        ins_seq[4].append([0] * len(tokens) + fea2idx_ph(adj_dict[ins_id][0][len(tokens):], phrase_dict))
        ins_seq[5].append(adj_dict[ins_id][1])  # adj_b
        ins_seq[6].append(adj_dict[ins_id][2])  # adj_c
        ins_seq[7].append(adj_dict[ins_id][3])  # adj_s

    return ins_seq


def bio2bioes(tags, flag='ADR'):
    tag_seq = ' '.join(tags)
    while 'O B-' + flag + ' O' in tag_seq or 'I-' + flag + ' O' in tag_seq:
        tag_seq = tag_seq.replace('O B-' + flag + ' O', 'O S-' + flag + ' O') \
            .replace('I-' + flag + ' O', 'E-' + flag + ' O')
    return tag_seq.split()


def tag2idx(tags, t_dict):
    idx_list = []
    for f in tags:
        if f not in t_dict:
            t_dict[f] = len(t_dict)
        idx_list.append(t_dict[f])
    return idx_list


def fea2idx(feas, f_dict):
    idx_list = []
    for f in feas:
        if f not in f_dict:
            idx_list.append(f_dict[unknow_word])
        else:
            idx_list.append(f_dict[f])
    return idx_list


def fea2idx_ph(feas, f_dict):
    idx_list = []
    for f in feas:
        if '_'.join(f.split()) not in f_dict:
            idx_list.append(f_dict[unknow_word])
        else:
            idx_list.append(f_dict['_'.join(f.split())])
    return idx_list


def get_kb(adr_t, drg_t):
    kb_tags = []
    for i in range(len(adr_t)):
        if adr_t[i] == drg_t[i]:
            kb_tags.append(kb_dict[adr_t[i]])
        elif adr_t[i] == 'O':
            kb_tags.append(kb_dict[drg_t[i]])
        elif drg_t[i] == 'O':
            kb_tags.append(kb_dict[adr_t[i]])
        else:
            print(adr_t[i], drg_t[i])
            kb_tags.append(kb_dict[adr_t[i]])
    return kb_tags


if __name__ == '__main__':

    word_emb, word_dict = emb_dict.read_emb(Config.opt.word_emb, unknow_word)  # load word_emb
    get_phrase_emb()
    print("get phrase embeddings  -> ../processed_file/phrase_100-avg.vec")
    phrase_emb, phrase_dict = emb_dict.read_emb(Config.phrase_emb, unknow_word)  # load phrase_emb

    with open(Config.adj_file, 'rb') as handle:
        adj_dict = pkl.load(handle)

    rf = open(Config.opt.dataset_bio, 'r')
    inses_dataset = ''.join(rf.readlines()).strip().split('\n\n')
    rf.close()

    fea_dict = []
    fea_dict = [get_fea(inses_dataset, adj_dict), inses_dataset]

    '''write files'''
    with open(Config.fea_file, 'wb') as handle:
        pkl.dump(fea_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
    print("get features  -> ../processed_file/fea_file.pkl")

    with open(Config.util_file, 'wb') as handle:
        pkl.dump([tag_dict, word_dict, word_emb, phrase_dict, phrase_emb, kb_dict],
                 handle, protocol=pkl.HIGHEST_PROTOCOL)
    print("get util_file  -> ../processed_file/util_file.pkl")

    print('Done...')