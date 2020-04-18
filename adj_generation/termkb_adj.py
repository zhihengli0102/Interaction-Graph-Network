import pickle as pkl
import re
from chunk_phrase import run_chunk
from preprocess.config import Config


def read_cadec(inses, term_list):
    adj_dict = {}
    for ins in inses:
        t_id, terms = ins.split('\n')[0], ins.split('\n')[1:]
        get_part = lambda x: [sample.strip().split('\t')[x] for sample in terms]
        tokens = get_part(0)

        chunk_list = run_chunk(tokens)
        node_dict = get_phrase(' '.join(tokens), term_list + chunk_list)
        node_dict['<sos>'] = [[-1]]
        node_dict['<eos>'] = [[-1]]

        node_pos = [n for n in tokens + list(node_dict.keys())]
        adj_W = get_adj_W(tokens, node_pos, node_dict)
        adj_B = get_adj_B(tokens, node_pos, node_dict)
        adj_S = get_adj_S(tokens, node_pos, node_dict)
        adj_dict[t_id] = [node_pos, adj_W, adj_B, adj_S]

    return adj_dict


def get_phrase(sen, phrase_list):
    node_dict = {}
    for ph in phrase_list:
        if ' ' + ph + ' ' in ' ' + sen + ' ':
            node_dict[ph] = []
            ph_pos = [[p.start(), p.end()] for p in re.finditer(' ' + ph + ' ', ' ' + sen + ' ')]
            for pos in ph_pos:
                s = len(sen[:pos[0]].split())
                assert sen.split()[s] in ph, print('sen.split()[s] not in ph')
                p_in = [s]
                while s + 1 < len(sen.split()) and sen.split()[s + 1] in ph:
                    p_in.append(s + 1)
                    s += 1
                node_dict[ph].append(p_in)
    return node_dict


def get_adj_W(tokens, node_pos, node_dict):
    adj = [[0] * len(node_pos) for row in range(len(node_pos))]
    for i_add in range(len(tokens), len(node_pos) - 2):
        node_a = node_pos[i_add]


        for pos_list in node_dict[node_a]:
            # the first and the last word link to the phrase
            adj[pos_list[0]][i_add] = 1
            adj[i_add][pos_list[0]] = 1

            adj[pos_list[len(pos_list) - 1]][i_add] = 1
            adj[i_add][pos_list[len(pos_list) - 1]] = 1

            # links between the words in a phrase
            if len(pos_list) > 1:
                for i_pos in range(0, len(pos_list) - 1):
                    adj[pos_list[i_pos]][pos_list[i_pos + 1]] = 1
                    adj[pos_list[i_pos + 1]][pos_list[i_pos]] = 1

    for i in range(len(node_pos)):
        adj[i][i] = 1    # self loop

    return adj


def get_adj_B(tokens, node_pos, node_dict):
    adj = [[0] * len(node_pos) for row in range(len(node_pos))]
    for i_add in range(len(tokens), len(node_pos) - 2):
        node_a = node_pos[i_add]
        for pos_list in node_dict[node_a]:  # term in entities
            if pos_list[0] == 0:
                adj[i_add][node_pos.index('<sos>')] = 1
                adj[node_pos.index('<sos>')][i_add] = 1
            elif pos_list[len(pos_list) - 1] == len(tokens) - 1:
                adj[i_add][node_pos.index('<eos>')] = 1
                adj[node_pos.index('<eos>')][i_add] = 1
            for i_tok in range(len(tokens)):
                if pos_list[0] > 0 and \
                        i_tok == pos_list[0] - 1 \
                    or pos_list[len(pos_list) - 1] < len(tokens) - 1 and \
                        i_tok == pos_list[len(pos_list) - 1] + 1:
                    adj[i_tok][i_add] = 1
                    adj[i_add][i_tok] = 1

    for i in range(len(node_pos)):
        adj[i][i] = 1  # self loop
        if i + 1 < len(tokens):
            adj[i][i + 1] = 1  # neighbor nodes
            adj[i + 1][i] = 1

    adj[0][node_pos.index('<sos>')] = 1
    adj[node_pos.index('<sos>')][0] = 1
    adj[len(tokens) - 1][node_pos.index('<eos>')] = 1
    adj[node_pos.index('<eos>')][len(tokens) - 1] = 1

    return adj


def get_adj_S(tokens, node_pos, node_dict):
    adj = [[0] * len(node_pos) for row in range(len(node_pos))]
    for i_add in range(len(tokens), len(node_pos) - 2):
        node_a = node_pos[i_add]

        for pos_list in node_dict[node_a]:
            for i_tok in pos_list:  # each word in a phrase links to the phrase
                adj[i_tok][i_add] = 1
                adj[i_add][i_tok] = 1

    for i in range(len(node_pos)):
        adj[i][i] = 1  # self loop
        if i + 1 < len(tokens):
            adj[i][i + 1] = 1  # neighbor nodes
            adj[i + 1][i] = 1

    adj[0][node_pos.index('<sos>')] = 1
    adj[node_pos.index('<sos>')][0] = 1
    adj[len(tokens) - 1][node_pos.index('<eos>')] = 1
    adj[node_pos.index('<eos>')][len(tokens) - 1] = 1

    return adj


if __name__ == '__main__':

    rf = open(Config.opt.adr_lexicon, 'r', encoding='UTF-8')
    term_list = []
    for line in rf:
        if len(line.strip().split('\t')) == 3:
            term_list.append(line.strip().split('\t')[1])

    rf = open(Config.opt.dataset_bio, 'r')
    lines = rf.readlines()
    rf.close()
    inses = ''.join(lines).strip().split('\n\n')
    adj_dict = read_cadec(inses, term_list)

    with open(Config.adj_file, 'wb') as handle:
        pkl.dump(adj_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

    print('get adjacency matrices for BCS-graphs -> ../processed_file/adj_bcs.pkl')