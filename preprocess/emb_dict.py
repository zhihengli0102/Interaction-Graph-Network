import numpy as np

'''
collect words from train file and return word_dict = {word: id}
find the word in embfile and return trained_emb = [[tensor], [tensor], ...]
'''


def read_emb(embname, unk):
    emb_dict = {}
    trained_emb = []

    emb_rf = open(embname, 'r', encoding='UTF-8')
    vec_size = int(emb_rf.readline().strip().split(' ')[1])
    emb_dict['padding-0'] = 0
    trained_emb.append([0.0] * vec_size)  # padding-0
    emb_dict[unk] = 1
    trained_emb.append(np.random.uniform(-1, 1, vec_size).tolist())  # UNKNOWN-1

    line = emb_rf.readline()
    while line:
        word = line.split()[0]
        vec = line[len(word) + 1:-1].strip().split()
        emb_dict[word] = len(emb_dict)
        trained_emb.append([float(v) for v in vec])
        line = emb_rf.readline()
    emb_rf.close()
    return trained_emb, emb_dict


def wid_emb(embname, word_dict):
    trained_emb = [[]] * len(word_dict)
    emb_rf = open(embname, 'r')
    vec_size = int(emb_rf.readline().strip().split(' ')[1])
    trained_emb[0] = [0.0] * vec_size  # padding-0
    trained_emb[1] = np.random.uniform(-1, 1, vec_size).tolist()  # UNKNOWN-1

    line = emb_rf.readline()
    while line:  # the word existing in both training set and emb file
        word = line.split()[0]
        if word in word_dict:
            vec = line[len(word) + 1:-1].strip().split()
            emb = []
            [emb.append(float(v)) for v in vec]
            trained_emb[word_dict[word]] = emb
        line = emb_rf.readline()
    emb_rf.close()

    for v_i in range(len(trained_emb)):  # the word not in emb file
        if len(trained_emb[v_i]) == 0:
            trained_emb[v_i] = np.random.uniform(-1, 1, vec_size).tolist()
    return trained_emb


def nid_emb(embname, idx_dict):
    trained_emb = [[]] * (len(idx_dict) + 1)
    emb_rf = open(embname, 'r')
    vec_size = int(emb_rf.readline().strip().split(' ')[1])
    trained_emb[0] = [0.0] * vec_size  # padding-0

    line = emb_rf.readline()
    while line:  # the word existing in both training set and emb file
        n_id = line.split()[0]
        if n_id in idx_dict:
            vec = line[len(n_id) + 1:-1].strip().split()
            emb = []
            [emb.append(float(v)) for v in vec]
            trained_emb[int(n_id)] = emb
        line = emb_rf.readline()
    emb_rf.close()

    for v_i in range(len(trained_emb)):  # the word not in emb file
        if len(trained_emb[v_i]) == 0:
            trained_emb[v_i] = np.random.uniform(-1, 1, vec_size).tolist()
    return trained_emb