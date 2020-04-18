import numpy as np
import torch
import torch.utils.data as data
import copy


class DatasetIns(data.Dataset):
    def __init__(self, sets):
        super(DatasetIns, self).__init__()
        # [tag, word, sen_len, char, char_len, bi_tag, adr_t, drg_t, tokens]
        self.tag = sets[0]
        self.w_idx = sets[1]
        self.sen_len = sets[2]
        self.adr_tag = sets[3]
        self.node_idx = sets[4]
        self.adj_b = sets[5]
        self.adj_b = sets[6]
        self.adj_s = sets[7]


    def __len__(self):
        return len(self.w_idx)
        
    def __getitem__(self, index):        
        tag = self.tag[index]
        w_idx = self.w_idx[index]
        sen_len = self.sen_len[index]

        adr_tag = self.adr_tag[index]

        node_idx = self.node_idx[index]
        adj_b = self.adj_b[index]
        adj_c = self.adj_b[index]
        adj_s = self.adj_s[index]

        return tag, w_idx, sen_len, adr_tag, node_idx, adj_b, adj_c, adj_s

    
def pad(batch):
    '''Pads to the longest sample'''
    # 0 tag, 1 w_idx, 2 sen_len,
    # 3 adr_tag, 4 node_idx, 5-7 adj

    f = lambda x: [sample[x] for sample in batch]
    sen_lens = f(2)
    adj_b = copy.deepcopy(f(5))
    adj_c = copy.deepcopy(f(6))
    adj_s = copy.deepcopy(f(7))
    sen_maxlen = np.array(sen_lens).max()
    graph_maxlen = np.array([len(adj) for adj in adj_b]).max()

    f_pad1d = lambda x, seqlen: [(sample[x] + [0] * (seqlen - len(sample[x])))[:seqlen] for sample in batch]  # 0: <pad>

    tag = f_pad1d(0, sen_maxlen)
    w_idx = f_pad1d(1, sen_maxlen)
    adr_tag = f_pad1d(3, sen_maxlen)

    n_idx = f_pad1d(4, graph_maxlen)
    adj_b = graph_pad(adj_b, graph_maxlen)
    adj_c = graph_pad(adj_c, graph_maxlen)
    adj_s = graph_pad(adj_s, graph_maxlen)
    
    f_tensor = torch.LongTensor
    return f_tensor(tag), f_tensor(w_idx), f_tensor(sen_lens), f_tensor(adr_tag), \
           f_tensor(n_idx), f_tensor(adj_b), f_tensor(adj_c), f_tensor(adj_s)


def graph_pad(adj_sist, maxlen):
    for adj in adj_sist:
        for n in adj:
            n += [0] * (maxlen - len(adj))
        adj += [[0] * maxlen]*(maxlen-len(adj))
    return adj_sist