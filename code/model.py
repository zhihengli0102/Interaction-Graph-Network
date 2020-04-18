import torch
import torch.nn as nn
from torch.autograd import Variable

from config import Config

from layer_word import Word_model
from layer_gat import GAT
from layer_crf import CRF


class Model(nn.Module):
    def __init__(self, word_embs, tag_idx, kb_size, cuda_idx, bs=1, node_embs=None):
        super(Model, self).__init__()
        tag_idx["<SOS>"] = len(tag_idx)
        tag_idx["<EOS>"] = len(tag_idx)
        self.tagset_size = len(tag_idx)
        self.tag_idx = tag_idx
        self.bs = bs
        self.cuda_idx = cuda_idx

        word_dim = word_embs.size(1)
        self.hd_dim = hd_dim = Config.hd_dim
        self.kb_dim = kb_dim = Config.kb_dim

        self.kb_flag = Config.kb_flag
        self.graph_flag = Config.graph_flag

        input_dim = word_dim
        '''kb embedding layer'''
        if self.kb_flag:
            input_dim = input_dim + kb_dim
            self.adr_embed = nn.Embedding(kb_size, kb_dim, padding_idx=0)

        self.input_dim = input_dim

        '''word embedding layer'''
        self.word_embed = nn.Embedding(word_embs.size(0), word_dim, padding_idx=0)
        self.word_embed.weight = nn.Parameter(word_embs, requires_grad=True)

        '''node embedding layer'''
        if self.graph_flag:
            node_dim = node_embs.size(1)
            self.node_embed = nn.Embedding(node_embs.size(0), node_dim, padding_idx=0)
            self.node_embed.weight = nn.Parameter(node_embs, requires_grad=True)
            self.node_trans = nn.Linear(node_dim, 2 * hd_dim, bias=False)

            '''gat layer'''
            # nfeat, nhid, nclass, dropout, alpha, nheads, layer
            self.gat_1 = GAT(2*hd_dim, hd_dim, self.tagset_size, dropout=0.5, alpha=0.1, nheads=3, layer=2)
            self.gat_2 = GAT(2*hd_dim, hd_dim, self.tagset_size, dropout=0.5, alpha=0.1, nheads=3, layer=2)
            self.gat_3 = GAT(2*hd_dim, hd_dim, self.tagset_size, dropout=0.5, alpha=0.1, nheads=3, layer=2)

            self.weight = nn.Linear(self.tagset_size*4, self.tagset_size)

        '''word lstm layer'''
        self.ner_lstm = Word_model(self.input_dim, hd_dim)

        '''crf layer'''
        self.crf_layer = CRF(tag_idx, self.cuda_idx)

        '''tag layer'''
        self.hidden2tag = nn.Linear(2 * hd_dim, self.tagset_size)

    def forward(self, batch, flag='train'):
        tag, w_seq, seq_len, adr_tag, n_idx, adj_b, adj_c, adj_s = self.batch_data(batch)

        word_mask = torch.ne(w_seq, 0.0)
        word_rep = self.word_embed(w_seq)
        input_rep = word_rep

        if self.kb_flag:
            input_rep = torch.cat((input_rep, self.adr_embed(adr_tag)), 2)

        lstm_fea, extr_senvec = self.ner_lstm.lstm(input_rep, word_mask)  # masked outs_lstm

        if not self.graph_flag:
            crf_feature = self.hidden2tag(lstm_fea)

        if self.graph_flag:
            node_rep = self.node_embed(n_idx)
            node_rep = self.node_trans(node_rep)  # trans node_rep dim into 2*hd_dim
            node_pad = torch.zeros(self.bs, node_rep.size(1) - lstm_fea.size(1), lstm_fea.size(2))
            if torch.cuda.is_available():
                node_pad = node_pad.cuda(self.cuda_idx)
            gat_input = torch.cat((lstm_fea, node_pad), 1) + node_rep

            max_len = max(list(seq_len))
            
            gat_fea_1 = self.gat_1(gat_input, adj_b)
            gat_fea_1 = gat_fea_1.transpose(0, 1)[:max_len].transpose(0, 1)
            
            gat_fea_2 = self.gat_2(gat_input, adj_c)
            gat_fea_2 = gat_fea_2.transpose(0, 1)[:max_len].transpose(0, 1)
            
            gat_fea_3 = self.gat_3(gat_input, adj_s)
            gat_fea_3 = gat_fea_3.transpose(0, 1)[:max_len].transpose(0, 1)
            
            lstm_fea = self.hidden2tag(lstm_fea)

            crf_feature = torch.cat((lstm_fea, gat_fea_1, gat_fea_2, gat_fea_3), dim=2)
            crf_feature = self.weight(crf_feature)

        if flag == 'train':
            score = self.crf_layer.train_crf(crf_feature, seq_len, tag)
            return score.mean()
        else:
            score, tag_seq = self.crf_layer.test_crf(crf_feature, seq_len)
            return score.mean(), tag_seq

    def batch_data(self, batch):
        tag, w_seq, seq_len, \
        adr_tag, n_idx, adj_w, adj_b, adj_s \
        = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), \
          Variable(batch[3]), Variable(batch[4]), Variable(batch[5]), \
          Variable(batch[6]), Variable(batch[7])

        tag = torch.cat((tag, tag[-1].unsqueeze(0).expand(self.bs - tag.size(0), tag.size(1))), 0)
        w_seq = torch.cat((w_seq, w_seq[-1].unsqueeze(0).expand(self.bs - w_seq.size(0), w_seq.size(1))), 0)
        seq_len = torch.cat((seq_len, seq_len[-1].expand(self.bs - seq_len.size(0))), 0)
        adr_tag = torch.cat((adr_tag, adr_tag[-1].unsqueeze(0).expand(self.bs - adr_tag.size(0), adr_tag.size(1))), 0)
        n_idx = torch.cat((n_idx, n_idx[-1].unsqueeze(0).expand(self.bs - n_idx.size(0), n_idx.size(1))), 0)
        adj_w = torch.cat((adj_w, adj_w[-1].unsqueeze(0).expand(self.bs - adj_w.size(0), adj_w.size(1), adj_w.size(2))), 0)
        adj_b = torch.cat((adj_b, adj_b[-1].unsqueeze(0).expand(self.bs - adj_b.size(0), adj_b.size(1), adj_b.size(2))), 0)
        adj_s = torch.cat((adj_s, adj_s[-1].unsqueeze(0).expand(self.bs - adj_s.size(0), adj_s.size(1), adj_s.size(2))), 0)

        if torch.cuda.is_available():
            tag, w_seq, seq_len, adr_tag, \
            n_idx, adj_w, adj_b, adj_s \
                = tag.cuda(self.cuda_idx), w_seq.cuda(self.cuda_idx), seq_len.cuda(self.cuda_idx), \
                  adr_tag.cuda(self.cuda_idx), \
                  n_idx.cuda(self.cuda_idx), adj_w.cuda(self.cuda_idx), adj_b.cuda(self.cuda_idx), adj_s.cuda(self.cuda_idx), \

        return tag, w_seq, seq_len, adr_tag, n_idx, adj_w, adj_b, adj_s