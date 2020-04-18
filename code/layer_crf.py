import torch
from torch.autograd import Variable
import torch.nn as nn


START_TAG = "<SOS>"
STOP_TAG = "<EOS>"


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.view(-1).data.tolist()[0]


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.LongTensor(idxs)


def log_sum_exp(vec, dim=0):
    max_score, idx = torch.max(vec, dim)
    max_score_broadcast = max_score.unsqueeze(-1).expand_as(vec)
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim))


class CRF(nn.Module):
    def __init__(self, tag_idx, cuda_idx):
        super(CRF, self).__init__()
        self.tagset_size = len(tag_idx)
        self.tag_idx = tag_idx
        self.cuda_idx = cuda_idx

        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_idx[START_TAG], :] = -10000
        self.transitions.data[:, tag_idx[STOP_TAG]] = -10000

    def train_crf(self, feas, seq_len, tag):
        norm_score = self._forward_alg(feas, seq_len)
        seq_score = self._score_sentence(feas, tag, seq_len)
        return norm_score - seq_score

    def test_crf(self, feas, seq_len):
        score, tag_seq = self._viterbi_decode(feas, seq_len)
        return score, tag_seq

    def _forward_alg(self, feats, lens):  # lens: [batch_size] LongTensor
        batchSize = feats.size(0)
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(batchSize, self.tagset_size).fill_(-10000.)
        # START_TAG has all of the score.
        init_alphas[:,self.tag_idx[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        alpha = Variable(init_alphas)
        if torch.cuda.is_available():
            alpha = alpha.cuda(self.cuda_idx)

        # Iterate through the sentence
        c_lens = lens.clone()
        feats = feats.transpose(0, 1)
        for feat in feats:
            feat_exp = feat.unsqueeze(-1).expand(batchSize, *self.transitions.size())
            alpha_exp = alpha.unsqueeze(1).expand(batchSize, *self.transitions.size())
            trans_exp = self.transitions.unsqueeze(0).expand_as(alpha_exp)

            mat = alpha_exp + trans_exp + feat_exp

            alpha_next = log_sum_exp(mat, 2).squeeze(-1)

            '''mask'''
            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_next + (1 - mask) * alpha
            c_lens = c_lens - 1

        alpha = alpha + self.transitions[self.tag_idx[STOP_TAG]].unsqueeze(0).expand_as(alpha)
        norm = log_sum_exp(alpha, 1).squeeze(-1)
        return norm

    def _score_sentence(self, feats, tags, lens):
        # Gives the score of a provided tag sequence
        transition_score = self.transition_score(tags, lens)
        bilstm_score = self._bilstm_score(feats, tags, lens)

        score = transition_score + bilstm_score
        return score

    def _bilstm_score(self, feats, tags, lens):
        tags_exp = tags.unsqueeze(-1)
        scores = torch.gather(feats, 2, tags_exp).squeeze(-1)
        mask = self.sequence_mask(lens, max_len=scores.size(1)).float()
        scores = scores * mask
        score = scores.sum(1).squeeze(-1)
        return score

    def transition_score(self, labels, lens):
        """
        Arguments:
             labels: [batch_size, seq_len] LongTensor
             lens: [batch_size] LongTensor
        """
        batch_size, seq_len = labels.size()

        # pad labels with <start> and <stop> indices
        labels_ext = Variable(labels.data.new(batch_size, seq_len + 2))
        labels_ext[:, 0] = self.tag_idx[START_TAG]
        labels_ext[:, 1:-1] = labels
        mask = self.sequence_mask(lens + 1, max_len=seq_len + 2).long()
        pad_stop = Variable(labels.data.new(1).fill_(self.tag_idx[STOP_TAG]))
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        labels_ext = (1 - mask) * pad_stop + mask * labels_ext
        labels = labels_ext

        trn = self.transitions

        # obtain transition vector for each label in batch and timestep
        # (except the last ones)
        trn_exp = trn.unsqueeze(0).expand(batch_size, *trn.size())
        lbl_r = labels[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), trn.size(0))
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        # obtain transition score from the transition vector for each label
        # in batch and timestep (except the first ones)
        lbl_lexp = labels[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)

        mask = self.sequence_mask(lens + 1, max_len=trn_scr.size(1)).float()
        trn_scr = trn_scr * mask
        score = trn_scr.sum(1).squeeze(-1)

        return score

    def _viterbi_decode(self, feats, lens):
        batchSize = feats.size(0)

        # Initialize the viterbi variables in log space
        init_vit = torch.Tensor(batchSize, self.tagset_size).fill_(-10000.)
        init_vit[:, self.tag_idx[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        vit = Variable(init_vit)
        if torch.cuda.is_available():
            vit = vit.cuda(self.cuda_idx)

        pointers = []
        c_lens = lens.clone()
        feats = feats.transpose(0, 1)
        for feat in feats:
            vit_exp = vit.unsqueeze(1).expand(batchSize, *self.transitions.size())
            trn_exp = self.transitions.unsqueeze(0).expand_as(vit_exp)
            vit_trn_sum = vit_exp + trn_exp
            vt_max, vt_argmax = vit_trn_sum.max(2)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + feat
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transitions[self.tag_idx[STOP_TAG]].unsqueeze(0).expand_as(vit_nxt)

            c_lens = c_lens - 1

        pointers = torch.cat(pointers)
        scores, idx = vit.max(1)
        idx = idx.squeeze(-1)
        paths = [idx.unsqueeze(1)]

        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths

    def sequence_mask(self, lens, max_len=None):
        batch_size = lens.size(0)

        if max_len is None:
            max_len = lens.data.max()

        ranges = torch.arange(0, max_len).long()
        ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
        ranges = Variable(ranges)

        if lens.data.is_cuda:
            ranges = ranges.cuda(self.cuda_idx)

        lens_exp = lens.unsqueeze(1).expand_as(ranges)
        mask = ranges < lens_exp

        # if torch.cuda.is_available():
        #     mask = mask.cuda(self.cuda_idx)

        return mask