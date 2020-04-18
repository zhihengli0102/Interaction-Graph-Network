import torch
import torch.nn as nn
from torch.nn.utils import rnn

import numpy as np


class Word_model(nn.Module):
    def __init__(self, input_dim, hd_dim):
        super(Word_model, self).__init__()
        self.lstm_fea = nn.LSTM(input_dim, hd_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.cnn_fea = nn.Conv1d(input_dim, 2*hd_dim, kernel_size=3, stride=1, padding=1)

        self.dropout5 = nn.Dropout(0.5)

    def cnn(self, word_rep):
        cnn_in = self.dropout5(word_rep)
        cnn_out = self.cnn_fea(torch.transpose(cnn_in, 1, 2))
        cnn_out = torch.transpose(cnn_out, 1, 2)
        return cnn_out

    def lstm(self, word_rep, word_mask):
        lstm_in = self.dropout5(word_rep)
        lstm_out, w_h = self.lstm_fea(lstm_in)
        lstm_senvec = torch.cat((w_h[0][0], w_h[0][1]), 1)  # bs * 2*hd_dim

        lstm_out = lstm_out.mul(word_mask.unsqueeze(2).expand(lstm_out.size()).float())
        return lstm_out, lstm_senvec
    
    def sort_lstm(self, word_rep, seq_len):
        '''sort instances to decrease only increase 0.1%'''
        x_sort_idx = np.argsort(-np.array(seq_len.cpu()))
        x_unsort_idx = torch.LongTensor(np.argsort(x_sort_idx))
        seq_sort_len = seq_len[x_sort_idx]
        w_sort_seq = word_rep[torch.LongTensor(x_sort_idx)]
        
        lstm_in = self.dropout2(w_sort_seq)
        lstm_in = rnn.pack_padded_sequence(lstm_in, seq_sort_len, batch_first=True)
        lstm_out, w_h = self.lstm_fea(lstm_in)
        lstm_out, _ = rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = lstm_out[x_unsort_idx]
        return lstm_out