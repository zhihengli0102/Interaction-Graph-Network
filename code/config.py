import argparse
import os


class Config:
    hd_dim = 100
    kb_dim = 10

    kb_flag = False
    graph_flag = True

    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
    parser.add_argument('--bs', type=int, default=32, help='training batch size')
    parser.add_argument('--nEpochs', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--cuda_idx', type=int, default=1, help='cuda index')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--model_f', type=str, default='../models/model.pkl')
    parser.add_argument('--param_f', type=str, default='../models/paramf/0_params.pkl')
    parser.add_argument('--res_f', type=str, default='../results/test.txt')
    opt = parser.parse_args()

    phrase_emb_file = '../processed_file/phrase_100-avg.vec'
    adj_file = '../processed_file/adj_bcs.pkl'
    fea_file = '../processed_file/fea_file.pkl'
    util_file = '../processed_file/util_file.pkl'