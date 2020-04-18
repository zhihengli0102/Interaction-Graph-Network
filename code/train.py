import pickle
import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import random

from config import Config

from dataset import DatasetIns, pad
from model import Model

torch.manual_seed(Config.opt.seed)
np.random.seed(Config.opt.seed)
random.seed(Config.opt.seed)

def model_train(epoch, traindata, model, model_optim):
    loss_ner_epoch = 0  # Reset every print_every

    for batch in traindata:
        model.train()
        model.zero_grad()
        model_optim.zero_grad()

        loss_ner = model(batch, 'train')
        loss_ner.backward()
        model_optim.step()

        loss_ner_epoch += loss_ner.cpu().data / len(batch[0])
    print("epoch = %d -- ner loss = %.4f\n" % (epoch, loss_ner_epoch/len(traindata)))


'''data loading'''
print("data loading ...")
with open(Config.fea_file, 'rb') as handle:
    fea_dict = pickle.load(handle)
with open(Config.util_file, 'rb') as handle:
    tag_dict, word_dict, word_emb, node_dict, node_emb, kb_dict = pickle.load(handle)

traindata = DataLoader(dataset=DatasetIns(fea_dict[0]), batch_size=Config.opt.bs, shuffle=True, collate_fn=pad)

print("model setting ...")
if Config.graph_flag:
    model = Model(torch.tensor(word_emb), tag_dict, len(kb_dict), Config.opt.cuda_idx, Config.opt.bs, torch.tensor(node_emb))
else:
    model = Model(torch.tensor(word_emb), tag_dict, len(kb_dict), Config.opt.cuda_idx, Config.opt.bs)
model_optim = optim.Adam(model.parameters(), lr=Config.opt.lr)
criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.opt.seed)
    model = model.cuda(Config.opt.cuda_idx)
    criterion = criterion.cuda(Config.opt.cuda_idx)

torch.save(model, Config.opt.model_f)

print("train and eval the model ...")
for epoch in range(Config.opt.nEpochs):
    print("-----%d-----" % epoch)
    '''train the model'''
    model_train(epoch, traindata, model, model_optim)

    paramf = '../models/paramf/'
    if not os.path.exists(paramf):
        os.makedirs(paramf)
    torch.save(model.state_dict(), paramf + str(epoch) + '_params.pkl')

print("Done...")




