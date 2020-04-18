import pickle
import torch
from config import Config
from torch.utils.data import DataLoader
from dataset import DatasetIns, pad
import output_file


def model_predict(input_data, model):
    model.eval()
    pred_tag_ner = []
    for batch in input_data:
        model.eval()
        loss, pred_ner = model(batch, 'test')

        seq_len = batch[2].tolist()
        for idx in range(len(seq_len)):
            pred_tag_ner.append(pred_ner[idx][:seq_len[idx]].cpu().data)

    return pred_tag_ner


with open(Config.util_file, 'rb') as handle:
    tag_dict, word_dict, word_emb, node_dict, node_emb, kb_dict = pickle.load(handle)
with open(Config.fea_file, 'rb') as handle:
    fea_dict = pickle.load(handle)
input_data = DataLoader(dataset=DatasetIns(fea_dict[0]), batch_size=Config.opt.bs, shuffle=False, collate_fn=pad)

model = torch.load(Config.opt.model_f)
model.load_state_dict(torch.load(Config.opt.param_f))
if torch.cuda.is_available():
    torch.load(Config.opt.model_f, map_location=lambda storage, loc: storage.cuda(Config.opt.cuda_idx))

pred_tag_ner = model_predict(input_data, model)
output_file.out_BIO(pred_tag_ner, tag_dict, fea_dict[1], Config.opt.res_f)
print('output prediction -> ../results/' + Config.opt.res_f)