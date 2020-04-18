import os
from conlleval import evaluate_conll_file
from sklearn.metrics import precision_recall_fscore_support as prf_support


epoch_ner = 0
P_ner = R_ner = F_ner = 0

def eval_bio(epoch, wf_flag):
    global epoch_ner, P_ner, R_ner, F_ner

    fileIterator = open(wf_flag)
    [p, r, f] = evaluate_conll_file(fileIterator)

    if F_ner <= f:
        P_ner = p
        R_ner = r
        F_ner = f
        epoch_ner = epoch
        os.rename(wf_flag, wf_flag.replace('.txt', '_best.txt'))
    print('epoch_ner_h = %d  P_h = %.4f, R_h = %.4f, F_h= %.4f' % (epoch_ner, P_ner, R_ner, F_ner))
    # return f


epoch_det = 0
P_det = R_det = F_det = 0
def eval_det(epoch, pred_det, real_det):
    global epoch_det, P_det, R_det, F_det

    tag_pred = []
    tag_real = []
    for i in range(len(pred_det)):
        tag_real.append(int(real_det[i]))
        tag_pred.append(list(pred_det[i]).index(max(list(pred_det[i]))))
    p, r, f, NUM = prf_support(tag_real, tag_pred, beta=1.0, labels=[1],
                                           average=None, warn_for=('precision', 'recall', 'f-score'),
                                           sample_weight=None)
    print('\nepoch_det = %d  P_h = %.4f, R_h = %.4f, F_h= %.4f' % (epoch, p, r, f))
    if F_det <= f:
        P_det = p
        R_det = r
        F_det = f
        epoch_det = epoch
    print('epoch_det_h = %d  P_h = %.4f, R_h = %.4f, F_h= %.4f' % (epoch_det, P_det, R_det, F_det))
    return f