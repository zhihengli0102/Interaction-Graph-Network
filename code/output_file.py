
def out_BIO(pred, tag_dict, inses, wf_flag):
    assert len(inses) == len(pred), print("len(inses) != len(pred)")

    idx_tag = {value: key for key, value in tag_dict.items()}
    output_test = []
    output = []
    for i in range(len(inses)):
        pred_ins = pred[i].tolist()
        ins = inses[i]
        tweet_id = ins.split('\n')[0]
        output.append(ins.split('\n')[0] + '\n')
        output_test.append(ins.split('\n')[0] + '\n')
        
        tokens = ins.split('\n')[1:]
        for j in range(len(tokens)):
            token, ori_word, start, end = tokens[j].split('\t')[:4]
            real_tag = tokens[j].split('\t')[-1]
            pred_tag = idx_tag[pred_ins[j]]
            
            output_test.append('\t'.join([token, ori_word, start, end, pred_tag]) + '\n')
            
            if real_tag != 'O':
                real_tag = real_tag + '-ADR'
            if pred_tag != 'O':
                pred_tag = pred_tag + '-ADR'
            output.append(token + '\t' + real_tag + '\t' + pred_tag + '\n')
        output.append('\n')
        output_test.append('\n')


        wf = open(wf_flag, 'w')
        for line in output_test:
            wf.write(line)
        wf.close()