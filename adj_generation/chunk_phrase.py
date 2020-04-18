import nltk
from nltk.tag import pos_tag


NP_label = ['NP', 'NNP', 'NNs', 'NJ', 'ICN', 'NV']
grammar = r"""
            NNs: {(<NN>|<NNS>)+}
            NJ: {<DT>(<NN>|<NNs>)<JJ>}
            NP: {<DT>?<JJ>*<NNs>}
            NNP: {<NNP>+}
            ICN: {<IN><CD><NP>}
            NV: {<NP><VBG>}
            """
cp = nltk.RegexpParser(grammar)


def get_chunk(tokens):
    phrase = []
    tokens_pos = pos_tag(tokens)
    result = cp.parse(tokens_pos)
    for res in result:
        if type(res) == nltk.Tree and res.label() in NP_label:
            ph_list = [a[0] for a in res.leaves()]
            if len(ph_list) > 1:
                phrase.append(' '.join(ph_list))
    return phrase


def run_chunk(tokens):
    phrase_list = get_chunk(tokens)
    return phrase_list