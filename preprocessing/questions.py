import itertools
import json
import pickle
import scipy.sparse as sparse

from common import clean_words, settings_file, split_line


with open(settings_file()) as settings_f:
    settings = json.load(settings_f)

    r = {}
    with open(settings['vocabulary']) as v_in_f:
        r.update(zip(map(lambda l: l.rstrip(), v_in_f), itertools.count()))

    dataset = 'train'
    line_ctr = itertools.count()
    data_tuples = list()
    with open(settings[dataset]) as in_f:
        for line in in_f:
            l = next(line_ctr)
            data_tuples.extend([(1, l, r[w]) for w in clean_words(split_line(line))])

    data, row, col = zip(*data_tuples)

    mx = sparse.csr_matrix((data, (row, col)))

    with open(settings['g_q_matrix'], mode='wb') as gq_out_f:
        pickle.dump(mx, gq_out_f)
