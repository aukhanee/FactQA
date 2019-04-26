import itertools
import json
import pickle
import scipy.sparse as sparse


from common import settings_file


with open(settings_file()) as settings_f:
    settings = json.load(settings_f)


    def process_fact(line):
        [entity, rel, obj] = line.rstrip().split('\t')[0:3]
        return entity, rel, obj.split(' ')


    def list_of_ordered_keys():
        all_keys = set()
        with open(settings['selfbase']) as f_in:
            for l in f_in:
                entity, rel, objs = process_fact(l)
                all_keys.update([entity, rel])
                all_keys.update(objs)
        keys = list(all_keys)
        keys.sort()
        return keys

    ordered_keys = list_of_ordered_keys()
    with open(settings['subjects'], mode="w") as s_out:
        for k in ordered_keys:
            print(k, file=s_out, sep='\t')

    cols = {}
    cols.update(zip(ordered_keys, itertools.count()))

    line_ctr = itertools.count()
    data_tuples = list()
    with open(settings['selfbase']) as f_in:
        for l in f_in:
            entity, rel, objs = process_fact(l)
            l = next(line_ctr)
            data_tuples.extend([(1, l, cols[w]) for w in (entity, rel, objs[0])])

    data, row, col = zip(*data_tuples)

    mx = sparse.csr_matrix((data, (row, col)))

    with open(settings['f_y_matrix'], mode='wb') as fy_out_f:
        pickle.dump(mx, fy_out_f)
