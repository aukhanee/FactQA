import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from numba import vectorize
import json
import itertools

from common import settings_file


with open(settings_file()) as settings_f:
    settings = json.load(settings_f)

# Learned hyperparameters (choice has been made by training)
LAMDA = 0.1 # loss hyperparamter
D = 128 # dimension of the embeddings
# Load Wv(bag of vocabulary) and Ws(bag of symbols)
Nv = 170 # size of the vocabulary
Ns = 150 # number of entities and relationships (see subjects.txt)
# Load learned weights
Wv = np.load('WvWs/Wv16.npy')
Ws = np.load('WvWs/Ws16.npy')

with open('f_y_matrixfact.pkl', 'rb') as pfile:
    f_y_matrix = pickle.load(pfile)

with open('g_q_matrix.pkl', 'rb') as pfile:
    g_q_matrix = pickle.load(pfile)

def padarray(A, size):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')

def S_qy(Wv, g_q, Ws, f_y):
    # S(q,y) = cos(Wv*g(q), Ws*f(y)), Wv and Ws are to be learned by SGD
    g_q_vec = np.transpose(g_q.toarray())
    f_y_vec = np.transpose(f_y.toarray())
    Wv_g_q = np.transpose(Wv.dot(g_q_vec))
    Ws_f_y = np.transpose(Ws.dot(f_y_vec))

    return cosine_similarity(Wv_g_q, Ws_f_y)[0]

def process_fact(line):
    [entity, rel, obj] = line.rstrip().split('\t')[0:3]
    entity = ' '.join(entity.lower().split('_'))
    rel = ' '.join(rel.lower().split('_'))  
    obj = ' '.join(obj.lower().split('_'))
    return entity, rel, obj

@vectorize(['float64(float64, float64)'], target='parallel')
def Add(a, b):
  return a + b

@vectorize(['float64(float64, int64)'], target='parallel')
def Multiply(a, b):
  return a * b

stop_words = ["what", "when", "where", "how", "who", "is", "are", "the"]

def alias_subjects(subjects_file):
    with open(subjects_file) as f:
        data = f.read().splitlines()
    parsed_subjects = []
    for s in data:
        s = ' '.join(s.lower().split('_'))
        parsed_subjects.append(s)
    with open("aliased_subjects.txt", "w") as f:
        for s in parsed_subjects:
            f.write(s + "\n")

def generate_ngrams(words_list, n):
    ngrams_list = []
 
    for num in range(0, len(words_list)):
        ngram = ' '.join(words_list[num:num + n])
        if len(ngram.split(' ')) == n:
            ngrams_list.append(ngram)
 
    return ngrams_list

def keep_aliased(aliased_subjects, q_words):
    bigrams = generate_ngrams(q_words, 2)
    q_words.extend(bigrams)
    aliased_retained_grams = [nword for nword in q_words if nword in aliased_subjects]
    return aliased_retained_grams

def cand_gen(q_words, aliased_subjects):
    grams = keep_aliased(aliased_subjects, q_words)

    cols = {}
    cols.update(zip(aliased_subjects, itertools.count()))

    line_ctr = itertools.count()
    data_tuples = list()
    idx = 0
    cand_objs = []
    with open("selfbaseSingleobjectonly.txt") as f_in:
        for l in f_in:
            entity, rel, objs = process_fact(l)
            # l = next(line_ctr)
            proc_entity = [entity]
            if entity in grams:
                proc_entity.extend([rel])
                entity_grams = [word for word in proc_entity if word in grams]
                if len(entity_grams) > 1:
                    data_tuples.extend([(1, idx, cols[w]) for w in (entity, rel, objs)])
                    cand_objs.append(objs)
                    idx += 1
    data, row, col = zip(*data_tuples)

    f_y = sparse.csr_matrix((data, (row, col)), shape=(idx, Ns))
    return f_y, cand_objs

def scoring():
    return

def response():
    return

if __name__=='__main__':
    q1 = "What are the geographic coordinates of Pakistan?"
    q = "How many judges serve in Supreme Court of Pakistan?"
    r = {}
    with open(settings['vocabulary']) as v_in_f:
        r.update(zip(map(lambda l: l.rstrip(), v_in_f), itertools.count()))
    s = ' '.join([word for word in q.strip('?.').lower().split() if word not in stop_words])
    q_words = s.split(' ')
    data_tuples = list()
    data_tuples.extend([(1, 0, r[w]) for w in q_words])
    data, row, col = zip(*data_tuples)
    g_q = sparse.csr_matrix((data, (row, col)), shape=(1, Nv))

    with open("aliased_subjects.txt") as ps:
        aliased_subjects = ps.read().splitlines()
    f_y_cands, answer_cands = cand_gen(q_words, aliased_subjects)
    min_score = 1
    for f_y, answer in zip(f_y_cands, answer_cands):
        score_fy = S_qy(Wv, g_q, Ws, f_y)
        print("\"{}\" score is {}".format(answer, score_fy))
        if score_fy < min_score:
            min_score = score_fy
            print(answer)

print("testing done.")
print("OK")