import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from numba import vectorize


# Learning hyperparameters (choice is random)
LAMDA = 0.1 # loss hyperparamter
D = 128 # dimension of the embeddings
ETA = 0.1 # learning rate (My own choice)
NITER = 20 # number of iterations over all training data
EPSILON = 0.10 # termination criteria
# Initialize Wv(bag of vocabulary) and Ws(bag of symbols)
Nv = 170 # size of the vocabulary
Ns = 150 # number of entities and relationships (see subjects.txt)
# Initialize weights
# Wv = np.zeros((D, Nv))
# Wv = 0.01* np.random.randn(D,Nv)
# # Ws = np.zeros((D, Ns))
# Ws = 0.01* np.random.randn(D,Ns)


with open('f_y_matrixfact.pkl', 'rb') as pfile:
    f_y_matrix = pickle.load(pfile)

with open('g_q_matrix.pkl', 'rb') as pfile:
    g_q_matrix = pickle.load(pfile)

def S_qy(Wv, g_q, Ws, f_y):
    # S(q,y) = cos(Wv*g(q), Ws*f(y)), Wv and Ws are to be learned by SGD
    g_q_vec = np.transpose(g_q.toarray())
    f_y_vec = np.transpose(f_y.toarray())
    Wv_g_q = np.transpose(Wv.dot(g_q_vec))
    Ws_f_y = np.transpose(Ws.dot(f_y_vec))

    return cosine_similarity(Wv_g_q, Ws_f_y)[0]

# l(q,y,y') = max(0, [lamda - S(q,y) + S(q,y')])
def loss(Wv, g_q, Ws, f_y, f_y_prime):
    return max(0, LAMDA - S_qy(Wv, g_q, Ws, f_y) + S_qy(Wv, g_q, Ws, f_y_prime))

@vectorize(['float64(float64, float64)'], target='parallel')
def Add(a, b):
  return a + b

@vectorize(['float64(float64, int64)'], target='parallel')
def Multiply(a, b):
  return a * b


def d_Wv_cossim(Wv_g_q_norm, Ws_f_y_norm, S_qy, ws, wv, f, g, j):
    ws_fy_i = 0
    wv_g_q_i = 0
    # for i in range(Ns):
    #     ws_fy_i += ws[i] * f[i]
    ws_fy_i = cuda_innsum(ws, f)
    # for i in range(Nv):
    #     wv_g_q_i += wv[i] *g[i]
    wv_g_q_i = cuda_innsum(wv, g)
    partial_derv = (ws_fy_i/(Wv_g_q_norm*Ws_f_y_norm) - S_qy*(wv_g_q_i)/(Wv_g_q_norm*Wv_g_q_norm))*g[j]
    return partial_derv

def d_Wv_cossim_innsum(ws, f):
    p = Multiply(ws, f)
    return np.sum(p)

def d_Ws_cossim_innsum(wv, g):
    p = Multiply(wv, g)
    return np.sum(p)

def cuda_innsum(w, x):
    p = Multiply(w, x)
    return np.sum(p)

def d_Ws_cossim(Ws_f_y_norm, Wv_g_q_norm, S_qy, wv, ws, g, f, j):
    ws_fy_i = 0
    wv_g_q_i = 0
    # for i in range(Nv):
    #     wv_g_q_i += wv[i] *g[i]
    wv_g_q_i = cuda_innsum(wv, g)
    # for i in range(Ns):
    #     ws_fy_i += ws[i] * f[i]
    ws_fy_i = cuda_innsum(ws, f)
    partial_derv = (wv_g_q_i/(Wv_g_q_norm*Ws_f_y_norm) - S_qy*(ws_fy_i)/(Ws_f_y_norm*Ws_f_y_norm))*f[j]
    return partial_derv

def update_Wv_cuda(Wv, g_q, f_y, f_y_prime, Ws):
    f = f_y.toarray()[0]
    f_prime = f_y_prime.toarray()[0]
    g = g_q.toarray()[0]
    for i in range(D):
        Wv[i] = Add(Wv[i], ETA * Add(Multiply(d_Wv_cossim_innsum(Ws[i], f), g), -1*Multiply(d_Wv_cossim_innsum(Ws[i], f_prime), g)))
    return Wv

def update_Wv(Wv, g_q, f_y, f_y_prime, Ws):
    print("sum Wv before update:", sum(sum(Wv)))
    f = f_y.toarray()[0]
    f_prime = f_y_prime.toarray()[0]
    g = g_q.toarray()[0]
    g_q_vec = np.transpose(g_q.toarray())
    f_y_vec = np.transpose(f_y.toarray())
    f_y_prime_vec = np.transpose(f_y_prime.toarray())
    Wv_g_q = np.transpose(Wv.dot(g_q_vec))
    Ws_f_y = np.transpose(Ws.dot(f_y_vec))
    Ws_f_y_prime = np.transpose(Ws.dot(f_y_prime_vec))
    Wv_g_q_norm = np.linalg.norm(Wv_g_q)
    Ws_f_y_norm = np.linalg.norm(Ws_f_y)
    Ws_f_y_prime_norm = np.linalg.norm(Ws_f_y_prime)
    Sqy = S_qy(Wv, g_q, Ws, f_y)[0]
    Sqy_prime = S_qy(Wv, g_q, Ws, f_y_prime)[0]
    for i in range(D):
        for j in range(Nv):
            delta_w = -1 * ETA*(-1* d_Wv_cossim(Wv_g_q_norm, Ws_f_y_norm, Sqy, Ws[i], Wv[i], f, g, j)+ d_Wv_cossim(Wv_g_q_norm, Ws_f_y_prime_norm, Sqy_prime, Ws[i], Wv[i], f_prime, g, j))
            Wv[i][j] = Wv[i][j] + delta_w
    print("sum Wv after update:", sum(sum(Wv)))
    return Wv


def update_Ws_cuda(Ws, g_q, f_y, f_y_prime, Wv):
    f = f_y.toarray()[0]
    f_prime = f_y_prime.toarray()[0]
    g = g_q.toarray()[0]
    for i in range(D):
        Ws[i] = Add(Ws[i], ETA * Add(Multiply(d_Ws_cossim_innsum(Wv[i], g), f), -1*Multiply(d_Ws_cossim_innsum(Wv[i], g), f_prime)))
    return Ws

def update_Ws(Ws, g_q, f_y, f_y_prime, Wv):
    print("sum Ws before update:", sum(sum(Ws)))
    f = f_y.toarray()[0]
    f_prime = f_y_prime.toarray()[0]
    g = g_q.toarray()[0]
    g_q_vec = np.transpose(g_q.toarray())
    f_y_vec = np.transpose(f_y.toarray())
    f_y_prime_vec = np.transpose(f_y_prime.toarray())
    Wv_g_q = np.transpose(Wv.dot(g_q_vec))
    Ws_f_y = np.transpose(Ws.dot(f_y_vec))
    Ws_f_y_prime = np.transpose(Ws.dot(f_y_prime_vec))
    Wv_g_q_norm = np.linalg.norm(Wv_g_q)
    Ws_f_y_norm = np.linalg.norm(Ws_f_y)
    Ws_f_y_prime_norm = np.linalg.norm(Ws_f_y_prime)
    Sqy = S_qy(Wv, g_q, Ws, f_y)[0]
    Sqy_prime = S_qy(Wv, g_q, Ws, f_y_prime)[0]
    for i in range(D):
        for j in range(Ns):
            delta_w = -1 * ETA*(-1* d_Ws_cossim(Ws_f_y_norm, Wv_g_q_norm, Sqy, Wv[i], Ws[i], g, f, j)+ d_Ws_cossim(Ws_f_y_prime_norm, Wv_g_q_norm, Sqy_prime, Wv[i], Ws[i], g, f_prime, j))
            Ws[i][j] = Ws[i][j] + delta_w
    print("sum Ws after update:", sum(sum(Ws)))
    return Ws

def alternate_sgd(Wv, Ws, g_q_matrix, f_y_matrix):
    train_loss = 0
    for idx, [g_q, f_y] in enumerate(zip(g_q_matrix, f_y_matrix)):
        for idy, f_y_prime in enumerate(f_y_matrix):
            if idx != idy:
                train_loss += loss(Wv, g_q, Ws, f_y, f_y_prime)
                sumWv = sum(sum(Wv))
                # Wv = update_Wv_cuda(Wv, g_q, f_y, f_y_prime, Ws)
                if loss(Wv, g_q, Ws, f_y, f_y_prime) > 0:
                    Wv = update_Wv(Wv, g_q, f_y, f_y_prime, Ws)
                sumWs = sum(sum(Ws))
                if np.linalg.norm(Wv) > 1:
                    Wv = Wv/np.linalg.norm(Wv, axis=0)
                # Ws = update_Ws_cuda(Ws, g_q, f_y, f_y_prime, Wv)
                if loss(Wv, g_q, Ws, f_y, f_y_prime) > 0:
                    Ws = update_Ws(Ws, g_q, f_y, f_y_prime, Wv)
                if np.linalg.norm(Ws) > 1:
                    Ws = Ws/np.linalg.norm(Ws, axis=0)
        with open("TrainingQuestionLoss.txt", "a") as loss_file:
            print("training loss after question {}".format(idx), train_loss, file=loss_file)
        print("training loss after question {}".format(idx), train_loss)
    return train_loss, Wv, Ws


if __name__=='__main__':
    # Initialize weights
    # Wv = np.zeros((D, Nv))
    Wv = np.random.randn(D,Nv)
    if np.linalg.norm(Wv) > 1:
        Wv = Wv/np.linalg.norm(Wv, axis=0)
    # Ws = np.zeros((D, Ns))
    Ws = np.random.randn(D,Ns)
    if np.linalg.norm(Ws) > 1:
        Ws = Ws/np.linalg.norm(Ws, axis=0)
    for iteration in range(NITER):
        iteration_loss, Wv, Ws = alternate_sgd(Wv, Ws, g_q_matrix, f_y_matrix)
        np.save('Wv' + str(iteration) + '.npy', Wv)
        np.save('Ws' + str(iteration) + '.npy', Ws)
        with open("TrainingIterationLoss.txt", "a") as loss_file:
            print("iteration loss after iteration {}:".format(iteration), iteration_loss, file=loss_file)
        print("iteration loss after iteration {}:".format(iteration), iteration_loss)

print("f_y_matrix shape", f_y_matrix.shape)
print("g_q_matrix shape", g_q_matrix.shape)
print("f_y[4]", f_y_matrix[4])
print("g_q[4]", g_q_matrix[4])

print("train_loss")
print("OK")