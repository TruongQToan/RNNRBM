import numpy as np
import matplotlib.pyplot as plt
from rbm import gibbs_sample, gradients
from utils import sigmoid
import pickle
import time


class RNNRBM(object):

    def __init__(self, learning_rate, visible_dim, hidden_dim, rnn_hidden_dim, num_epochs):
        self.w = np.random.normal(loc=0, scale=0.01, size=(visible_dim, hidden_dim))
        self.wuu = np.random.normal(loc=0, scale=0.01, size=(rnn_hidden_dim, rnn_hidden_dim))
        self.wuv = np.random.normal(loc=0, scale=0.01, size=(rnn_hidden_dim, visible_dim))
        self.wuh = np.random.normal(loc=0, scale=0.01, size=(rnn_hidden_dim, hidden_dim))
        self.wvu = np.random.normal(loc=0, scale=0.01, size=(visible_dim, rnn_hidden_dim))
        self.bv = np.zeros((1, visible_dim))
        self.bh = np.zeros((1, hidden_dim))
        self.bu = np.zeros((1, rnn_hidden_dim))
        self.u0 = np.zeros((1, rnn_hidden_dim))
        self.lr = learning_rate
        self.num_epochs = num_epochs
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim


    def _forward_pass(self, visible, caches):
        ## forward pass for each sample (multiple steps)
        ## return list caches, which each element is 
        ## the result from one timestep
        def get_bias(u_tm1):
            bh_t = self.bh + np.matmul(u_tm1, self.wuh)
            bv_t = self.bv + np.matmul(u_tm1, self.wuv)
            return bh_t, bv_t
        
        def get_rnn_hidden(v_t, u_tm1):
            activation = np.matmul(v_t, self.wvu) + np.matmul(u_tm1, self.wuu) + self.bu
            return sigmoid(activation)

        time_steps = visible.shape[0]
        u_tm1 = self.u0
        for t in range(1, time_steps):
            v_t = visible[t]
            v_tm1 = visible[t - 1]
            bh_t, bv_t = get_bias(u_tm1)
            ## gibbs sampling start from v_tm1
            negative_sample, h_probs0, h_probs1 = gibbs_sample(v_tm1, self.w, bh_t, bv_t)
            caches.append((v_t, negative_sample, h_probs0, h_probs1, bh_t, bv_t))
            ut = get_rnn_hidden(v_t, u_tm1)
            u_tm1 = ut
        return caches


    def _backward_pass(self, caches, time_steps):
        ## backward pass for each sample
        grads = []
        for t in reversed(list(range(time_steps))):
            v_t, negative_sample, h_probs0, h_probs1, bh_t, bv_t = caches[t]
            dw_t, dbh_t, dbv_t = gradients(v_t, negative_sample, h_probs0, h_probs1)
            grads.append((dw_t, dbh_t, dbv_t))
        return grads
    

    def _update_params(self, *args):
        dw, dbh, dbv, dwuh, dwuv, du0, dbu, dwuu, dwvu = args
        self.w -= self.lr * dw
        self.bh -= self.lr * dbh
        self.bv -= self.lr * dbv
        self.wuh -= self.lr * dwuh
        self.wuv -= self.lr * dwuv
        self.u0 -= self.lr * du0
        self.wuh -= self.lr * dwuh
        self.bu -= self.lr * dbu
        self.wuu -= self.lr * dwuu
        self.wvu -= self.lr * dwvu
    

    def _train_step(self, X):
        ## perform train on each epoch
        num_samples = X.shape[0]
        preds = []
        for i, sample in enumerate(X):
            ## sample have form: time_steps x features
            time_steps = sample.shape[0]
            sample = np.reshape(sample, (time_steps, self.visible_dim))
            caches = []
            ## zeros vector to perform gibbs sampling for v0
            sample = np.vstack((np.zeros((1, self.visible_dim)), sample))
            caches = self._forward_pass(sample, caches)
            preds.append(caches[-1][1][0][0])
            grads = self._backward_pass(caches, time_steps)
            ## compute gradients of rbm parameters
            ## the character s in the name just means plural
            dw = np.sum(np.reshape(np.array([grad[0] for grad in grads]), \
                [time_steps, self.visible_dim, self.hidden_dim]), \
                axis=0)
            u_ts = [cache[4] for cache in caches]
            u_ts.insert(0, self.u0)
            u_ts = np.reshape(np.array(u_ts[:-1]), [time_steps, self.rnn_hidden_dim])
            dbh_ts = np.reshape(np.array([grad[1] for grad in grads]), [time_steps, self.hidden_dim])
            dwuh = np.matmul(u_ts.T, dbh_ts)
            dbv_ts = np.reshape(np.array([grad[2] for grad in grads]), [time_steps, self.visible_dim])
            dwuv = np.matmul(u_ts.T, dbv_ts)
            dbh = np.reshape(np.sum(dbh_ts, axis=0), [1, self.hidden_dim])
            dbv = np.reshape(np.sum(dbv_ts, axis=0), [1, self.visible_dim])

            ## compute gradients of rnn papameters
            du_ts = [np.zeros((1, self.rnn_hidden_dim))]
            for i in reversed(list(range(time_steps))):
                ## maybe this is where bugs are found later
                addend1 = np.matmul(u_ts[i], (1 - u_ts[i]).T) # (1xr) * (rx1) * (1xr) * (rxr)
                addend1 = np.reshape(addend1, (1,1))
                addend1 = np.matmul(addend1, np.matmul(du_ts[-1], self.wuu))
                addend2 = np.matmul(dbh_ts[i], self.wuh.T)
                addend3 = np.matmul(dbv_ts[i], self.wuv.T)
                du_t = addend1 + addend2 + addend3
                du_ts.append(du_t)
            du_ts = np.array(list(reversed(du_ts)))
            du_ts = np.reshape(du_ts, [time_steps + 1, self.rnn_hidden_dim])
            du0 = du_ts[0]
            h_hm1 = np.matmul(u_ts, (1 - u_ts).T) # h_t * (1 - h_t)
            dbu = np.sum(np.matmul(h_hm1, du_ts[1:]), axis=0)
            dbu = np.reshape(dbu, (1, self.rnn_hidden_dim))
            u_tm1s = np.vstack((self.u0, u_ts[:-1]))
            dwuu = np.matmul(np.matmul(u_tm1s.T, h_hm1), du_ts[1:])
            dwvu = np.matmul(np.matmul(sample[1:].T, h_hm1), du_ts[1:])
            self._update_params(dw, dbh, dbv, dwuh, dwuv, du0, dbu, dwuu, dwvu)
        return preds


    def _pickle(self):
        with open("model.pkl", "wb") as f:
            pickle.dump(self, f)


    def train(self, X, y):
        costs = []
        for epoch in range(self.num_epochs):
            np.random.shuffle(X)
            start = time.time()
            preds = self._train_step(X)
            end = time.time()
            mae = 0.0
            for i in range(len(preds)):
                mae += np.abs(preds[i] - y[i])
            mae /= len(preds)
            costs.append(mae)
            print ("Loss after epoch %d is %f, time %f" % (epoch, mae, end - start))
        self._pickle()
        return preds, y, mae