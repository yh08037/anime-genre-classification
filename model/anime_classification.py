# ID: 2018115809 (undergraduate)
# NAME: Dohun Kim   
# File name: anime_classification.py
# Platform: Python 3.9.5 on Ubuntu Linux 18.04
# Required Package(s): numpy=1.20.3, pandas=1.2.4 matplotlib=3.4.2

'''
anime_classification.py :
    Term Project: Predict Genres of Anime from Synopsis
'''

############################## import required packages ##############################

import os
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


################################## define functions ##################################

def cross_entropy_error(y, t):
    assert t.size == y.size
    return -np.mean(np.sum(t*np.log(y+1e-7), axis=1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x


#################################### define layers ###################################

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
        
    def forward(self, x):
        W, b = self.params
        out = np.matmul(x, W) + b
        self.x = x
        return out
    
    def backward(self, dout):
        W, b = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        db = np.sum(dout, axis=0)
        
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None
        
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = self.out * (1.0 - self.out) * dout
        return dx


class CrossEntropyLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # sigmoid의 출력
        self.t = None  # 정답 데이터

        self.pt = None
        self.loss = None

    def forward(self, y, t):
        self.t = t
        self.y = y

        self.pt = np.where(self.t==1, y, 1-y)
        self.out = -self.t * np.log(self.pt + 1e-7)

        return np.mean(np.sum(self.out, axis=1))

    def backward(self, dout=1):
        dx = -(self.t / self.pt) * dout
        return dx


class BinaryCrossEntropyLoss:
    pass


class FocalLoss:
    pass


class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # sigmoid의 출력
        self.t = None  # 정답 데이터

    def forward(self, x, t):
        self.t = t
        self.y = sigmoid(x)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx


class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flag = True

    def forward(self, x):
        if self.train_flag:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


#################################### define model ####################################

class BaseModel:
    def __init__(self):
        self.params, self.grads = None, None

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    def save_params(self, file_name=None):
        if not file_name:
            file_name = self.__class__.__name__ + '.pkl'

        params = [p.astype(np.float16) for p in self.params]
        # if GPU:
        #     params = [to_cpu(p) for p in params]

        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name=None):
        if not file_name:
            file_name = self.__class__.__name__ + '.pkl'

        if '/' in file_name:
            file_name = file_name.replace('/', os.sep)

        if not os.path.exists(file_name):
            raise IOError('No file: ' + file_name)

        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        params = [p.astype('f') for p in params]
        # if GPU:
        #     params = [to_gpu(p) for p in params]

        for i, param in enumerate(self.params):
            param[...] = params[i]


class MultiLabelClassifier(BaseModel):
    def __init__(self, input_size, output_size):
        
        # initialize weights
        W1 = np.random.randn(input_size, 300)
        b1 = np.zeros(300)

        W2 = np.random.randn(300, 300)
        b2 = np.zeros(300)

        W3 = np.random.randn(300, output_size)
        b3 = np.zeros(output_size)

        # create layers
        self.layers = [
            Affine(W1, b1), Dropout(), Sigmoid(),
            Affine(W2, b2), Dropout(), Sigmoid(),
            Affine(W3, b3), Sigmoid()
        ]
        self.dropout_layer = [self.layers[1], self.layers[4]]
        self.loss_layer = CrossEntropyLoss()

        # collect params and grads
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
        
    def set_train_flag(self, flag):
        for layer in self.dropout_layer:
            layer.train_flag = flag

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return sigmoid(xs)

    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout


################################## define optimizer ##################################

class Adam:
    '''
    Adam(Adaptive Moment Estimation, http://arxiv.org/abs/1412.6980v8)
        m <- m + (1 - beta1)*(dL/dW - m)
        v <- v + (1 - beta2)*[(dL/dW)*(dL/dW) - v]
        W <- W - lr * [sqrt(1-beta2^iter)/(1-beta1^iter)] * (m / sqrt(v + eps))
    '''
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for i in  range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])

            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)


################################### define trainer ###################################

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

        self.train_loss_list = []
        self.val_loss_list = []

    def fit(self, x, t, x_val, t_val, max_epoch=10,
            batch_size=32, max_grad=None):
        data_size = len(x)
        max_iters = data_size // batch_size
        model, optimizer = self.model, self.optimizer
        
        start_time = time.time()
        for epoch in range(max_epoch):
            # 뒤섞기
            idx = np.random.permutation(np.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]

                # 기울기 구해 매개변수 갱신
                model.forward(batch_x, batch_t)
                model.backward()
                # params, grads = remove_duplicate(model.params, model.grads)  # 공유된 가중치를 하나로 모음
                # if max_grad is not None:
                #     clip_grads(grads, max_grad)
                optimizer.update(model.params, model.grads)

            # 평가
            model.set_train_flag(False)
            train_loss = model.forward(x, t)
            val_loss = model.forward(x_val, t_val)
            model.set_train_flag(True)
            
            self.train_loss_list.append(float(train_loss))
            self.val_loss_list.append(float(val_loss))
            
            elapsed_time = time.time() - start_time            

            print('| epoch %3d | time %3d[s] | train loss %4.2f | val loss %4.2f'
                    % (epoch + 1, elapsed_time, train_loss, val_loss))

    def plot(self, ylim=None):
        x = np.arange(len(self.train_loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.train_loss_list, label='train')
        plt.plot(x, self.val_loss_list, label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()



if __name__ == '__main__':
    pass