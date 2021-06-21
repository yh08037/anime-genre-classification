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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 정답 데이터가 원핫 벡터일 경우 정답 레이블 인덱스로 변환
    if t.size == y.size:
        t = t.argmax(axis=1)
    
    batch_size = y.shape[0]

    cross_entropy = np.log(y[np.arange(batch_size), t] + 1e-7)
    loss = -np.sum(cross_entropy) / batch_size
    
    return loss


#################################### define layers ###################################

class Affine:
    def __init__(self, input_size, output_size, weight_init=None):
        self.name = f'Affine(input_size={input_size}, ' \
                    + f'output_size={output_size}, ' \
                    + f'weight_init={weight_init})'

        if weight_init == 'xavier':
            scale = np.sqrt(1.0 / input_size)
        elif weight_init == 'he':
            scale = np.sqrt(2.0 / input_size)
        else:
            scale = 1

        _W = scale * np.random.randn(input_size, output_size)
        _b = np.zeros(output_size)

        self.params = [_W, _b]
        self.grads = [np.zeros_like(_W), np.zeros_like(_b)]
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


class Relu:
    def __init__(self):
        self.name = 'Relu()'

        self.params, self.grads = [], []
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self):
        self.name = 'Sigmoid()'

        self.params, self.grads = [], []
        self.out = None
        
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = self.out * (1.0 - self.out) * dout
        return dx


class SigmoidWithLoss:
    def __init__(self):
        self.name = 'SigmoidWithLoss()'

        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoid의 출력
        self.t = None  # 정답 데이터

    def forward(self, x, t):
        self.t = t
        self.y = sigmoid(x)

        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx


class CrossEntropyLoss:
    def __init__(self):
        self.name = 'CrossEntropyLoss()'

        self.params, self.grads = [], []
        self.y = None  # output of sigmoid
        self.t = None  # true value
        self.pt = None # p_t from Focal Loss paper

    def forward(self, x, t):
        self.t = t
        self.y = sigmoid(x)

        self.pt = np.where(self.t==1, self.y, 1-self.y)
        log = np.log(self.pt + 1e-7)

        loss = np.mean(np.sum(-log, axis=1))
        return loss

    def backward(self, dout=1):
        sign = np.where(self.t==1, 1, -1)
        dx = sign*(self.pt-1)*dout
        return dx


class FocalLoss:
    '''
    https://arxiv.org/abs/1708.02002
    '''
    def __init__(self, gamma=2):
        self.name = f'FocalLoss(gamma={gamma})'

        self.params, self.grads = [], []
        self.y = None  # output of sigmoid
        self.t = None  # true value
        
        self.gamma = gamma # focusing parameter
        self.pt  = None # p_t from Focal Loss paper
        self.log = None # log(p_t)
        self.mod = None # focal modulating factor

    def forward(self, x, t):
        self.t = t
        self.y = sigmoid(x)

        self.pt  = np.where(self.t==1, self.y, 1-self.y)
        self.log = np.log(self.pt + 1e-7)
        self.mod = (1-self.pt) ** self.gamma

        loss = np.mean(np.sum(-self.mod*self.log, axis=1))
        return loss

    def backward(self, dout=1):
        sign = np.where(self.t==1, 1, -1)
        dx = sign*self.mod*(self.gamma*self.pt*self.log+self.pt-1)*dout
        return dx



class MSELoss:
    '''
    Mean Squared Error Loss
    '''
    def __init__(self):
        self.name = 'MSELoss()'

        self.params, self.grads = [], []
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        self.t = t
        self.y = x

        return np.mean(np.square(x - t))

    def backward(self, dout=1):
        dx = 2 * (self.y - self.t) / self.y.size
        return dx


class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.name = f'Dropout(dropout_ratio={dropout_ratio})'

        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flag = True

    def forward(self, x):
        if self.train_flag:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            # return x * (1.0 - self.dropout_ratio)
            return x

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, size_layer, momentum=0.9):
        self.name = f'BatchNormalization(size_layer={size_layer})'

        _gamma = np.ones(size_layer)
        _beta = np.zeros(size_layer)

        self.params = [_gamma, _beta]
        self.grads = [np.zeros_like(_gamma), np.zeros_like(_beta)]
        
        self.momentum = momentum
        self.input_shape = None # 합성곱 계층은 4차원, 완전연결 계층은 2차원  

        # 시험할 때 사용할 평균과 분산
        self.train_flag = True
        self.running_mean = None
        self.running_var = None  
        
        # backward 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.std = None

    def forward(self, x):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x):
        gamma, beta = self.params

        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if self.train_flag:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = gamma * xn + beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        gamma, _ = self.params

        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.grads[0][...] = dgamma
        self.grads[1][...] = dbeta
        
        return dx


#################################### define model ####################################

class BaseModel:
    def __init__(self):
        self.params, self.grads = None, None

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError
    
    def summary(self, *args):
        raise NotImplementedError

    def save_params(self, file_name=None):
        if not file_name:
            file_name = self.__class__.__name__ + '.pkl'

        params = [p.astype(np.float16) for p in self.params]
        
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

        for i, param in enumerate(self.params):
            param[...] = params[i]


class MultiLabelClassifier(BaseModel):
    def __init__(self, input_size, hidden_size_list, output_size,
                 use_dropout=False, dropout_ratio=0.5, use_batchnorm=False,
                 use_focal_loss=False, focal_gamma=2):
        
        # create layers
        size_list = [input_size] + hidden_size_list
        self.turn_off_list = []
        self.layers = []
        for i in range(len(size_list)-1):
            self.layers += [Affine(size_list[i], size_list[i+1], 'he')]
            if use_batchnorm:
                self.turn_off_list.append(len(self.layers))
                self.layers += [BatchNormalization(size_list[i+1])]
            self.layers += [Relu()]
            if use_dropout:
                self.turn_off_list.append(len(self.layers))
                self.layers += [Dropout(dropout_ratio)]
        self.layers += [Affine(size_list[-1], output_size, 'xavier')]
        if use_focal_loss:
            self.loss_layer = FocalLoss(gamma=focal_gamma)
        else:
            self.loss_layer = CrossEntropyLoss()

        # collect params and grads
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
        
    def set_train_flag(self, flag):
        for idx in self.turn_off_list:
            self.layers[idx].train_flag = flag

    def predict(self, xs):
        self.set_train_flag(False)
        for layer in self.layers:
            xs = layer.forward(xs)
        self.set_train_flag(True)
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
    
    def summary(self):
        print('-'*60)
        for layer in self.layers:
            print(layer.name)
            print('-'*60)
        print(self.loss_layer.name)
        print('-'*60)


################################## define optimizer ##################################

class CosineAnnealingScheduler():
    '''
    https://arxiv.org/abs/1608.03983v5
    '''
    def __init__(self, period, lr_min, lr_max):
        self.period = period
        self.lr_min = lr_min
        self.lr_max = lr_max

    def next_lr(self):
        return self.lr_min


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

        for i in range(len(params)):
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

