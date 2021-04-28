# -*- coding: UTF-8 -*-
import numpy as np
import sys


class AdamOptim():
    def __init__(self, w_size, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # parameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lr = eta 
        self.w_size = w_size
        # Momentum and RMSProp initial
        # m_dw: Momentum for weight's gradient 
        # v_dw: RMSProp for weight's gradient
        self.m_dw, self.v_dw = np.zeros(self.w_size), np.zeros(self.w_size)
            
    def step(self, parms, grad, t):
        ''' weights '''
        self.m_dw = self.beta1 * self.m_dw + (1-self.beta1) * grad
        self.v_dw = self.beta2 * self.v_dw + (1-self.beta2) * np.square(grad)
        self.m_dw /= (1-self.beta1**t)
        self.v_dw /= (1-self.beta2**t)
        parms += self.lr * self.m_dw / (np.sqrt(self.v_dw) + self.epsilon)
        return parms

class MatrixFactorization:
    def __init__(self, U_size, V_size, W, epoch, lr, max_patience, numpy_version=True):        
        # parameters
        self.epoch = epoch
        self.lr = lr
        self.max_patience = max_patience
        self.W = W
        
        # random initial
        self.U = self._set_example(U_size)
        self.V = self._set_example(V_size)
        
    def _set_example(self, matrix_size):
        # standard normal distribution (mean=0, std=1)
        return np.random.normal(0, 1, size=matrix_size)
        
    def _model(self, inputs, weights):
        return np.dot(inputs, weights)
    
    def _criterion(self, target_matrix, real_matrix):
        # E: error
        self.E = np.subtract(target_matrix, real_matrix)
        return np.sum(np.square(self.E))
    
    def _backward(self, t):
        # alpha: learning_rate
        # dU = -alpha* E*V.T, U_new = U_old + dU
        # dV = -alpha* U.T*E, V_new = V_old + dV 
        
        # GD (gradient descent)
        # self.U += self.lr * np.dot(self.E, self.V.T)
        # self.V += self.lr * np.dot(self.U.T, self.E)
        
        # Adam (adaptive momentum estimation) = Momentum + RMSProp
        # t: time step
        self.U = self.adam_U.step(self.U, np.dot(self.E, self.V.T), t)
        self.V = self.adam_V.step(self.V, np.dot(self.U.T, self.E), t)
        
    def train(self):
        # early stopping parameters
        patience = 0
        last_loss = 0
        
        for e in range(self.epoch):
        
            ''' Optimizer '''
            self.adam_U = AdamOptim(self.U.shape, eta=self.lr)
            self.adam_V = AdamOptim(self.V.shape, eta=self.lr)
            
            ''' forward '''
            # outputs: ^W, real value
            # inner product: Y = f(Wx)
            outputs = self._model(self.U, self.V)
            
            # loss: Σ(W-^W)^2
            # if use E = W-^W, then adjust parameters must add '+'
            # if use E = ^W-W, then adjust parameters must sub '-'
            loss = self._criterion(self.W, outputs)
            print("epoch:{e}/{total_e} loss = {l}".format(e=e+1, total_e=self.epoch, l=loss))
            
            ''' backward '''
            # update parameters with gradient descent
            self._backward(e+1)
            
            ''' early stopping '''
            if last_loss and loss > last_loss:
                patience += 1
                if patience >= self.max_patience:
                    print("\nEarlystop at epoch {e}".format(e=e))
                    break
            else:
                patience = 0   
            last_loss = loss
        print()

if __name__ == '__main__':

    # np.random.seed(42)
    ''' train setting '''
    epoch = 300
    learning_rate = 1e-2
    max_patience = 1
    
    # W ~= U*V
    # W:3x4
    U_size = (3, 5)
    V_size = (5, 4)
    W = np.array([ [1, 3, 5, -2],
                   [2, 4, -3, -1], 
                   [-1, 6, 3, -5]  ])
                   
    ''' Matrix Factorization '''
    mf = MatrixFactorization(U_size=U_size, V_size=V_size, W=W, epoch=epoch, lr=learning_rate, max_patience=max_patience)
    mf.train()
    print("------ U ------\n", mf.U)
    print("------ V ------\n", mf.V)
    print("------ W ------\n", mf.W)
    print("------ U*V ------\n", np.dot(mf.U, mf.V))
    