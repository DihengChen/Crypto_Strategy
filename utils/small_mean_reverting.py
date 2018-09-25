# -*- coding: utf-8 -*-
# @Author: Sky Zhang
# @Date:   2018-09-22 22:50:26
# @Last Modified by:   Sky Zhang
# @Last Modified time: 2018-09-24 21:27:08

import numpy as np
import statsmodels.api as sm
import pandas as pd
import os
import sqlite3
import matplotlib.pyplot as plt


def CCA_Box_Tiao(data):

    df_lag = data.shift(1).dropna()
    df = data.drop(data.index[0]).dropna()
    n = data.shape[1]

    X = df_lag.as_matrix()
    X_I = sm.add_constant(X)
    Y = df.as_matrix()
    l1 = sm.OLS(Y, X_I).fit()
    A = l1.params[1:(n + 1)]

    cov = data.cov().values
    Q = np.linalg.inv(cov) @ A.T @ cov @ A

    eig_val, eig_vec = np.linalg.eig(Q)

    # choose the eigenvector corresponding to the smallest eigenvalue
    weights = eig_vec[:, eig_val.argmin()].reshape(-1, 1)

    return weights


class SparseCCA:
    """
    Solver for Sparse decomposition algorithms
    reference: 'Identifying Small Mean Reverting Portfolios'
    """

    def __init__(self, ret):
        """
        ret should be a pd.DataFrame without missing value
        """
        assert isinstance(ret, pd.DataFrame)
        assert ret.isnull().values.sum() == 0

        self.ret = ret

        S = ret.iloc[1:, :].values
        S_lag = ret.iloc[:-1, :].values
        gamma = ret.iloc[1:, :].cov().values  # covariance matrix of S
        A = np.linalg.inv(S_lag.T @ S_lag) @ S_lag.T @ S

        self._gamma = gamma
        self._A = A

    # may not need this
    def _mat_check(self, A, lamb_da=0.01):
        eig_val, eig_vec = np.linalg.eig(A)
        if eig_val.max() / eig_val.min() > 1000:
            print('mat need ridge shrinkage')
            return A + np.diag([lamb_da] * len(A))
        else:
            return A

    def solve(self, k, opt='max', method='greedy search'):
        """
        Parameters
        ==========
        k: cardinality of weights vector

        opt:
            'min': minimize predictability, find mean reverting portfolio
            'max': maximize predictability, find momentum portfolio
        method:
            'greedy search'
            'semidefinite relaxation'

        Return
        ==========
        weights vector, np.array
        """
        A = self._A.T @ self._gamma @ self._A
        B = self._gamma

        def target(A, B, x):
            return (x.T @ A @ x) / (x.T @ B @ x)

        if method == 'greedy search':
            I = []
            I_c = [i for i in range(len(self._A))]
            for i in range(k):
                if len(I) == 0:
                    if opt == 'max':
                        index = (np.diag(A) / np.diag(B)).argmax()
                    elif opt == 'min':
                        index = (np.diag(A) / np.diag(B)).argmin()

                    I.append(index)
                    I_c.remove(index)
                else:
                    gain = -np.inf if opt == 'max' else np.inf
                    for j in I_c:

                        ran_ge = np.ix_(I + [j], I + [j])
                        a = A[ran_ge]
                        b = B[ran_ge]
                        b_inv = np.linalg.inv(b)
                        mat = b_inv @ a
                        eig_val, eig_vec = np.linalg.eig(mat)

                        if opt == 'max':
                            x = eig_vec[:, eig_val.argmax()].reshape(-1, 1)
                            if target(a, b, x) > gain:
                                gain = target(a, b, x)
                                z = x
                                index = j
                        elif opt == 'min':
                            x = eig_vec[:, eig_val.argmin()].reshape(-1, 1)
                            if target(a, b, x) < gain:
                                gain = target(a, b, x)
                                z = x
                                index = j

                    I.append(index)
                    I_c.remove(index)

            weights = np.zeros((len(self._A), 1))
            weights[I, 0] = z.ravel()
            return weights, sorted(I)

        elif method == 'semidefinite relaxation':
            # TODO
            # https://github.com/TrishGillett/pysdpt3glue
            pass

    def _ou_estimator(self, x):
        assert isinstance(x, np.ndarray)
        p = x.ravel()
        p_t = p[1:]
        p_lag = p[:-1]
        mu = p_t.mean()
        lamb_da = -np.log(np.sum((p_t - mu) * (p_lag - mu)) /
                          np.sum((p_t - mu) * (p_t - mu)))
        sigma = (np.sum(((1 - np.exp(-lamb_da)) * (p_t - mu)) ** 2) * 2 *
                 lamb_da / (1 - np.exp(-2 * lamb_da)) / (len(p) - 2))
        half_life = np.log(2) / lamb_da
        return mu, lamb_da, sigma, half_life

    def _cov_selection(self, alphas=4, n_refinements=4, cv=None):
        from sklearn.covariance import GraphLassoCV
        gl = GraphLassoCV(alphas=alphas, n_refinements=n_refinements,
                          cv=cv, assume_centered=True)
        gl.fit(self.ret)
        return gl.covariance_, gl.precision_


def test_1():   # test for greedy search
    con = sqlite3.connect(os.getcwd()[:-6] + "/data/hist_data.db")
    tic = "('BTC','DASH','LTC','ETH','ZEC','XRP','XMR','NEO','ADA','EOS')"
    sql = "select time,tic,open,low,high,close \
           from hist_data where tic in {} and time>'2017-9-21'".format(tic)
    data = pd.read_sql(sql, con=con)
    con.close()

    price = data.pivot_table(index='time', columns='tic', values='close')
    temp_price = price.dropna()
    log_price = np.log(temp_price)
    ret = (log_price - log_price.shift()).dropna()

    card = 3
    solver = SparseCCA(ret)
    weights1, I1 = solver.solve(card, opt='min')
    weights2, I2 = solver.solve(card, opt='max')

    print(weights1, I1)
    print(weights2, I2)
    plt.subplot(2, 1, 1)
    port1 = ret.values @ weights1
    lab1 = 'mean_revert, lam={}'.format(int(solver._ou_estimator(port1)[1]))
    plt.plot(port1, label=lab1)
    port2 = ret.values @ weights2
    lab2 = 'momentum, lam={}'.format(int(solver._ou_estimator(port2)[1]))
    plt.plot(port2, label=lab2)
    plt.title("return")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(np.cumsum(ret.values @ weights1), label='mean_revert')
    plt.plot(np.cumsum(ret.values @ weights2), label='momentum')
    plt.legend()
    plt.title("cumulative return")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_1()
