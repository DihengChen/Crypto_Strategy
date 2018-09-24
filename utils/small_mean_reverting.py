# -*- coding: utf-8 -*-
# @Author: Sky Zhang
# @Date:   2018-09-22 22:50:26
# @Last Modified by:   Sky Zhang
# @Last Modified time: 2018-09-23 19:59:59

import numpy as np
import statsmodels.api as sm
import pandas as pd


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


class SparseSolver:
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
        gamma = ret.iloc[1:, :].cov()  # covariance matrix of S
        A = np.linalg.inv(S_lag.T @ S_lag) @ S_lag.T @ S

        self._gamma = gamma
        self._A = A

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
                    gain = []
                    for j in I_c:

                        ran_ge = np.ix_(I + [j], I + [j])
                        a, b = A[ran_ge], B[ran_ge]
                        b_inv_sqrt = np.sqrt(np.linalg.inv(b))
                        mat = b_inv_sqrt @ a @ b_inv_sqrt
                        eig_val, eig_vec = np.linalg.eig(mat)

                        if opt == 'max':
                            z = eig_vec[:, eig_val.argmax()].reshape(-1, 1)
                        elif opt == 'min':
                            z = eig_vec[:, eig_val.argmin()].reshape(-1, 1)

                        x = (b_inv_sqrt @ z).reshape(-1, 1)
                        gain.append(target(a, b, x))

                    if opt == 'max':
                        index = I_c[np.array(gain).argmax()]
                    elif opt == 'min':
                        index = I_c[np.array(gain).argmax()]

                    I.append(index)
                    I_c.remove(index)

            ran_ge = np.ix_(I, I)
            a, b = A[ran_ge], B[ran_ge]
            b_inv_sqrt = np.sqrt(np.linalg.inv(b))
            mat = b_inv_sqrt @ a @ b_inv_sqrt
            eig_val, eig_vec = np.linalg.eig(mat)

            if opt == 'max':
                z = eig_vec[:, eig_val.argmax()].reshape(-1, 1)
            elif opt == 'min':
                z = eig_vec[:, eig_val.argmin()].reshape(-1, 1)

            x = (b_inv_sqrt @ z).reshape(-1, 1)

            return x

        elif method == 'semidefinite relaxation':
            # TODO
            # https://github.com/TrishGillett/pysdpt3glue
            pass

