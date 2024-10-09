# _*_ encoding:utf-8 _*_
import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None

    def fit(self, X, eta=0.01, n_iters=1e4):

        def demean(X):
            return X - np.mean(X, axis=0)

        def f(w, X):
            return np.sum((X.dot(w) ** 2)) / len(X)

        def df(w, X):
            return X.T.dot(X.dot(w)) * 2. / len(X)

        def direction(w):
            t = w / np.linalg.norm(w)
            return t

        def first_component(df, X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
            cur_iter = 0
            w = direction(initial_w)
            while cur_iter < n_iters:
                gradient = df(w, X)
                last_w = w
                w = w + eta * gradient
                w = direction(w)
                if (abs(f(w, X) - f(last_w, X)) < epsilon):
                    break
                cur_iter += 1
            return w

        def first_n_components(n, X, eta=0.01, n_iters=1e4, epsilon=1e-8):
            X_pca = X.copy()
            X_pca = demean(X_pca)
            res = []
            for i in range(n):
                initial_w = np.random.random(X_pca.shape[1])
                w = first_component(df, X_pca, initial_w, eta)
                res.append(w)
                X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w
            return res

        X_pca = demean(X)
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))
        self.components_ = first_n_components(self.n_components, X)
        self.components_ = np.array(self.components_)
        return self

    def transform(self, X):
        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        return X.dot(self.components_)

    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components
