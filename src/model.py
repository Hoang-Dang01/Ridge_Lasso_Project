# src/model.py
class RidgeLassoPure:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n_samples = len(X)
        self.n_features = len(X[0])

    def _transpose(self, A):
        return [list(i) for i in zip(*A)]

    def _mat_mul(self, A, B):
        return [[sum(a * b for a, b in zip(r, c)) for c in zip(*B)] for r in A]

    def _invert_matrix(self, A):
        n = len(A)
        AM = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(A)]
        for i in range(n):
            pivot = AM[i][i]
            AM[i] = [x / pivot for x in AM[i]]
            for j in range(n):
                if i != j:
                    factor = AM[j][i]
                    AM[j] = [a - factor * b for a, b in zip(AM[j], AM[i])]
        return [row[n:] for row in AM]

    def fit_ridge(self, alpha):
        """Fits Ridge regression (L2 regularization) and returns weights as a list."""
        XT = self._transpose(self.X)
        XTX = self._mat_mul(XT, self.X)
        for i in range(self.n_features):
            XTX[i][i] += alpha
        
        y_col = [[val] for val in self.y]
        XTy = self._mat_mul(XT, y_col)
        inv_XTX = self._invert_matrix(XTX)
        weights_col = self._mat_mul(inv_XTX, XTy)
        return [w[0] for w in weights_col]

    def fit_lasso(self, alpha, iterations=1000):
        """Fits Lasso regression (L1 regularization) using Coordinate Descent."""
        w = [0.0] * self.n_features
        for _ in range(iterations):
            for j in range(self.n_features):
                rho = 0.0
                for i in range(self.n_samples):
                    r_i = self.y[i] - sum(w[k] * self.X[i][k] for k in range(self.n_features) if k != j)
                    rho += self.X[i][j] * r_i
                
                threshold = alpha * self.n_samples
                if rho > threshold:
                    w[j] = (rho - threshold) / self.n_samples
                elif rho < -threshold:
                    w[j] = (rho + threshold) / self.n_samples
                else:
                    w[j] = 0.0
        return w
