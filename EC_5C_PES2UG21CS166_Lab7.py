import torch
import numpy as np

class GMM:
    def __init__(self, n_components):
        self.n_components = n_components
        self.weights = torch.ones(n_components) / n_components
        self.means = torch.randn(n_components, 3)
        self.covariances = torch.eye(3).repeat(n_components, 1, 1)

    def fit(self, X, max_iters=100, tol=1e-4):
        n_samples = X.shape[0]
        n_features = X.shape[1]

        for iteration in range(max_iters):
            # Expectation step
            responsibilities = self._e_step(X)

            # Maximization step
            self._m_step(X, responsibilities)

            if self._is_converged(X, responsibilities, tol):
                break

    def _e_step(self, X):
        # Compute the responsibilities
        responsibilities = torch.zeros(self.n_components, X.shape[0])
        for k in range(self.n_components):
            diff = X - self.means[k]
            inv_covariance = self._inverse(self.covariances[k])
            exponent = -0.5 * torch.einsum('ij,ij->i', [diff, torch.matmul(inv_covariance, diff.T).T])
            responsibilities[k] = self.weights[k] * torch.exp(exponent)

        responsibilities = responsibilities / responsibilities.sum(0)
        return responsibilities

    def _m_step(self, X, responsibilities):
        # Update the weights, means, and covariances
        total_responsibilities = responsibilities.sum(1)
        self.weights = total_responsibilities / X.shape[0]
        for k in range(self.n_components):
            weighted_sum = torch.sum(responsibilities[k].unsqueeze(1) * X, dim=0)
            self.means[k] = weighted_sum / total_responsibilities[k]
            diff = X - self.means[k]
            self.covariances[k] = torch.einsum('ij,ik->jk', [diff, diff * responsibilities[k].unsqueeze(1)]) / total_responsibilities[k]

    def _inverse(self, matrix):
        return torch.inverse(matrix + torch.eye(matrix.shape[0]) * 1e-6)

    def _is_converged(self, X, responsibilities, tol):
        prev_log_likelihood = self._log_likelihood(X, responsibilities)
        responsibilities = self._e_step(X)
        current_log_likelihood = self._log_likelihood(X, responsibilities)
        return abs(current_log_likelihood - prev_log_likelihood) < tol

    def _log_likelihood(self, X, responsibilities):
        log_likelihood = torch.log(responsibilities.sum(0)).sum()
        return log_likelihood

    def predict(self, X):
        responsibilities = self._e_step(X)
        labels = torch.argmax(responsibilities, dim=0)
        return labels

    def get_cluster_means(self):
        return self.means

    def get_cluster_covariances(self):
        return self.covariances

GMMModel = GMM