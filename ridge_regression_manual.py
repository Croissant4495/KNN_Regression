import numpy as np

class RidgeRegressor:
    def __init__(self, x, t, lam,M):
        self.features = x
        self.target = t
        self.lambda_reg = lam
        self.complexity = M
        self.weights = np.zeros((self.complexity, 1))
        self.noFeatures = self.features.shape[0]
        self.noSamples = self.target.shape[0]
    
    def get_weights(self):
        return self.weights

    def closed_form(self):
        I= np.identity(self.features.shape[1])
        I[0,0]= 0
        self.weights = np.linalg.inv(self.features.T @ self.features + self.lambda_reg * I) @ (self.features.T @ self.target).reshape(-1,1)

    def predict(self, x):
        return np.dot(x, self.weights).reshape(-1, 1)

    def calc_loss(self, x, t):
        error = self.predict(x) - t
        mse= (error**2).sum() / (2*self.noSamples)
        mae = np.mean(np.abs(error))
        return error, mse, mae

    def reg_loss(self):
        reg_loss = self.lambda_reg / (2 * self.noSamples) * (self.weights[1:] ** 2).sum()
        return reg_loss

    def gradient_descent(self, learn_rate, tolerance= 1e-6, iterations= 10000):
        self.weights= np.zeros((self.complexity, 1))
        losses= []
        reg_factor= self.lambda_reg / self.noSamples
        for i in range(iterations):
            error, mse, mae = self.calc_loss(self.features, self.target)
            reg_loss = self.reg_loss()
            losses.append(mse + reg_loss)
            if i > 1 and abs(losses[-1] - losses[-2]) < tolerance:
                break

            gradient= (self.features.T @ error) / self.noSamples + reg_factor * self.weights.reshape(-1, 1)
            gradient[0, 0] -= reg_factor * self.weights[0]
            self.weights -= learn_rate * gradient
        
        return self.weights