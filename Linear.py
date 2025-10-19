import numpy as np
import matplotlib.pyplot as plt
import data_preprocessing as dp


class LinearRegressor:
    def __init__(self, x, t, M):
        self.features = np.array(x)
        self.target = np.array(t)
        self.complexity = M
        self.weights = np.zeros(self.complexity)
        self.noFeatures = self.features.shape[0]
        self.noSamples = self.target.shape[0]

    def get_weights(self):
        return self.weights
    
    def calc_direct_sol(self):
        inv = np.linalg.inv(np.dot(self.features.T, self.features))
        self.weights = np.dot(inv, np.dot(self.features.T, self.target)).reshape(-1)

    def predict(self, x):
        predictions = np.dot(x, self.weights).reshape(-1, 1)
        return predictions

    def calc_loss(self, x, t):
        predictions = self.predict(x)
        error = predictions - t
        mse = np.mean(error ** 2)
        mae = np.mean(np.abs(error))
        return error, mse, mae

    def optimize_gd(self, learning_rate=0.01, max_iter=10000, tolerance=1e-6):
        for iter in range(max_iter):
            error, mse, mae = self.calc_loss(self.features, self.target)
            if iter % 100 == 0:
                print(f"Iteration {iter}: MSE = {mse:.2f}, MAE = {mae:.2f}")
            if mse < tolerance:
                print(f"Iteration {iter}: MSE = {mse:.2f}, MAE = {mae:.2f}")
                break
            gradient = (2 / self.noSamples) * np.dot(self.features.T, error).reshape(-1)
            self.weights -= learning_rate * gradient

def print_array_2d(arr):
    """
    Prints a NumPy array with all float values formatted to 2 decimal places.
    Works for any array shape.
    """ 
    formatted = np.array2string(
        arr, 
        formatter={'float_kind': lambda x: f"{x:.4f}"}
    )
    print(formatted)

if __name__ == '__main__':
    x_train, t_train = dp.get_training_data()
    print("Shape of training data:", x_train.shape, t_train.shape)
    x_validation, t_validation = dp.get_validation_data()
    L = LinearRegressor(x_train, t_train, x_train.shape[1])
    L.optimize_gd(learning_rate=0.01, max_iter=1000, tolerance=0.001)
    print_array_2d(L.get_weights())
    print("Direct Solution:")    
    L.calc_direct_sol()
    print_array_2d(L.get_weights())

    print("Validation Loss:")
    error, mse, mae = L.calc_loss(x_validation, t_validation)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    test_x, test_t = dp.get_test_data()
    print("Test Loss:")
    error, mse, mae = L.calc_loss(test_x, test_t)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Visualizing predictions vs actuals for test set
    predictions = L.predict(test_x)
    #Denomralize
    y_mean, y_std = dp.get_pred_denorm_factors()
    test_t = test_t * y_std + y_mean
    predictions = predictions * y_std + y_mean
    plt.scatter(test_t, predictions, alpha=0.5)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Linear Regression: Actual vs Predicted on Test Set")
    plt.plot([test_t.min(), test_t.max()], [test_t.min(), test_t.max()], 'k--', lw=2)
    plt.show()
