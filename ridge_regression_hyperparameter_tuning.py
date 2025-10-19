import numpy as np
import matplotlib as plt
import data_preprocessing
from ridge_regression_manual import RidgeRegressor

X, t= data_preprocessing.get_training_data()
X_val, t_val= data_preprocessing.get_validation_data()
lambdas= [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
learning_rates= [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128]

models = np.array([RidgeRegressor(X, t, lam, X.shape[1]) for lam in lambdas])

for model in models:
    model.closed_form()
W_closed_form= np.column_stack([model.get_weights() for model in models])
errors_closed_form= X_val @ W_closed_form - t_val
mse_closed_form= np.mean(errors_closed_form ** 2, axis= 0) 
mae_closed_form= np.mean(np.abs(errors_closed_form), axis= 0)

best_mse_index = np.argmin(mse_closed_form)
best_model = models[best_mse_index]

W_grad_descent = np.column_stack([best_model.gradient_descent(lr) for lr in learning_rates])
errors_grad_descent= X_val @ W_grad_descent - t_val
mse_grad_descent= np.mean(errors_grad_descent ** 2, axis= 0) 
mae_grad_descent= np.mean(np.abs(errors_grad_descent), axis= 0)
mean, std= data_preprocessing.get_pred_denorm_factors()

print(f"Best MSE index (closed form): {best_mse_index}")
print(mae_closed_form)
print(mse_closed_form)
print(mae_grad_descent)
print(mse_grad_descent)
