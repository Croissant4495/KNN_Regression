import numpy as np
import matplotlib as plt
import data_preprocessing
import ridge_regression_manual

X, t= data_preprocessing.get_training_data()
X_val, t_val= data_preprocessing.get_validation_data()
lambdas= [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
learning_rates= [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128]

W_closed_form= np.column_stack([ridge_regression_manual.closed_form_ridge_regression(X, t, lam) for lam in lambdas])
W_grad_descent= np.column_stack([ridge_regression_manual.gradient_descent_ridge_regression(X, t, 0.0001, lr) for lr in learning_rates])
errors_closed_form= X_val @ W_closed_form - t_val
mse_closed_form= np.mean(errors_closed_form ** 2, axis= 0) 
mae_closed_form= np.mean(np.abs(errors_closed_form), axis= 0)
errors_grad_descent= X_val @ W_grad_descent - t_val
mse_grad_descent= np.mean(errors_grad_descent ** 2, axis= 0) 
mae_grad_descent= np.mean(np.abs(errors_grad_descent), axis= 0)
mean, std= data_preprocessing.get_pred_denorm_factors()
print(mae_closed_form)
print(mse_closed_form)
print(mae_grad_descent)
print(mse_grad_descent)
