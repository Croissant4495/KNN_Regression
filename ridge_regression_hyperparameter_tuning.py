import numpy as np
import matplotlib as plt
import data_preprocessing
import ridge_regression_manual

X, t= data_preprocessing.get_training_data()
X_val, t_val= data_preprocessing.get_validation_data()
lambdas= [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
learning_rates= [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128]

W= np.column_stack([ridge_regression_manual.closed_form_ridge_regression(X, t, lam) for lam in lambdas])
errors= X_val @ W - t_val
mse= np.mean(errors ** 2, axis= 0) 
mae= np.mean(np.abs(errors), axis= 0)
print(mae)
print(mse)
