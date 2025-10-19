from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import data_preprocessing

X, t= data_preprocessing.get_training_data()
X_val, t_val= data_preprocessing.get_validation_data()
ridge_model= Ridge(alpha= 0.001, fit_intercept= True)
ridge_model.fit(X,t.ravel())
y_val= ridge_model.predict(X_val)
mse= mean_squared_error(t_val, y_val)
mae= mean_absolute_error(t_val, y_val)
print(mse, mae)
