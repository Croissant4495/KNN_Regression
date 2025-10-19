import pandas as pd
from sklearn.model_selection import train_test_split

## importing the dataset
houses_dataset= pd.read_csv("California_Houses/California_Houses.csv")
x_training= houses_dataset.to_numpy()

## splitting data into training, validation and test
x_training, x_test= train_test_split(x_training, test_size= 0.3, random_state= 42)
x_test, x_validation= train_test_split(x_test, test_size= 0.5, random_state= 42)

## normalizing the data using standard normalization, then splitting the data
## to a feature matrix and a target vector,and adding a ones column for bias 
## in each feature matrix
means= x_training.mean(axis=0)
stds= x_training.std(axis=0)
t_training= x_training[:, 0].reshape(-1, 1)
t_validation= x_validation[:, 0].reshape(-1, 1)
t_test= x_test[:, 0].reshape(-1, 1)
x_training= (x_training - means) / stds
x_validation= (x_validation - means) / stds
x_test= (x_test - means) / stds
x_training[:, 0]= 1
x_validation[:, 0]= 1
x_test[:, 0]= 1
y_mean= means[0]
y_std= stds[0]

t_training= (t_training - y_mean) / y_std
t_validation= (t_validation - y_mean) / y_std

def get_training_data():
    return x_training, t_training
def get_validation_data():
    return x_validation, t_validation
def get_test_data():
    return x_test, t_test
def get_pred_denorm_factors():
    return y_mean, y_std
