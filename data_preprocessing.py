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
x_training= (x_training - means) / stds
x_validation= (x_validation - means) / stds
x_test= (x_test - means) / stds
t_training= x_training[:, 0]
x_training[:, 0]= 1
t_validation= x_validation[:, 0]
x_validation[: 0]= 1
t_test= x_test[:, 0]
x_test[:, 0]= 1
