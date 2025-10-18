Assignment 1

This repository contains our solution for the first assignment in the Introduction to Machine Learning course. The project focuses on implementing two fundamental machine learning tasks from the ground up and comparing them to industry-standard libraries.

---

## 🔭 Problem 1: K-NN Classification

In this problem, we build a **K-Nearest Neighbors (K-NN)** classifier to distinguish between gamma and hadron particles from the **MAGIC Gamma Telescope dataset**.

Our approach involves two key implementations:
* **From Scratch:** A manual build using NumPy to understand the algorithm's core mechanics.
* **Using Scikit-Learn:** A standard implementation to compare performance and efficiency.

The goal is to tune the hyperparameter `k` and evaluate both models on key metrics like accuracy, precision, and recall.

---

## 🏠 Problem 2: Regression for House Price Prediction

Here, we predict median house values using the **California Housing dataset** by implementing **Linear, Ridge, and Lasso regression models**.

Similar to the first problem, we tackle this with two methods:
* **From Scratch:** A manual implementation using matrix operations for the **Normal Equation** and **Gradient Descent**.
* **Using Scikit-Learn:** An implementation using Scikit-Learn's powerful regression tools.

The final analysis will compare the models based on their **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)**, with a focus on the effect of regularization.
