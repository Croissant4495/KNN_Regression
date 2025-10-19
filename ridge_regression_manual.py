import numpy as np

def closed_form_ridge_regression(X, t, lam):
    I= np.identity(X.shape[1])
    I[0,0]= 0
    w= np.linalg.inv(X.T @ X + lam * I) @ (X.T @ t)
    return w

def gradient_descent_ridge_regression(X, t, lam, learn_rate, tolerance= 1e-6, iterations= 10000):
    n,d= X.shape
    losses= []
    w= np.zeros((d,1))
    reg_factor= lam / n
    for i in range(iterations):
        error= X @ w - t
        loss= (error**2).sum / 2*n
        reg_loss= reg_factor/2 * (w[1:]**2).sum()
        loss += reg_loss
        losses.append(loss)
        if i > 2 and abs(losses[-1] - losses[-2]) < tolerance and abs(losses[-2] - losses[-3]) < tolerance and abs(losses[-3] - losses[-4]) < tolerance:
            break
        
        gradient= (X.T @ error) / n + reg_factor * w
        gradient[0,0] -= reg_factor * w[0,0]
        w = w - learn_rate * gradient
    return w, losses
