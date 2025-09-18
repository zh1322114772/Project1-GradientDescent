#!/usr/bin/env python
# coding: utf-8

# # Introduction

# # Methods Section - Gradient Descent
# 
# The vanilla Gradient Descent (GD) method optimizes functions by first computing the gradient of the function. Geometrically, the gradient indicates the direction of the steepest ascent. By negating the gradient and multiplying it by a constant learning rate $\alpha$, the algorithm gradually moves the function towards its minimum value.
# 
# Mathematically, it can be written as:
# 
# $$
# \theta^{(t+1)} = \theta^{(t)} - \alpha \, \nabla_{\theta} F(\theta^{(t)})
# $$
# 
# where  
# - $\theta^{(t)}$ are the parameters at iteration $t$
# - $\alpha$ is the learning rate  
# - $\nabla_{\theta} F(\theta^{(t)})$ is the gradient of the function with respect to $\theta$.  

# ### Pseudocode
# 
# $F(·):=$ Objective function  
# $x:=$ Input
# 
# Do  
# $\hspace{2em}x_{prev}=x $  
# $\hspace{2em}x= x_{prev} - \alpha\nabla F(x_{prev}) $
# 
# While $ |x - x_{prev}| < $ 1e-6 
# 
# return $x$

# ### Example Code
# 
# We first define base classes for C1 and C2 differentiable functions

# In[27]:


import numpy as np

class C1Differentiable:
    """base class for C1-differentiable functions."""

    def __init__(self):
        self.__history = []

    def get_history(self):
        return np.array(self.__history)

    def add_history(self, x):
        self.__history.append(x)

    def clear_history(self):
        self.__history = []

    def derivative(self, x):
        pass

    def forward(self, x):
        pass



class C2Differentiable(C1Differentiable):
    """base class for C2-differentiable functions."""

    def second_derivative(self, x):
        pass


# In[ ]:


class GradientDescent:
    """Gradient Descent optimizer."""

    def __init__(self, f: C1Differentiable, x0, alpha=0.01, tol=1e-6, max_iter=2000):
        super().__init__()

        self.f = f # function to minimize
        self.x = x0 # initial point
        self.alpha = alpha # learning rate
        self.tol = tol # tolerance for stopping criterion
        self.max_iter = max_iter # maximum number of iterations

    def optimize(self):
        self.f.clear_history()

        for _ in range(self.max_iter):
            grad = self.f.derivative(self.x)
            y = self.f.forward(self.x)

            new_x = self.x - self.alpha * grad
            self.f.add_history([self.x[0], self.x[1], y])

            if abs(self.f.forward(new_x) - self.f.forward(self.x)) < self.tol:
                break
            self.x = new_x

        return self.x


# # Methods Section - Newton's Method 
# 
# 
# Newton's method outperforms vanilla Gradient Descent by introducing an adaptive learning rate. Instead of using a fixed learning rate $\alpha$, Newton's method replaces it with the inverse of the Hessian matrix. This allows the step size to adapt to the local curvature of the function: when the curvature is steep, the algorithm takes smaller steps; when the curvature is flat, it takes larger steps. This adaptive adjustment effectively mitigates the zig-zag problem commonly observed in Gradient Descent.  
# 
# Mathematically, it can be written as:
# 
# $$
# \theta^{(t+1)} = \theta^{(t)} - H^{-1}(\theta^{(t)}) \, \nabla_{\theta} F(\theta^{(t)})
# $$
# 
# where  
# - $\theta^{(t)}$ are the parameters at iteration $t$  
# - $H(\theta^{(t)})$ is the Hessian matrix of second derivatives at iteration $t$  
# - $ \nabla_{\theta} F(\theta^{(t)}) $ is the gradient of the function with respect to $\theta$.

# ### Pseudocode
# 
# $F(·):=$ Objective function  
# $x:=$ Input
# 
# Do  
# $\hspace{2em}x_{prev}=x $  
# $\hspace{2em}x= x_{prev} - H^{-1}(x_{prev}) \, \nabla F(x_{prev}) $
# 
# While $ |x - x_{prev}| < $  1e-6 
# 
# return $x$

# ### Example Code

# In[29]:


class GradientDescentNewton:
    """Gradient Descent optimizer."""

    def __init__(self, f: C1Differentiable, x0, tol=1e-6, max_iter=2000):
        super().__init__()

        self.f = f # function to minimize
        self.x = x0 # initial point
        self.tol = tol # tolerance for stopping criterion
        self.max_iter = max_iter # maximum number of iterations

    def optimize(self):
        self.f.clear_history()

        for _ in range(self.max_iter):
            grad = self.f.derivative(self.x)
            H_inverse = np.linalg.inv(self.f.second_derivative(self.x))
            y = self.f.forward(self.x)

            new_x = self.x - H_inverse @ grad
            self.f.add_history([self.x[0], self.x[1], y])

            if abs(self.f.forward(new_x) - self.f.forward(self.x)) < self.tol:
                break

            self.x = new_x

        return self.x


# # Methods Section - AdaGrad
# 
# 
# Newton's method often outperforms vanilla Gradient Descent by introducing an adaptive learning rate through the inverse of the Hessian matrix. However, computing the inverse of the Hessian has a time complexity of $O(d^3)$, which becomes impractical when dealing with a large number of parameters. 
# 
# To address this, AdaGrad introduces an alternative adaptive learning rate by leveraging historical gradients. The key idea is that if certain parameters consistently have large gradients, their learning rate should shrink, while parameters with infrequent updates should maintain a relatively larger learning rate. This helps balance convergence across all parameters.
# 
# Mathematically, it can be written as
# 
# $$
# \theta^{(t+1)} = \theta^{(t)} - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot \nabla_{\theta} F(\theta^{(t)})
# $$
# 
# where
# - $\theta^{(t)}$ are the parameters at iteration $t$
# - $\eta$ is the initial learning rate  
# - $\nabla_{\theta} F(\theta^{(t)})$ is the gradient of the function with respect to $\theta$ at iteration $t$  
# - $G_t = \sum_{\tau=1}^t \big( \nabla_{\theta} F(\theta^{(\tau)}) \big)^2$ is the sum of squared historical gradients (accumulated per parameter)  
# - $\epsilon$ is a small constant added to avoid singularity
# - $\odot$ denotes element-wise division  

# ### Pseudocode
# 
# $F(·):=$ Objective function  
# $x:=$ Input  
# $G:=$ 0 (same shape as θ)  
# $\epsilon:=$ 1e-8
# 
# Do  
# $\hspace{2em}x_{prev}=x $  
# $\hspace{2em}g=\nabla F(x_{prev}) $  
# $\hspace{2em}G=G + g \odot g $  
# $\hspace{2em}x= x_{prev} - \frac{\eta}{\sqrt{G + \epsilon}} \odot g $
# 
# While $ |x - x_{prev}| <$  1e-6 
# 
# return $x$

# ### Example Code

# In[ ]:


class AdaGrad:
    """AdaGrad optimizer."""

    def __init__(self, f: C1Differentiable, x0, alpha=0.1, tol=1e-6, max_iter=2000):
        super().__init__()

        self.f = f # function to minimize
        self.x = np.asarray(x0, dtype=np.float64)
        self.G = np.zeros_like(self.x, dtype=np.float64)
        # self.x = x0 # initial point
        self.alpha = alpha # initial learning rate
        self.tol = tol # tolerance for stopping criterion
        self.max_iter = max_iter # maximum number of iterations
        self.epsilon = 1e-8 # small constant to avoid singularity
        # self.G = np.zeros_like(x0) # sum of squared historical gradients

    def optimize(self):
        self.f.clear_history()

        for _ in range(self.max_iter):
            grad = self.f.derivative(self.x)
            y = self.f.forward(self.x)

            self.G += grad * grad
            adjusted_grad = grad / (np.sqrt(self.G) + self.epsilon)
            new_x = self.x - self.alpha * adjusted_grad
            self.f.add_history([self.x[0], self.x[1], y])

            if abs(self.f.forward(new_x) - self.f.forward(self.x)) < self.tol:
                break

            self.x = new_x

        return self.x


# # Methods Section - Adam
# 
# The Adam (Adaptive Moment Estimation) optimizer improves upon vanilla Gradient Descent by combining the benefits of **Momentum** and **RMSProp**.  
# Adam maintains two moving averages:  
# 
# 1. **First moment (mean of gradients)** — captures the direction of past gradients (like Momentum).  
# 2. **Second moment (uncentered variance of gradients)** — scales the step size by the magnitude of past gradients (like RMSProp).  
# 
# Both estimates are bias-corrected to counteract initialization at zero. This adaptive mechanism allows Adam to adjust learning rates individually for each parameter, ensuring stable and efficient convergence, especially in high-dimensional or sparse settings.
# 
# Mathematically, it can be written as
# 
# 
# **First moment estimate:**
# $$
# m_t = \beta_1 m_{t-1} + (1 - \beta_1)\nabla_{\theta} F(\theta^{(t)})
# 
# $$ 
# **Second moment estimate:**
# $$
# v_t = \beta_2 v_{t-1} + (1 - \beta_2)\big(\nabla_{\theta} F(\theta^{(t)})\big)^2
# $$
# 
# **Bias correction:**
# $$
# \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
# $$
# 
# **Final parameter update:**
# $$
# \theta^{(t+1)} = \theta^{(t)} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
# $$
# 
# where  
# - $\theta^{(t)}$: parameters at iteration $t$  
# - $\nabla_{\theta} F(\theta^{(t)})$: gradient of the loss function with respect to $\theta$  
# - $m_t$: first moment (exponential moving average of gradients)  
# - $v_t$: second moment (exponential moving average of squared gradients)  
# - $\hat{m}_t, \hat{v}_t$: bias-corrected estimates  
# - $\alpha$: learning rate  
# - $\beta_1, \beta_2$: decay rates for the moving averages (commonly $\beta_1 = 0.9, \beta_2 = 0.999$)  
# - $\epsilon$: small constant for numerical stability 

# ### Pseudocode
# 
# $F(·):=$ Objective function  
# $x:=$ Input  
# $\alpha:= $ Learning rate  
# $\beta_1:= $ 0.9  
# $\beta_2:= $ 0.999  
# $\epsilon:=$ 1e-8  
# $t:= $ 1
# 
# Do  
# $\hspace{2em}x_{prev}=x $  
# $\hspace{2em}g=\nabla F(x_{prev}) $  
# $\hspace{2em}m=\beta_1 * m + (1-\beta_1) * g $  
# $\hspace{2em}v=\beta_2 * v + (1-\beta_2) * (g \odot g) $  
# $\hspace{2em} \hat{m} = \frac{m}{(1-(\beta_{1})^{t})}$  
# $\hspace{2em} \hat{v} = \frac{v}{(1-(\beta_{2})^{t})}$  
# $\hspace{2em}x= x_{prev} - \frac{\alpha * \hat{m}}{\sqrt{\hat{v}} + \epsilon} $  
# $\hspace{2em}t=t+1$
# 
# While $ |x - x_{prev}| <$  1e-6 
# 
# return $x$

# ### Example Code

# In[ ]:


class Adam:
    def __init__(self, f: C1Differentiable, x0, alpha=0.01, beta1=0.9, beta2=0.999, eps=1e-8, tol=1e-6, max_iter=2000):
        super().__init__()
        self.f, self.x = f, x0
        self.alpha, self.beta1, self.beta2 = alpha, beta1, beta2
        self.eps, self.tol, self.max_iter = eps, tol, max_iter

    def optimize(self):
        self.f.clear_history()
        m = np.zeros_like(self.x); v = np.zeros_like(self.x); t = 0
        for _ in range(self.max_iter):
            g = self.f.derivative(self.x); y = self.f.forward(self.x); t += 1
            m = self.beta1*m + (1-self.beta1)*g
            v = self.beta2*v + (1-self.beta2)*(g*g)
            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)
            new_x = self.x - self.alpha * m_hat / (np.sqrt(v_hat) + self.eps)
            self.f.add_history([self.x[0], self.x[1], y])
            if abs(self.f.forward(new_x) - y) < self.tol: break
            self.x = new_x
        return self.x







class ConvexBowl(C2Differentiable):
    """Convex bowl function."""
    def forward(self, x):
        return x[0]**2 + x[1]**2
    def derivative(self, x):
        return np.array([2*x[0], 2*x[1]])
    def second_derivative(self, x):
        return np.array([[2, 0], [0, 2]])


class Rosenbrock(C2Differentiable):
    """Rosenbrock (banana valley) function.
    f(x,y) = (1-x)² + 100(y-x²)²
    """
    def forward(self, x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def derivative(self, x):
        df_dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
        df_dy = 200 * (x[1] - x[0]**2)
        return np.array([df_dx, df_dy])
    
    #! NOT SURE IF THIS IS CORRECT SECOND DERIVATIVE
    def second_derivative(self, x):
        d2f_dx2 = 2 - 400 * x[1] + 1200 * x[0]**2
        d2f_dy2 = 200
        d2f_dxdy = -400 * x[0]
        
        return np.array([[d2f_dx2, d2f_dxdy], 
                        [d2f_dxdy, d2f_dy2]])

class CosineBumps(C2Differentiable):
    """Multimodal cosine bumps.
    f(x,y) = x² + y² + 10cos(x) + 10cos(y)
    """
    def forward(self, x):
        return x[0]**2 + x[1]**2 + 10*np.cos(x[0]) + 10*np.cos(x[1])
    
    def derivative(self, x):
        df_dx = 2*x[0] - 10*np.sin(x[0])
        df_dy = 2*x[1] - 10*np.sin(x[1])
        return np.array([df_dx, df_dy])
    
    def second_derivative(self, x):
        d2f_dx2 = 2 - 10*np.cos(x[0])
        d2f_dy2 = 2 - 10*np.cos(x[1])
        d2f_dxdy = 0 
        
        return np.array([[d2f_dx2, d2f_dxdy], 
                        [d2f_dxdy, d2f_dy2]])