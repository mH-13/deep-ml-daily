---

## 4. Inverse & Determinant

### What it is

* **Determinant**: A scalar value that, among other things, tells you whether a square matrix is invertible (nonzero ⇒ invertible).
* **Inverse**: For a square matrix $A\in\mathbb{R}^{n\times n}$, $A^{-1}$ satisfies $A\,A^{-1} = I$.

### Python Code

```python
import numpy as np
from numpy.linalg import inv, det

A = np.array([[4.0, 7.0],
              [2.0, 6.0]])

# Determinant
d = det(A)           # 4*6 - 7*2 = 10

# Inverse (only if d ≠ 0)
A_inv = inv(A)       # [[ 0.6, -0.7], [-0.2, 0.4]]
```

### How it works under the hood

* **Determinant** uses LU decomposition (or cofactors for small n) to compute the product of U’s diagonal entries.
* **Inverse** typically uses Gaussian elimination (via LU or QR factorization) to solve $A X = I$ column by column.

### Real-World Scenario

> **Problem:** You want to “undo” a linear transformation in data preprocessing—e.g., you standardized a batch by multiplying by a scaling matrix $S$, and now you need to restore original units.
> **Solution:** Compute $S^{-1}$ once and apply it to your normalized data:
>
> ```python
> S = np.diag(scales)      # scale factors along each feature
> inv_S = np.linalg.inv(S)
> original = normalized_data @ inv_S
> ```

---

## 5. Solving Linear Systems

### What it is

Finding $x$ that satisfies $A x = b$, where $A\in\mathbb{R}^{n\times n}$ (or even rectangular) and $b\in\mathbb{R}^n$.

### Python Code

```python
from scipy.linalg import solve

# Square system
A = np.array([[3, 1],
              [1, 2]])
b = np.array([9, 8])

x = solve(A, b)    # solves exactly: x = [2, 3]
```

### How it works under the hood

* Uses optimized LAPACK routines (e.g. `dgesv`) to perform LU factorization and forward/back substitution in $O(n^3)$.

### Real-World Scenario

> **Problem:** Calibrating a sensor network: you have linear equations relating sensor offsets to observed errors, and you need the offset vector $x$.
> **Solution:** Pack your coefficients into `A`, observations into `b`, and call `solve(A, b)` to get all offsets in one step.

---

## 6. Least Squares & Pseudoinverse

### What it is

* When $A x = b$ has no exact solution (overdetermined), the **least-squares** solution minimizes $\|A x - b\|_2$.
* The **Moore–Penrose pseudoinverse** $A^+$ generalizes inversion: $x = A^+ b$.

### Python Code

```python
from numpy.linalg import pinv
from scipy.linalg import lstsq

# Overdetermined system: more equations than unknowns
A = np.array([[1, 1],
              [1, 2],
              [1, 3]])
b = np.array([6, 0, 0])

# Method 1: pseudoinverse
x_pinv = pinv(A) @ b

# Method 2: direct least-squares
x_ls, residuals, rank, s = lstsq(A, b)
```

### How it works under the hood

* **Least-squares** via QR decomposition or SVD to solve $\min_x \|A x - b\|$.
* **Pseudoinverse** computed from the SVD: $A = UΣV^T ⇒ A^+ = VΣ^+U^T$ (where Σ⁺ inverts nonzero singular values).

### Real-World Scenario

> **Problem:** Fitting a linear regression model with more data points than features.
> **Solution:**
>
> ```python
> # X: (n_samples, n_features), y: (n_samples,)
> w_opt = np.linalg.pinv(X) @ y
> ```
>
> This gives the best-fit weights that minimize mean squared error.

---

## 7. Eigenvalues/Vectors & SVD

### What it is

* **Eigen-decomposition**: $A v = λ v$. Yields eigenvalues $λ$ and eigenvectors $v$.
* **SVD**: $A = UΣV^T$, where columns of $U$, $V$ are orthonormal and Σ holds singular values.

### Python Code

```python
# Eigen-decomposition
eigvals, eigvecs = np.linalg.eig(A)

# Singular Value Decomposition
U, S, Vt = np.linalg.svd(A, full_matrices=False)
```

### How it works under the hood

* Eigen: QR algorithm iteratively shifts and orthogonalizes to converge to eigenvalues.
* SVD: bidiagonalization + iterative methods to extract singular values/vectors.

### Real-World Scenario

> **Problem:** Dimensionality reduction via PCA on feature matrix `X`.
> **Solution:**
>
> ```python
> # Center data
> Xc = X - X.mean(axis=0)
> # Compute covariance
> cov = Xc.T @ Xc / (Xc.shape[0] - 1)
> # Eigen-decomp of covariance
> vals, vecs = np.linalg.eig(cov)
> # Project onto top k components
> X_pca = Xc @ vecs[:, :k]
> ```
>
> Or, more efficiently, use SVD directly on `Xc`.

---


**Next up:**
Putting It All Together: End-to-End Mini Project (data generation → regression → PCA → reconstruction).



---

## 8. Gradient Computations & Matrix Calculus

### What it is

* **Gradient**: Vector of partial derivatives of a scalar loss $L$ with respect to parameters $W$.
* **Jacobian**: Matrix of partial derivatives of a vector-valued function.
* **Hessian**: Matrix of second derivatives—used in optimization diagnostics.

### How it works under the hood

* Given a loss $L(W) = \tfrac12\|XW - y\|^2$, its gradient is

  $$
    \nabla_W L = X^T (XW - y).
  $$
* You can derive this by applying the chain rule:

  $$
    \frac{\partial}{\partial W} (XW - y)^T (XW - y)
    = 2\,X^T (XW - y).
  $$

### Python Code (Manual)

```python
import numpy as np

# Data
X = np.random.randn(50, 5)
true_w = np.arange(1, 6)
y = X @ true_w + 0.1 * np.random.randn(50)

# Initialize weights
W = np.zeros(5)

# Compute loss and gradient
def loss_and_grad(W, X, y):
    pred = X @ W
    residual = pred - y
    loss = 0.5 * np.mean(residual**2)
    grad = (X.T @ residual) / X.shape[0]
    return loss, grad

# Single step of gradient descent
lr = 0.1
loss, grad = loss_and_grad(W, X, y)
W -= lr * grad
print(f"Loss: {loss:.4f}, Gradient norm: {np.linalg.norm(grad):.4f}")
```

### Real‑World Scenario

> **Problem:** Training a linear regression “layer” by gradient descent instead of solving in closed form.
> **Insight:** You see that each update is a combination of matrix–vector products—the same batched operations that power deep‑network backprop.

---

## 9. GPU Acceleration with CuPy & JAX

### What it is

* **CuPy**: NumPy‑compatible library that runs on NVIDIA GPUs.
* **JAX**: Google library offering both GPU/TPU‑backed array operations and automatic differentiation.

### Python Code (CuPy)

```python
import cupy as cp

# Transfer data to GPU
X_gpu = cp.asarray(X)
y_gpu = cp.asarray(y)
W_gpu = cp.zeros(5)

# One step of gradient descent on GPU
grad_gpu = (X_gpu.T @ (X_gpu @ W_gpu - y_gpu)) / X_gpu.shape[0]
W_gpu -= lr * grad_gpu

# Bring back to CPU if needed
W_new = cp.asnumpy(W_gpu)
```

### Python Code (JAX)

```python
import jax.numpy as jnp
from jax import grad

# Define loss in JAX
def jax_loss(W, X, y):
    pred = X @ W
    return 0.5 * jnp.mean((pred - y)**2)

# Data on device
X_jax = jnp.array(X)
y_jax = jnp.array(y)
W_jax = jnp.zeros(5)

# Compute gradient function automatically
grad_fn = grad(jax_loss)

# Single update
g = grad_fn(W_jax, X_jax, y_jax)
W_jax = W_jax - lr * g
```

### Real‑World Scenario

> **Problem:** You have a large dataset and high‑dimensional parameters—CPU alone is too slow.
> **Solution:** Swap in CuPy or JAX for all NumPy calls to transparently accelerate batches on GPU, and use JAX’s `grad` for backprop without writing any derivative code yourself.

---

## 10. Practical Tips & Pitfalls

1. **Vectorize Everything**

   * Avoid Python loops over samples or features; rely on batched matrix ops.

2. **Watch Your Shapes**

   * Keep track of dimensions; mismatches are the most common errors.

3. **Condition Numbers & Regularization**

   * Compute `np.linalg.cond(A)` to detect ill‑conditioning.
   * For regression, add a small ridge term:

     $$
       w = (X^T X + λI)^{-1} X^T y.
     $$

4. **Initialization Tricks**

   * Use orthogonal or Xavier/He initializations based on matrix SVD insights to stabilize deep networks.

5. **Profiling & BLAS Tuning**

   * Ensure you’re using a high‑performance BLAS (like Intel MKL) for NumPy/SciPy.
   * Profile with `%timeit` or line profilers to spot slow operations.

---

## 11. Further Resources

* **Books & Tutorials**

  * *Deep Learning* by Goodfellow, Bengio & Courville (Chapters on linear algebra & optimization)
  * Stanford’s CS231n lecture notes (Matrix calculus appendix)

* **Online Courses**

  * MIT 18.06 Linear Algebra (Gilbert Strang)
  * JAX official tutorials on automatic differentiation

* **Libraries to Explore**

  * **PyTorch**: Another GPU‑backed tensor library with `autograd`
  * **TensorFlow**: Built‑in linear algebra ops and auto‑diff

---
