# Algebraic Sigmoid + Newton Strategy

This document outlines the `algebraic-sigmoid+newton` strategy, a root-finding algorithm designed for functions on the interval $(0, 1)$. This method is particularly useful for objective functions, such as the difference of Mahalanobis distances, where the scale can vary dramatically at the boundaries, causing standard Newton methods to become unstable.

To ensure that the optimization variable remains within the $(0, 1)$ domain, this strategy employs a variable transformation known as the **Algebraic Sigmoid function**. This reparameterization allows Newton's method to be applied in an unbounded space, enhancing stability.

## 1. Mathematical Formulation

### 1.1. Variable Transformation

The core of the strategy is a mapping from the unbounded real line $u 
 \in \mathbb{R}$ to the interval $x 
 \in (0, 1)$. This is achieved using the algebraic sigmoid function:

$$
 x(u) = \frac{1}{2} \left( 1 + \frac{u}{\sqrt{1 + u^2}} \right) 
$$

This function smoothly maps $(-\infty, \infty)$ to $(0, 1)$ and has a fixed point at $x(0) = 0.5$.

### 1.2. Newton's Method in the Transformed Space

Let the original objective function be $f(x)$, defined on $x 
 \in (0, 1)$. We define a new function $F(u)$ in the transformed space:

$$ 
 F(u) = f(x(u)) 
$$

To apply Newton's method to $F(u)$, we require its derivative, which is computed using the chain rule:

$$ 
 F'(u) = \frac{dF}{du} = f'(x(u)) \cdot \frac{dx}{du} 
$$

The derivative of the transformation, $x'(u)$, is:

$$ 
 x'(u) = \frac{dx}{du} = \frac{1}{2} (1 + u^2)^{-3/2} 
$$

This derivative decays polynomially, which makes it more robust against the vanishing gradient problem compared to exponential-based transforms like `tanh`.

The Newton update rule in the $u$-space is then:

$$ 
 u_{new} = u_{old} - \frac{F(u_{old})}{F'(u_{old})} = u_{old} - \frac{f(x(u_{old}))}{f'(x(u_{old})) \cdot x'(u_{old})} 
$$

The iteration starts with an initial value of $u=0$, which corresponds to $x=0.5$.

## 2. Implementation Details and Numerical Stability

### 2.1. Backtracking Line Search

A naive application of the Newton step can still be unstable if the function landscape is very flat, causing the derivative $F'(u)$ to be close to zero and the step size to become excessively large. To mitigate this, a **Backtracking Line Search** is implemented.

1.  **Calculate the full Newton step:**
    $$ 
     \Delta u = - \frac{F(u)}{F'(u)} 
    $$

2.  **Iteratively find an acceptable step size:** Starting with a step coefficient $\alpha = 1.0$, the algorithm checks if the candidate point $u_{next} = u + \alpha \Delta u$ satisfies the Armijo condition (i.e., results in a sufficient decrease in the residual):
    $$ 
     |f(x(u_{next}))| < |f(x(u))| 
    $$

3.  If the condition is not met, $\alpha$ is reduced (e.g., $\alpha \leftarrow 0.5 \alpha$), and the check is repeated. This process ensures that each step leads to progress without overshooting.

### 2.2. Numerical Safeguards

-   **Boundary Clipping:** The implementation of $x(u)$ includes a small epsilon-clipping to prevent the value of $x$ from becoming exactly $0$ or $1$ due to floating-point limitations. This avoids `NaN` results from downstream function evaluations (e.g., `_target(x)`).

-   **Computational Guard:** Although the backtracking logic provides stability, as an ultimate safeguard against extreme floating-point behavior, the value of $u$ is clamped to a large range (e.g., $[-10^7, 10^7]$) after each update. This prevents $u$ from growing to a magnitude where `u**2` would cause an `OverflowError`.

By combining the algebraic sigmoid transformation with these robustification techniques, the solver can reliably find roots even for challenging functions with extreme curvature near the boundaries.
