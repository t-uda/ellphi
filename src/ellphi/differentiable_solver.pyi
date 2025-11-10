import numpy as np
from .solver import MethodName

__all__ = ['solve_mu_gradients', 'solve_mu_numerical_diff']

def solve_mu_numerical_diff(p: np.ndarray, q: np.ndarray, h: float = 1e-06) -> tuple[np.ndarray, np.ndarray]: ...
def solve_mu_gradients(p: np.ndarray, q: np.ndarray, *, mu: float | None = None, method: MethodName = 'brentq+newton', bracket: tuple[float, float] = (0.0, 1.0), x0: float | None = None) -> tuple[float, np.ndarray, np.ndarray]: ...
