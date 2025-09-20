#!/usr/bin/env python
# coding: utf-8

# # Performance Benchmark: Serial vs. Parallel `pdist_tangency`

# In[1]:


import numpy as np
import timeit
import pandas as pd
import matplotlib.pyplot as plt

from ellphi.ellcloud import EllipseCloud
from ellphi.solver import pdist_tangency
from ellphi.geometry import coef_from_cov


# In[2]:


def generate_ellipses(n_ellipses, seed=42):
    np.random.seed(seed)
    means = np.random.rand(n_ellipses, 2) * 100
    covs_list = []
    for _ in range(n_ellipses):
        a = np.random.rand() * 5 + 1
        b = np.random.rand() * 5 + 1
        angle = np.random.rand() * np.pi
        rot = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        cov = rot @ np.diag([a, b]) @ rot.T
        covs_list.append(cov)
    covs = np.array(covs_list)
    coefs = coef_from_cov(means, covs)
    # Create dummy nbd and k, as they are not used in pdist_tangency
    dummy_nbd = np.array([[] for _ in range(n_ellipses)])
    return EllipseCloud(coef=coefs, mean=means, cov=covs, k=0, nbd=dummy_nbd)


# In[3]:


ellipse_counts = [10, 50, 100, 150, 200]
results = []

for n in ellipse_counts:
    print(f"Running benchmark for {n} ellipses...")
    ellipses = generate_ellipses(n)

    # Time serial execution
    serial_time = (
        timeit.timeit(lambda: pdist_tangency(ellipses, parallel=False), number=5) / 5
    )

    # Time parallel execution
    parallel_time = (
        timeit.timeit(
            lambda: pdist_tangency(ellipses, parallel=True, n_jobs=-1), number=5
        )
        / 5
    )

    results.append(
        {"n_ellipses": n, "serial_time": serial_time, "parallel_time": parallel_time}
    )

df_results = pd.DataFrame(results)
print(df_results)


# In[4]:


plt.figure(figsize=(10, 6))
plt.plot(
    df_results["n_ellipses"],
    df_results["serial_time"],
    marker="o",
    label="Serial",
)
plt.plot(
    df_results["n_ellipses"],
    df_results["parallel_time"],
    marker="o",
    label="Parallel (all cores)",
)
plt.title("pdist_tangency Performance: Serial vs. Parallel")
plt.xlabel("Number of Ellipses")
plt.ylabel("Execution Time (seconds)")
plt.legend()
plt.grid(True)
plt.show()
