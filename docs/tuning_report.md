# Hybrid Backend Tuning Report (Final)

## Executive Summary
Based on density plot distribution analysis with clearly separated method groups:
- **Normal Low-Dim**: **Hybrid** superior (1e-15 precision vs Brent's 1e-10~1e-11 tail)
- **Extreme Low-Dim**: **Brent** required (Hybrid 37% failure)
- **Normal High-Dim**: **Hybrid** slightly better precision, both acceptable
- **Extreme High-Dim**: **Brent** only (Hybrid 98% failure)

**Color Legend**: Hybrid (Blue), Brent (Green), Other/Bisect/Newton (Gray)

---

## Scenario Analysis

### 1. Normal Low-Dim (2D, 3D) - **Hybrid Superior**

![Normal Low-Dim](file:///Users/uda/.gemini/antigravity/brain/5d4165cf-cb3a-437e-ada1-6a3abbdadccb/normal_lowdim-hybrid_time_vs_error_density_python.png)

**Distribution Observations**:
- **Hybrid (Blue)**: Extremely tight concentration at **1e-15 error**. Clean, compact distribution.
  - Both `hybrid_8x3` and `hybrid_28x3` show nearly identical behavior (plotted together)
- **Brent (Green)**: Primary concentration around 1e-14, **BUT with visible tail extending to 1e-10~1e-11**. This tail represents a significant fraction of cases.
- **Other/Bisect (Gray)**: Clearly visible at 1e-11, slower time

**Critical Finding**: Brent shows a **problematic tail at 1e-10~1e-11**, which represents cases with insufficient precision. Hybrid avoids this tail entirely.

**Numerical Summary**:
| Method | Median Error | Error Distribution | Median Time | Failures |
|:---|---:|:---|---:|---:|
| Hybrid 8x3 | 1.3e-15 | Tight at 1e-15 | 0.777ms | 0/10000 |
| Hybrid 28x3 | 1.3e-15 | Tight at 1e-15 | 0.774ms | 0/10000 |
| Brentq | 7.4e-15 | 1e-14 with 1e-10~1e-11 tail | 0.59ms | 0/10000 |

**Recommendation**: **`hybrid_8x3`** or **`hybrid_28x3`** (nearly identical) - The 1e-10~1e-11 tail in Brent is unacceptable for precision-critical applications. The speed advantage of Brent is negligible.

---

### 2. Extreme Low-Dim (2D, 3D) - **Brent Required**

![Extreme Low-Dim](file:///Users/uda/.gemini/antigravity/brain/5d4165cf-cb3a-437e-ada1-6a3abbdadccb/extreme_lowdim-hybrid_time_vs_error_density_python.png)

**Distribution Observations**:
- **Hybrid (Blue)**: Sparse distribution at 1e-14 (only successful cases shown) - **37% failure rate**
- **Brent (Green)**: Dense, reliable distribution at 1e-13~1e-14 with **0% failures**

**Recommendation**: **`brentq`** - 37% failure rate is catastrophic

---

### 3. Normal High-Dim (10D, 20D) - **Hybrid Slightly Better**

![Normal High-Dim](file:///Users/uda/.gemini/antigravity/brain/5d4165cf-cb3a-437e-ada1-6a3abbdadccb/normal_highdim-hybrid_time_vs_error_density_python.png)

**Distribution Observations**:
- **Hybrid (Blue)**: Tight at **1e-15**
  - Both `hybrid_8x3` and `hybrid_28x3` show nearly identical behavior
- **Brent (Green)**: Main distribution at 1e-14~1e-15, less tail than in Low-Dim case

**Numerical Summary**:
| Method | Median Error | Median Time | Failures |
|:---|---:|---:|---:|
| Hybrid 8x3 | 1.3e-15 | 0.855ms | 1/10000 (0.01%) |
| Hybrid 28x3 | 1.3e-15 | 0.869ms | 1/10000 (0.01%) |
| Brentq | 5.2e-15 | 0.68ms | 0/10000 |

**Recommendation**: **`hybrid_8x3`** (or `hybrid_28x3`, nearly identical) for precision, **`brentq`** for speed. Both acceptable (0.01% failure negligible).

---

### 4. Extreme High-Dim (10D, 20D) - **Brent Only**

![Extreme High-Dim](file:///Users/uda/.gemini/antigravity/brain/5d4165cf-cb3a-437e-ada1-6a3abbdadccb/extreme_highdim-hybrid_time_vs_error_density_python.png)

**Distribution Observations**:
- **Hybrid**: 98% failure - not viable
- **Brent (Green)**: Robust distribution at 1e-13

**Recommendation**: **`brentq`**

---

## Final Recommendation

### **Primary: `hybrid_8x3` with `brentq` fallback**

**Rationale**:
1. **Normal cases dominate** real-world usage
2. **Hybrid avoids 1e-10~1e-11 tail** seen in Brent (Normal Low-Dim)
3. **Fallback ensures robustness** for extreme geometries
4. **Speed difference is acceptable**: 0.76ms vs 0.59ms (~28%) is negligible

### Implementation
```python
def tangency_optimized(p, q, **kwargs):
    """Hybrid with fallback for robustness."""
    try:
        result = tangency(p, q, method="brentq+newton",
                         hybrid_bracket_maxiter=8,
                         hybrid_newton_maxiter=3, **kwargs)
        if result.converged:
            return result
    except:
        pass
    # Fallback for extreme cases
    return tangency(p, q, method="brentq", **kwargs)
```

### Alternative: Conservative
- **Default**: `brentq` everywhere
- **Trade-off**: Accept 1e-10~1e-11 tail in Normal Low-Dim for simplicity

---

## Summary Table

| Scenario | Hybrid 8x3 | Hybrid 28x3 | Brentq | Winner |
|:---|:---|:---|:---|:---|
| Normal Low-Dim | 1e-15, 0.78ms, 0% fail | 1e-15, 0.77ms, 0% fail | 1e-14 w/ **1e-10 tail**, 0.59ms, 0% fail | **Hybrid** |
| Extreme Low-Dim | 1e-14, **37% fail** | 1e-14, **37% fail** | 1e-13, 0% fail | **Brent** |
| Normal High-Dim | 1e-15, 0.86ms, 0.01% fail | 1e-15, 0.87ms, 0.01% fail | 1e-14, 0.68ms, 0% fail | **Hybrid** (precision) |
| Extreme High-Dim | **98% fail** | **97% fail** | 1e-13, 0% fail | **Brent** |

**Conclusion**: Hybrid (8x3 or 28x3 are equivalent) with Brent fallback provides optimal precision for normal cases while maintaining robustness.
