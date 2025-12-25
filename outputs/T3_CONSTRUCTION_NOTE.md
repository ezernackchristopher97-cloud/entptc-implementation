# T³ Construction Implementation Note

**Date**: 2025-12-24 

---

## Overview

The EntPTC model requires a **3-torus (T³)** state space, NOT a 2-torus (T²). specifies the exact construction of T³ coordinates (θ_x, θ_y, θ_z) from EEG/fMRI data.

---

## T³ Coordinate Construction

### Definition

T³ is the 3-dimensional torus, parameterized by three angular coordinates:

**T³ = S¹ × S¹ × S¹**

where each S¹ is a circle (1-sphere) with angular coordinate in [0, 2π).

### Coordinate Extraction from Neural Data

Each angular coordinate is derived from **phase dynamics at different timescales**:

#### θ₁: Sub-delta phase (0.14-0.33 Hz)
- **Slowest timescale** - EntPTC control frequency from Stage B
- Extracted via bandpass filter (0.14-0.33 Hz) + Hilbert transform
- Represents **global coherence mode**

#### θ₂: Delta phase (0.5-4.0 Hz)
- **Intermediate timescale** - classical delta band
- Extracted via bandpass filter (0.5-4.0 Hz) + Hilbert transform
- Represents **regional coordination mode**

#### θ₃: Theta phase (4.0-8.0 Hz)
- **Fastest timescale** - hippocampal theta rhythm
- Extracted via bandpass filter (4.0-8.0 Hz) + Hilbert transform
- Represents **local oscillatory mode**

### Mathematical Formulation

For each ROI *i* and time *t*:

1. **Bandpass filter** the signal x_i(t) in each frequency band:
 - x_{i,1}(t) = BP_{0.14-0.33}[x_i(t)]
 - x_{i,2}(t) = BP_{0.5-4.0}[x_i(t)]
 - x_{i,3}(t) = BP_{4.0-8.0}[x_i(t)]

2. **Hilbert transform** to get analytic signal:
 - z_{i,k}(t) = x_{i,k}(t) + j · H[x_{i,k}(t)]

3. **Extract instantaneous phase**:
 - θ_{i,k}(t) = arg(z_{i,k}(t)) ∈ [-π, π]

4. **Map to [0, 2π)**:
 - θ_{i,k}(t) ← (θ_{i,k}(t) + 2π) mod 2π

### Result

For each ROI *i* at time *t*, the analysis a point on T³:

**(θ_{i,1}(t), θ_{i,2}(t), θ_{i,3}(t)) ∈ T³**

---

## T³ → R³ Projection

After constructing T³ coordinates, projecting to R³ for visualization and invariant computation.

### Projection Types

#### 1. Stereographic Projection (Default)

Conformal mapping that preserves angular relationships:

```
x = sin(θ₁) · cos(θ₂)
y = sin(θ₁) · sin(θ₂)
z = cos(θ₁) · sin(θ₃)
```

**Properties**:
- Conformal (preserves angles)
- Smooth embedding
- Good for phase relationship analysis

#### 2. Cylindrical Projection

Preserves one angular coordinate directly:

```
x = θ₁ / (2π) (normalized)
y = cos(θ₂)
z = sin(θ₃)
```

**Properties**:
- One dimension is linear (θ₁)
- Good for temporal sequence visualization
- Simpler geometry

#### 3. Embedding Projection (via PCA)

Standard torus embedding in R⁶, then PCA to R³:

```
R⁶ embedding:
v = [cos(θ₁), sin(θ₁), cos(θ₂), sin(θ₂), cos(θ₃), sin(θ₃)]

R³ projection:
(x, y, z) = PCA₃(v)
```

**Properties**:
- Preserves maximal variance
- Data-driven projection
- Good for exploratory analysis

---

## Normalization

After projection to R³, coordinates are normalized using one of three methods:

### 1. Unit Sphere (Default)

Project onto unit sphere (preserves angles):

```
(x, y, z) ← (x, y, z) / ||(x, y, z)||
```

### 2. Unit Variance

Standardize to zero mean, unit variance per dimension:

```
x ← (x - μ_x) / σ_x
y ← (y - μ_y) / σ_y
z ← (z - μ_z) / σ_z
```

### 3. Unit Cube

Scale to [0, 1]³:

```
x ← (x - min(x)) / (max(x) - min(x))
y ← (y - min(y)) / (max(y) - min(y))
z ← (z - min(z)) / (max(z) - min(z))
```

---

## Topology Verification

After constructing T³ coordinates, verifying proper toroidal topology:

### 1. Angular Coverage

Each dimension should span [0, 2π):

```
Coverage_k = (max(θ_k) - min(θ_k)) / (2π) × 100%
```

**Expected**: > 90% for each dimension

### 2. Circular Variance

Measures uniformity of phase distribution:

```
CircVar_k = 1 - |mean(exp(j·θ_k))|
```

**Expected**: > 0.5 (not concentrated at one phase)

### 3. Cross-Dimensional Coupling

Measures independence of dimensions:

```
Coupling_{k,l} = |mean(exp(j·(θ_k - θ_l)))|
```

**Expected**: < 0.3 (weakly coupled, not identical)

---

## Invariant Computation

### On T³ (Angular Coordinates)

1. **Phase Velocity**: |dθ_k/dt| for each dimension
2. **Phase Winding**: Circulation around adjacent ROIs
3. **Trajectory Curvature**: |d²θ_k/dt²|
4. **Entropy Flow**: Rate of change of phase entropy

### On R³ (Projected Coordinates)

1. **Trajectory Length**: Integrated path length
2. **Spatial Spread**: Variance in each dimension
3. **Neighbor Distance**: Mean distance to adjacent ROIs
4. **Trajectory Alignment**: Correlation of velocity vectors

---

## Implementation

### Code Location

`/home/ubuntu/entptc-implementation/entptc/t3_to_r3_mapping.py`

### Main Function

```python
from entptc.t3_to_r3_mapping import entptc_t3_to_r3_pipeline

results = entptc_t3_to_r3_pipeline(
 data, # (n_rois, n_samples) array
 fs, # sampling rate (Hz)
 adjacency, # (n_rois, n_rois) adjacency matrix
 projection_type='stereographic', # or 'cylindrical', 'embedding'
 normalization='unit_variance' # or 'unit_sphere', 'unit_cube'
)
```

### Returns

```python
{
 't3_coords': (3, n_rois, n_samples), # T³ coordinates
 'r3_coords': (3, n_rois, n_samples), # R³ projection
 't3_verification': {...}, # Topology checks
 't3_invariants': {...}, # Invariants on T³
 'r3_invariants': {...} # Invariants on R³
}
```

---

## Relation to Grid Cells

### Stage A: Grid Cell T² Mapping

In Stage A, extracting T² (2-torus) coordinates from grid cell hexagonal firing patterns:

- θ_x: Phase along x-axis of grid
- θ_y: Phase along y-axis of grid

This is **naturally T²** because grid cells tile 2D space.

### Stage C: EEG/fMRI T³ Extension

In Stage C, extending to T³ by adding a third dimension:

- θ₁: Sub-delta phase (EntPTC control frequency)
- θ₂: Delta phase (regional coordination)
- θ₃: Theta phase (local oscillations)

This is **T³** because the analysis **three independent phase coordinates**.

### Why T³, Not T²?

The EntPTC model proposes that **experiential coherence** emerges from **T³ dynamics**, not just T²:

1. **Grid cells provide T² foundation** (spatial navigation)
2. **Temporal dynamics add third dimension** (oscillatory hierarchy)
3. **T³ → R³ projection** captures full experiential manifold

---

## Validation Criteria

### T³ Construction is Valid If:

1. ✅ **Angular coverage** > 90% in all three dimensions
2. ✅ **Circular variance** > 0.5 in all three dimensions
3. ✅ **Cross-coupling** < 0.3 between dimensions (weakly independent)
4. ✅ **Invariants stable** across projections (stereographic vs cylindrical vs embedding)

### T³ Construction Fails If:

1. ❌ Any dimension has < 50% angular coverage (collapsed to point)
2. ❌ Any dimension has circular variance < 0.2 (concentrated at one phase)
3. ❌ Cross-coupling > 0.7 between dimensions (redundant, not T³)
4. ❌ Invariants change > 50% across projections (projection-dependent, not intrinsic)

---

## Commit Hash

Implementation committed at: `6a3c5b4`

---

## References

1. EntPTC.tex (mathematical specification)
2. Hafting et al. (2005) - Grid cell hexagonal firing patterns
3. Buzsáki & Draguhn (2004) - Neuronal oscillations in cortical networks
4. Skaggs et al. (1996) - Theta phase precession in hippocampal place cells

---

**End of T³ Construction Implementation Note**
