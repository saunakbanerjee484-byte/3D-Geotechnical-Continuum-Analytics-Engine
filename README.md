# 🌌 3D Geotechnical Continuum Analytics Engine

A high-performance, 100% user-driven multi-physics simulation and geotechnical engineering framework built on top of **NumPy** and **SciPy**. This engine completely decouples from algebraic dependencies (0% SymPy) and operates on vectorized matrices, allowing complex 3D stress field computation and consolidation mechanics to run efficiently on standard hardware (e.g., Core i3 setups).

---

## 🛠️ Key Architectural Features

1. **Flexible Soil Classification (IS:1498 / USCS):**
   * Processes laboratory sieve parameters (% passing 75μm and 4.75mm) and Atterberg limits ($LL$, $PL$) using nested logical filters.
   * Features interactive calibration loops allowing engineers to choose typical empirical configurations or seed exact custom laboratory parameters ($c, \phi$).

2. **Multi-Variable Footing Capacity Optimizer:**
   * Calculates specific footing profiles (Square, Rectangular, and Circular) with precise analytical shape, depth, and inclination adjustments based on IS:6403 formulations.
   * Incorporates a dynamic ground water table vector ($W'$) to automatically offset safe structural loading boundaries.

3. **Multi-Footing Adjacent Stress Bulb Tensor Solver:**
   * Simulates real-world construction limits by mapping dual adjacent footing geometries and horizontal placement spacing.
   * Computes complex multi-axis overlapping vertical stress tensors ($\sigma_{zz}$) using an interactive boundary selection:
     * **Boussinesq Continuum:** For isotropic/granular soil models.
     * **Westergaard Field:** For highly stratified clay strata.
   * Utilizes rapid matrix slicing techniques with zero loop overhead.

4. **Decoupled Dual-Phase Consolidation Timeline Solver:**
   * Resolves hydrostatic dissipation matrices using a strict Finite Difference Method (FDM) time-stepping setup ($\alpha = \frac{C_v \cdot dt}{dz^2}$).
   * Isolates the structural settlement mechanics into two distinct graphics lines:
     * **Primary Consolidation Phase:** Driven by excess pore water pressure dissipation.
     * **Secondary Compression (Plastic Creep Phase):** Re-activates at the 98% dissipation boundary using multi-year logarithmic skeleton deformation equations.

---

## 📊 3-Panel Research Dashboard Visualization

The system generates a synchronous, aligned interactive Matplotlib control layout mapping three deep continuous metrics at once:
* **Panel 1 (Stress Bulb):** Color-mapped contours showing localized stress field propagation and structural proximity intersection boundaries.
* **Panel 2 (Primary Stage):** Tracks immediate liquid pressure decay versus structural dhasan (settlement in mm).
* **Panel 3 (Secondary Creep):** A separate dedicated log-timeline visualizing long-term skeletal plastic creeping over a 50-year horizon.

---

## 💻 Prerequisites & Local Deployment

Ensure you have your environment ready to handle core vectorized matrix arrays:

```bash
pip install numpy scipy matplotlib
