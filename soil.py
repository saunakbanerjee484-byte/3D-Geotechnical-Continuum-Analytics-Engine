"""
3D Geotechnical Continuum Analytics Engine (Research Edition)
Modules Included:
  1. USCS / IS Soil Classification Logic Engine (Seeding Module)
  2. Multi-Variable Bearing Capacity Engine (Dynamic Footing Shape Factors)
  3. 3D Adjacent Footing Stress Bulb Overlap Tensor Engine (NumPy Continuum Mesh)
  4. Terzaghi 1D/3D Consolidation Solver via FDM Time-Stepping Loop (Time vs Settlement)
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

class UltimateGeotechEngine:
    def __init__(self):
        print("\n" + "="*70)
        print(" 🌌 WELCOME TO THE INTEGRATED ADVANCED CONTINUUM & FDM TIME SOLVER 🌌")
        print("="*70)
        self.soil_tag = "Unknown"
        self.cohesion = 0.0
        self.phi = 0.0
        self.gamma = 18.0
        self.q_nu = 0.0
        self.mv = 0.0005  # Coefficient of volume compressibility (m2/kN) - default seed
        self.cv = 2.5     # Coefficient of consolidation (m2/year) - default seed

    # =========================================================================
    # MODULE 1: SOIL CLASSIFICATION ENGINE
    # =========================================================================
    def run_soil_classifier(self):
        print("\n--- [MODULE 1: SOIL CLASSIFICATION GRID] ---")
        try:
            p_75u = float(input("Enter % passing 75-micron sieve (0-100): "))
            ll = float(input("Enter Liquid Limit (LL) in %: "))
            pl = float(input("Enter Plastic Limit (PL) in %: "))
            
            pi = ll - pl
            a_line = 0.73 * (ll - 20.0)
            
            if p_75u < 50.0:
                print(" -> Category identified: Coarse-Grained Soil")
                p_4_75m = float(input("Enter % passing 4.75mm sieve (0-100): "))
                if p_4_75m >= 50.0:
                    if pi > a_line and pi > 7: self.soil_tag = "SC (Clayey Sand)"
                    elif pi < a_line or pi < 4: self.soil_tag = "SM (Silty Sand)"
                    else: self.soil_tag = "SC-SM (Dual Sand)"
                else:
                    if pi > a_line and pi > 7: self.soil_tag = "GC (Clayey Gravel)"
                    elif pi < a_line or pi < 4: self.soil_tag = "GM (Silty Gravel)"
                    else: self.soil_tag = "GC-GM (Dual Gravel)"
            else:
                print(" -> Category identified: Fine-Grained Soil")
                if ll < 35.0:
                    if pi > a_line and pi > 7: self.soil_tag = "CL (Low Plastic Clay)"
                    else: self.soil_tag = "ML (Low Plastic Silt)"
                elif 35.0 <= ll <= 50.0:
                    if pi > a_line and pi > 7: self.soil_tag = "CI (Intermediate Clay)"
                    else: self.soil_tag = "MI (Intermediate Silt)"
                else:
                    if pi > a_line and pi > 7: self.soil_tag = "CH (Highly Plastic Clay)"
                    else: self.soil_tag = "MH (Highly Plastic Silt)"

            print(f"✅ Soil Tag Assigned: {self.soil_tag}")
            
            # Smart Seeding for Advanced Mechanics Layers
            if "C" in self.soil_tag:
                self.cohesion, self.phi = 30.0, 10.0
                self.mv, self.cv = 0.0008, 1.2  # Compressible highly active clay matrix seed
            else:
                self.cohesion, self.phi = 2.0, 32.0
                self.mv, self.cv = 0.0001, 8.5  # Rapid drainage granular matrix seed
                
            return p_75u, ll, pl
        except ValueError:
            print("❌ Input validation error.")
            return None

    # =========================================================================
    # MODULE 2: DYNAMIC SHAPE BEARING CAPACITY OPTIMIZER
    # =========================================================================
    def compute_bearing_capacity(self):
        print("\n--- [MODULE 2: BEARING CAPACITY SHAPE SOLVER] ---")
        try:
            print("Select Footing Geometry Profile:")
            print("1. Square Footing\n2. Rectangular Footing\n3. Circular Footing")
            shape_choice = input("Enter choice (1/2/3): ")

            if shape_choice == "1":
                self.footing_shape = "Square"
                b = float(input("Enter Width B (m): ")); l = b
                s_c, s_q, s_gamma = 1.3, 1.2, 0.8
            elif shape_choice == "2":
                self.footing_shape = "Rectangular"
                b = float(input("Enter Width B (m): "))
                l = float(input("Enter Length L (m): "))
                s_c, s_q = 1.0 + 0.2*(b/l), 1.0 + 0.2*(b/l)
                s_gamma = 1.0 - 0.4*(b/l)
            else:
                self.footing_shape = "Circular"
                b = float(input("Enter Diameter D (m): ")); l = b
                s_c, s_q, s_gamma = 1.3, 1.2, 0.6

            d_f = float(input("Enter Depth Df (m): "))
            d_w = float(input("Enter Water Table Depth (m): "))
            
            # Analytical Factor Mechanics (IS:6403 Equations)
            phi_rad = np.radians(self.phi)
            if self.phi == 0:
                n_c, n_q, n_gamma = 5.14, 1.0, 0.0
            else:
                n_q = np.exp(np.pi * np.tan(phi_rad)) * (np.tan(np.radians(45) + phi_rad/2.0)**2)
                n_c = (n_q - 1.0) / np.tan(phi_rad)
                n_gamma = 1.4 * (n_q - 1.0) * np.tan(phi_rad)

            w_prime = 0.5 if d_w <= d_f else (0.5 + 0.5*((d_w-d_f)/b) if d_f < d_w < (d_f+b) else 1.0)
            
            self.q_nu = (self.cohesion * n_c * s_c) + (self.gamma * d_f * (n_q - 1.0) * s_q) + (0.5 * self.gamma * b * n_gamma * s_gamma * w_prime)
            print(f"✅ Net Ultimate Bearing Capacity (q_nu): {self.q_nu:.2f} kPa")
            return b, l, d_f
        except Exception as e:
            print(f"❌ Computation fault in Layer 2: {e}")
            return None

    # =========================================================================
    # MODULE 3: ADJACENT MULTI-FOOTING STRESS BULB OVERLAP ENERGETICS
    # =========================================================================
    def execute_adjacent_stress_overlap(self, b1, l1, d_f):
        print("\n--- [MODULE 3: MULTI-FOOTING ADJACENT OVERLAP SOLVER] ---")
        try:
            s_load1 = float(input("Enter Load for Footing 1 (kPa): "))
            spacing = float(input("Enter Center-to-Center Spacing between Footings (meters): "))
            s_load2 = float(input("Enter Load for Adjacent Footing 2 (kPa): "))
            
            # Setting up our cross-sectional spatial monitoring space grid mesh (Y = 0 cut plane)
            x_grid = np.linspace(-10.0, 10.0, 100)
            z_grid = np.linspace(d_f + 0.1, 12.0, 100) # Deep profile mapping for FDM
            X, Z = np.meshgrid(x_grid, z_grid)
            
            # Equivalent point loads from distributed pressure areas
            p1 = s_load1 * (b1 * l1)
            p2 = s_load2 * (b1 * l1) # Assuming uniform geometries for proximity mapping
            
            # Position coordinates vectors tracking for both structures boundary arrays
            x_f1 = -spacing / 2.0
            x_f2 = spacing / 2.0
            
            R1 = np.sqrt((X - x_f1)**2 + Z**2)
            R2 = np.sqrt((X - x_f2)**2 + Z**2)
            
            # Boussinesq Fields Calculations Vectorization
            with np.errstate(divide='ignore', invalid='ignore'):
                sigma_z1 = (3.0 * p1 / (2.0 * np.pi * Z**2)) * ((Z / R1) ** 5)
                sigma_z2 = (3.0 * p2 / (2.0 * np.pi * Z**2)) * ((Z / R2) ** 5)
            
            np.nan_to_num(sigma_z1, copy=False)
            np.nan_to_num(sigma_z2, copy=False)
            
            # CRITICAL VECTOR OVERLAP AREA: Summing the separate stress matrices
            total_sigma_z = sigma_z1 + sigma_z2
            print("✅ Stress Tensor Fields combined successfully. Localized intersection mapped.")
            return X, Z, total_sigma_z, z_grid
        except Exception as e:
            print(f"❌ Structural computation fault in Layer 3: {e}")
            return None

    # =========================================================================
    # MODULE 4: TERZAGHI FDM TIME-STEPPING CONSOLIDATION SOLVER
    # =========================================================================
    def execute_fdm_consolidation_loop(self, z_grid, total_sigma_z):
        print("\n--- [MODULE 4: FDM TIME-DEPENDENT SETTLEMENT LOOP] ---")
        # Extracting the integrated delta vertical stress vector right below the center point (X=0)
        # axis index 50 corresponds to spatial center x=0.0 in linspace(-10,10,100)
        delta_sigma_z_profile = total_sigma_z[:, 50] 
        
        # Consolidation Layer Attributes
        H = z_grid[-1] - z_grid[0] # Total thickness of deep compressible clay bed
        nodes = len(z_grid)
        dz = H / (nodes - 1)
        
        # Finite Difference Time Steps Setup Configuration
        t_max = 15.0 # Max simulation boundary time line (Years)
        dt = 0.01    # Safe fraction step stability matrix parameter (Years)
        time_steps = int(t_max / dt)
        
        # Stability check parameter calculation: alpha = (Cv * dt) / (dz^2)
        alpha = (self.cv * dt) / (dz ** 2)
        if alpha >= 0.5:
            print(f"⚠️ FDM Stability Warning: Alpha value {alpha:.3f} >= 0.5. Re-tuning dt parameters automatically...")
            dt = 0.45 * (dz ** 2) / self.cv
            time_steps = int(t_max / dt)
            alpha = (self.cv * dt) / (dz ** 2)
            
        print(f"ℹ️ Core Parameters: Spatial Nodes = {nodes}, dz = {dz:.3f}m, Time Steps Vector = {time_steps}, Alpha = {alpha:.4f}")
        
        # Initial condition array generation: Pore Water Pressure (u) field is initially matching load stress profile
        u = delta_sigma_z_profile.copy()
        
        # Array to store settlement progress profile across years tracking
        monitored_years = [1, 5, 10]
        settlement_tracking = {year: 0.0 for year in monitored_years}
        time_axis_plot = []
        settlement_axis_plot = []

        # Real-time multi-variable time stepping matrix updates loop execution
        current_time = 0.0
        for step in range(time_steps):
            u_next = u.copy()
            # FDM central difference formulation solver loops across inner continuum nodes
            for i in range(1, nodes - 1):
                u_next[i] = u[i] + alpha * (u[i+1] - 2.0*u[i] + u[i-1])
            
            # Boundary Conditions: Double Drainage Interface Profile (u=0 at top and bottom surfaces)
            u_next[0] = 0.0
            u_next[-1] = 0.0
            u = u_next.copy()
            current_time += dt
            
            # Calculate instant settlement via integration across compressed intervals: Sc = Sum(mv * (Δσ_z - u) * dz)
            pore_pressure_loss = delta_sigma_z_profile - u
            instant_settlement = np.sum(self.mv * pore_pressure_loss * dz) * 1000.0 # Convert meters to mm matrix
            
            time_axis_plot.append(current_time)
            settlement_axis_plot.append(instant_settlement)
            
            # Log exact metrics at required milestone years
            for target_year in monitored_years:
                if abs(current_time - target_year) < (dt / 2.0):
                    settlement_tracking[target_year] = instant_settlement

        print("\n📊 --- MULTI-YEAR COMPRESSION SETTLEMENT CONSOLE REPORT ---")
        for yr, val in settlement_tracking.items():
            print(f" 🕒 Settlement after {yr:2d} Year(s): {val:6.2f} mm")
        print("-" * 57)
        return time_axis_plot, settlement_axis_plot

    # =========================================================================
    # RENDER UNIFIED ENGINE INDUSTRIAL REPORT INTERFACE
    # =========================================================================
    def render_unified_dashboard(self, mesh_data, time_data):
        X, Z, total_sigma_z = mesh_data
        time_axis, settlement_axis = time_data
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot A: Dynamic Dual Footing Overlapping Stress Bulb Contours Map
        contour = axs[0].contourf(X, Z, total_sigma_z, levels=25, cmap='viridis')
        fig.colorbar(contour, ax=axs[0], label='Combined Vertical Stress Field σ_zz (kPa)')
        axs[0].set_title("1. Adjacent Footings Overlapping Stress Bulb Map")
        axs[0].set_xlabel("Horizontal Baseline Coordinates X (meters)")
        axs[0].set_ylabel("Depth Z below Foundation Interface (meters)")
        axs[0].invert_yaxis()
        axs[0].grid(True, linestyle=':', alpha=0.6)

        # Plot B: Research Grade Terzaghi FDM Time vs Settlement Curve
        axs[1].plot(time_axis, settlement_axis, color='#e74c3c', lw=3, label='Primary Consolidation Path')
        axs[1].set_title("2. Primary Consolidation Settlement Curve over Time")
        axs[1].set_xlabel("Time Horizon Processing Pipeline (Years)")
        axs[1].set_ylabel("Total Settlement Compression Magnitude (mm)")
        axs[1].grid(True, linestyle='--', alpha=0.5)
        axs[1].legend()

        plt.suptitle(f"Unified Advanced Soil Continuum Report\nSoil Type Identifier: {self.soil_tag} | Dynamic FDM Engine Status: COMPLETED", fontsize=13, fontweight='bold')
        plt.tight_layout()
        print("\n🚀 Pop-up Render Pipeline Triggered successfully. Close Matplotlib framework screen to terminate loop safely.")
        plt.show()

# =========================================================================
# PROGRAM GATEWAY DRIVER INITIALIZATION
# =========================================================================
if __name__ == "__main__":
    engine = UltimateGeotechEngine()
    
    # Run pipeline sequences sequentially
    class_res = engine.run_soil_classifier()
    if class_res:
        bc_res = engine.compute_bearing_capacity()
        if bc_res:
            b1, l1, d_f = bc_res
            mesh_res = engine.execute_adjacent_stress_overlap(b1, l1, d_f)
            if mesh_res:
                X, Z, total_sigma_z, z_grid = mesh_res
                time_res = engine.execute_fdm_consolidation_loop(z_grid, total_sigma_z)
                
                # Execute unified graphical board interface
                engine.render_unified_dashboard((X, Z, total_sigma_z), time_res)