import numpy as np
import matplotlib.pyplot as plt


class PlanarDiffusionSolver:
    """
    1D planar diffusion solver using explicit finite differences,
    with optional current calculation at the electrode surface.

    User-facing units:
        - Length in cm
        - Diffusion coefficient in cm²/s
        - Concentration in mM

    Internal units:
        - Length converted to meters
        - D converted to m²/s
        - Concentration converted to mol/m³ (1 mM = 1 mol/m³)

    PDE solved:
        ∂C/∂t = D ∂²C/∂x²

    Nondimensionalization:
        X = x / L
        τ = D t / L²
        c = C / C_bulk

    Boundary conditions:
        c(0, τ) = 0       (perfect sink)
        c(1, τ) = 1

    Attributes
    ----------
    C_bulk_mM : float
        Bulk concentration in mM.
    D : float
        Diffusion coefficient (converted to m²/s).
    L_m : float
        Simulation domain length in meters.
    c_history : ndarray
        Dimensionless concentration matrix with shape (NT, NX).
    C_mM_history : ndarray
        Physical concentration matrix in mM.
    i_t : ndarray
        Calculated current vs. time.
    """

    def __init__(self, D_cm2_s, C_bulk_mM, t_final_s,
                 NX=101, lambda_value=0.4,
                 n_elec=1, A_cm2=1.0):
        """
        Initialize the diffusion solver and convert user units to SI.

        Parameters
        ----------
        D_cm2_s : float
            Diffusion coefficient in cm²/s.
        C_bulk_mM : float
            Bulk concentration in mM.
        t_final_s : float
            Final simulation time in seconds.
        NX : int
            Number of spatial grid points.
        lambda_value : float
            Explicit FD stability parameter (<0.5).
        n_elec : int
            Number of electrons per molecule for current calcs.
        A_cm2 : float
            Electrode area in cm² used to compute current.
        """
        self.C_bulk_mM = C_bulk_mM
        self.t_final_s = t_final_s
        self.NX = NX
        self.lambda_value = lambda_value
        self.n = n_elec
        self.A_cm2 = A_cm2
        self.F = 96485  # C/mol

        if lambda_value >= 0.5:
            raise ValueError("lambda_value must be < 0.5 for stability.")

        # Unit conversions
        self.cm_to_m = 1e-2
        self.cm2_to_m2 = 1e-4
        self.D = D_cm2_s * self.cm2_to_m2

        self._setup_domain()
        self._allocate_arrays()

    # ------------------------------------------------------------
    # INTERNAL METHODS
    # ------------------------------------------------------------

    def _setup_domain(self):
        """
        Define the physical domain and dimensionless variables.

        The domain length is set to 6× the diffusion layer thickness:
            δ = sqrt(D*t_final)

        Defines:
            - L_m : physical domain length in meters
            - X   : dimensionless spatial grid
            - dX  : spatial step in X
            - dtau : dimensionless timestep
            - NT   : number of steps
        """
        self.delta_m = np.sqrt(self.D * self.t_final_s)
        self.L_m = 6.0 * self.delta_m

        self.X = np.linspace(0, 1, self.NX)
        self.dX = 1.0 / (self.NX - 1)

        self.dtau = self.lambda_value * self.dX**2
        self.tau_final = self.D * self.t_final_s / self.L_m**2
        self.NT = int(self.tau_final / self.dtau) + 1

    def _allocate_arrays(self):
        """
        Allocate memory for concentration arrays.

        Creates:
            c          : dimensionless concentration
            c_new      : temporary update array
            c_history  : time × space array storing solution
        """
        self.c = np.ones(self.NX)
        self.c_new = np.zeros(self.NX)

        self.c[0] = 0.0
        self.c[-1] = 1.0

        self.c_history = np.zeros((self.NT, self.NX))
        self.c_history[0, :] = self.c.copy()

    # ------------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------------

    def solve(self):
        """
        Solve the diffusion PDE using explicit finite differences.

        After solving:
            - Converts concentration to mM
            - Computes current vs. time
        """
        for n in range(1, self.NT):
            self.c_new[1:-1] = (
                self.c[1:-1]
                + self.lambda_value *
                (self.c[2:] - 2*self.c[1:-1] + self.c[:-2])
            )

            self.c_new[0] = 0.0
            self.c_new[-1] = 1.0

            self.c[:] = self.c_new[:]
            self.c_history[n, :] = self.c

        self._convert_to_physical()
        self._compute_current()

    def _convert_to_physical(self):
        """
        Convert dimensionless arrays into physical units.

        Produces:
            x_m      : position in meters
            x_cm     : position in cm
            t_s      : time in seconds
            C_mM_history : concentration in mM
        """
        self.x_m = self.X * self.L_m
        self.x_cm = self.x_m / self.cm_to_m

        tau_array = np.arange(self.NT) * self.dtau
        self.t_s = tau_array * self.L_m**2 / self.D

        self.C_mM_history = self.c_history * self.C_bulk_mM

    def _compute_current(self):
        """
        Compute the diffusion-limited current from the surface flux.

        Current definition:
            i(t) = n F A J(t)
            J(t) = -D * (dC/dx)|x=0

        Uses first-order finite difference at the boundary.
        """
        A_m2 = self.A_cm2 * 1e-4  # cm² → m²
        C_mol_m3 = self.C_mM_history  # 1 mM = 1 mol/m³

        self.i_t = np.zeros(self.NT)

        for k in range(self.NT):
            dCdx = (C_mol_m3[k, 1] - C_mol_m3[k, 0]) / (self.x_m[1] - self.x_m[0])
            J = -self.D * dCdx
            self.i_t[k] = self.n * self.F * A_m2 * J

    # ------------------------------------------------------------
    # PLOTTING / REPORTING
    # ------------------------------------------------------------

    def plot_profile(self):
        """
        Plot the final concentration profile.
        x-axis: micrometers (µm)
        y-axis: concentration in mM
        """
        x_um = self.x_m * 1e6  # meters → micrometers

        plt.figure(figsize=(7, 4))
        plt.plot(x_um, self.C_mM_history[-1, :])
        plt.xlabel("Position x [µm]")
        plt.ylabel("Concentration C [mM]")
        plt.title("Final Concentration Profile (1D Diffusion)")
        plt.grid(True)
        plt.show()

    def plot_current(self):
        """
        Plot diffusion-limited current vs time.
        Current plotted in mA instead of A.
        """
        i_mA = self.i_t * 1e3  # A → mA

        plt.figure(figsize=(7, 4))
        plt.plot(self.t_s, i_mA)
        plt.xlabel("Time [s]")
        plt.ylabel("Current [mA]")
        plt.title("Diffusion-Limited Current vs Time")
        plt.grid(True)
        plt.show()

    
    def summary(self):
        """
        Print a summary of key simulation parameters and results.
        """
        print("----- Simulation Summary -----")
        print(f"D (cm²/s) = {self.D / self.cm2_to_m2}")
        print(f"C_bulk = {self.C_bulk_mM} mM")
        print(f"Final time = {self.t_final_s} s")
        print(f"Diffusion thickness δ = {self.delta_m*1e6:.2f} um")
        print(f"Domain length L = {self.L_m*1e6:.2f} um")
        print(f"Nx = {self.NX}, Nt = {self.NT}")
        print(f"lambda = {self.lambda_value}")
        print("------------------------------")


# Example
if __name__ == "__main__":
    solver = PlanarDiffusionSolver(
        D_cm2_s=1e-5,
        C_bulk_mM=1.0,
        t_final_s=1.0,
        NX=101,
        lambda_value=0.4,
        n_elec=1,
        A_cm2=1.0
    )

    solver.solve()
    solver.plot_profile()
    solver.plot_current()
    solver.summary()

