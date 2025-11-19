import numpy as np
import matplotlib.pyplot as plt


class PlanarDiffusionSolver:
    """
    Solver for 1D planar diffusion using explicit finite differences.

    User inputs:
    - Lengths in cm
    - Diffusion coefficient in cm^2/s
    - Concentration in mM
    Internally everything is converted to SI (meters, m^2/s)
    for correct physics, then converted back for plotting.

    PDE solved:
        ∂C/∂t = D ∂²C/∂x²

    Nondimensionalization:
        X = x / L
        τ = Dt / L²
        c = C / C_bulk

    Parameters
    ----------
    D_cm2_s : float
        Diffusion coefficient [cm^2/s].
    C_bulk_mM : float
        Bulk concentration in mM.
    t_final_s : float
        Final physical time [s].
    NX : int
        Number of spatial grid points.
    lambda_value : float
        Stability parameter (< 0.5).
    """

    def __init__(self, D_cm2_s, C_bulk_mM, t_final_s,
                 NX=101, lambda_value=0.4):

        self.C_bulk_mM = C_bulk_mM
        self.t_final_s = t_final_s
        self.NX = NX
        self.lambda_value = lambda_value

        if lambda_value >= 0.5:
            raise ValueError("lambda_value must be < 0.5 for stability.")

        # --------------------------------------------
        # Convert units to SI
        # --------------------------------------------
        self.cm_to_m = 1e-2
        self.cm2_to_m2 = 1e-4

        self.D = D_cm2_s * self.cm2_to_m2   # convert cm²/s → m²/s

        self._setup_domain()
        self._allocate_arrays()

    def _setup_domain(self):
        """Configure physical and nondimensional domain."""

        # Diffusion thickness δ = sqrt(D * t)
        self.delta_m = np.sqrt(self.D * self.t_final_s)

        # Domain length L = 6 δ (in meters)
        self.L_m = 6.0 * self.delta_m

        # Dimensionless spatial grid
        self.X = np.linspace(0, 1, self.NX)
        self.dX = 1.0 / (self.NX - 1)

        # Dimensionless time step
        self.dtau = self.lambda_value * self.dX**2

        # Dimensionless final time
        self.tau_final = self.D * self.t_final_s / self.L_m**2

        # Number of time steps
        self.NT = int(self.tau_final / self.dtau) + 1

    def _allocate_arrays(self):
        """Allocate concentration arrays."""
        self.c = np.ones(self.NX)
        self.c_new = np.zeros(self.NX)

        # Dimensionless BCs
        self.c[0] = 0.0
        self.c[-1] = 1.0

        self.c_history = np.zeros((self.NT, self.NX))
        self.c_history[0, :] = self.c.copy()

    def solve(self):
        """Solve the diffusion PDE."""
        for n in range(1, self.NT):
            self.c_new[1:-1] = (
                self.c[1:-1] +
                self.lambda_value *
                (self.c[2:] - 2 * self.c[1:-1] + self.c[:-2])
            )

            self.c_new[0] = 0.0
            self.c_new[-1] = 1.0

            self.c[:] = self.c_new[:]
            self.c_history[n, :] = self.c

        self._convert_to_physical()

    def _convert_to_physical(self):
        """Convert dimensionless results to physical units (cm, mM)."""

        # Spatial coordinates in meters and cm
        self.x_m = self.X * self.L_m
        self.x_cm = self.x_m / self.cm_to_m

        # Physical time array (s)
        tau_array = np.arange(self.NT) * self.dtau
        self.t_s = tau_array * self.L_m**2 / self.D

        # Concentration in mM
        self.C_mM_history = self.c_history * self.C_bulk_mM

    def plot_profile(self):
        """Plot final concentration profile in cm and mM."""
        plt.figure(figsize=(7, 4))
        plt.plot(
            np.array(self.x_m)*1e6,
            np.array(self.C_mM_history).T,
        )
        plt.xlabel("Position x [um]")
        plt.ylabel("Concentration C [mM]")
        plt.title("Final Concentration Profile (1D Planar Diffusion)")
        plt.grid(True)
        plt.show()

    def summary(self):
        """Print simulation summary."""
        print("----- Simulation Summary -----")
        print(f"D (cm²/s) = {self.D / self.cm2_to_m2}")
        print(f"C_bulk = {self.C_bulk_mM} mM")
        print(f"Final time = {self.t_final_s} s")
        print(f"Diffusion thickness δ = {self.delta_m*1e6:.2f} um")
        print(f"Domain length L = {self.L_m*1e6:.2f} um")
        print(f"Nx = {self.NX}, Nt = {self.NT}")
        print(f"lambda = {self.lambda_value}")
        print("------------------------------")


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    solver = PlanarDiffusionSolver(
        D_cm2_s=1e-5,     # diffusion coefficient in cm²/s
        C_bulk_mM=1.0,    # concentration in mM
        t_final_s=1.0,    # time in seconds
        NX=21,
        lambda_value=0.4
    )

    solver.solve()
    solver.plot_profile()
    solver.summary()

