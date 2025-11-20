import numpy as np
import matplotlib.pyplot as plt
from .mesh1d import Mesh1D


class PlanarDiffusionSolver:
    """
    Explicit FD solver for 1D planar diffusion using separated mesh class.

    User-facing units:
        - D in cm^2/s
        - Concentration in mM
        - Output lengths in µm
        - Output current in mA
    """

    def __init__(self, D_cm2_s, C_bulk_mM, t_final_s,
                 NX=101, lambda_value=0.4,
                 n_elec=1, A_cm2=1.0):
        """
        Parameters
        ----------
        D_cm2_s : float
            Diffusion coefficient [cm^2/s].
        C_bulk_mM : float
            Bulk concentration [mM].
        t_final_s : float
            Final simulation time [s].
        NX : int
            Number of spatial nodes.
        lambda_value : float
            Explicit FD stability parameter (<0.5).
        n_elec : int
            Number of electrons for current.
        A_cm2 : float
            Electrode area [cm^2].
        """
        self.C_bulk_mM = C_bulk_mM
        self.t_final_s = t_final_s
        self.NX = NX
        self.lambda_value = lambda_value
        self.n = n_elec
        self.A_cm2 = A_cm2
        self.F = 96485  # Faraday constant

        if lambda_value >= 0.5:
            raise ValueError("lambda must be < 0.5 for explicit FD.")

        # Convert user units to SI
        self.D = D_cm2_s * 1e-4  # cm²/s → m²/s

        # Domain length = 6 * diffusion thickness
        self.delta_m = np.sqrt(self.D * t_final_s)
        L_m = 6 * self.delta_m

        # Create mesh object
        self.mesh = Mesh1D(L_m=L_m, NX=NX)

        # Allocate arrays
        self._allocate_arrays()

    # ------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------
    def _allocate_arrays(self):
        """Allocate dimensionless concentration arrays."""
        self.c = np.ones(self.NX)
        self.c_new = np.zeros(self.NX)

        # Boundary conditions: c(0)=0, c(L)=1
        self.c[0] = 0.0
        self.c[-1] = 1.0

        NT = self._nt_steps()
        self.c_history = np.zeros((NT, self.NX))
        self.c_history[0] = self.c.copy()

    def _nt_steps(self):
        """Number of time steps based on nondimensional time."""
        tau_final = self.D * self.t_final_s / self.mesh.L_m**2
        dtau = self.lambda_value * self.mesh.dX**2
        return int(tau_final / dtau) + 1

    # ------------------------------------------------------------
    # MAIN SOLVER
    # ------------------------------------------------------------
    def solve(self):
        """Explicit FD solution of ∂c/∂τ = ∂²c/∂X²."""
        dtau = self.lambda_value * self.mesh.dX**2
        tau_final = self.D * self.t_final_s / self.mesh.L_m**2
        NT = int(tau_final / dtau) + 1

        for k in range(1, NT):
            self.c_new[1:-1] = (
                self.c[1:-1]
                + self.lambda_value *
                (self.c[2:] - 2*self.c[1:-1] + self.c[:-2])
            )

            # BCs
            self.c_new[0] = 0.0
            self.c_new[-1] = 1.0

            self.c[:] = self.c_new[:]
            self.c_history[k] = self.c.copy()

        self._convert_to_physical()
        self._compute_current()

    # ------------------------------------------------------------
    # UNIT CONVERSION
    # ------------------------------------------------------------
    def _convert_to_physical(self):
        """Convert dimensionless concentration to mM, time to seconds."""
        dtau = self.lambda_value * self.mesh.dX**2
        tau_array = np.arange(self.c_history.shape[0]) * dtau
        self.t_s = tau_array * self.mesh.L_m**2 / self.D

        # c in [0,1] → mM
        self.C_mM_history = self.c_history * self.C_bulk_mM

    # ------------------------------------------------------------
    # CURRENT
    # ------------------------------------------------------------
    def _compute_current(self):
        """Compute Faradaic current: i(t) = n F A (-D dC/dx)|₀."""
        A_m2 = self.A_cm2 * 1e-4  # cm² → m²
        NT = self.c_history.shape[0]

        self.i_t = np.zeros(NT)

        for k in range(NT):
            dCdx = (self.C_mM_history[k, 1] - self.C_mM_history[k, 0]) / self.mesh.dx_m
            J = -self.D * dCdx
            self.i_t[k] = self.n * self.F * A_m2 * J  # A

    # ------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------
    def plot_profile(self):
        """Plot final concentration profile in μm and mM."""
        x_um = self.mesh.to_um()

        plt.figure(figsize=(7, 4))
        plt.plot(x_um, self.C_mM_history[-1])
        plt.xlabel("Position x [μm]")
        plt.ylabel("Concentration [mM]")
        plt.grid(True)
        plt.title("Final Concentration Profile")
        plt.show()

    def plot_current(self):
        """Plot current vs time in mA."""
        i_mA = self.i_t * 1e3

        plt.figure(figsize=(7, 4))
        plt.plot(self.t_s, i_mA)
        plt.xlabel("Time [s]")
        plt.ylabel("Current [mA]")
        plt.grid(True)
        plt.title("Diffusion-Limited Current")
        plt.show()
