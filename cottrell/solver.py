import numpy as np
import matplotlib.pyplot as plt
from .mesh1d import Mesh1D


class PlanarDiffusionSolver:
    """
    Explicit FD solver for 1D planar diffusion using a separate mesh class.

    User-facing units:
        - D in cm^2/s
        - Concentration in mM
        - Output lengths in µm
        - Output current in mA
    """

    def __init__(
        self,
        D_cm2_s,
        C_bulk_mM,
        t_final_s,
        dt_s=None,                 # ★ NEW: user-supplied time step
        NX=101,
        lambda_value=0.4,
        n_elec=1,
        A_cm2=1.0,
        grid_type="uniform",
        stretch_factor=1.05,
    ):
        """
        Parameters
        ----------
        D_cm2_s : float
            Diffusion coefficient [cm^2/s].
        C_bulk_mM : float
            Bulk concentration [mM].
        t_final_s : float
            Final simulation time [s].
        dt_s : float or None
            Physical time step [s]. If None, dt is computed automatically
            from stability.
        NX : int
            Number of spatial nodes.
        lambda_value : float
            Explicit FD stability parameter (<0.5).
        n_elec : int
            Number of electrons for current.
        A_cm2 : float
            Electrode area [cm^2].
        grid_type : {"uniform", "expanding"}
            Type of spatial mesh.
        stretch_factor : float
            Geometric ratio for expanding grid (only used if grid_type="expanding").
        """
        if lambda_value >= 0.5:
            raise ValueError("lambda_value must be < 0.5 for explicit FD.")

        self.C_bulk_mM = C_bulk_mM
        self.t_final_s = t_final_s
        self.dt_s = dt_s            # ★ store dt_s
        self.NX = NX
        self.lambda_value = lambda_value
        self.n = n_elec
        self.A_cm2 = A_cm2
        self.grid_type = grid_type
        self.stretch_factor = stretch_factor
        self.F = 96485.0  # Faraday constant [C/mol]

        # Convert user units to SI
        self.D = D_cm2_s * 1e-4  # cm²/s → m²/s

        # Domain length = 6 * diffusion thickness
        self.delta_m = np.sqrt(self.D * t_final_s)
        L_m = 6 * self.delta_m

        # Create mesh object
        self.mesh = Mesh1D(
            L_m=L_m,
            NX=NX,
            grid_type=grid_type,
            stretch_factor=stretch_factor,
        )

        # Allocate arrays and compute number of time steps
        self._allocate_arrays()

    # ------------------------------------------------------------
    # TIME STEP COMPUTATION
    # ------------------------------------------------------------
    def _compute_dtau_and_NT(self):
        """
        Compute dimensionless time step dtau and number of time steps NT.

        If user supplies dt_s:
            - convert to dtau
            - check explicit stability
        Otherwise:
            - compute dtau automatically from λ and minimum spacing
        """
        X = self.mesh.X
        dX = np.diff(X)
        dX_min = np.min(dX)

        # Maximum dtau allowed by explicit stability
        dtau_max = self.lambda_value * dX_min**2

        if self.dt_s is not None:
            # Convert physical dt_s → dimensionless dtau
            dtau = self.dt_s * self.D / (self.mesh.L_m**2)

            if dtau >= dtau_max:
                raise ValueError(
                    f"dt_s={self.dt_s:.3e} is too large for explicit stability.\n"
                    f"Maximum stable dt_s is {(dtau_max * self.mesh.L_m**2 / self.D):.3e} s."
                )

            NT = int(self.t_final_s / self.dt_s) + 1
            return dtau, NT

        # If dt_s not provided, compute dtau automatically
        dtau = dtau_max
        tau_final = self.D * self.t_final_s / (self.mesh.L_m**2)
        NT = int(tau_final / dtau) + 1

        # Store equivalent dt_s for later use in conversions
        self.dt_s = dtau * (self.mesh.L_m**2) / self.D

        return dtau, NT

    # ------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------
    def _allocate_arrays(self):
        """Allocate dimensionless concentration arrays and history."""
        self.c = np.ones(self.NX)
        self.c_new = np.zeros(self.NX)

        # Boundary conditions: c(0)=0, c(L)=1
        self.c[0] = 0.0
        self.c[-1] = 1.0

        self._dtau, NT = self._compute_dtau_and_NT()

        self.c_history = np.zeros((NT, self.NX))
        self.c_history[0] = self.c.copy()

    # ------------------------------------------------------------
    # MAIN SOLVER
    # ------------------------------------------------------------
    def solve(self):
        """
        Explicit FD solution of ∂c/∂τ = ∂²c/∂X²
        on a possibly non-uniform X-grid.
        """
        X = self.mesh.X
        dX = np.diff(X)

        dtau = self._dtau
        NT = self.c_history.shape[0]

        for k in range(1, NT):
            for i in range(1, self.NX - 1):
                dX_left = dX[i - 1]
                dX_right = dX[i]

                d2cdX2 = (
                    2.0 / (dX_left + dX_right)
                    * ((self.c[i + 1] - self.c[i]) / dX_right
                       - (self.c[i] - self.c[i - 1]) / dX_left)
                )

                self.c_new[i] = self.c[i] + dtau * d2cdX2

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
        """Convert dimensionless concentration to mM; time to seconds."""
        NT = self.c_history.shape[0]

        # Physical time using dt_s (user or automatic)
        self.t_s = np.arange(NT) * self.dt_s

        # c in [0,1] → concentration [mM]
        self.C_mM_history = self.c_history * self.C_bulk_mM

    # ------------------------------------------------------------
    # CURRENT
    # ------------------------------------------------------------
    def _compute_current(self):
        """
        Compute Faradaic current:
        i(t) = n F A (-D dC/dx)|_{x=0}
        Using forward difference at the boundary.
        """
        A_m2 = self.A_cm2 * 1e-4
        x = self.mesh.x_m
        dx0 = x[1] - x[0]

        NT = self.c_history.shape[0]
        self.i_t = np.zeros(NT)

        for k in range(NT):
            C_line = self.C_mM_history[k, :]
            dCdx = (C_line[1] - C_line[0]) / dx0
            J = -self.D * dCdx
            self.i_t[k] = self.n * self.F * A_m2 * J  # A

    # ------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------
    def plot_profile(self):
        x_um = self.mesh.to_um()

        plt.figure(figsize=(7, 4))
        plt.plot(x_um, self.C_mM_history[-1])
        plt.xlabel("Position x [μm]")
        plt.ylabel("Concentration [mM]")
        plt.grid(True)
        plt.title("Final Concentration Profile")
        plt.show()

    def plot_current(self):
        i_mA = self.i_t * 1e3

        plt.figure(figsize=(7, 4))
        plt.plot(self.t_s, i_mA)
        plt.xlabel("Time [s]")
        plt.ylabel("Current [mA]")
        plt.grid(True)
        plt.title("Diffusion-Limited Current")
        plt.show()
