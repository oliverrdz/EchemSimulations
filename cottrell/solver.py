import numpy as np
import matplotlib.pyplot as plt
from .mesh1d import Mesh1D

# Optional: SciPy for solve_ivp backend
try:
    from scipy.integrate import solve_ivp
except ImportError:
    solve_ivp = None


class PlanarDiffusionSolver:
    """
    1D planar diffusion solver with multiple time-stepping schemes.

    PDE (dimensionless):
        ∂c/∂τ = ∂²c/∂X²

    User-facing units:
        - D in cm^2/s
        - Concentration in mM
        - Output lengths in µm
        - Output current in mA

    Methods:
        - "explicit"          : Forward Euler in time
        - "rk2"               : 2nd-order Runge–Kutta (midpoint)
        - "rk4"               : 4th-order Runge–Kutta
        - "implicit"          : Backward Euler (unconditionally stable)
        - "crank-nicolson"    : Crank–Nicolson (2nd order, A-stable)
        - "solve_ivp"         : Uses scipy.integrate.solve_ivp on the semi-discrete ODE system
    """

    def __init__(
        self,
        D_cm2_s,
        C_bulk_mM,
        t_final_s,
        dt_s=None,                 # Physical time step [s]; if None, chosen automatically
        NX=101,
        lambda_value=0.4,          # Explicit stability parameter (< 0.5)
        n_elec=1,
        A_cm2=1.0,
        grid_type="uniform",
        stretch_factor=1.05,
        method="explicit",         # "explicit", "rk2", "rk4", "implicit", "crank-nicolson", "solve_ivp"
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
            Physical time step [s]. If None, dt is computed automatically.
        NX : int
            Number of spatial nodes.
        lambda_value : float
            Explicit FD stability parameter (<0.5). Used as guideline for
            time step selection.
        n_elec : int
            Number of electrons for current.
        A_cm2 : float
            Electrode area [cm^2].
        grid_type : {"uniform", "expanding"}
            Type of spatial mesh.
        stretch_factor : float
            Geometric ratio for expanding grid (only used if grid_type="expanding").
        method : str
            Time integration method.
        """

        if lambda_value >= 0.5:
            raise ValueError("lambda_value must be < 0.5 for explicit schemes.")

        self.C_bulk_mM = C_bulk_mM
        self.t_final_s = t_final_s
        self.dt_s = dt_s
        self.NX = NX
        self.lambda_value = lambda_value
        self.n = n_elec
        self.A_cm2 = A_cm2
        self.grid_type = grid_type
        self.stretch_factor = stretch_factor
        self.F = 96485.0  # Faraday constant [C/mol]

        self.method = method.lower()
        self._explicit_methods = ["explicit", "rk2", "rk4"]
        self._implicit_methods = ["implicit", "crank-nicolson", "solve_ivp"]

        if self.method not in self._explicit_methods + self._implicit_methods:
            raise ValueError(
                "method must be one of: "
                "'explicit', 'rk2', 'rk4', 'implicit', 'crank-nicolson', 'solve_ivp'"
            )

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

        Explicit methods:
            - enforce classic stability dtau <= lambda * (min dX)^2.

        Implicit / CN / solve_ivp:
            - no stability limit, but we still use the same dtau as a
              reasonable time resolution unless the user specifies dt_s.
        """
        X = self.mesh.X
        dX = np.diff(X)
        dX_min = np.min(dX)

        # Maximum dtau guided by explicit stability
        dtau_max = self.lambda_value * dX_min**2

        tau_final = self.D * self.t_final_s / (self.mesh.L_m**2)

        # User-specified dt_s
        if self.dt_s is not None:
            dtau = self.dt_s * self.D / (self.mesh.L_m**2)

            if self.method in self._explicit_methods and dtau >= dtau_max:
                raise ValueError(
                    f"dt_s={self.dt_s:.3e} is too large for explicit stability.\n"
                    f"Maximum stable dt_s is "
                    f"{(dtau_max * self.mesh.L_m**2 / self.D):.3e} s."
                )

            NT = int(self.t_final_s / self.dt_s) + 1
            return dtau, NT

        # dt_s not provided: choose dtau
        # For simplicity, we use dtau_max for all methods as a
        # reasonable resolution; explicit methods get a stable dtau,
        # implicit/CN/solve_ivp simply inherit this resolution.
        dtau = dtau_max
        NT = int(tau_final / dtau) + 1

        # Store equivalent dt_s for later conversions
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

        # Build discrete Laplacian matrix (for implicit / CN) once
        self._build_L_matrix()

    # ------------------------------------------------------------
    # SPATIAL DISCRETIZATION
    # ------------------------------------------------------------
    def _laplacian(self, c):
        """
        Compute ∂²c/∂X² on a non-uniform grid using second-order
        finite differences.

        Parameters
        ----------
        c : ndarray, shape (NX,)
            Dimensionless concentration.

        Returns
        -------
        d2 : ndarray, shape (NX,)
            Discrete Laplacian.
        """
        X = self.mesh.X
        dX = np.diff(X)
        NX = self.NX

        d2 = np.zeros_like(c)

        for i in range(1, NX - 1):
            dX_L = dX[i - 1]
            dX_R = dX[i]

            d2[i] = (
                2.0 / (dX_L + dX_R)
                * ((c[i + 1] - c[i]) / dX_R
                   - (c[i] - c[i - 1]) / dX_L)
            )

        # Boundaries: Dirichlet, no PDE applied
        d2[0] = 0.0
        d2[-1] = 0.0
        return d2

    def _build_L_matrix(self):
        """
        Build the full N x N discrete Laplacian matrix L such that:

            (Lc)_i ≈ ∂²c/∂X² at node i (interior),
            rows 0 and -1 are zero (BC enforced separately).
        """
        X = self.mesh.X
        dX = np.diff(X)
        N = self.NX

        L = np.zeros((N, N))

        for i in range(1, N - 1):
            dX_L = dX[i - 1]
            dX_R = dX[i]

            alpha = 2.0 / (dX_L * (dX_L + dX_R))
            gamma = 2.0 / (dX_R * (dX_L + dX_R))
            beta = -alpha - gamma

            L[i, i - 1] = alpha
            L[i, i] = beta
            L[i, i + 1] = gamma

        # Boundary rows remain zeros; Dirichlet is enforced in the
        # time-stepping matrices / RHS.
        self._L = L

    # ------------------------------------------------------------
    # MAIN SOLVER DISPATCH
    # ------------------------------------------------------------
    def solve(self):
        """
        Integrate the diffusion equation using the selected time
        integration method.
        """
        if self.method == "explicit":
            self._solve_explicit()
        elif self.method == "rk2":
            self._solve_rk2()
        elif self.method == "rk4":
            self._solve_rk4()
        elif self.method == "implicit":
            self._solve_implicit()
        elif self.method == "crank-nicolson":
            self._solve_crank_nicolson()
        elif self.method == "solve_ivp":
            self._solve_solve_ivp()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Convert to physical units and compute current
        self._convert_to_physical()
        self._compute_current()

    # ------------------------------------------------------------
    # EXPLICIT EULER
    # ------------------------------------------------------------
    def _solve_explicit(self):
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

    # ------------------------------------------------------------
    # RK2 (MIDPOINT)
    # ------------------------------------------------------------
    def _solve_rk2(self):
        """
        2nd-order Runge–Kutta (midpoint) for the semi-discrete system:
            dc/dτ = L(c)
        """
        dtau = self._dtau
        NT = self.c_history.shape[0]
        c = self.c.copy()

        for k in range(1, NT):
            k1 = self._laplacian(c)
            k2 = self._laplacian(c + 0.5 * dtau * k1)

            c_new = c + dtau * k2

            # Dirichlet BC
            c_new[0] = 0.0
            c_new[-1] = 1.0

            c[:] = c_new
            self.c_history[k] = c.copy()

        self.c = c

    # ------------------------------------------------------------
    # RK4
    # ------------------------------------------------------------
    def _solve_rk4(self):
        """
        4th-order Runge–Kutta for the semi-discrete system:
            dc/dτ = L(c)
        """
        dtau = self._dtau
        NT = self.c_history.shape[0]
        c = self.c.copy()

        for k in range(1, NT):
            k1 = self._laplacian(c)
            k2 = self._laplacian(c + 0.5 * dtau * k1)
            k3 = self._laplacian(c + 0.5 * dtau * k2)
            k4 = self._laplacian(c + dtau * k3)

            c_new = c + dtau / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

            # Dirichlet BC
            c_new[0] = 0.0
            c_new[-1] = 1.0

            c[:] = c_new
            self.c_history[k] = c.copy()

        self.c = c

    # ------------------------------------------------------------
    # IMPLICIT (BACKWARD EULER)
    # ------------------------------------------------------------
    def _solve_implicit(self):
        """
        Backward Euler:
            (I - dtau L) c^{n+1} = c^n
        with Dirichlet BC enforced by setting boundary rows to identity.
        """
        dtau = self._dtau
        NT = self.c_history.shape[0]
        N = self.NX

        I = np.eye(N)
        L = self._L

        M = I - dtau * L

        # Enforce Dirichlet BC in the system matrix
        M[0, :] = 0.0
        M[0, 0] = 1.0
        M[-1, :] = 0.0
        M[-1, -1] = 1.0

        c = self.c.copy()

        for k in range(1, NT):
            rhs = c.copy()
            rhs[0] = 0.0
            rhs[-1] = 1.0

            c_new = np.linalg.solve(M, rhs)

            c[:] = c_new
            self.c_history[k] = c.copy()

        self.c = c

    # ------------------------------------------------------------
    # CRANK–NICOLSON
    # ------------------------------------------------------------
    def _solve_crank_nicolson(self):
        """
        Crank–Nicolson:
            (I - 0.5 dtau L) c^{n+1} = (I + 0.5 dtau L) c^n
        with Dirichlet BC enforced via boundary rows.
        """
        dtau = self._dtau
        NT = self.c_history.shape[0]
        N = self.NX

        I = np.eye(N)
        L = self._L

        M_lhs = I - 0.5 * dtau * L
        M_rhs = I + 0.5 * dtau * L

        # Enforce BC in the left-hand-side matrix
        M_lhs[0, :] = 0.0
        M_lhs[0, 0] = 1.0
        M_lhs[-1, :] = 0.0
        M_lhs[-1, -1] = 1.0

        c = self.c.copy()

        for k in range(1, NT):
            rhs = M_rhs @ c

            # Override boundary entries to enforce Dirichlet BC exactly
            rhs[0] = 0.0
            rhs[-1] = 1.0

            c_new = np.linalg.solve(M_lhs, rhs)

            c[:] = c_new
            self.c_history[k] = c.copy()

        self.c = c

    # ------------------------------------------------------------
    # SOLVE_IVP BACKEND (SCIPY)
    # ------------------------------------------------------------
    def _solve_solve_ivp(self):
        """
        Use scipy.integrate.solve_ivp to integrate:
            dc/dτ = L(c)

        We integrate in dimensionless time τ and then map to
        physical time using dt_s.
        """
        if solve_ivp is None:
            raise ImportError(
                "scipy is required for method='solve_ivp' but is not installed."
            )

        dtau = self._dtau
        NT = self.c_history.shape[0]

        tau_eval = np.arange(NT) * dtau
        tau_final_eff = tau_eval[-1]

        c0 = self.c.copy()

        def rhs(tau, c):
            dc = self._laplacian(c)
            # Keep Dirichlet BC fixed
            dc[0] = 0.0
            dc[-1] = 0.0
            return dc

        sol = solve_ivp(
            rhs,
            t_span=(0.0, tau_final_eff),
            y0=c0,
            t_eval=tau_eval,
            method="BDF",
            vectorized=False,
        )

        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")

        # sol.y has shape (NX, NT)
        self.c_history = sol.y.T

        # Enforce BC exactly
        self.c_history[:, 0] = 0.0
        self.c_history[:, -1] = 1.0

        self.c = self.c_history[-1].copy()

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
