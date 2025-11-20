import numpy as np

class Mesh1D:
    """
    1D mesh for planar diffusion problems.

    Supports:
    - uniform grid (default)
    - expanding (geometric) grid with higher density near x = 0
    """

    def __init__(self, L_m, NX, grid_type="uniform", stretch_factor=1.05):
        """
        Parameters
        ----------
        L_m : float
            Total domain length [m].
        NX : int
            Number of grid points.
        grid_type : str
            "uniform"  -> equally spaced
            "expanding" -> geometric stretching with small spacing at x=0
        stretch_factor : float
            Geometric ratio (>1). Only used if grid_type="expanding".
            Typical values: 1.02–1.1
        """
        self.L_m = L_m
        self.NX = NX
        self.grid_type = grid_type
        self.stretch_factor = stretch_factor

        if grid_type == "uniform":
            self._make_uniform_grid()
        elif grid_type == "expanding":
            self._make_expanding_grid()
        else:
            raise ValueError("grid_type must be 'uniform' or 'expanding'")

        # Dimensionless coordinate X in [0, 1]
        self.X = self.x_m / self.L_m

    # ------------------------------------------------------------
    # GRID BUILDERS
    # ------------------------------------------------------------
    def _make_uniform_grid(self):
        """Standard linear grid."""
        self.x_m = np.linspace(0, self.L_m, self.NX)
        self.dx_m = self.x_m[1] - self.x_m[0]
        self.dX = self.dx_m / self.L_m

    def _make_expanding_grid(self):
        """
        Geometric mesh:
        dx, dx*r, dx*r^2, ...

        Where dx is determined so that the sum equals L.
        """
        r = self.stretch_factor
        N = self.NX

        # Solve dx * (1 - r^N) / (1 - r) = L
        if abs(r - 1) < 1e-12:
            raise ValueError("stretch_factor must not be 1 for geometric mesh")

        dx0 = self.L_m * (1 - r) / (1 - r**N)

        # Build cumulative sum of geometric increments
        increments = dx0 * r**np.arange(N)
        self.x_m = np.cumsum(increments)
        self.x_m -= self.x_m[0]   # ensure x(0) = 0

        # Uniform-like values for dimensionless spacing
        # (only needed by solver stability condition)
        self.dx_m = increments[0]
        self.dX = self.dx_m / self.L_m

    # ------------------------------------------------------------
    # UNIT CONVERSIONS
    # ------------------------------------------------------------
    def to_um(self):
        return self.x_m * 1e6

    def to_cm(self):
        return self.x_m * 1e2
