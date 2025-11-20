import numpy as np


class Mesh1D:
    """
    1D mesh for diffusion problems (planar or spherical radial).

    Supports:
    - uniform grid (default)
    - expanding (geometric) grid with higher density near the inner boundary
    """

    def __init__(
        self,
        L_m,
        NX,
        x0_m=0.0,
        grid_type="uniform",
        stretch_factor=1.05,
        geometry="planar",
    ):
        """
        Parameters
        ----------
        L_m : float
            Total domain length [m].
            The physical domain is [x0_m, x0_m + L_m].
        NX : int
            Number of grid points.
        x0_m : float
            Left/inner boundary position [m].
            - planar: usually 0.0 (electrode at x=0)
            - spherical: usually r0 (electrode radius)
        grid_type : str
            "uniform"   -> equally spaced
            "expanding" -> geometric stretching with small spacing at inner boundary
        stretch_factor : float
            Geometric ratio (>1). Only used if grid_type="expanding".
            Typical values: 1.02–1.1
        geometry : {"planar", "spherical"}
            Stored for convenience; solver uses it to interpret the mesh.
        """
        self.L_m = L_m
        self.NX = NX
        self.x0_m = x0_m
        self.grid_type = grid_type
        self.stretch_factor = stretch_factor
        self.geometry = geometry

        if grid_type == "uniform":
            self._make_uniform_grid()
        elif grid_type == "expanding":
            self._make_expanding_grid()
        else:
            raise ValueError("grid_type must be 'uniform' or 'expanding'")

        # Dimensionless coordinate X in [0, 1], measured from inner boundary
        self.X = (self.x_m - self.x0_m) / self.L_m

    # ------------------------------------------------------------
    # GRID BUILDERS
    # ------------------------------------------------------------
    def _make_uniform_grid(self):
        """Standard linear grid over [x0_m, x0_m + L_m]."""
        self.x_m = np.linspace(self.x0_m, self.x0_m + self.L_m, self.NX)
        self.dx_m = self.x_m[1] - self.x_m[0]
        self.dX = self.dx_m / self.L_m

    def _make_expanding_grid(self):
        """
        Geometric mesh:
        dx, dx*r, dx*r^2, ...

        Where dx is determined so that the sum equals L.
        The mesh is then shifted to start at x0_m.
        """
        r = self.stretch_factor
        N = self.NX

        # Solve dx * (1 - r^N) / (1 - r) = L
        if abs(r - 1) < 1e-12:
            raise ValueError("stretch_factor must not be 1 for geometric mesh")

        dx0 = self.L_m * (1 - r) / (1 - r**N)

        increments = dx0 * r**np.arange(N)
        x_local = np.cumsum(increments)
        x_local -= x_local[0]  # ensure inner node is at 0 in local coordinates

        self.x_m = self.x0_m + x_local

        # Use smallest spacing as representative dx
        self.dx_m = increments[0]
        self.dX = self.dx_m / self.L_m

    # ------------------------------------------------------------
    # UNIT CONVERSIONS
    # ------------------------------------------------------------
    def to_um(self):
        return self.x_m * 1e6

    def to_cm(self):
        return self.x_m * 1e2
