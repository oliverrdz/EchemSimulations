import numpy as np

class Mesh1D:
    """1D mesh for diffusion problems (planar, thin_layer, rde, spherical)."""

    def __init__(self, L_m, NX, x0_m=0.0, grid_type="uniform",
                 stretch_factor=1.05, geometry="planar"):
        self.L_m = L_m
        self.NX = NX
        self.x0_m = x0_m
        self.grid_type = grid_type
        self.stretch_factor = stretch_factor
        self.geometry = geometry.lower().replace("-", "_")

        if grid_type == "uniform":
            self._make_uniform_grid()
        elif grid_type == "expanding":
            self._make_expanding_grid()
        else:
            raise ValueError("grid_type must be uniform or expanding")

        self.X = (self.x_m - self.x0_m) / self.L_m

    def _make_uniform_grid(self):
        self.x_m = np.linspace(self.x0_m, self.x0_m + self.L_m, self.NX)
        self.dx_m = self.x_m[1] - self.x_m[0]
        self.dX = self.dx_m / self.L_m

    def _make_expanding_grid(self):
        r = self.stretch_factor
        N = self.NX
        if abs(r-1) < 1e-12:
            raise ValueError("stretch_factor cannot be 1")
        dx0 = self.L_m * (1 - r) / (1 - r**N)
        increments = dx0 * r**np.arange(N)
        x_local = np.cumsum(increments)
        x_local -= x_local[0]
        self.x_m = self.x0_m + x_local
        self.dx_m = increments[0]
        self.dX = self.dx_m / self.L_m

    def to_um(self):
        return self.x_m * 1e6

    def to_cm(self):
        return self.x_m * 1e2
