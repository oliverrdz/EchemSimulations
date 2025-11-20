import numpy as np


class Mesh1D:
    """
    1D mesh for planar diffusion problems.

    Handles:
    - Physical domain length in meters (L_m)
    - Dimensionless grid X in [0, 1]
    - Mapping between X and physical coordinates
    """

    def __init__(self, L_m, NX):
        """
        Parameters
        ----------
        L_m : float
            Physical domain length in meters.
        NX : int
            Number of grid points.
        """
        self.L_m = L_m
        self.NX = NX

        # Dimensionless grid
        self.X = np.linspace(0, 1, NX)

        # Physical grid
        self.x_m = self.X * self.L_m

        # Grid spacing
        self.dX = 1.0 / (NX - 1)
        self.dx_m = self.x_m[1] - self.x_m[0]

    def to_um(self):
        """Return mesh coordinates in micrometers."""
        return self.x_m * 1e6

    def to_cm(self):
        """Return mesh coordinates in centimeters."""
        return self.x_m * 1e2
