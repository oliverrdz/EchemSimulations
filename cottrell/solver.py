import numpy as np
from .mesh1d import Mesh1D
try:
    from scipy.integrate import solve_ivp
except ImportError:
    solve_ivp = None

class PlanarDiffusionSolver:
    def __init__(self, D_cm2_s, C_bulk_mM, t_final_s,
                 dt_s=None, NX=101, lambda_value=0.4,
                 n_elec=1, A_cm2=1.0, grid_type="uniform",
                 stretch_factor=1.05, method="explicit",
                 geometry="planar", r0_cm=0.01,
                 L_thinlayer_cm=1e-3, omega_rpm=1000.0,
                 nu_cm2_s=0.01):

        self.geometry = geometry.lower().replace("-","_")
        self.C_bulk_mM = C_bulk_mM
        self.t_final_s = t_final_s
        self.dt_s = dt_s
        self.NX = NX
        self.lambda_value = lambda_value
        self.n = n_elec
        self.A_cm2 = A_cm2
        self.grid_type = grid_type
        self.stretch_factor = stretch_factor
        self.F = 96485.0
        self.r0_m = r0_cm*1e-2
        self.L_thinlayer_cm = L_thinlayer_cm
        self.omega_rpm = omega_rpm
        self.nu_cm2_s = nu_cm2_s
        self.method = method.lower()

        # Diffusion coefficient to SI
        self.D = D_cm2_s*1e-4
        # Characteristic diffusion length
        self.delta_m = np.sqrt(self.D*t_final_s)

        # Domain length by geometry
        if self.geometry in ("planar","spherical","cylindrical","microband"):
            L_m = 6*self.delta_m
        elif self.geometry=="thin_layer":
            L_m = self.L_thinlayer_cm*1e-2
        elif self.geometry=="rde":
            L_m = self._compute_rde_delta()
        elif self.geometry=="rce":
            L_m = self._compute_rce_delta()
        else:
            raise ValueError("bad geometry")

        # Inner coordinate (planar at x=0, radial at r=r0)
        x0_m = self.r0_m if self.geometry in ("spherical","cylindrical","microband") else 0.0

        # Mesh
        self.mesh = Mesh1D(L_m, NX, x0_m, grid_type, stretch_factor, self.geometry)

        # Allocate arrays, time grid
        self._allocate_arrays()

    def _compute_rde_delta(self):
        """RDE effective diffusion layer thickness (Levich-type)."""
        nu = self.nu_cm2_s*1e-4        # cm^2/s -> m^2/s
        omega = 2*np.pi*self.omega_rpm/60.0
        return 1.61*(self.D**(1.0/3.0))*(nu**(1.0/6.0))*(omega**-0.5)

    def _compute_rce_delta(self):
        """RCE effective diffusion layer thickness (similar scaling to RDE)."""
        nu = self.nu_cm2_s*1e-4
        omega = 2*np.pi*self.omega_rpm/60.0
        # slightly different prefactor; here we just use 1.47 as an example
        return 1.47*(self.D**(1.0/3.0))*(nu**(1.0/6.0))*(omega**-0.5)

    def _allocate_arrays(self):
        self.c = np.ones(self.NX)*self.C_bulk_mM
        self.i_t = []
        self.t_s = []

        if self.dt_s is None:
            dx = self.mesh.dx_m
            self.dt_s = self.lambda_value*dx*dx/self.D
        self.NT = int(self.t_final_s/self.dt_s)+1

    def _compute_current(self):
        dc_dx = (self.c[1]-self.c[0])/self.mesh.dx_m
        # Use planar area unless spherical; A_cm2 is user-defined for all non-spherical
        if self.geometry == "spherical":
            A_m2 = 4*np.pi*self.r0_m**2
        else:
            A_m2 = self.A_cm2*1e-4
        return -self.n*self.F*A_m2*self.D*dc_dx

    def solve(self):
        for k in range(self.NT):
            self.t_s.append(k*self.dt_s)
            self.i_t.append(self._compute_current())
            self._step()
        self.t_s = np.array(self.t_s)
        self.i_t = np.array(self.i_t)

    def _step(self):
        # explicit diffusion update with geometry-dependent Laplacian
        c = self.c
        D = self.D
        dt = self.dt_s
        dx = self.mesh.dx_m
        x = self.mesh.x_m
        c_new = c.copy()

        geom = self.geometry

        for i in range(1, self.NX-1):
            if geom in ("planar","thin_layer","rde","rce"):
                # standard 1D planar diffusion
                d2 = (c[i+1]-2*c[i]+c[i-1])/(dx*dx)
                c_new[i] = c[i] + D*dt*d2
            elif geom == "spherical":
                r = x[i]
                dcdr = (c[i+1]-c[i-1])/(2*dx)
                d2cdr2 = (c[i+1]-2*c[i]+c[i-1])/(dx*dx)
                rhs = d2cdr2 + 2.0*dcdr/r
                c_new[i] = c[i] + D*dt*rhs
            elif geom in ("cylindrical","microband"):
                r = x[i]
                dcdr = (c[i+1]-c[i-1])/(2*dx)
                d2cdr2 = (c[i+1]-2*c[i]+c[i-1])/(dx*dx)
                rhs = d2cdr2 + 1.0*dcdr/r
                c_new[i] = c[i] + D*dt*rhs
            else:
                # fallback: planar
                d2 = (c[i+1]-2*c[i]+c[i-1])/(dx*dx)
                c_new[i] = c[i] + D*dt*d2

        # Inner boundary: electrode surface (Dirichlet c=0)
        c_new[0] = 0.0

        # Outer boundary: bulk concentration for all current geometries
        c_new[-1] = self.C_bulk_mM

        self.c = c_new
