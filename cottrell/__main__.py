from .solver import PlanarDiffusionSolver

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from time import time

    def spherical_current_time(t, n, F, D, C_bulk_mol_m3, r_m):
        """
        Analytical spherical electrode current (diffusion-controlled region):

            i(t) = n F A D C* [ 1/sqrt(pi D t) + 1/r ]

        Parameters
        ----------
        t : float or array (s)
        n : int
        F : float (C/mol)
        D : float (m^2/s)
        C_bulk_mol_m3 : float (mol/m^3)
        r_m : float (m)

        Returns
        -------
        Current in A.
        """
        A = 4 * np.pi * r_m**2
        term1 = 1.0 / np.sqrt(np.pi * D * t)
        term2 = 1.0 / r_m
        return n * F * A * D * C_bulk_mol_m3 * (term1 + term2)

    # -------------------------------
    # Simulation settings
    # -------------------------------
    methods = ["solve_ivp", "solve_ivp"]

    # Choose geometry: "planar" or "spherical"
    geometries = ["planar", "spherical"]

    # For spherical geometry
    #A = (r0_cm * 2)**2                     # dummy planar area
    #A_planar_cm2 = 1.0                   # choose any realistic planar area
    r0_cm = 0.01                           # 100 μm radius
    A_spherical_cm2 = 4 * np.pi * r0_cm**2
    A_planar_cm2 = A_spherical_cm2

    # Prepare figure
    plt.figure()

    # -------------------------------
    # Run simulations (planar & spherical)
    # -------------------------------
    for m, g in zip(methods, geometries):
    
        if g == "planar":
           A_used = A_planar_cm2
        else:
            A_used = A_spherical_cm2   # ignored internally, but consistent

        solver = PlanarDiffusionSolver(
            D_cm2_s=1e-5,
            C_bulk_mM=1.0,
            t_final_s=10.0,
            dt_s=None,
            NX=101,
            A_cm2=A_used,
            grid_type="expanding",
            stretch_factor=1.03,
            method=m,
            geometry=g,
            r0_cm=r0_cm,
        )


        t_start = time()
        solver.solve()
        t_end = time()

        print(f"{m} ({g}): {(t_end - t_start):.2f} s")
        print(f"dt = {solver.dt_s:.6f} s")

        # Plot simulated current
        plt.plot(solver.t_s[1:],
                 solver.i_t[1:],
                 "-o", label=f"{g} simulated")

    # -------------------------------
    # Analytical currents
    # -------------------------------

    F = 96485.0
    n = 1

    # Convert D (cm²/s) → D (m²/s)
    D = 1e-5 * 1e-4

    # Concentration (mM → mol/m³)
    C_bulk_mM = 1.0
    C_bulk_mol_m3 = C_bulk_mM

    # Areas
    A_planar_m2 = A_planar_cm2 * 1e-4
    A_spherical_m2 = 4 * np.pi * (r0_cm * 1e-2)**2

    # Avoid t=0
    t_plot = solver.t_s[1:]

    # ---- Analytical planar (Cottrell) ----
    def i_planar_analytical(t):
        return -n * F * A_planar_m2 * C_bulk_mol_m3 * np.sqrt(D / (np.pi * t))

    plt.plot(t_plot,
             i_planar_analytical(t_plot),
             "k--",
             label="planar analytical")

    # Spherical analytical (direct equation)
    r_m = r0_cm * 1e-2
    i_sphere_direct = -spherical_current_time(t_plot, n, F, D, C_bulk_mol_m3, r_m)

    plt.plot(t_plot,
             i_sphere_direct,
             "g--",
             label="spherical analytical (direct)")


    # -------------------------------
    # Final plot formatting
    # -------------------------------
    plt.xlabel("t [s]")
    plt.ylabel("i [A]")
    plt.grid()
    plt.title("Simulated vs Analytical Diffusion Currents")
    plt.legend()
    plt.show()
