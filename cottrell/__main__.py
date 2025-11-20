from .solver import PlanarDiffusionSolver

if __name__ == "__main__":
    solver = PlanarDiffusionSolver(
        D_cm2_s=1e-5,
        C_bulk_mM=1.0,
        t_final_s=1.0,
        NX=101,
        lambda_value=0.4,
        n_elec=1,
        A_cm2=1.0
    )

    solver.solve()
    solver.plot_profile()
    solver.plot_current()
