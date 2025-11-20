from .solver import PlanarDiffusionSolver

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from time import time
    
    methods = ["explicit", "implicit", "crank-nicolson", "rk2", "rk4", "solve_ivp"]
    plt.figure()
    for m in methods:
        solver = PlanarDiffusionSolver(
        D_cm2_s=1e-5,
        C_bulk_mM=1.0,
        t_final_s=1.0,
        dt_s=None,
        NX=101,
        grid_type="expanding",
        stretch_factor=1.03,
        method=m
        )
        t_start = time()
        solver.solve()
        t_end = time()
        print(f"{m}: {(t_end-t_start):.2f} s")
        print(f"dt = {solver.dt_s:.6f} s")
        plt.plot(solver.t_s, solver.i_t*1e3)

    plt.xlabel("t / s")
    plt.ylabel("i / mA")
    plt.grid()
    plt.show()
