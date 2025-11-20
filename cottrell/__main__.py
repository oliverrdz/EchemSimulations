from .solver import PlanarDiffusionSolver

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    NX = [11, 21, 51]
    plt.figure(1)
    for nx in NX:
        solver = PlanarDiffusionSolver(
            D_cm2_s=1e-5,
            C_bulk_mM=1,
            t_final_s=1.0,
            dt_s = 0.001,
            NX=nx,
            grid_type="expanding",
            stretch_factor=1.03   # mild stretching
        )
        solver.solve()
        #plt.plot(solver.mesh.to_um(), solver.C_mM_history[-1])
        plt.plot(solver.t_s, solver.i_t*1e3)
    plt.xlabel("t / s")
    plt.ylabel("i / mA")
    plt.grid()
    plt.show()
    

    #solver.solve()
    #solver.plot_profile()
    #solver.plot_current()
