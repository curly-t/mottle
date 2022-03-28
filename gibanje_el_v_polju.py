from calc_magnetic_field import get_Bz_tot_func, get_Brho_tot_func
from calc_electric_field_analytic import get_Ez, get_Erho
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from time import perf_counter
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d import Axes3D
from pathos.multiprocessing import Pool     # Veliko več objektov zna Serializirat za multiproccessing!
from functools import cache


# GLOBALS - TRUE CONSTANTS ONLY ----------------------------------------------------------------------------------------
m = 9.1e-31
e0 = 1.6e-19
mu0 = 4 * np.pi * 1e-7
# ----------------------------------------------------------------------------------------------------------------------


def get_gibalna_en(Bz, Brho, Ez=None, Erho=None):
    if Ez is None or Erho is None:
        def gibalna_en(t, y):
            # y je vektor trenutne lokacije in hitrosti delca [rho, phi, z, rho', phi', z']
            # Gibalna enačba vrne odvod vektorja y po času
            # Nevem zakaj je samo v prvem delu pri rho''[t] "-" drugje pa ne! - to je čudno, ampak tako dela pri mathematici
            # e0 je sicer +1e-19 ampak elektrone mamo tu zato je - zamenjan s plusom
            return np.array([y[3], y[4], y[5],
                             - e0/m * y[0]*y[4]*Bz(y[0], y[2]) + y[0]*np.square(y[4]),
                             (e0/m * (y[3] * Bz(y[0], y[2]) - y[5] * Brho(y[0], y[2])) - 2*y[3]*y[4]) / y[0],
                             e0/m * y[4] * y[0] * Brho(y[0], y[2])])
    else:
        def gibalna_en(t, y):
            # Dodano še za električno polje
            return np.array([y[3], y[4], y[5],
                             - e0/m * (y[0]*y[4]*Bz(y[0], y[2]) + Erho(y[0], y[2])) + y[0]*np.square(y[4]),
                             (e0/m * (y[3] * Bz(y[0], y[2]) - y[5] * Brho(y[0], y[2])) - 2*y[3]*y[4]) / y[0],
                             e0/m * (y[4] * y[0] * Brho(y[0], y[2]) - Ez(y[0], y[2]))])

    return gibalna_en


def get_stopping_conditions(Zf, Rt, Rs, Zb, Tf):
    def padel_na_senzor(t, y):
        return (y[2] - (Zf - 1e-6)) + (1 + np.sign(y[0] - Rs))

    def zadel_koncno_steno(t, y):
        return y[2] - Zf

    def zadel_ob_rob_cevi(t, y):
        return y[0] - Rt

    def zbezal_nazaj(t, y):
        return y[2] - Zb

    def potekel_cas(t, y):
        return t - Tf * 0.999

    padel_na_senzor.terminal = True
    zadel_koncno_steno.terminal = True
    zadel_ob_rob_cevi.terminal = True
    zbezal_nazaj.terminal = True
    potekel_cas.terminal = True

    stopping_conds = [padel_na_senzor, zadel_koncno_steno, zadel_ob_rob_cevi, zbezal_nazaj, potekel_cas]

    def human_readable_end_modes(end_modes):
        return [stopping_conds[mode].__name__ for mode in end_modes]

    return stopping_conds, human_readable_end_modes


@cache
def data_for_cylinder_along_z(center_x, center_y, radius, start_Z, height_z):
    z = np.linspace(start_Z, height_z, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center_x
    y_grid = radius * np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid


def draw_sim(res, Rs):
    Xc, Yc, Zc = data_for_cylinder_along_z(0., 0., Rt, Zb, Zf)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(res.y[0] * np.cos(res.y[1]), res.y[0] * np.sin(res.y[1]), res.y[2])
    ax.plot_surface(Xc, Yc, Zc, alpha=0.3)
    p_sens = Circle((0, 0), Rs, alpha=0.3)
    ax.add_patch(p_sens)
    art3d.pathpatch_2d_to_3d(p_sens, z=Zf, zdir="z")
    plt.show()


def get_single_y0_simulation_function(gibalna_en, stopping_conds, Tf=1e-5):
    def sim_single_y0(y0):
        return solve_ivp(gibalna_en, t_span=(0, Tf), y0=y0, events=stopping_conds, method="DOP853", rtol=1e-7, atol=1e-11)
    return sim_single_y0


def run_sim(y0s, gibalna_en, stopping_conds, Tf, draw_trajectories=False, Rs=None, num_processes=4):
    """Vhod je array pravilnih y0 vektorjev, izhod je array časov preleta za te y0.
    Not lahko damo tudi en sam začetni pogoj."""
    start = perf_counter()
    y0s = np.atleast_2d(y0s)

    final_times = []
    ending_mode = []
    success = []
    sim_single_y0 = get_single_y0_simulation_function(gibalna_en, stopping_conds, Tf)

    if draw_trajectories:
        # If drawing is needed - run single core one at a time
        for i, y0 in enumerate(y0s):
            res = sim_single_y0(y0)

            if draw_trajectories:
                draw_sim(res, Rs)

            ending_mode.append(np.flatnonzero([len(event) for event in res.t_events])[0])
            if ending_mode[-1] == 0:
                print(f"Success at i:{i}.  {i / (len(y0s) - 1)} done.  ETA: {((perf_counter() - start) / (max(i, 1) / (len(y0s) - 1)) - (perf_counter() - start))/60:.2f}min")
                success.append(i)
                final_times.append(res.t_events[0][0])

    else:   # Else - run parallel processes
        with Pool(num_processes) as p:
            results = p.map(sim_single_y0, y0s)    # Trenutno žal ne podpira napodovanja kulko še traja!

        for i, res in enumerate(results):
            ending_mode.append(np.flatnonzero([len(event) for event in res.t_events])[0])
            if ending_mode[-1] == 0:
                success.append(i)
                final_times.append(res.t_events[0][0])

    return np.array(final_times), np.array(success), np.array(ending_mode)


def get_y0s_xy_plane(rho0, z0, v0, num=1000):
    thetas = np.linspace(0, np.pi, num=num)
    return np.array([rho0*np.ones_like(thetas), np.zeros_like(thetas), z0*np.ones_like(thetas), np.sin(thetas)*v0, np.zeros_like(thetas), np.cos(thetas)*v0]).T


def get_y0s_omni(rho0, phi0, z0, v0, num=1000):
    spherical_angles = construct_omni_spherical_angles(num)
    return transform_spherical_angles_into_y0s(rho0, phi0, z0, v0, spherical_angles)


def get_y0s_sphere_skeleton(rho0, phi0, z0, v0, num=1000):
    spherical_angles = construct_skeleton_spherical_angles(num)
    return transform_spherical_angles_into_y0s(rho0, phi0, z0, v0, spherical_angles)


def transform_spherical_angles_into_y0s(rho0, phi0, z0, v0, spherical_angles):
    theta_sph, phi_sph = spherical_angles
    num = len(theta_sph)

    # Pretvorba iz sferičnih (os phi_sperical poravnana z globalno osjo x), v cilindrične koordinate
    vz = v0 * np.sin(theta_sph) * np.sin(phi_sph)
    vphi = v0 * (np.cos(phi_sph)*np.sin(theta_sph)*np.cos(phi0) - np.sin(phi0)*np.cos(theta_sph)) / rho0
    vrho = v0 * (np.cos(theta_sph)*np.cos(phi0) + np.cos(phi_sph)*np.sin(phi0)*np.sin(theta_sph))

    # y0 je začetni vektor lokacije in hitrosti delca [rho, phi, z, rho', phi', z']
    rho = np.ones(num) * rho0
    z = np.ones(num) * z0
    phi = np.ones(num) * phi0
    return np.array([rho, phi, z, vrho, vphi, vz]).T


def construct_omni_spherical_angles(num):
    # Uniformna porazdelitev po sferi, os phi=0, vedno obrnjena v smeri globalne osi x (smer X žarkov)
    theta_sph = np.arccos(1 - 2 * np.random.random(num))
    phi_sph = 2 * np.pi * np.random.random(num)
    return theta_sph, phi_sph


def construct_skeleton_spherical_angles(num):
    theta_sph = np.concatenate([np.ones(num // 3) * np.pi / 2, np.random.random(num//3) * np.pi,
                                np.random.random(int(num - 2 * num // 3)) * np.pi])
    phi_sph = np.concatenate([2 * np.pi * np.random.random(num // 3), np.random.randint(0, 2, num // 3) * np.pi,
                              (np.random.randint(0, 2, int(num - 2 * num // 3)) * 2 - 1) * np.pi / 2])
    return theta_sph, phi_sph


def visualize_y0s(y0s, t_prop):
    def propagate_y0s(y0s, t_prop):
        zero_field = lambda rho, phi: 0
        gibalna_en = get_gibalna_en(zero_field, zero_field)

        final_positions = []
        for i, y0 in enumerate(y0s):
            T0 = t_prop
            t_eval = np.linspace(0, T0, num=3)
            res = solve_ivp(gibalna_en, t_span=(0, T0), y0=y0, t_eval=t_eval)

            final_positions.append(res.y[:3, -1])
        return np.array(final_positions)

    fin_pos = propagate_y0s(y0s, t_prop)

    # 3D plot rešitev
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot([0, 1], [0, 0], [0, 0], 'k-')
    ax.plot([0, 0], [0, 1], [0, 0], 'k--')
    ax.plot([0, 0], [0, 0], [0, 1], 'k-.')
    ax.scatter(y0s[0, 0]*np.cos(y0s[0, 1]), y0s[0, 0]*np.sin(y0s[0, 1]), y0s[0, 2], marker="o", color="k")
    ax.scatter(fin_pos[:, 0]*np.cos(fin_pos[:, 1]), fin_pos[:, 0]*np.sin(fin_pos[:, 1]), fin_pos[:, 2], marker=".", color="b")
    plt.show()


if __name__ == "__main__":
    # NASTAVITVE SIMULACIJE --------------------------------------------------------------------------------------------
    # TULJAVA
    L = 1.99200  # Dolžina tuljave v m
    RT = 0.08  # Polmer tuljave v m    # PUSTIL ENAKO
    tokT = 2000  # Razlog je v pdf simulacijaMB_Ziga.nb
    BT = mu0 * tokT / L
    # KROŽNA ZANKA
    a = 0.0035  # Polmer zanke v m
    tok = 3000  # Tok v ta mali tuljavi
    Bmax = mu0 * tok / (2 * a)  # Največje polje zanke v T
    # MEDSEBOJNA POZICIJA ZANKE IN TULJAVE
    d = 0.09560  # Razadlja med zanko in začetkom tuljave v m
    # ELEKTRONSKI PARAMETRI
    EE = 35  # Hitrost elektorna z energijo EE eV v m/s
    v0 = np.sqrt(2 * EE / 27.212) / 137 * 3e8
    # DIMENZIJE DETEKTORJA
    Zf = L + d + 0.01080   # Dolžina cele magnetne steklenice v metrih od vzorca pa do detektorja
    Rt = 0.073/2            # Diameter of the solenoid tube
    Rs = 0.045/2          # Diameter of the sensor -> Estimate
    Zb = -0.1
    Tf = 1e-5           # Končni čas simulacije
    # DIMENZIJE ZASUTAVLJALNE LEČE
    Rl = 0.025/2        # Polmer zaustavljalne leče elektrostatske [m]
    hl = 0.006          # Polovica debeline zaustavljalne elektrostatske leče [m]
    U0 = 1              # Napetost na zaustavljalni leči
    dl = 0.079          # Oddaljenost središča leče od vzorca [m]
    # PREDPOSTAVKA da kot leča delujeta ta oddaljena dva cilindra
    # ------------------------------------------------------------------------------------------------------------------

    # Trenutno nastavljene natančnosti rtol=1e-7 in atol=1e-11  kar nam da:
    # Pri majhnem konst polju 0.030T čas izračuna cca. 0.65s za 5e-7s simulacije. Lažji del simualcije.
    # Radij kroženja bi mogel biti 0.00033291980816174986, je pa start: 0.0003329198052185638, end: 0.000332918217777505
    #
    # Pri velikem konst polju 0.6T čas izračuna cca. 10s za 5e-7s simulacije. Zahteven del simualcije.
    # Radij bi mogel biti 1.6645990408087493e-05, je pa start: 1.6645985230218617e-05, end:  1.6643178399020596e-05
    #
    # Nekak tako smo zadali bilanco med natančnostjo in časovno hitrostjo izračuna. Metoda DOP853
    #
    # Rezultirajoč čas izračuna na trajektorijo je cca 2s.

    # # MAGNETIC FIELD ONLY
    # run_name = str(input("Name this simulation run: "))
    # # y0s = get_y0s_omni(0.0001, 0.0, 0.001, v0, num=10)
    # y0s = get_y0s_xy_plane(0.0001, 0.0001, v0, num=5)
    # # y0s = get_y0s_sphere_skeleton(0.0001, 0.0001, 0.0001, v0, num=1000)
    # visualize_y0s(y0s, t_prop=1e-7)
    # Bz_tot = get_Bz_tot_func(d, a, Bmax, L, RT, BT)
    # Brho_tot = get_Brho_tot_func(d, a, Bmax, L, RT, BT)
    # stopping_conds, hr_endmodes = get_stopping_conditions(Zf, Rt, Rs, Zb, Tf)
    # times, successes, end_mode = run_sim(y0s, Bz_tot, Brho_tot, stopping_conds, Tf, draw_trajectories=True, Rs=Rs)
    # # np.savez(f"sim_results/{run_name}.npz", y0s=y0s, times=times, successes=successes, end_mode=end_mode)
    # #
    # # data = np.load(f"sim_results/{run_name}.npz")
    # # print(data["times"], data["successes"],  hr_endmodes(data["end_mode"]))


    # names = ["pregled_po_kotu_z_0_0001_1eV.npz", "pregled_po_kotu_z_0_0005_1eV.npz", "pregled_po_kotu_z_0_001_1eV.npz", "pregled_po_kotu_z_0_01_1eV.npz"]
    # fig, axes = plt.subplots(1, 5)
    # for i, name in enumerate(names):
    #     data = np.load("sim_results/" + name)
    #     visualize_y0s(y0s[data["successes"]], 1e-7)
    #
    #     times = data["times"]
    #     y0s = data["y0s"]
    #     successes = data["successes"]
    #     end_mode = data["end_mode"]
    #     print(f"Final efficiency: {len(successes) / len(y0s)}")
    #     axes[i].plot(np.linspace(0, 180, num=len(end_mode)), end_mode)
    #     axes[i].set_title("z = " + name[18:-8].replace("_", ","))
    #     axes[-1].hist(times, bins="auto", label=name[18:-8].replace("_", ","), alpha=(1-i/6))
    # plt.legend()
    # plt.show()

    # ELECTRIF FILED ALSO
    # run_name = str(input("Name this simulation run: "))
    # y0s = get_y0s_omni(0.0001, 0.0, 0.001, v0, num=10)
    y0s = get_y0s_xy_plane(0.0001, 0.0001, v0, num=5)
    # y0s = get_y0s_sphere_skeleton(0.0001, 0.0001, 0.0001, v0, num=1000)
    visualize_y0s(y0s, t_prop=1e-7)
    Bz_tot = get_Bz_tot_func(d, a, Bmax, L, RT, BT)
    Brho_tot = get_Brho_tot_func(d, a, Bmax, L, RT, BT)
    Ez = get_Ez(Rl, hl, U0, dl=dl)
    Erho = get_Erho(Rl, hl, U0, dl=dl)
    gibalna_en = get_gibalna_en(Bz_tot, Brho_tot, Ez, Erho)
    stopping_conds, hr_endmodes = get_stopping_conditions(Zf, Rt, Rs, Zb, Tf)
    times, successes, end_mode = run_sim(y0s, gibalna_en, stopping_conds, Tf, draw_trajectories=True, Rs=Rs)
    # np.savez(f"sim_results/{run_name}.npz", y0s=y0s, times=times, successes=successes, end_mode=end_mode)
    #
    # data = np.load(f"sim_results/{run_name}.npz")
    # print(data["times"], data["successes"],  hr_endmodes(data["end_mode"]))

