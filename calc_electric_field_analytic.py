import numpy as np
from scipy.optimize import root
from scipy.special import factorial
from fdm import central_fdm
import matplotlib.pyplot as plt
from functools import cache
from scipy.interpolate import RectBivariateSpline


@cache
def get_E_on_axis_func(Rr):
    @cache
    def E_on_axis(zr):
        zr = np.abs(zr)

        # Če je zr PREVELIK metoda ne skonvergira, zato priležemo gor kr en eksponent!
        if zr > 15:
            Ear_at_border = E_on_axis(15)
            # Opažena odvisnost e**(2zr)
            return Ear_at_border * 255/(zr**2)

        # DOBI Ur in Vr
        @cache
        def get_nonlin_UrVr_eqs(R_rel, z_rel):
            # UrVr = np.array([Ur, Vr])
            def UrVr_eqs(UrVr):
                return np.array([(np.exp(UrVr[1]) * np.cos(UrVr[0]) + UrVr[1] + 1) / np.pi - R_rel,
                                 (np.exp(UrVr[1]) * np.sin(UrVr[0]) + UrVr[0]) / np.pi - z_rel])
            return UrVr_eqs

        # Izgleda da dela kr vredu za velik razpon števil
        # TODO: Sprobaj fsolve da vidiš če slučajno dela drugače, če ne, sprobaj če dela hitreje!!
        eqs = get_nonlin_UrVr_eqs(Rr, zr)
        res = root(eqs, np.array([np.pi-0.5, np.pi-1]))
        # print(f"{res.message[:-1]} to values {res.x}.")
        assert res.success
        assert -1e-10 <= res.x[0] <= np.pi and 0 <= res.x[1]

        Ur, Vr = res.x[0], res.x[1]

        # DOBI Epzr
        Epzr = (np.exp(Vr) * np.cos(Ur) + 1)/(np.exp(2*Vr) + 2*np.exp(Vr) * np.cos(Ur) + 1)

        # Dobi Koeficient Kdp
        # Prvo dobi VrP
        VrP = Vr       # ???? Pri koordinatah xr = Rr in zr = zr smo že poračunali.... A uporabim kr to?
        # DODATNO: Poglej če izvira El flux line iz zgornjega dela plošče
        # print("Preverba če fluw line izhaja od zgoraj?", Rr > 2/np.pi)
        # Dobi lrP
        lrP = (1 + VrP - np.exp(VrP)) / np.pi
        # print("lrP:", lrP)
        # Končni dobi UrP tako da rešiš sistem enačb, tokrat xr = 2Rr + |lrP|, zr = 1
        eqs_P = get_nonlin_UrVr_eqs(2*Rr + np.abs(lrP), 1)
        # print(f"Vhodni podatki druge minimazacije: Rr {2*Rr + np.abs(lrP)}, zr 1")
        res = root(eqs_P, np.array([np.pi-0.5, np.pi-0.5]))
        # print(f"{res.message[:-1]} to values {res.x}.")
        assert res.success
        assert -1e-10 <= res.x[0] <= np.pi and 0 <= res.x[1]
        UrP = res.x[0]
        Kdp = 1. / (1 + UrP/np.pi)      # TULE JE MANJKAL /pi KER SEVEDA - rešitve (7, 8) dajo UrP = UP/U0 * np.pi

        # Dobi Kd (za krožno luknjo in ne za dva Plate-a)
        Kd = 2*Kdp - 1

        # Izračunaj polje na osi okrogle luknje!
        Ear = np.pi * Kd * Epzr

        # print(f"Epzr: {Epzr}\nKdp: {Kdp}\nKd: {Kd}\nEar: {Ear}")

        return Ear
    return E_on_axis


@cache
def axisymetric_coefs(k_num):
    coef_fs = np.zeros(k_num)
    for k in range(k_num):
        coef_fs[k] = (-1)**k / (factorial(k)**2 * np.power(2, 2*k))
    return coef_fs


def get_Ez(R, h, U0, expansion_order=3, dl=0):
    @cache
    def Ez(rho, z):
        z = np.abs(z-dl)
        # Če je zr PREVELIK metoda ne skonvergira, zato ga tam kar odrežemo.
        if z/h > 15:
            return 0.

        coefs = axisymetric_coefs(expansion_order)
        E_val = get_E_on_axis_func(R/h)(z/h)
        for k in range(1, expansion_order):
            E_val += coefs[k] * (rho/h)**(2*k) * central_fdm(2*k + 2, deriv=2*k)(get_E_on_axis_func(R/h), z/h)
        return - E_val * U0 / h     # Ker smo v izpeljavi dali +grad(U) namesto -grad(U)
    return Ez


def get_Erho(R, h, U0, expansion_order=3, dl=0):
    @cache
    def Erho(rho, z):
        z = np.abs(z-dl)
        # Če je zr PREVELIK metoda ne skonvergira, zato ga tam kar odrežemo.
        if z/h > 15:
            return 0.

        coefs = axisymetric_coefs(expansion_order)
        E_val = 0.
        for k in range(1, expansion_order):
            E_val += coefs[k] * (rho/h)**(2*k - 1) * 2 * k * central_fdm(2*k + 1, deriv=2*k - 1)(get_E_on_axis_func(R/h), z/h)
        return - E_val * U0 / h       # Ker smo v izpeljavi dali +grad(U) namesto -grad(U)
    return Erho


def load_simion_data(filepath, skiprows, max_rows):
    # Kako dobimo to številko za max_rows? :
    # Dolžina fila (grep -c '^' chamber_20pix_per_mm.patxt) - 2 (spusti zadnji dve) - 3 (sprednji trije)
    # V filu je zapis: x, y, z, is_electrode, potential
    # Funkcija vrne [[x, y, z, is_electrode, potential]_1, []_2, ...]
    # Simion ma x tam kjer mamo mi z in naša xy ravnina je njegova yz ravnina
    return np.loadtxt(filepath, skiprows=skiprows, max_rows=max_rows)


def reposition_simion_data(data, z_offset, points_per_mm):
    xsize_pre = int(np.max(data[:, 0]) + 1)
    mirrored = np.array(sorted(data[xsize_pre:], key=lambda el: (-el[1], el[0])))   # Pri rho=0 ne smeš dat zraven 2x
    mirrored[:, 1] *= -1
    positioned_data = np.concatenate((mirrored, data))
    xsize, ysize = int(np.max(positioned_data[:, 0]) + 1), int(np.max(positioned_data[:, 1]) * 2 + 1)

    # positioned_data = np.copy(data)
    positioned_data[:, 0] = positioned_data[:, 0] * 0.001 / points_per_mm + z_offset
    positioned_data[:, 1] = positioned_data[:, 1] * 0.001 / points_per_mm
    return positioned_data, xsize, ysize


def get_field_funcs_simion(data, U0, xsize, ysize):
    # U0 je v voltih
    spl_rho, spl_z = data[::xsize, 1], data[:xsize, 0]
    spline_rep = RectBivariateSpline(spl_rho, spl_z, data[:, 4].reshape(ysize, xsize))     # Ker hočemo (rho, z) koordinate
    Erho = -spline_rep(spl_rho, spl_z, dx=1)        # E = -grad(U)
    Ez = -spline_rep(spl_rho, spl_z, dy=1)          # E = -grad(U)

    Erho_splrep = RectBivariateSpline(spl_rho, spl_z, Erho * U0)
    Ez_splrep = RectBivariateSpline(spl_rho, spl_z, Ez * U0)

    fig, axes = plt.subplots(1, 3)

    axes[0].plot(Ez_splrep(0., spl_z)[0])

    axes[1].imshow(Ez_splrep(spl_rho, spl_z))

    axes[2].imshow(Ez)
    plt.show()

    def Erho_simion(rho, z):
        z = np.abs(z)
        rho = np.abs(rho)
        if (rho > spl_rho[-1]) or (z > spl_z[-1]):
            return 0.
        return Erho_splrep(rho, z)[0][0]

    def Ez_simion(rho, z):
        z = np.abs(z)
        rho = np.abs(rho)
        if (rho > spl_rho[-1]) or (z > spl_z[-1]):
            return 0.
        return Ez_splrep(rho, z)[0][0]

    return Erho_simion, Ez_simion


def get_simion_field_funcs(filepath="simion/chamber_20pix_per_mm.patxt", skiprows=3, max_rows=7020000,
                      z_offset=0.010, points_per_mm=20, U0=1):
    # Kako dobimo številko za max_rows? :
    # Dolžina fila (grep -c '^' chamber_20pix_per_mm.patxt) - 2 (spusti zadnji dve) - 3 (sprednji trije)
    # V filu je zapis: x, y, z, is_electrode, potential
    # Funkcija vrne [[x, y, z, is_electrode, potential]_1, []_2, ...]
    # Simion ma x tam kjer mamo mi z in naša xy ravnina je njegova yz ravnina
    data = load_simion_data(filepath, skiprows, max_rows)
    pos_data, xsize, ysize = reposition_simion_data(data, z_offset, points_per_mm)
    return get_field_funcs_simion(pos_data, U0, xsize, ysize)


if __name__ == "__main__":
    # # Test 0.1
    # Rr = 0.8
    # zr = 0.9
    # E_on_axis(Rr, zr)

    R = 0.025/2
    h = 0.006

    # Ez = get_Ez(R, h/2 + 0.001, 1/2)        # V Algoritmu je 2*U0 in 2*h med ploščama zato h/2 in U0/2
    # Erho = get_Erho(R, h/2, 1/2)
    #
    # Erho_sim, Ez_sim = get_simion_field_funcs()
    #
    # # TEST 1
    # zs = np.linspace(-5*R, 5*R, num=200)  # Do sem še gre - nato razširi z eksponentno? al pa kr trdo odrežeš?
    # # zs = np.linspace(-100*R, 100*R, num=50000)  # Do sem še gre - nato razširi z eksponentno? al pa kr trdo odrežeš?
    # print("intep!")
    # polja = np.zeros_like(zs)
    # polja_sim = np.zeros_like(zs)
    # for i, z in enumerate(zs):
    #     polja[i] = Ez(0., z)
    #     polja_sim[i] = Ez_sim(0., z + 0.079)    # Da prikaže prav mormo tole računat zamaknjeno za 79 mm
    # plt.plot(zs, polja, label="Near axis priblizek")
    # plt.plot(zs, polja_sim, label="Simion")
    # plt.legend()
    # plt.show()


    # # TEST 2
    # rhos = np.linspace(-R*2, 2*R, num=100)
    # polja = np.zeros_like(rhos)
    # polja_sim = np.zeros_like(rhos)
    # for i, rho in enumerate(rhos):
    #     polja[i] = Erho(rho, h/3)
    #     polja_sim[i] = Erho_sim(rho, h/3 + 0.079)
    # plt.plot(rhos, polja, label="Near axis priblizek")
    # plt.plot(rhos, polja_sim, label="Simion")
    # plt.legend()
    # plt.show()

    # # Test 3
    # rhos = np.linspace(-R, R, num=11)
    # zs = np.linspace(-5*R, 5*R, num=500)
    # krivulje = np.zeros((11, 500))
    # for i, rho in enumerate(rhos):
    #     for j, z in enumerate(zs):
    #         krivulje[i, j] = Erho(rho, z)
    #     plt.plot(zs, krivulje[i], color=(0, 1-i/(len(rhos)-1), i/(len(rhos)-1)))
    # plt.xlabel("z")
    # plt.title("Zeleno -> Modra ... rho = -R -> R")
    # plt.show()

    # # Test 4
    # zs = np.linspace(-5*R, 5*R, num=200)
    # rhos = np.linspace(0, R, num=10)
    # krivulje = np.zeros((10, 200))
    # for i, rho in enumerate(rhos):
    #     for j, z in enumerate(zs):
    #         krivulje[i, j] = Ez(rho, z)
    #     plt.plot(zs, krivulje[i], color=(0, 1-i/(len(rhos)-1), i/(len(rhos)-1)))
    # plt.xlabel("zs")
    # plt.title("Zelena -> Modra ... Rho = -R -> R")
    # plt.show()
