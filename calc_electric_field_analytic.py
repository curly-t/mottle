import numpy as np
from scipy.optimize import root
from scipy.special import factorial
from fdm import central_fdm
import matplotlib.pyplot as plt
from functools import cache


@cache
def get_E_on_axis_func(Rr):
    @cache
    def E_on_axis(zr):
        zr = np.abs(zr)

        # Če je zr PREVELIK metoda ne skonvergira, zato priležemo gor kr en eksponent!
        if zr > 20:
            Ear_at_border = E_on_axis(20)
            # Opažena odvisnost e**(2zr)
            return Ear_at_border * 400/(zr**2)

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
        if z/h > 20:
            return 0.

        coefs = axisymetric_coefs(expansion_order)
        E_val = get_E_on_axis_func(R/h)(z/h)
        for k in range(1, expansion_order):
            E_val += coefs[k] * rho**(2*k) * central_fdm(2*k + 2, deriv=2*k)(get_E_on_axis_func(R/h), z/h)
        return E_val * U0
    return Ez


def get_Erho(R, h, U0, expansion_order=3, dl=0):
    @cache
    def Erho(rho, z):
        z = np.abs(z-dl)
        # Če je zr PREVELIK metoda ne skonvergira, zato ga tam kar odrežemo.
        if z/h > 20:
            return 0.

        coefs = axisymetric_coefs(expansion_order)
        E_val = 0.
        for k in range(1, expansion_order):
            E_val += coefs[k] * rho**(2*k - 1) * 2 * k * central_fdm(2*k + 1, deriv=2*k - 1)(get_E_on_axis_func(R/h), z/h)
        return E_val * U0
    return Erho


if __name__ == "__main__":
    # # Test 0.1
    # Rr = 0.8
    # zr = 0.9
    # E_on_axis(Rr, zr)

    R = 0.025/2
    h = 0.006

    # Ez = get_Ez(R, h, 1)
    # Erho = get_Erho(R, h, 1)
    #
    # # TEST 1
    # zs = np.linspace(-5*R, 5*R, num=200)  # Do sem še gre - nato razširi z eksponentno? al pa kr trdo odrežeč?
    # polja = np.zeros_like(zs)
    # for i, z in enumerate(zs):
    #     polja[i] = Ez(0., z)
    # plt.plot(zs, polja)
    # plt.show()
    #
    # # TEST 2
    # rhos = np.linspace(-R/2, R/2, num=100)
    # polja = np.zeros_like(rhos)
    # for i, rho in enumerate(rhos):
    #     polja[i] = Erho(rho, h/2)
    # plt.plot(rhos, polja)
    # plt.show()
    #
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
    #
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

