import numpy as np
from scipy.optimize import root
from fdm import central_fdm
import matplotlib.pyplot as plt


def E_on_axis(Rr, zr):
    zr = np.abs(zr)
    # DOBI Ur in Vr
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



def Ez(rho, z, Rr, h, fdm_order=3):
    # TODO Pogruntaj kakšni so koeficienti 1/4 1/64 ...? da lahko razviješ do poljubenga reda!
    return E_on_axis(Rr, z/h) - 1/4 * central_fdm(fdm_order, deriv=2)(lambda x: E_on_axis(Rr, x), z/h) * rho**2 + 1/64 * central_fdm(fdm_order+2, deriv=4)(lambda x: E_on_axis(Rr, x), z/h) * rho**4


def Erho(rho, z, Rr, h, fdm_order=2):
    return -1/2 * central_fdm(fdm_order, deriv=1)(lambda x: E_on_axis(Rr, x), z/h)*rho + 1/16 * central_fdm(fdm_order+2, deriv=3)(lambda x: E_on_axis(Rr, x), z/h)*rho**3


if __name__ == "__main__":
    # # Test 1
    # Rr = 0.8
    # zr = 0.9
    # E_on_axis(Rr, zr)



    R = 0.025/2
    h = 0.006
    z = 0.0
    print(Ez(0, z, R/h, h))

    zs = np.linspace(-5*R, 5*R, num=100)
    polja = np.zeros_like(zs)
    for i, z in enumerate(zs):
        polja[i] = Ez(0., z, R/h, h)
    plt.plot(zs, polja)
    plt.show()

    rhos = np.linspace(-R/2, R/2, num=100)
    polja = np.zeros_like(rhos)
    for i, rho in enumerate(rhos):
        polja[i] = Ez(rho, 0., R/h, h)
    plt.plot(rhos, polja)
    plt.show()