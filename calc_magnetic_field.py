import numpy as np
import math as m
import scipy as sp
from scipy.special import ellipk, ellipe, elliprf, elliprj
import matplotlib.pyplot as plt

# GLOBAL CONSTANT
mu0 = 4 * np.pi * 1e-7


def ellippi(n, m):
    # Based on:
    # https://mathworld.wolfram.com/CarlsonEllipticIntegrals.html
    # Where m == k**2
    return elliprf(0., 1. - m, 1.) + n * elliprj(0., 1. - m, 1., 1. - n) / 3.


def _kappa_sqd(rho, ceta, RT):
    return 4 * RT * rho / (np.square(RT + rho) + np.square(ceta))


def _inner_part_BTz(rho, ceta, h_sqd, RT):
    kappa_sqd = _kappa_sqd(rho, ceta, RT)
    return ceta * np.sqrt(kappa_sqd) * (ellipk(kappa_sqd) + (RT - rho)/(RT + rho) * ellippi(h_sqd, kappa_sqd))


def BTz(rho, z, L, RT, BT):
    # Pozor - rezultat divergira ko je z == 0 ali rho == 0
    # Problem če je rho < 0 kar je itak nefizikalno, mathematica zvozi zato ker dela analitično kulker daleč lahko
    ceta_p = z + L/2
    ceta_m = z - L/2
    h_sqd = 4 * RT * rho / np.square(RT + rho)
    return BT/(4. * np.pi * np.sqrt(RT * rho)) *\
           (_inner_part_BTz(rho, ceta_p, h_sqd, RT) - _inner_part_BTz(rho, ceta_m, h_sqd, RT))


def _inner_part_BTrho(rho, ceta, RT):
    kappa_sqd = _kappa_sqd(rho, ceta, RT)
    return (kappa_sqd - 2)/np.sqrt(kappa_sqd) * ellipk(kappa_sqd) + 2 / np.sqrt(kappa_sqd) * ellipe(kappa_sqd)


def BTrho(rho, z, L, RT, BT):
    # Problem če je rho < 0 kar je itak nefizikalno, mathematica zvozi zato ker dela analitično kulker daleč lahko
    ceta_p = z + L/2
    ceta_m = z - L/2
    return BT / (2. * np.pi) * np.sqrt(RT / rho) * (_inner_part_BTrho(rho, ceta_p, RT) - _inner_part_BTrho(rho, ceta_m, RT))


def B(z, to, a):
    # Odvisnost polja zanke v osi zanke
    return mu0 * a * a * to / (2 * np.power(z*z + a*a, 3/2))


def _alpha_sqd(a, rho, z):
    return a*a + np.square(rho) + np.square(z) - 2*a*rho


def _beta_sqd(a, rho, z):
    return a*a + np.square(rho) + np.square(z) + 2*a*rho


def _k_sqd(alpha_sqd, beta_sqd):
    return 1 - alpha_sqd/beta_sqd


def Bz(rho, z, a, Bmax):
    # Komponenta polja krozne zanke po prostoru
    alpha_sqd = _alpha_sqd(a, rho, z)
    beta_sqd = _beta_sqd(a, rho, z)
    k_sqd = _k_sqd(alpha_sqd, beta_sqd)
    return Bmax * a / (np.pi * alpha_sqd * np.sqrt(beta_sqd)) *\
           ((a*a - np.square(rho) - np.square(z)) * ellipe(k_sqd) + alpha_sqd * ellipk(k_sqd))


def Brho(rho, z, a, Bmax):
    alpha_sqd = _alpha_sqd(a, rho, z)
    beta_sqd = _beta_sqd(a, rho, z)
    k_sqd = _k_sqd(alpha_sqd, beta_sqd)
    return Bmax * a * z / (np.pi * alpha_sqd * np.sqrt(beta_sqd) * rho) *\
           ((a*a + np.square(rho) + np.square(z)) * ellipe(k_sqd) - alpha_sqd * ellipk(k_sqd))


def Bz_tot(rho, z, d, a, Bmax, L, RT, BT):
    return Bz(rho, z, a, Bmax) + BTz(rho, z - L/2 - d, L, RT, BT)


def Brho_tot(rho, z, d, a, Bmax, L, RT, BT):
    return Brho(rho, z, a, Bmax) + BTrho(rho, z - L/2 - d, L, RT, BT)


def get_Bz_tot_func(d, a, Bmax, L, RT, BT):
    def Bz_tot_func(rho, z):
        return Bz_tot(rho, z, d, a, Bmax, L, RT, BT)
    return Bz_tot_func


def get_Brho_tot_func(d, a, Bmax, L, RT, BT):
    def Brho_tot_func(rho, z):
        return Brho_tot(rho, z, d, a, Bmax, L, RT, BT)
    return Brho_tot_func


if  __name__ == "__main__":
    # TULJAVA
    L = 1.99200  # Dolžina tuljave v m
    RT = 0.08  # Polmer tuljave v m    # PUSTIL ENAKO
    tokT = 2000  # Razlog je v pdf simulacijaMB_Ziga.nb
    BT = mu0 * tokT / L
    # KROŽNA ZANKA
    a = 0.0035  # Polmer zanke v m
    tok = 3000  # Tok v ta mali tuljavi
    Bmax = mu0 * tok / (2 * a)  # Največje polje zanke v T

    # # Test 1
    # zs = np.linspace(-3 * L / 4, 3 * L / 4, num=1000)
    # plt.plot(zs, BTz(1e-7, zs, L, RT, BT))
    # # Podtest - premaknjena tuljava
    # zs = np.linspace(0, 3*L/2, num=1000)
    # d = 0.1
    # plt.plot(zs, BTz(1e-7, zs - d - L/2, L, RT, BT))
    # plt.show()
    #
    # # Test 2
    # rhos = np.linspace(0, 2*RT, num=1000)
    # plt.plot(rhos, BTrho(rhos, 0.001, L, RT, BT))
    # plt.show()
    #
    # # Test 3
    # zs = np.linspace(0, 3*a, num=100)
    # plt.plot(zs, B(zs, tok, a), 'y')
    # plt.plot(zs, Bz(0, zs, a, Bmax), 'k--')
    # plt.show()
    #
    # # Test 4
    # rhos = np.linspace(-5 *a, 5*a, num=1000)
    # plt.plot(rhos, Bz(rhos, a/30, a, Bmax))
    # plt.show()
    #
    # # Test 5
    # # TALE ČUDNE REZULATTE MEČE - namesto 2 je 2000, sicer zgleda OK
    # rhos = np.linspace(0, 3*a, num=1000)
    # plt.plot(rhos, Brho(rhos, 0.1*a, a, Bmax))
    # plt.show()
    #
    # # Test 6
    # # Obe polji skupaj
    # zs = np.linspace(0, 0.5, num=5000)
    # # Zji so v izračunu BTz zamaknjeni za (L/2 + d) ker se koordinatni sistem začne pri ta mali tuljavi
    # plt.plot(zs, B(zs, tok, a), zs, BTz(1e-7, zs - L/2 - 0.1, L, RT, BT), zs, Bz_tot(1e-7, zs, 0.1, a, Bmax, L, RT, BT))
    # plt.ylim(0, 0.002)
    # plt.xlim(0, 0.5)
    # plt.show()

    print("Done!")

