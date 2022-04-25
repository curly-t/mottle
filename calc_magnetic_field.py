import numpy as np
import math as m
import scipy as sp
from scipy.special import ellipk, ellipe, elliprf, elliprj
import matplotlib.pyplot as plt
from functools import cache
from numba import njit


# GLOBAL CONSTANT
mu0 = 4 * np.pi * 1e-7


@cache
def ellippi(n, m):
    # Based on:
    # https://mathworld.wolfram.com/CarlsonEllipticIntegrals.html
    # Where m == k**2
    return elliprf(0., 1. - m, 1.) + n * elliprj(0., 1. - m, 1., 1. - n) / 3.


@njit(cache=True)
def _kappa_sqd(rho, ceta, RT):
    return 4 * RT * rho / (np.square(RT + rho) + np.square(ceta))


@cache
def _inner_part_BTz(rho, ceta, h_sqd, RT):
    kappa_sqd = _kappa_sqd(rho, ceta, RT)
    return ceta * np.sqrt(kappa_sqd) * (ellipk(kappa_sqd) + (RT - rho)/(RT + rho) * ellippi(h_sqd, kappa_sqd))


@cache
def BTz(rho, z, L, RT, BT):
    # Pozor - rezultat divergira ko je z == 0 ali rho == 0
    # Problem če je rho < 0 kar je itak nefizikalno, mathematica zvozi zato ker dela analitično kulker daleč lahko
    ceta_p = z + L/2
    ceta_m = z - L/2
    h_sqd = 4 * RT * rho / np.square(RT + rho)
    return BT/(4. * np.pi * np.sqrt(RT * rho)) *\
           (_inner_part_BTz(rho, ceta_p, h_sqd, RT) - _inner_part_BTz(rho, ceta_m, h_sqd, RT))


@cache
def _inner_part_BTrho(rho, ceta, RT):
    kappa_sqd = _kappa_sqd(rho, ceta, RT)
    return (kappa_sqd - 2)/np.sqrt(kappa_sqd) * ellipk(kappa_sqd) + 2 / np.sqrt(kappa_sqd) * ellipe(kappa_sqd)


@cache
def BTrho(rho, z, L, RT, BT):
    # Problem če je rho < 0 kar je itak nefizikalno, mathematica zvozi zato ker dela analitično kulker daleč lahko
    if rho == 0.:
        return 0.
    ceta_p = z + L/2
    ceta_m = z - L/2
    return BT / (2. * np.pi) * np.sqrt(RT / rho) * (_inner_part_BTrho(rho, ceta_p, RT) - _inner_part_BTrho(rho, ceta_m, RT))


def B(z, to, a):
    # Odvisnost polja zanke v osi zanke
    return mu0 * a * a * to / (2 * np.power(z*z + a*a, 3/2))


@njit(cache=True)
def _alpha_sqd(a, rho, z):
    return a*a + np.square(rho) + np.square(z) - 2*a*rho


@njit(cache=True)
def _beta_sqd(a, rho, z):
    return a*a + np.square(rho) + np.square(z) + 2*a*rho


@njit(cache=True)
def _k_sqd(alpha_sqd, beta_sqd):
    return 1 - alpha_sqd/beta_sqd


@cache
def Bz(rho, z, a, Bmax):
    if rho == 0.:
        return (Bmax * (a ** 3)) / ((a ** 2 + z ** 2) ** (3 / 2))
    # Komponenta polja krozne zanke po prostoru
    alpha_sqd = _alpha_sqd(a, rho, z)
    beta_sqd = _beta_sqd(a, rho, z)
    k_sqd = _k_sqd(alpha_sqd, beta_sqd)
    return Bmax * a / (np.pi * alpha_sqd * np.sqrt(beta_sqd)) *\
           ((a*a - np.square(rho) - np.square(z)) * ellipe(k_sqd) + alpha_sqd * ellipk(k_sqd))


@cache
def Brho(rho, z, a, Bmax):
    # ČE rho == 0, Bz izražen z obosno formulo
    if rho == 0.:
        return 0.
    if abs(rho) < (0.01 * a):   # Neka reasonable meja, sem preveril da je vsaj v testu na 15 decimalk točno
        return 3 * (a ** 3) * Bmax * 0.1 * a * rho / (2 * (a ** 2 + (0.1 * a) ** 2) ** (5 / 2))
    alpha_sqd = _alpha_sqd(a, rho, z)
    beta_sqd = _beta_sqd(a, rho, z)
    k_sqd = _k_sqd(alpha_sqd, beta_sqd)
    return Bmax * a * z / (np.pi * alpha_sqd * np.sqrt(beta_sqd) * rho) *\
           ((a*a + np.square(rho) + np.square(z)) * ellipe(k_sqd) - alpha_sqd * ellipk(k_sqd))


@cache
def Bz_tot(rho, z, d, a, Bmax, L, RT, BT):
    return Bz(rho, z, a, Bmax) + BTz(rho, z - L/2 - d, L, RT, BT)


@cache
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
    # BTzs = np.zeros(1000)
    # for i, z in enumerate(zs):
    #     BTzs[i] = BTz(1e-7, z, L, RT, BT)
    # plt.plot(zs, BTzs)
    # plt.show()
    #
    # # Test 2
    # rhos = np.linspace(1e-10, 2*RT, num=1000)
    # BTrhos = np.zeros(1000)
    # for i, rho in enumerate(rhos):
    #     BTrhos[i] = BTrho(rho, 0.001, L, RT, BT)
    # plt.plot(rhos, BTrhos)
    # plt.show()
    #
    # # Test 3
    # zs = np.linspace(0, 3*a, num=100)
    # Bzs = np.zeros(100)
    # for i, z in enumerate(zs):
    #     Bzs[i] = Bz(0, z, a, Bmax)
    # plt.plot(zs, B(zs, tok, a), 'y')
    # plt.plot(zs, Bzs, 'k--')
    # plt.show()
    #
    # # Test 4
    # rhos = np.linspace(0, 5*a, num=1000)
    # Bzs = np.zeros(1000)
    # for i, rho in enumerate(rhos):
    #     Bzs[i] = Bz(rho, a/30, a, Bmax)
    # plt.plot(rhos, Bzs)
    # plt.show()
    #
    # # Test 5
    # rhos = np.linspace(0, 3*a, num=1000)
    # Brhos = np.zeros(1000)
    # for i, rho in enumerate(rhos):
    #     Brhos[i] = Brho(rho, 0.1*a, a, Bmax)
    # plt.plot(rhos, Brhos)
    # plt.show()
    # # Glede na tak hiter pregled pri 0.01*a lahko preklopimo na near axis approximation,
    # # da je hitreje, in še zmeraj do 15 decimalk točno, vsaj za (z=0.1*a) tem testnem primeru.
    #
    # # Test 6
    # # Obe polji skupaj
    # zs = np.linspace(1e-10, 0.5, num=5000)
    # # Zji so v izračunu BTz zamaknjeni za (L/2 + d) ker se koordinatni sistem začne pri ta mali tuljavi
    # BTzs = np.zeros(5000)
    # Bz_tots = np.zeros(5000)
    # for i, z in enumerate(zs):
    #     BTzs[i] = BTz(1e-7, z - L/2 - 0.1, L, RT, BT)
    #     Bz_tots[i] = Bz_tot(1e-7, z, 0.1, a, Bmax, L, RT, BT)
    # plt.plot(zs, B(zs, tok, a), zs, BTzs, zs, Bz_tots)
    # plt.ylim(0, 0.002)
    # plt.xlim(0, 0.5)
    # plt.show()


