import pathlib
from numba import jit
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from alive_progress import alive_bar

# from typing import
dpisiz = 1000


def savim(dir: str, name: str) -> None:
    """Save a picture of the current image. If dpisiz is too high, save a lower resolution photo as well.

    Args:
        dir (str): directory to save the image
        name (str): name of the image
    """

    if dpisiz >= 200:
        path = pathlib.Path(f"./{dir}/low_res")
        path.mkdir(
            exist_ok=True,  # Without exist_ok=True,
            # FileExistsError show up if folder already exists
            parents=True,
        )
        path = pathlib.Path(f"./{dir}/high_res")
        path.mkdir(
            exist_ok=True,  # Without exist_ok=True,
            # FileExistsError show up if folder already exists
            parents=True,
        )  # Missing parents of the path are created.
        plt.savefig(f"./{dir}/low_res/{name}.png", dpi=200)
        plt.savefig(f"./{dir}/high_res/{name}.png", dpi=dpisiz)
        print(f"Saved image at location: ./{dir}/low_res/{name}.png")
        print(f"Saved image at location: ./{dir}/high_res/{name}.png")
    else:
        path = pathlib.Path(f"./{dir}")
        path.mkdir(
            exist_ok=True,  # Without exist_ok=True,
            # FileExistsError show up if folder already exists
            parents=True,
        )
        print(f"Saved image at location: ./{dir}/{name}.png")


def P(n: int, x: np.ndarray) -> float:
    if n == 0:
        return 1  # P0 = 1
    elif n == 1:
        return x  # P1 = x
    else:
        if n < 0:
            raise ValueError("n can't be less than 1")
        return (((2 * n) - 1) * x * P(n - 1, x) - (n - 1) * P(n - 2, x)) / float(n)


# @jit()
def P4(n: int, m: int, x: np.ndarray) -> float:
    if m == 0:
        return P(n, x)
    if abs(m) > abs(n):
        return 0
    if n < 0:
        return P4(abs(n) - 1, m, x)
    if m < 0:
        return (
            (-1) ** m
            * (np.math.factorial(n - m))
            / (np.math.factorial(n + m))
            * P4(n, abs(m), x)
        )

    return x * P4(n - 1, m, x) - (n + m - 1) * np.sqrt(1 - x**2) * P4(n - 1, m - 1, x)


def plot_init():
    plt.figure()
    plt.grid()
    plt.xlabel("X")
    plt.ylabel("Pn")
    plt.tight_layout()


def legendre_plotter(
    n_m: int, range_wanted: tuple, choose_fixed: str, number_of_values: int = 1000
) -> None:
    range_start, range_end = range_wanted
    x = np.linspace(-1, 1, number_of_values)
    if range_start > n_m or range_end > n_m:
        print(f"For values that m>n, the result is 0")
        array = [range_start, range_end, n_m]
        array.sort()
        range_end = array[1]
    if choose_fixed == "n":
        with alive_bar(range_end - range_start) as bar:
            for i in range(range_start, range_end):
                legendre_func = legendre_calc_for_plot(
                    n_m, i, x, choose_fixed, number_of_values
                )
                # Labelling according to order
                plt.plot(x, legendre_func, label=f"$P_{{{n_m}}}^{{{i}}}$")
                bar()
        plt.title(f"Associated Legendre functions for n={n_m}")
        plt.legend(loc="best")
        plt.tight_layout()
    elif choose_fixed == "m":
        with alive_bar(range_end - range_start) as bar:
            for i in trange(range_start, range_end):
                legendre_func = legendre_calc_for_plot(
                    i, n_m, x, choose_fixed, number_of_values
                )
                # Labelling according to order
                plt.plot(x, legendre_func, label=f"$P_{{{i}}}^{{{n_m}}}$")
                bar()
        if n_m == 0:
            plt.title(f"Legendre functions")
        else:
            plt.title(f"Associated Legendre functions for m={n_m}")
        plt.legend(loc="best")
        plt.tight_layout()
    else:
        raise ValueError(f"Choose_fixed is {choose_fixed} which is not 'n' or 'm'")


def legendre_calc_for_plot(
    n_m: int,
    i: int,
    x: np.ndarray,
    choose_fixed: str,
    number_of_values: int,
) -> np.ndarray:
    if choose_fixed == "n":
        if i == 0:
            yaxisplt = np.full(number_of_values, P4(n_m, i, x))
        if n_m > i:
            yaxisplt = np.full(number_of_values, P4(n_m, i, x))
        else:
            yaxisplt = np.full(number_of_values, P4(n_m, i, x))
    elif choose_fixed == "m":
        if i == 0:
            yaxisplt = np.full(number_of_values, P4(i, n_m, x))
        if n_m > i:
            yaxisplt = np.full(number_of_values, P4(i, n_m, x))
        else:
            yaxisplt = np.full(number_of_values, P4(i, n_m, x))
    else:
        raise ValueError(f"Choose_fixed is {choose_fixed} which is not 'n' or 'm'")

    return yaxisplt


def numerical_solution_of_legendre(n, m, returnplot=False):
    CosTheta = np.zeros((1999), float)
    Plm = np.zeros((1999), float)
    y = [0] * (2)
    dCos = 0.001
    el = n  # m intger  m<=el,   m = 1,2,3,...
    if el == 0 or el == 2:
        y[0] = 1
    if el > 2 and (el) % 2 == 0:
        if m == 0:
            y[0] = -1
        elif m > 0:
            y[0] = 1
        elif m < 0 and abs(m) % 2 == 0:
            y[0] = 1
        elif m < 0 and abs(m) % 2 == 1:
            y[0] = -1
    if el > 2 and el % 2 == 1:
        if m == 0:
            y[0] = 1
        elif m > 0:
            y[0] = -1
        elif m < 0:
            y[0] = 1
    y[1] = 1

    def f(Cos, y):  # RHS of equation
        rhs = np.zeros(2)  # Declare array dimension
        rhs[0] = y[1]
        rhs[1] = 2 * Cos * y[1] / (1 - Cos**2) - (
            el * (el + 1) - m**2 / (1 - Cos**2)
        ) * y[0] / (1 - Cos**2)
        return rhs

    f(0, y)  # Call function for xi = 0 with init conds.
    i = -1

    def rk4Algor(t, h, N, y, f):
        k1 = np.zeros(N)
        k2 = np.zeros(N)
        k3 = np.zeros(N)
        k4 = np.zeros(N)
        k1 = h * f(t, y)
        k2 = h * f(t + h / 2.0, y + k1 / 2.0)
        k3 = h * f(t + h / 2.0, y + k2 / 2.0)
        k4 = h * f(t + h, y + k3)
        y = y + (k1 + 2 * (k2 + k3) + k4) / 6.0
        return y

    for Cos in np.arange(-0.999999, 1 - dCos, dCos):
        i = i + 1
        CosTheta[i] = Cos
        y = rk4Algor(Cos, dCos, 2, y, f)  # call runge kutt
        Plm[i] = y[0]  #
    if returnplot == True:
        plt.plot(CosTheta, Plm, label=f"$P_{{{el}}}^{{{m}}}$ with RK4")
        plt.legend(loc="best")
        plt.tight_layout()
    else:
        return CosTheta, Plm
