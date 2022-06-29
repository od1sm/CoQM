import numpy as np
from itertools import product
from typing import Optional
from fractions import Fraction

__docformat__ = "google"


def gellmann_func(j: int, k: int, d: int) -> np.ndarray:
    """Function that returns generalized Gell-Mann matrices acting on qudits
    based on
    https://doi.org/10.1088/1751-8113/41/23/23530,\t     https://arxiv.org/abs/0806.1174 \n
    Gell-Mann matrices are used to study the internal color rotations of the gluon fields associated with QCD
    coloured quarks. A gauge color rotation is a spacetime-depentent SU(3) group element
    $$U=\\exp \\left(i \\theta^{k}(\\mathbf{r}, t) \\lambda_{k} / 2\\right)$$
    where summation over the eight indices k is implied.
    Args:
        j (int): Generalized Gell-Mann matrix first index
        k (int): Generalized Gell-Mann matrix second index
        d (int): Dimension of the generalized Gell-Mann matrix
    """
    if j > k:
        gellman_vals = np.zeros((d, d), dtype=np.complex64)
        gellman_vals[j - 1][k - 1] = 1
        gellman_vals[k - 1][j - 1] = 1
    elif k > j:
        gellman_vals = np.zeros((d, d), dtype=np.complex64)
        gellman_vals[j - 1][k - 1] = complex(0, -1)
        gellman_vals[k - 1][j - 1] = complex(0, 1)
    elif j == k and j < d:
        gellman_vals = np.sqrt(2 / (j * (j + 1))) * np.diag(
            [
                1 + 0j if n <= j else (-j + 0j if n == (j + 1) else 0 + 0j)
                for n in range(1, d + 1)
            ]
        )
    else:
        gellman_vals = np.diag([1 + 0.0j for _ in range(1, d + 1)])
    return gellman_vals


def gellman_matrices(d: int) -> list:
    """get_basis
    Returns an orthogonal Hermitian traceless basis of dimension d, with the identity element at the end.\n
    They provide a Lie-algebra-generator basis acting on the fundamental representation of su(d).\n
    The order of the individual Gell-Mann matrices is not the same with the gellmann_func one for 3 dimensions.
    For 3D:\n
    L1 = glm[3] \tL2 = glm[1] \t    L3 = glm[0]\n
    L4 = glm[6]\t     L5 = glm[2]\t    L6 = glm[7]\n
    L7 = glm[5]\t  L8 = glm[4]

    Args:
        d (int): dimensions

    Returns:
        list: Contains all of the individual Gell-Mann matrices. Last one is the identity element
    """
    return [gellmann_func(j, k, d) for j, k in product(range(1, d + 1), repeat=2)]


def prettier_format(array: np.ndarray) -> str:
    """prettier_format A brute-force method for making complex number arrays look better when printed

    Args:
        array (np.ndarray): An array of complex numbers. This is how it looks when printed with Python's print function.
                        e.g.:   [[0.+0.j 0.+0.j 0.+0.j]
                                [0.+0.j 0.+0.j 1.+0.j]
                                [0.+0.j 1.+0.j 0.+0.j]]

    Returns:
        str: A "prettified" string representation of an array of complex numbers.
        The output of the above array should be
            e.g.:   [[0 0 0]
                    [0 0 1]
                    [0 1 0]]
    """
    d_replace = {
        "0.+": "0+",
        "0.-": "0-",
        "0.j": "0 ",
        "1.j": "i",
        "+0 ": "",
        "0+": "",
        "0-": "-",
        " 0. ": " 0 ",
        " 1. ": " 1 ",
        " -1. ": " -1 ",
        "[0. ": "[0 ",
        "[1. ": "[1 ",
        "0.]": "0]",
        "1.]": "1]",
    }
    temp = str(array)
    for key, value in d_replace.items():
        temp = temp.replace(key, value)
    return temp


def set_quarks(dimensions: int = 3) -> dict:
    """Configure the quarks as arrays for 3D dimensions.

    Args:
        dimensions (int, optional): Fail if not dimensions=3. Defaults to 3.

    Raises:
        ValueError:  dimensions = 3 check to make sure that dimensions=3

    Returns:
        dict: dictionary of up, down, strange quarks and their anti-quarks
    """
    if dimensions != 3:
        raise ValueError("This function works only for 3 dimensions")
    u: np.ndarray = np.array([1, 0, 0])  # Up quark
    d: np.ndarray = np.array([0, 1, 0])  # Down quark
    s: np.ndarray = np.array([0, 0, 1])  # Strange quark
    quarks: dict = {
        "up": u,
        "down": d,
        "strange": s,
        "anti-up": -u,
        "anti-down": -d,
        "anti-strange": -s,
    }
    return quarks


def ladder_result(d: int = 3) -> None:
    """Calculates the dot product between every operator and quark and
    if the result is equal to a quark or an anti-quark it would print the result

    Args:
        d (int, optional): dimensions. Defaults to 3.

    Raises:
        ValueError: Because quarks and operators are only configured for three dimensions, the function is only intended to run in those dimensions.
    """
    if d != 3:
        raise ValueError("This function works only for 3 dimensions")
    quarks: dict = set_quarks()
    operators: dict = set_operators()
    for qrk in quarks:
        for opr in operators:
            newquark = check_which_quark(np.dot(quarks[qrk], operators[opr]))
            if newquark != None:
                print(f"Operator {opr} on {qrk} results to {newquark}")


def L_set_3D_only(dimensions: int = 3) -> dict:
    """Define Gell-Mann matrices as specified in the literature.

    Args:
        dimensions (int, optional): dimensions. Defaults to 3.

    Raises:
        ValueError: Gell-Mann matrices are only configured for three dimensions,
        the function is only intended to run in those dimensions and only

    Returns:
        dict: Gell-Mann matrices for 3 dimensions
    """
    if dimensions != 3:
        raise ValueError("This function works only for 3 dimensions")
    vals: list = [[2, 1], [1, 2], [1, 1], [3, 1], [1, 3], [3, 2], [2, 3], [2, 2]]
    L_: dict = {
        i: gellmann_func(vals[i - 1][0], vals[i - 1][1], dimensions)
        for i in range(1, 9)
    }
    return L_


def set_operators(dimensions: int = 3) -> dict:
    """Define the Ladder operators (raising and lowering operators) for 3 dimensions
        $$T_{\pm}=\\frac{1}{2}\\left(\\lambda_{1} \\pm i \\lambda_{2}\\right)$$
        $$V_{\pm}=\\frac{1}{2}\\left(\\lambda_{4} \\pm i \\lambda_{5}\\right)$$
        $$U_{\pm}=\\frac{1}{2}\\left(\\lambda_{6} \\pm i \\lambda_{7}\\right)$$
    Ladder operators allow to move around inside a isospin multiplet.

    Args:
        dimensions (int, optional): dimensions. Defaults to 3.

    Raises:
        ValueError: operators are only configured for three dimensions,
        the function is only intended to run in those dimensions and only.

    Returns:
        dict: dictionary of raising operators and lowering operators
    """
    if dimensions != 3:
        raise ValueError("This function works only for 3 dimensions")
    L_: dict = L_set_3D_only()
    operators: dict = {
        # Raising operators
        "Tp": 0.5 * (L_[1] + 1j * L_[2]),
        "Up": 0.5 * (L_[6] + 1j * L_[7]),
        "Vp": 0.5 * (L_[4] + 1j * L_[5]),
        # Lowering operators
        "Tm": 0.5 * (L_[1] - 1j * L_[2]),
        "Um": 0.5 * (L_[6] - 1j * L_[7]),
        "Vm": 0.5 * (L_[4] - 1j * L_[5]),
    }
    return operators


def calculate_casimir_value(glm: list) -> None:
    """Calculates the Casimir value for a given list of Gell-Mann matrices.
    The squared sum of the Gell-Mann matrices gives the quadratic Casimir operator, a group invariant,
    $$C=\\sum_{i=1}^{8} \\lambda_{i} \\lambda_{i}=\\frac{16}{3} I$$
    $I$ is 3×3 identity matrix. There is another, independent, cubic Casimir operator, as well.


    Args:
        glm (list): All Gell-Mann matrices.
    """
    casimir_value = 0
    for i in range(len(glm) - 1):
        casimir_value += np.sum((np.dot(glm[i], glm[i])))
    print(
        "Casimir Value:",
        (Fraction(np.sum(casimir_value.real) / 3).limit_denominator(1000)),
    )
    return None


def check_which_quark(probably_quark: np.ndarray) -> Optional[str]:
    """Check if x is in the set of SU(3) quarks .

    Args:
        probably_quark (np.ndarray):  the quark to be checked.

    Returns:
        Optional[str]: returns either the quark or None if it is not a quark
    """
    quarks = set_quarks()
    for qrk in quarks:
        if np.array_equal(probably_quark, quarks[qrk]):
            return qrk
    return None
