import numpy as np, matplotlib.pylab as plt
from legendre import savim, plot_init, numerical_solution_of_legendre, P, P4


def main():
    plt.figure(figsize=(15, 15))
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    number_of_values = 1000
    x = np.linspace(-1, 1, number_of_values)
    for i in range(2, 6):
        n = 0
        m = 2
        if i == 0:
            yaxisplt = np.full(number_of_values, P4(i, m, x))
        if m > n:
            yaxisplt = np.full(number_of_values, P4(i, m, x))
        else:
            yaxisplt = np.full(number_of_values, P4(i, m, x))
        # Labelling according to order
        axs[0].plot(x, yaxisplt, label=f"$P_{{{i}}}^{{{m}}}$ analytical")
        CosTheta, Plm = numerical_solution_of_legendre(
            i,
            m,
        )
        axs[1].plot(CosTheta, Plm, label=f"$P_{{{i}}}^{{{m}}}$ non-analytical")
    axs[0].set_title("Analytical solution")
    axs[1].set_title("RK4 Numerical solution")
    axs[0].legend(loc="best")
    axs[1].legend(loc="best")
    plt.tight_layout()
    savim("images", "mixed_2_5")


if __name__ == "__main__":
    main()
