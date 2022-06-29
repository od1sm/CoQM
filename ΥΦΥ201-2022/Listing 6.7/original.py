from legendre import savim, plot_init, numerical_solution_of_legendre


def main():
    plot_init
    n = 0
    m = 0
    numerical_solution_of_legendre(n, m, returnplot=True)
    savim("images", "numerical_0_0")


if __name__ == "__main__":
    main()
