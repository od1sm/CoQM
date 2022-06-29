import matplotlib.pyplot as plt
import numpy as np
from legendre import savim, legendre_plotter, plot_init


def main():
    # Creating an array of x values
    plot_init()
    legendre_plotter(n_m=5, range_wanted=(0, 10), choose_fixed="n")
    savim("images", "an_n_fixed_5_0_5")


if __name__ == "__main__":
    main()
