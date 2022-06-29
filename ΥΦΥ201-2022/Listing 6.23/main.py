import gellman as gl


def main():
    """main Using a generalized Gell-Mann basis generator function and using numpy to construct
    the matrices representing the operators of the symmetry group SU(3) to describbe constituen quarks.
    It also calculates the Casimir value for an SU(3) basis
    """
    glm = gl.gellman_matrices(3)
    gl.ladder_result()
    gl.calculate_casimir_value(glm)


if __name__ == "__main__":
    main()
