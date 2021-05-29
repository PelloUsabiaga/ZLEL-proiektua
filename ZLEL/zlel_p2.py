#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. module:: zlel_main.py
    :synopsis:

.. moduleauthor:: YOUR NAME AND E-MAIL


"""

import numpy as np
import sys
import matplotlib.pyplot as plt
"""
if __name__ == "__main__":
    import zlel_p1 as zl1
else:
    import zlel.zlel_p1 as zl1
"""


def getTransposeMatrix(matrix):
    
    transpose = np.zeros((len(matrix[0]), len(matrix)))
    for row in range(len(matrix)):
        for el in range(len(matrix[0])):
            transpose[el][row] = matrix[row][el]
    return transpose


def print_solution(sol, b, n):
    """ This function prints the solution with format.

        Args:
            sol: np array with the solution of the Tableau equations
            (e_1,..,e_n-1,v_1,..,v_b,i_1,..i_b)
            b: # of branches
            n: # of nodes

    """

    # The instructor solution needs to be a numpy array of numpy arrays of
    # float. If it is not, convert it to this format.
    if sol.dtype == np.float64:
        np.set_printoptions(sign=' ') # Only from numpy 1.14
        tmp = np.zeros([np.size(sol), 1], dtype=float)
        for ind in range(np.size(sol)):
            tmp[ind]=np.array(sol[ind])                  
        sol = tmp
    print("\n========== Nodes voltage to reference ========")
    for i in range(1, n):
        print("e" + str(i) + " = ", sol[i-1])
    print("\n========== Branches voltage difference ========")
    for i in range(1, b+1):
        print("v" + str(i) + " = ", sol[i+n-2])
    print("\n=============== Branches currents ==============")
    for i in range(1, b+1):
        print("i" + str(i) + " = ", sol[i+b+n-2])

    print("\n================= End solution =================\n")


def build_csv_header(tvi, b, n):
    """ This function build the csv header for the output files.
        First column will be v or i if .dc analysis or t if .tr and it will
        be given by argument tvi.
        The header will be this form,
        t/v/i,e_1,..,e_n-1,v_1,..,v_b,i_1,..i_b

    Args:
        tvi: "v" or "i" if .dc analysis or "t" if .tran
        b: # of branches
        n: # of nodes

    Returns:
        header: The header in csv format as string
    """
    header = tvi
    for i in range(1, n):
        header += ",e" + str(i)
    for i in range(1, b+1):
        header += ",v" + str(i)
    for i in range(1, b+1):
        header += ",i" + str(i)
    return header


def save_as_csv(b, n, filename):
    """ This function gnerates a csv file with the name filename.
        First it will save a header and then, it loops and save a line in
        csv format into the file.

    Args:
        b: # of branches
        n: # of nodes
        filename: string with the filename (incluiding the path)
    """
    # Sup .tr
    header = build_csv_header("t", b, n)
    with open(filename, 'w') as file:
        print(header, file=file)
        # Get the indices of the elements corresponding to the sources.
        # The freq parameter cannot be 0 this is why we choose cir_tr[0].
        t = 0
        while t < 10:
            # for t in tr["start"],tr["end"],tr["step"]
            # Recalculate the Us for the sinusoidal sources

            sol = np.full(2*b+(n-1), t+1, dtype=float)
            # Inserte the time
            sol = np.insert(sol, 0, t)
            # sol to csv
            sol_csv = ','.join(['%.5f' % num for num in sol])
            print(sol_csv, file=file)
            t = t + 1


def plot_from_cvs(filename, x, y, title):
    """ This function plots the values corresponding to the x string of the
        file filename in the x-axis and the ones corresponding to the y
        string in the y-axis.
        The x and y strings must mach with some value of the header in the
        csv file filename.

    Args:
        filename: string with the name of the file (including the path).
        x: string with some value of the header of the file.
        y: string with some value of the header of the file.

    """
    data = np.genfromtxt(filename, delimiter=',', skip_header=0,
                         skip_footer=1, names=True)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(data[x], data[y], color='r', label=title)
    ax1.set_xlabel(x)
    ax1.set_ylabel(y)
    plt.show()


"""
https://stackoverflow.com/questions/419163/what-does-if-name-main-do
https://stackoverflow.com/questions/19747371/
python-exit-commands-why-so-many-and-when-should-each-be-used
"""

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "../cirs/all/1_zlel_V_R_dc.cir"

    b = 2
    n = 2
    matrix = np.array([[1, 2], [3, 4], [7, 8]])
    print(getTransposeMatrix(matrix))
    filename = filename[:-3] + "tr"
    save_as_csv(b, n, filename)
    plot_from_cvs(filename, "t", "e1", "")
