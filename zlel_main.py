#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. module:: zlel_main.py
    :synopsis:

.. moduleauthor:: YOUR NAME AND E-MAIL


"""

import zlel.zlel_p1 as zl1
import sys

"""
https://stackoverflow.com/questions/419163/what-does-if-name-main-do
https://stackoverflow.com/questions/19747371/
python-exit-commands-why-so-many-and-when-should-each-be-used
"""
if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "cirs/examples/0_zlel_V_R_Q.cir"
    # Parse the circuit
    [cir_el, cir_nd, cir_val, cir_ctr, orders] = zl1.cir_parser(filename)

    # Get nodes list
    nd_list = zl1.get_nd_list(cir_nd)

    # get element quantity
    el_num = zl1.get_element_N(cir_el)

    # get branchs list
    branch_list = zl1.get_branches(cir_el, cir_nd, cir_val, cir_ctr)

    # get Aa and A
    Aa = zl1.get_Aa(nd_list, branch_list)
    A = zl1.get_A(Aa)

    # get M N Us T and U arrays
    M = zl1.get_M_matrix(branch_list)
    N = zl1.get_N_matrix(branch_list)
    Us = zl1.get_Us_matrix(branch_list)
    T = zl1.get_T_matrix(M, N, A)
    U = zl1.get_U_matrix(A, Us)

    # for .DC and .TR
    filenameDC = filename[:-3] + "dc"
    filenameTR = filename[:-3] + "tr"

    # solve orders
    zl1.solve_orders(T, U, orders, branch_list, nd_list, el_num, Aa)
