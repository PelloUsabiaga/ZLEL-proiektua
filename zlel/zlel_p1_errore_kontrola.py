#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. module:: zlel_main.py
    :synopsis:

.. moduleauthor:: YOUR NAME AND E-MAIL


"""

import numpy as np
import sys
if __name__ == "__main__":
    import zlel_p2 as p2
else:
    import zlel.zlel_p2 as p2
import math


def check_file_correct(filename, branch_list, cir_nd, cir_el):
    correct = True
    correct *= check_zero_node_exists(branch_list)
    correct *= check_parallel_voltage_source(branch_list)
    correct *= check_not_series_current_sources(branch_list)
    correct *= check_not_connected_node(branch_list)
    correct *= check_controled_elements_correct(branch_list)
    return correct


def check_controled_elements_correct(branch_list):
    controled_elements = {"G", "E"}
    for branch in branch_list:
        if branch.name[0] in controled_elements:
            if branch.control == ['0']:
                sys.exit(branch.printName
                         + " controled sorce have not controler.")
    return True


def check_not_series_current_sources(branch_list):
    cir_correct = True
    for branch in branch_list:
        if branch.name[0] == "I":
            error_in_branch = False
            in_node = branch.incoming_node
            out_node = branch.outcoming_node
            nodes = [in_node, out_node]
            for node in nodes:
                conected = []
                node_correct = False
                for branch in branch_list:
                    if branch.is_node_in(node):
                        conected.append(branch)
                for branch in conected:
                    if branch.name[0] != "I":
                        node_correct = True
                        break
                if not node_correct:
                    out_curr = 0
                    for branch in conected:
                        out_curr += branch.get_out_current(node)
                    if out_curr != 0:
                        sys.exit("I sources in series at node " + str(node)
                                 + ".")
            if error_in_branch:
                cir_correct = False
                break
    return cir_correct


def check_not_connected_node(branch_list):
    nd_cuantity = dict()
    for branch in branch_list:
        if branch.incoming_node in nd_cuantity:
            nd_cuantity[branch.incoming_node] += 1
        else:
            nd_cuantity[branch.incoming_node] = 1
        if branch.outcoming_node in nd_cuantity:
            nd_cuantity[branch.outcoming_node] += 1
        else:
            nd_cuantity[branch.outcoming_node] = 1
    for nd, cuantity in zip(nd_cuantity.keys(), nd_cuantity.values()):
        if cuantity == 1:
            sys.exit("Node " + str(nd) + " is floating.")
    return True


def get_node_qty(cir_nd):
    node_qty = {}
    for node_list in cir_nd:
        for node in node_list:
            if node in node_qty:
                node_qty[node] += 1
            else:
                node_qty[node] = 1

    return node_qty


def check_zero_node_exists(branch_list):
    for branch in branch_list:
        if branch.incoming_node == 0 or branch.outcoming_node == 0:
            return True
    sys.exit("Reference node \"0\" is not defined in the circuit.")


def check_parallel_voltage_source(branch_list):
    voltage_sources = []
    for branch in branch_list:
        if branch.name[0] == "V":
            for vsource in voltage_sources:
                if (branch.incoming_node == vsource.incoming_node
                        and branch.outcoming_node == vsource.outcoming_node):
                    if branch.value[0] != vsource.value[0]:
                        sys.exit("Parallel V sources at branches "
                                 + str(branch.incoming_node) + " and "
                                 + str(branch.outcoming_node) + ".")
                if (branch.incoming_node == vsource.outcoming_node
                        and branch.outcoming_node == vsource.incoming_node):
                    if branch.value[0] != -vsource.value[0]:
                        sys.exit("Parallel V sources at branches "
                                 + str(branch.incoming_node) + " and "
                                 + str(branch.outcoming_node) + ".")
            voltage_sources.append(branch)
    return True


def cir_parser(filename):
    """
        This function takes a .cir test circuit and parse it into
        4 matices.
        If the file has not the proper dimensions it warns and exit.

    Args:
        filename: string with the name of the file

    Returns:
        cir_el: np array of strings with the elements to parse. size(1,b)
        cir_nd: np array with the nodes to the circuit. size(b,4)
        cir_val: np array with the values of the elements. size(b,3)
        cir_ctrl: np array of strings with the element which branch
        controls the controlled sources. size(1,b)

    Rises:
        SystemExit

    """
    try:
        file = np.array(np.loadtxt(filename, dtype=str))
    except ValueError:
        sys.exit("File corrupted: .cir size is incorrect.")
    index = 0
    for line in file:
        if line[0].find(".") != -1:
            break
        index += 1

    cir = file[:index]
    orders = file[index:]
    for line in cir:
        if line.size != 9:
            raise ValueError

    try:
        cir_el = np.array(cir[:, 0:1])
        cir_nd = np.array(cir[:, 1:5], dtype=int)
        cir_val = np.array(cir[:, 5:8], dtype=float)
        cir_ctrl = np.array(cir[:, 8:9])
    except:
        raise ValueError

    return cir_el, cir_nd, cir_val, cir_ctrl, orders


def get_nd_list(cir_nd):
    """
        This function takes a circuits nodes matrix, and returns a list with
        the diferent nodes numbers.

    Args:
        cir_nd: np array with the nodes to the circuit. size(b,4)

    Returns:
        nd_list: np array with the circuits diferent nodes numbers.

    """
    return np.unique(cir_nd)


def get_element_N(cir_el):
    """
        This function calculates the number of elements at a circuit.

    Args:
        cir_el: np array of strings with the elements to parse. size(1,b)

    Returns:
        element_N: int the number of elements at a circuit.

    """
    return cir_el.size


class branch:
    """
        This class is used to represent branches, and to work creating and
        manipulating them.
    """

    N = None
    name = None
    value = None
    control = None
    outcoming_node = None
    incoming_node = None
    printName = None
    lineal = None
    printN = None
    dinamic = None

    def __init__(self, name, nd, N, printN, value, control, lineal=True,
                 dinamic=False):
        self.name = name.upper()
        self.printName = name
        self.outcoming_node = nd[0]
        self.incoming_node = nd[1]
        self.N = N
        self.value = value
        self.control = control
        self.lineal = lineal
        self.printN = printN
        self.dinamic = dinamic
        for i in range(len(control)):
            control[i] = control[i].upper()

    def insert_DC_value(self, value):
        self.value[0] = value

    def get_out_current(self, node):
        if self.name[0] != "I":
            sys.exit("current source function was called by not current "
                     + "source element")
            raise(ValueError)
        if node == self.outcoming_node:
            return self.value[0]
        else:
            return -self.value[0]

    def __str__(self):
        return("\t" + str(self.printN) + ". branch:\t" + self.printName +
               ",\ti" + str(self.printN) +
               ",\tv" + str(self.printN) +
               "=e" + str(self.outcoming_node) +
               "-e" + str(self.incoming_node) + "\n")

    def __repr__(self):
        return("\t" + str(self.printN) + ". branch:\t" + self.printName +
               ",\ti" + str(self.printN) +
               ",\tv" + str(self.printN) +
               "=e" + str(self.outcoming_node) +
               "-e" + str(self.incoming_node) + "\n")

    def is_node_in(self, node):
        return node == self.outcoming_node or node == self.incoming_node

    def get_oposite_node(self, node):
        if node == self.outcoming_node:
            return self.incoming_node
        else:
            return self.outcoming_node

    def get_M_row(self, branch_list, vi=0.6, vbe=0.6, vbc=0.6, din_val=0, h=1):
        row = np.zeros(len(branch_list))

        if self.name[0] == "R":
            row[self.N] = 1

        elif self.name[0] == "V":
            row[self.N] = 1

        elif self.name[0] == "I":
            pass
        elif self.name[0] == "G":
            row[get_branch_by_name(self.control,
                                   branch_list).N] = self.value[0]

        elif self.name[0] == "E":
            row[self.N] = -1
            row[get_branch_by_name(self.control,
                                   branch_list).N] = self.value[0]

        elif self.name[0] == "H":
            row[self.N] = -1

        elif self.name[0] == "F":
            pass
        elif self.name[0] == "B":
            row[self.N] = 1

        elif self.name[0] == "Y":
            pass

        elif self.name[0] == "D":
            row[self.N] = self.diod_values(vi)[0]

        elif self.name[0] == "Q":
            # BE branch
            if self.name[-1] == "E":
                row[self.N] = self.transistor_values(vbe, vbc)[0]
                row[self.N + 1] = self.transistor_values(vbe, vbc)[1]

                # BC branch
            elif self.name[-1] == "C":
                row[self.N - 1] = self.transistor_values(vbe, vbc)[2]
                row[self.N] = self.transistor_values(vbe, vbc)[3]

        elif self.name[0] == "C":
            row[self.N] = 1

        elif self.name[0] == "L":
            row[self.N] = -1*h/self.value[0]

        elif self.name[0] == "A":
            if self.name[-1] == "N":
                row[self.N] = 1
            if self.name[-1] == "U":
                pass

        return row

    def get_n_row(self, branch_list, Vi=0.6, din_val=0, h=1):
        row = np.zeros(len(branch_list))

        if self.name[0] == "R":
            row[self.N] = -1*self.value[0]

        elif self.name[0] == "V":
            pass
        elif self.name[0] == "I":
            row[self.N] = 1

        elif self.name[0] == "G":
            row[self.N] = 1

        elif self.name[0] == "E":
            pass
        elif self.name[0] == "H":
            row[get_branch_by_name(self.control,
                                   branch_list).N] = self.value[0]

        elif self.name[0] == "F":
            row[self.N] = -1
            row[get_branch_by_name(self.control,
                                   branch_list).N] = self.value[0]

        elif self.name[0] == "B":
            pass
        elif self.name[0] == "Y":
            row[self.N] = 1

        elif self.name[0] == "D":
            row[self.N] = 1

        elif self.name[0] == "Q":
            # BE branch
            if self.name[-1] == "E":
                row[self.N] = 1

                # BC branch
            elif self.name[-1] == "C":
                row[self.N] = 1

        elif self.name[0] == "C":
            row[self.N] = -1*h/self.value[0]

        elif self.name[0] == "L":
            row[self.N] = 1

        elif self.name[0] == "A":
            if self.name[-1] == "N":
                pass
            if self.name[-1] == "U":
                row[self.N - 1] = 1

        return row

    def get_us_row(self, t, vi=0.6, vbc=0.6, din_val=0, h=1):
        row = np.zeros(1)
        if self.name[0] == "R":
            pass
        elif self.name[0] == "V":
            row[0] = self.value[0]

        elif self.name[0] == "I":
            row[0] = self.value[0]

        elif self.name[0] == "G":
            pass
        elif self.name[0] == "E":
            pass
        elif self.name[0] == "H":
            pass
        elif self.name[0] == "F":
            pass
        elif self.name[0] == "B":
            row[0] = self.value[0] * np.sin(2 * np.pi * self.value[1]*t
                                            + np.pi/180 * self.value[2])

        elif self.name[0] == "Y":
            row[0] = self.value[0] * np.sin(2 * np.pi * self.value[1] * t
                                            + np.pi/180 * self.value[2])

        elif self.name[0] == "D":
            row[0] = self.diod_values(vi)[1]

        elif self.name[0] == "Q":
            vbe = vi
            if self.name[-1] == "E":
                row[0] = self.transistor_values(vbe, vbc)[4]

                # BC branch
            elif self.name[-1] == "C":
                row[0] = self.transistor_values(vbe, vbc)[5]

        elif self.name[0] == "C":
            row[0] = din_val

        elif self.name[0] == "L":
            row[0] = din_val

        elif self.name[0] == "A":
            if self.name[-1] == "N":
                row[0] = 0
            if self.name[-1] == "U":
                row[0] = 0

        return row

    def diod_values(self, vj):
        if self.name[0] != "D":
            sys.exit("diod function was called by not diod element")
            raise(ValueError)
        Vt = self.value[1] * 8.6173324e-5 * 300
        I0 = self.value[0]
        Gj = - (I0 / Vt) * math.exp(vj / Vt)
        Ij = I0 * math.exp(vj / Vt) + Gj * vj
        return (Gj, Ij)

    def transistor_values(self, vbe=0.6, vbc=0.6):
        if self.name[0] != "Q":
            sys.exit("diod function was called by not diod element")
            raise(ValueError)
        # if self.control[0] == 1:
        #     vbe = -1*vbe
        #     vbc = -1*vbc
        vt = 8.6173324e-5 * 300
        ies = self.value[0]
        ics = self.value[1]
        b = self.value[2]
        alpha_f = b / (1 + b)
        alpha_r = (ies / ics) * alpha_f
        g11 = -(ies / vt) * math.exp(vbe / vt)
        g22 = -(ics / vt) * math.exp(vbc / vt)
        g12 = -alpha_r * g22
        g21 = -alpha_f * g11
        ie = (g11 * vbe + g12 * vbc + ies * (math.exp(vbe / vt) - 1)
              - alpha_r * ics * (math.exp(vbc / vt) - 1))
        ic = (g21 * vbe + g22 * vbc - alpha_f * ies * (math.exp(vbe / vt) - 1)
              + ics * (math.exp(vbc / vt) - 1))
        return (g11, g12, g21, g22, ie, ic)


def get_branch_by_name(brch_name, branch_list):
    for branch in branch_list:
        if brch_name.__class__ == np.str_:
            brch_name = np.str_.upper(brch_name)
        elif brch_name.__class__ == str:
            brch_name = brch_name.upper()
        if branch.name == brch_name:
            return branch


def get_m_matrix(branch_list, non_lineal_voltages="kaixo",
                 dinamic_values="kaixo", h=0):

    """
        This function takes the branches and especial characteristics of the circuit and parse they to
        M matrix, voltage (Vn) matrix.

    Args:
        branch_list: list of branches of the current circuit.
        non_lineal_voltages: indicates if current circuit has non linear elements. If don´t have it contains "kaixo" str,
            otherwise a list of float with the value for each branch
        dinamic_values: indicates if current circuit has dynamic values. If don´t have it contains "kaixo" str,
            otherwise a list of float with the value for each branch

    Returns:
        matrix : np array matrix of floats with the value for specific Vn
    """

    matrix = np.zeros([len(branch_list), len(branch_list)])
    if type(non_lineal_voltages) == str and type(dinamic_values) == str:
        # circuit is resistive and lineal
        for i in range(len(branch_list)):
            matrix[i] = branch_list[i].get_M_row(branch_list)
    elif type(non_lineal_voltages) == str:
        # circuit is dinamic and lineal
        for i in range(len(branch_list)):
            if math.isnan(dinamic_values[i]):
                matrix[i] = branch_list[i].get_M_row(branch_list)
            else:
                if (branch_list[i].name[0] == "C"
                        or branch_list[i].name[0] == "L"):
                    matrix[i] = branch_list[i].get_M_row(branch_list, din_val=dinamic_values[i], h=h)

    elif type(dinamic_values) == str:
        # circuit is resistive and non lineal
        for i in range(len(branch_list)):
            if math.isnan(non_lineal_voltages[i]):
                matrix[i] = branch_list[i].get_M_row(branch_list)
            else:
                if branch_list[i].name[0] == "Q":
                    if branch_list[i].name[-1] == "E":
                        matrix[i] = branch_list[i].get_M_row(branch_list, vbe=non_lineal_voltages[i], vbc=non_lineal_voltages[i+1])
                    elif branch_list[i].name[-1] == "C":
                        matrix[i] = branch_list[i].get_M_row(branch_list, vbe=non_lineal_voltages[i-1], vbc=non_lineal_voltages[i])
                elif branch_list[i].name[0] == "D":
                    matrix[i] = branch_list[i].get_M_row(branch_list,
                                                         non_lineal_voltages[i])

    else:
        # circuit is dinamic and non lineal
        for i in range(len(branch_list)):
            if math.isnan(non_lineal_voltages[i]) and math.isnan(dinamic_values[i]):
                matrix[i] = branch_list[i].get_M_row(branch_list)
            else:
                if branch_list[i].name[0] == "Q":
                    if branch_list[i].name[-1] == "E":
                        matrix[i] = branch_list[i].get_M_row(branch_list, vbe=non_lineal_voltages[i], vbc=non_lineal_voltages[i+1])
                    elif branch_list[i].name[-1] == "C":
                        matrix[i] = branch_list[i].get_M_row(branch_list, vbe=non_lineal_voltages[i-1], vbc=non_lineal_voltages[i])
                elif branch_list[i].name[0] == "D":
                    matrix[i] = branch_list[i].get_M_row(branch_list,
                                                         non_lineal_voltages[i])
                if branch_list[i].name[0] == "C" or branch_list[i].name[0] == "L":
                    matrix[i] = branch_list[i].get_M_row(branch_list, din_val=dinamic_values[i], h=h)
    return matrix


def get_n_matrix(branch_list, non_lineal_voltages="kaixo",
                 dinamic_values="kaixo", h=0):
    matrix = np.zeros([len(branch_list), len(branch_list)])
    if type(non_lineal_voltages) == str and type(dinamic_values) == str:
        # circuit is resistive and lineal
        for i in range(len(branch_list)):
            matrix[i] = branch_list[i].get_n_row(branch_list)
    elif type(non_lineal_voltages) == str:
        # circuit is dinamic and lineal
        for i in range(len(branch_list)):
            if math.isnan(dinamic_values[i]):
                matrix[i] = branch_list[i].get_n_row(branch_list)
            else:
                if (branch_list[i].name[0] == "C"
                        or branch_list[i].name[0] == "L"):
                    matrix[i] = branch_list[i].get_n_row(branch_list, din_val=dinamic_values[i], h=h)

    elif type(dinamic_values) == str:
        # circuit is resistive and non lineal
        for i in range(len(branch_list)):
            if math.isnan(non_lineal_voltages[i]):
                matrix[i] = branch_list[i].get_n_row(branch_list)
            else:
                if branch_list[i].name[0] == "Q":
                    if branch_list[i].name[-1] == "E":
                        matrix[i] = branch_list[i].get_n_row(branch_list)
                    elif branch_list[i].name[-1] == "C":
                        matrix[i] = branch_list[i].get_n_row(branch_list)
                elif branch_list[i].name[0] == "D":
                    matrix[i] = branch_list[i].get_n_row(branch_list, non_lineal_voltages[i])

    else:
        # circuit is dinamic and non lineal
        for i in range(len(branch_list)):
            if math.isnan((non_lineal_voltages[i])
                          and math.isnan(dinamic_values[i])):
                matrix[i] = branch_list[i].get_n_row(branch_list)
            else:
                if branch_list[i].name[0] == "Q":
                    if branch_list[i].name[-1] == "E":
                        matrix[i] = branch_list[i].get_n_row(branch_list)
                    elif branch_list[i].name[-1] == "C":
                        matrix[i] = branch_list[i].get_n_row(branch_list)
                elif branch_list[i].name[0] == "D":
                    matrix[i] = branch_list[i].get_n_row(branch_list,
                                                         non_lineal_voltages[i])
                if (branch_list[i].name[0] == "C"
                        or branch_list[i].name[0] == "L"):
                    matrix[i] = branch_list[i].get_n_row(branch_list, din_val=dinamic_values[i], h=h)
    return matrix


def get_us_matrix(branch_list, t=0, non_lineal_voltages="kaixo",
                  dinamic_values="kaixo", h=0):
    matrix = np.zeros([len(branch_list), 1])
    if type(non_lineal_voltages) == str and type(dinamic_values) == str:
        # circuit is resistive and lineal
        for i in range(len(branch_list)):
            matrix[i] = branch_list[i].get_us_row(t)
    elif type(non_lineal_voltages) == str:
        # circuit is dinamic and lineal
        for i in range(len(branch_list)):
            if math.isnan(dinamic_values[i]):
                matrix[i] = branch_list[i].get_us_row(t)
            else:
                if (branch_list[i].name[0] == "C"
                        or branch_list[i].name[0] == "L"):
                    matrix[i] = branch_list[i].get_us_row(t, din_val=dinamic_values[i], h=h)

    elif type(dinamic_values) == str:
        # circuit is resistive and non lineal
        for i in range(len(branch_list)):
            if math.isnan(non_lineal_voltages[i]):
                matrix[i] = branch_list[i].get_us_row(t)
            else:
                if branch_list[i].name[0] == "Q":
                    if branch_list[i].name[-1] == "E":
                        matrix[i] = branch_list[i].get_us_row(t, non_lineal_voltages[i], non_lineal_voltages[i+1])
                    elif branch_list[i].name[-1] == "C":
                        matrix[i] = branch_list[i].get_us_row(t, non_lineal_voltages[i-1], non_lineal_voltages[i])
                elif branch_list[i].name[0] == "D":
                    matrix[i] = branch_list[i].get_us_row(t,
                                                          non_lineal_voltages[i])

    else:
        # circuit is dinamic and non lineal
        for i in range(len(branch_list)):
            if math.isnan((non_lineal_voltages[i])
                          and math.isnan(dinamic_values[i])):
                matrix[i] = branch_list[i].get_us_row(t)
            else:
                if branch_list[i].name[0] == "Q":
                    if branch_list[i].name[-1] == "E":
                        matrix[i] = branch_list[i].get_us_row(t, non_lineal_voltages[i], non_lineal_voltages[i+1])
                    elif branch_list[i].name[-1] == "C":
                        matrix[i] = branch_list[i].get_us_row(t, non_lineal_voltages[i-1], non_lineal_voltages[i])
                elif branch_list[i].name[0] == "D":
                    matrix[i] = branch_list[i].get_us_row(t,
                                                          non_lineal_voltages[i])
                if (branch_list[i].name[0] == "C"
                        or branch_list[i].name[0] == "L"):
                    matrix[i] = branch_list[i].get_us_row(t, din_val=dinamic_values[i], h=h)
    return matrix


def get_t_matrix(m, n, a):
    at = -1 * np.transpose(a)
    la = len(a)
    lat = len(at)
    lm = len(m)
    t = np.zeros([la + lm*2, la + lat + lm])
    lt = len(t)

    t[0: la, la + lm: lt] = a
    t[la: la + lat, 0: la] = at
    t[la + lat: la + lat + lm, la: la + lm] = m
    t[la + lat: la + lat + lm, la + lm: la + lm*2] = n
    t[la: la + lat, la: la + lm] = np.eye(lat)
    return(t)


def get_u_matrix(A, Us):
    la = len(A)
    lat = len(np.transpose(A))
    lus = len(Us)
    U = np.zeros([la + lat + lus, 1])
    U[la + lat: la + lat + lus] = Us
    return U


def get_el_branches(name, nd, N, val, ctr, nonLinealBranchN):
    """
        This function takes an elements name and nodes, and return its
        branch(s) list.

    Args:
        name: the elements name string.
        nd: string with elements nodes, cir_nd-s format.

    Returns:
        branch objects np array.

    """

    # el name ->
    # ((first brnch out nd pos, inc nd pos, chars add to el name),(...))
    el_to_branch = {"V": ((0, 1, ""),),
                    "I": ((0, 1, ""),),
                    "R": ((0, 1, ""),),
                    "D": ((0, 1, ""),),
                    "Q": ((1, 2, "_be"), (1, 0, "_bc"),),
                    "A": ((0, 1, "_in"), (2, 3, "_ou")),
                    "E": ((0, 1, ""),),
                    "G": ((0, 1, ""),),
                    "H": ((0, 1, ""),),
                    "F": ((0, 1, ""),),
                    "B": ((0, 1, ""),),
                    "Y": ((0, 1, ""),),
                    "C": ((0, 1, ""),),
                    "L": ((0, 1, ""),)}

    nonLineals = {"D", "Q"}
    dinamics = ["C", "L"]
    lineal = True
    if name[0][0].upper() in nonLineals:
        lineal = False
    dinamic = False
    if name[0][0].upper() in dinamics:
        dinamic = True
    brch_array = np.array([])
    for brch_tpl in el_to_branch[name[0][0].upper()]:
        brch_array = np.append(brch_array,
                               branch(name[0]+brch_tpl[2],
                                      [nd[brch_tpl[0]], nd[brch_tpl[1]]],
                                      N, N + 1, val, ctr,
                                      lineal, dinamic))
        N += 1
    return (brch_array, 0)


def get_branches(cir_el, cir_nd, cir_val, cir_ctr):
    """
    This function takes cir_el and cir_nd arrays, and returns their branches
    list.

    Args:
        cir_el: np array of strings with the elements to parse. size(1,b)
        cir_nd: np array with the nodes to the circuit. size(b,4)

    Returns:
        branch objects np array.

    """

    nonLinealBranchN = 0
    branch_list = np.array([])
    for i in range(cir_el.size):
        N = len(branch_list)
        (brch_array, nonLineals) = get_el_branches(cir_el[i], cir_nd[i], N,
                                                   cir_val[i], cir_ctr[i],
                                                   nonLinealBranchN)
        branch_list = np.append(branch_list, brch_array)
        nonLinealBranchN += nonLineals
    return branch_list


def get_aa(nd_list, branch_list):
    nd_map = {}
    i = 0
    for nd in nd_list:
        nd_map[nd] = i
        i += 1
    Aa = np.zeros((nd_list.size, branch_list.size), int)
    current_col = 0
    for brch in branch_list:
        Aa[nd_map[brch.outcoming_node]][current_col] = 1
        Aa[nd_map[brch.incoming_node]][current_col] = -1
        current_col += 1
    return Aa


def get_A(Aa):
    return Aa[1:]


def print_cir_info(nd_list, el_num, branch_list, Aa):
    nonLinealN = 0
    for branch in branch_list:
        if not branch.lineal:
            nonLinealN += 1

    print(str(el_num) + " Elements")
    print(str(len(nd_list)) + " Different nodes: " + str(nd_list))
    print("\n" + str(len(branch_list)) + " Branches:")
    for branch in branch_list:
        print(branch, end="")
    b = len(branch_list)
    print("\n" + str(2*b + (len(nd_list)-1)) + " variables: ")
    # Print all the nodes but the first (0 because is sorted)
    for i in nd_list[1:]:
        print("e"+str(i)+", ", end="", flush=True)
    for i in range(b):
        print("i"+str(i+1)+", ", end="", flush=True)
    # Print all the branches but the last to close it properly
    # It works because the minuimum amount of branches in a circuit must be 2.
    for i in range(b-1):
        print("v"+str(i+1)+", ", end="", flush=True)
    print("v"+str(b))
    print("\nIncidence Matrix:")
    print(Aa)


def print_MNUs(M, N, Us):
    print("\nM Matrix:")
    print(M)
    print("\nN Matrix:")
    print(N)
    print("\nUs Matrix:")
    print(Us)


def solve_orders(t, u, orders, branch_list, nd_list, el_num, Aa, filenameDC,
                 filenameTR, a):
    b = len(branch_list)
    n = len(nd_list)
    cir_lineal = True
    for branch in branch_list:
        cir_lineal *= branch.lineal
    for order in orders:
        orderID = order[0].upper()

        if orderID == ".OP":
            if cir_lineal:
                p2.print_solution(np.linalg.solve(T, U), b, n)
            else:
                p2.print_solution(solve_non_lineal_op(b, n, t=t, u=u,
                                                   branch_list=branch_list,
                                                   a=a), b, n)

        elif orderID == ".DC":
            solveDC(order, b, n, cir_lineal, filenameDC, u, branch_list, t, a)

        elif orderID == ".TR":
            solveTR(order, b, n, cir_lineal, branch_list, filenameTR, a, t)

        elif orderID == ".PR":
            print_cir_info(nd_list, el_num, branch_list, Aa)


def solveDC(order, b, n, cir_lineal, filenameDC, U, branch_list, T, A):
    with open(filenameDC, 'w') as file:
        header = p2.build_csv_header(np.str_.upper(order[8][0]), b, n)
        print(header, file=file)
        values = order[5:8].astype(np.float)
        current_value = values[0]
        step = values[2]
        final_value = values[1]
        while current_value <= final_value:
            Ui = U
            index_name = order[8]
            changeableBranch = get_branch_by_name(index_name, branch_list)
            changeableBranch.insert_DC_value(current_value)
            index = changeableBranch.N
            Ui[len(Ui) - len(branch_list) + index] = current_value
            if cir_lineal:
                solution = np.append(np.array([current_value]),
                                     np.linalg.solve(T, Ui))
            else:
                solution = np.append(np.array([current_value]),
                                     solve_non_lineal_op(b, n, t=T, u=Ui,
                                                      branch_list=branch_list,
                                                      a=A))
            sol_csv = ','.join(['%.9f' % num for num in solution])
            print(sol_csv, file=file)
            current_value += step


def solveTR(order, b, n, cir_lineal, branch_list, filenameTR, A, T):
    cir_dinamic = False
    for branch in branch_list:
        if branch.dinamic:
            cir_dinamic = True
    with open(filenameTR, 'w') as file:
        header = p2.build_csv_header("t", b, n)
        print(header, file=file)
        values = order[5:8].astype(np.float)
        current_value = values[0]
        step = values[2]
        final_value = values[1]
        h = order[7]
        h = h.astype(np.float)
        dinamic_values = np.zeros(b)
        dinamic_index = []
        dinamic_names = []
        for branch in branch_list:
            if branch.dinamic:
                # initial V value for NR
                dinamic_index.append(branch.N)
                dinamic_names.append(branch.name)
                dinamic_values[branch.N] = branch.value[1]
            else:
                dinamic_values[branch.N] = None
        while current_value <= final_value:
            if cir_dinamic:
                Mi = get_m_matrix(branch_list, dinamic_values=dinamic_values,
                                  h=h)
                Ni = get_n_matrix(branch_list, dinamic_values=dinamic_values,
                                  h=h)
                Usi = get_us_matrix(branch_list, current_value,
                                    dinamic_values=dinamic_values, h=h)
                Ti = get_t_matrix(Mi, Ni, A)
                Ui = get_u_matrix(A, Usi)
                if cir_lineal:
                    solution = np.append(np.array([current_value]),
                                         np.linalg.solve(Ti, Ui))
                else:
                    solution = np.append(np.array([current_value]),
                                         solve_non_lineal_op(b, n, T=Ti, U=Ui,
                                                          t=current_value,
                                                          branch_list=branch_list,
                                                          A=A))
                for index, name in zip(dinamic_index, dinamic_names):
                    if name[0] == "C":
                        dinamic_values[index] = solution[index+n]
                    if name[0] == "L":
                        dinamic_values[index] = solution[index+n+b]
            else:
                Usi = get_us_matrix(branch_list, current_value)
                Ui = get_u_matrix(A, Usi)
                if cir_lineal:
                    solution = np.append(np.array([current_value]),
                                         np.linalg.solve(T, Ui))
                else:
                    solution = np.append(np.array([current_value]),
                                         solve_non_lineal_op(b, n, T=T, U=Ui,
                                                          t=current_value,
                                                          branch_list=branch_list,
                                                          A=A))
            sol_csv = ','.join(['%.9f' % num for num in solution])
            print(sol_csv, file=file)
            current_value += step


def solve_non_lineal_op(b, n, t, u, branch_list, a, time=0):
    non_lineal_voltages = np.zeros(b)
    non_lineal_index = []
    for branch in branch_list:
        if not branch.lineal:
            # initial V value for NR
            non_lineal_index.append(branch.N)
            non_lineal_voltages[branch.N] = 0.6
        else:
            non_lineal_voltages[branch.N] = None
    # NR maximun error
    e = 1e-5
    ti = t
    ui = u
    current_sol = np.linalg.solve(ti, ui)
    iteration = 1
    max_iteration = 10000
    while True:
        for index in non_lineal_index:
            non_lineal_voltages[index] = current_sol[index+n-1]

        Mi = get_m_matrix(branch_list, non_lineal_voltages)
        Ni = get_n_matrix(branch_list, non_lineal_voltages)
        Usi = get_us_matrix(branch_list,
                            non_lineal_voltages=non_lineal_voltages, t=time)
        Ti = get_t_matrix(Mi, Ni, a)
        Ui = get_u_matrix(a, Usi)
        prev_sol = current_sol
        current_sol = np.linalg.solve(Ti, Ui)
        correct = True
        for index in non_lineal_index:
            err = abs(prev_sol[index+n-1]-current_sol[index+n-1])
            if err > e:
                correct = False
        iteration += 1
        if correct:
            return current_sol
            break
        elif iteration > max_iteration:
            print("Iterazio maximo kopurua gainditu da")


def solveCircuit(filename):
    # Parse the circuit
    [cir_el, cir_nd, cir_val, cir_ctr, orders] = cir_parser(filename)

    # Get nodes list
    nd_list = get_nd_list(cir_nd)

    # get element quantity
    el_num = get_element_N(cir_el)

    # get branchs list
    branch_list = get_branches(cir_el, cir_nd, cir_val, cir_ctr)

    # check circuit correct
    if not check_file_correct(filename, branch_list, cir_nd, cir_el):
        sys.exit("file have some error, unable to solve")
        raise(ValueError)

    # get Aa and A
    Aa = get_aa(nd_list, branch_list)
    A = get_A(Aa)

    # get M N Us T and U arrays
    M = get_m_matrix(branch_list)
    N = get_n_matrix(branch_list)
    Us = get_us_matrix(branch_list)
    T = get_t_matrix(M, N, A)
    U = get_u_matrix(A, Us)

    # for .DC and .TR
    filenameDC = filename[:-3] + "dc"
    filenameTR = filename[:-3] + "tr"

    # solve orders
    solve_orders(T, U, orders, branch_list, nd_list, el_num, Aa, filenameDC,
                 filenameTR, A)