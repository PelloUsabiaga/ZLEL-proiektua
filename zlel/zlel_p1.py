#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. module:: zlel_main.py
    :synopsis: This module is used to solve .cir circuits.

.. module_author:: Ander Dokando (anddokan@gmail.com) and Pello Usabiaga
(pellousabiaga@gmail.com).

"""

import numpy as np
import sys
import math

if __name__ == "__main__":
    import zlel_p2 as p2
else:
    import zlel.zlel_p2 as p2


def check_file_correct(branch_list):
    """
        This function checks if the circuit is correct, checking if zero node
        exists, if there is any parallel voltage source or series current
        current source, if there is any node not conected, or if all the
        controled elements have a correct controller.

    Args:
        branch_list : np.array with circuits branches.

    Returns:
        True if circuit is correct.
    Rises:
        SystemExit

    """
    correct = True
    correct *= check_zero_node_exists(branch_list)
    correct *= check_parallel_voltage_source(branch_list)
    correct *= check_not_series_current_sources(branch_list)
    correct *= check_not_connected_node(branch_list)
    correct *= check_controlled_elements_correct(branch_list)
    return correct


def check_controlled_elements_correct(branch_list):
    """
        This function checks if all the controlled elemets have a controller
        element. If not, program stops and returns "X controlled source have
        not controler." from sys.exit.

    Args:
        branch_list : np.array with circuits branches.

    Returns:
        True if all controlled elements are correct.
    Rises:
        SystemExit

    """
    controlled_elements = {"G", "E"}
    for branch in branch_list:
        if branch.name[0] in controlled_elements:
            if branch.control == ['0']:
                sys.exit(branch.printName
                         + " controlled source have not controler.")
    return True


def check_not_series_current_sources(branch_list):
    """
        This function checks for series current sources in the circuit. If
        any founded, program stops and returns "I sources in series at node
        i." from sys.exit.

    Args:
        branch_list : np.array with circuits branches.

    Returns:
        True if not series current sources founded.
    Rises:
        SystemExit

    """
    cir_correct = True
    for branch in branch_list:
        if branch.name[0] == "I":
            error_in_branch = False
            in_node = branch.incoming_node
            out_node = branch.outcoming_node
            nodes = [in_node, out_node]
            for node in nodes:
                connected = []
                node_correct = False
                for branch in branch_list:
                    if branch.is_node_in(node):
                        connected.append(branch)
                for branch in connected:
                    if branch.name[0] != "I":
                        node_correct = True
                        break
                if not node_correct:
                    out_curr = 0
                    for branch in connected:
                        out_curr += branch.get_out_current(node)
                    if out_curr != 0:
                        sys.exit("I sources in series at node " + str(node)
                                 + ".")
            if error_in_branch:
                cir_correct = False
                break
    return cir_correct


def check_not_connected_node(branch_list):
    """
        This function checks for any not conected node in the circuit. If any
        founded, program stops and returns "Node i is floating." from sys.exit.

    Args:
        branch_list : np.array with circuits branches.

    Returns:
        True if none not conected node founded.
    Rises:
        SystemExit

    """
    nd_quantity = dict()
    for branch in branch_list:
        if branch.incoming_node in nd_quantity:
            nd_quantity[branch.incoming_node] += 1
        else:
            nd_quantity[branch.incoming_node] = 1
        if branch.outcoming_node in nd_quantity:
            nd_quantity[branch.outcoming_node] += 1
        else:
            nd_quantity[branch.outcoming_node] = 1
    for nd, quantity in zip(nd_quantity.keys(), nd_quantity.values()):
        if quantity == 1:
            sys.exit("Node " + str(nd) + " is floating.")
    return True


def get_node_qty(cir_nd):
    """
        this function returns the total different nodes number.

    Parameters:
        cir_nd : np.array with nodes directly parsed from file.

    Returns:
        node_qty : int, represents the different nodes number.
    Rises:


    """
    node_qty = {}
    for node_list in cir_nd:
        for node in node_list:
            if node in node_qty:
                node_qty[node] += 1
            else:
                node_qty[node] = 1

    return node_qty


def check_zero_node_exists(branch_list):
    """
        This function checks for zero node in the circuit. If not founded,
        program stops and returns "Reference node "0" is not defined in the
        circuit." from sys.exit.

    Args:
        branch_list : np.array with circuits branches.

    Returns:
        True if zero node founded.
    Rises:
        SystemExit

    """
    for branch in branch_list:
        if branch.incoming_node == 0 or branch.outcoming_node == 0:
            return True
    sys.exit("Reference node \"0\" is not defined in the circuit.")


def check_parallel_voltage_source(branch_list):
    """
        This function checks for parallel voltage sources in the circuit. If
        any founded, program stops and returns "Parallel V sources at branches
        i and j.".

    Args:
        branch_list : np.array with circuits branches.

    Returns:
        True if not parallel voltage sources founded.
    Rises:
        SystemExit

    """
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
        4 matrix.
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
    except ValueError:
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


def get_element_n(cir_el):
    """
        This function calculates the number of elements at a circuit.

    Args:
        cir_el: np array of strings with the elements to parse. size(1,b)

    Returns:
        element_N: int the number of elements at a circuit.

    """
    return cir_el.size


class Branch:
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
    dynamic = None

    def __init__(self, name, nd, n, print_n, value, control, lineal=True,
                 dynamic=False):
        """
            Branch class initializer, asigns some parameters to local fields.

        Parameters:
            name: String, with branchs name.
            nd: tuple, with the incoming node and the outcoming node ints.
            n: int, the branchs ordinal position.
            print_n: String, the name, only for printing.
            value: nd.array, with branchs values, descriveing the element.
            control: String, the controler info.
            lineal: boolean, represent if branch is lineal. Optional. The
            default is True.
            dynamic: boolean, represent if branch is dynamic. Optional. The
            default is False.

        Returns:

        Rises:
        """
        self.name = name.upper()
        self.printName = name
        self.outcoming_node = nd[0]
        self.incoming_node = nd[1]
        self.N = n
        self.value = value
        self.control = control
        self.lineal = lineal
        self.printN = print_n
        self.dynamic = dynamic
        for i in range(len(control)):
            control[i] = control[i].upper()

    def insert_dc_value(self, value):
        """
            Changes the branchs first value, sould be used for .DC analisis.

        Parameters:
        value : float, the new value for element.

        Returns:

        Rise:

        """
        self.value[0] = value

    def get_out_current(self, node):
        """
            Returns the branch normalized current considering a direction. For
            current sources checking.

        Parameters:
            node : int, the node referent to which the current will be geaven.

        Returns:
            float current normalized value.

        Raises:
            ValueError if called from no current source element.

        """
        if self.name[0] != "I":
            sys.exit("current source function was called by not current "
                     + "source element")
            raise ValueError
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
        """
            Checks if a node is in the branch.

        Parameters:
            node : int, node to check

        Returns:
            true if node in, false else.
        """
        return node == self.outcoming_node or node == self.incoming_node

    def get_m_row(self, branch_list, vi=0.6, vbe=0.6, vbc=0.6, din_val=0, h=1):
        """
            Gets the correspondient row of the branch for the M matrix
            construction.

        Parameters:
            branch_list : nd.array, with the circuit branches.
            vi : float, for non lineal elements current voltages. Optional.
            The default is 0.6.
            vbe : float, for non lineal elements current voltages. Optional.
            The default is 0.6.
            vbc : float, for non lineal elements current voltages. Optional.
            The default is 0.6.
            din_val : float, for dinamic element current voltages. Optional.
            The default is 0.
            h : float, for dinamic elements, representing the transient
            analisis steps size. Optional. The default is 1.

        Returns:
            row : np.array, with the correspondient row of the branch.

        Rises:
        """
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

    def get_n_row(self, branch_list, vi=0.6, din_val=0, h=1):
        """
            Gets the correspondient row of the branch for the N matrix
            construction.

        Parameters:
            branch_list : nd.array, with the circuit branches.
            vi : float, for non lineal elements current voltages. Optional.
            The default is 0.6.
            din_val : float, for dinamic element current voltages. Optional.
            The default is 0.
            h : float, for dinamic elements, representing the transient
            analisis steps size. Optional. The default is 1.

        Returns:
            row : np.array, with the correspondient row of the branch.

        Rises:

        """
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
        """
            Gets the correspondient row of the branch for the Us matrix
            construction.

        Parameters:
            branch_list : nd.array, with the circuit branches.
            vi : float, for non lineal elements current voltages. Optional.
            The default is 0.6.
            vbc : float, for non lineal elements current voltages. Optional.
            The default is 0.6.
            din_val : float, for dinamic element current voltages. Optional.
            The default is 0.
            h : float, for dinamic elements, representing the transient
            analisis steps size. Optional. The default is 1.

        Returns:
            row : np.array, with the correspondient row of the branch.

        Rises:

        """
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
            raise ValueError
        vt = self.value[1] * 8.6173324e-5 * 300
        i0 = self.value[0]
        gj = - (i0 / vt) * math.exp(vj / vt)
        ij = i0 * math.exp(vj / vt) + gj * vj
        return gj, ij

    def transistor_values(self, vbe=0.6, vbc=0.6):
        """
            This function takes the vbe and vbc voltages and computes the
            transistor parameters for this
            specific voltages following Ebers-Mollen transistor model.

        Args:
            vbe: float, voltage difference between base and emitter.
            vbc: float, voltage difference between base and collector.

        Returns:
            g11: float, transistors parameter.
            g12: float, transistors parameter.
            g21: float, transistors parameter.
            g22: float, transistors parameter.
            ie: float, current passing away collector.
            ic: float, current passing away collector.
        """

        if self.name[0] != "Q":
            sys.exit("diod function was called by not diod element")
            raise ValueError
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
        return g11, g12, g21, g22, ie, ic


def get_branch_by_name(brch_name, branch_list):
    """
        Get a branch from the circuit by its name.

    Parameters:
        brch_name : String, with the searched branchs name.
        branch_list : np.array, with the circuit branches.

    Returns
        branch : Branch, the branch correspondign to the name.

    """
    for branch in branch_list:
        if brch_name.__class__ == np.str_:
            brch_name = np.str_.upper(brch_name)
        elif brch_name.__class__ == str:
            brch_name = brch_name.upper()
        if branch.name == brch_name:
            return branch


def get_m_matrix(branch_list, non_lineal_voltages="kaixo",
                 dynamic_values="kaixo", h=0):
    """
        This function takes the branches and especial characteristics of the
        circuit and parse they to
        M matrix, voltage (Vn) matrix.

    Args:
        branch_list: np.array of branches of the current circuit.
        non_lineal_voltages: indicates if current circuit has non linear
        elements.
            If don´t have it contains "kaixo" str, otherwise a list of float
            with the value for each branch
        dynamic_values: indicates if current circuit has dynamic values. If
        don´t have it contains "kaixo" str,
            otherwise a list of float with the value for each branch
        h: Int of jump in Euler backward method.

    Returns:
        matrix : np array matrix of floats with the value for specific Vn
    """

    matrix = np.zeros([len(branch_list), len(branch_list)])
    if type(non_lineal_voltages) == str and type(dynamic_values) == str:
        # circuit is resistive and lineal
        for i in range(len(branch_list)):
            matrix[i] = branch_list[i].get_m_row(branch_list)
    elif type(non_lineal_voltages) == str:
        # circuit is dynamic and lineal
        for i in range(len(branch_list)):
            if math.isnan(dynamic_values[i]):
                matrix[i] = branch_list[i].get_m_row(branch_list)
            else:
                if (branch_list[i].name[0] == "C"
                        or branch_list[i].name[0] == "L"):
                    matrix[i] = branch_list[i].get_m_row(branch_list, din_val=dynamic_values[i], h=h)

    elif type(dynamic_values) == str:
        # circuit is resistive and non lineal
        for i in range(len(branch_list)):
            if math.isnan(non_lineal_voltages[i]):
                matrix[i] = branch_list[i].get_m_row(branch_list)
            else:
                if branch_list[i].name[0] == "Q":
                    if branch_list[i].name[-1] == "E":
                        matrix[i] = branch_list[i].get_m_row(
                            branch_list, vbe=non_lineal_voltages[i], vbc=non_lineal_voltages[i+1]
                        )
                    elif branch_list[i].name[-1] == "C":
                        matrix[i] = branch_list[i].get_m_row(
                            branch_list, vbe=non_lineal_voltages[i-1], vbc=non_lineal_voltages[i]
                        )
                elif branch_list[i].name[0] == "D":
                    matrix[i] = branch_list[i].get_m_row(branch_list,
                                                         non_lineal_voltages[i])

    else:
        # circuit is dynamic and non lineal
        for i in range(len(branch_list)):
            if (math.isnan(non_lineal_voltages[i])
                    and math.isnan(dynamic_values[i])):
                matrix[i] = branch_list[i].get_m_row(branch_list)
            else:
                if branch_list[i].name[0] == "Q":
                    if branch_list[i].name[-1] == "E":
                        matrix[i] = branch_list[i].get_m_row(
                            branch_list, vbe=non_lineal_voltages[i], vbc=non_lineal_voltages[i+1]
                        )
                    elif branch_list[i].name[-1] == "C":
                        matrix[i] = branch_list[i].get_m_row(
                            branch_list, vbe=non_lineal_voltages[i-1], vbc=non_lineal_voltages[i]
                        )
                elif branch_list[i].name[0] == "D":
                    matrix[i] = branch_list[i].get_m_row(branch_list,
                                                         non_lineal_voltages[i])
                if (branch_list[i].name[0] == "C"
                        or branch_list[i].name[0] == "L"):
                    matrix[i] = branch_list[i].get_m_row(branch_list, din_val=dynamic_values[i], h=h)
    return matrix


def get_n_matrix(branch_list, non_lineal_voltages="kaixo",
                 dynamic_values="kaixo", h=0):
    """
        This function takes the branches and especial characteristics of the
        circuit and parse they to
        N matrix, current (In) matrix.

    Args:
        branch_list: list of branches of the current circuit.
        non_lineal_voltages: indicates if current circuit has non linear
        elements.
            If don´t have it contains "kaixo" str, otherwise a list of float
        with the value for each branch
        dynamic_values: indicates if current circuit has dynamic values. If
        don´t have it contains "kaixo" str,
            otherwise a list of float with the value for each branch
        h: Int of jump in Euler backward method.

    Returns:
        matrix : np array matrix of floats with the value for specific In
    """

    matrix = np.zeros([len(branch_list), len(branch_list)])
    if type(non_lineal_voltages) == str and type(dynamic_values) == str:
        # circuit is resistive and lineal
        for i in range(len(branch_list)):
            matrix[i] = branch_list[i].get_n_row(branch_list)
    elif type(non_lineal_voltages) == str:
        # circuit is dynamic and lineal
        for i in range(len(branch_list)):
            if math.isnan(dynamic_values[i]):
                matrix[i] = branch_list[i].get_n_row(branch_list)
            else:
                if (branch_list[i].name[0] == "C"
                        or branch_list[i].name[0] == "L"):
                    matrix[i] = branch_list[i].get_n_row(branch_list, din_val=dynamic_values[i], h=h)

    elif type(dynamic_values) == str:
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
        # circuit is dynamic and non lineal
        for i in range(len(branch_list)):
            if math.isnan((non_lineal_voltages[i])
                          and math.isnan(dynamic_values[i])):
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
                    matrix[i] = branch_list[i].get_n_row(branch_list, din_val=dynamic_values[i], h=h)
    return matrix


def get_us_matrix(branch_list, t=0, non_lineal_voltages="kaixo",
                  dynamic_values="kaixo", h=0):
    """
        This function takes the branches and especial characteristics of the
        circuit and parse they to Us matrix.

    Args:
        branch_list: list of branches of the current circuit.
        t: float, for t-dependient elements.
        non_lineal_voltages: indicates if current circuit has non linear
        elements.
            If don´t have it contains "kaixo" str, otherwise a list of float
        with the value for each branch
        dynamic_values: indicates if current circuit has dynamic values. If
        don´t have it contains "kaixo" str,
            otherwise a list of float with the value for each branch
        h: Int of jump in Euler backward method.

    Returns:
        matrix : np array matrix of floats with the value for specific Us.
    """
    matrix = np.zeros([len(branch_list), 1])
    if type(non_lineal_voltages) == str and type(dynamic_values) == str:
        # circuit is resistive and lineal
        for i in range(len(branch_list)):
            matrix[i] = branch_list[i].get_us_row(t)
    elif type(non_lineal_voltages) == str:
        # circuit is dynamic and lineal
        for i in range(len(branch_list)):
            if math.isnan(dynamic_values[i]):
                matrix[i] = branch_list[i].get_us_row(t)
            else:
                if (branch_list[i].name[0] == "C"
                        or branch_list[i].name[0] == "L"):
                    matrix[i] = branch_list[i].get_us_row(t, din_val=dynamic_values[i], h=h)

    elif type(dynamic_values) == str:
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
        # circuit is dynamic and non lineal
        for i in range(len(branch_list)):
            if math.isnan((non_lineal_voltages[i])
                          and math.isnan(dynamic_values[i])):
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
                    matrix[i] = branch_list[i].get_us_row(t, din_val=dynamic_values[i], h=h)
    return matrix


def get_t_matrix(m, n, a):
    """
        Builds the T matrix using voltages matrix M, currents matrix N, and
        and morfologic matrix A.

    Parameters
        m : np.array, M matrix
        n : np.array, N matrix
        a : np.array, A matrix

    Returns
        t : np.array, T matrix of the circuit.

    """
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
    return t


def get_u_matrix(a, us):
    """
        Builds the U matrix using matrix Us. Function needs A matrix to
        get dimensions.

    Parameters
        us : np.array, Us matrix
        a : np.array, A matrix

    Returns
        u : np.array, U matrix of the circuit.

    """
    la = len(a)
    lat = len(np.transpose(a))
    lus = len(us)
    u = np.zeros([la + lat + lus, 1])
    u[la + lat: la + lat + lus] = us
    return u


def get_el_branches(name, nd, n, val, ctr):
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

    non_lineals = {"D", "Q"}
    dynamics = ["C", "L"]
    lineal = True
    if name[0][0].upper() in non_lineals:
        lineal = False
    dynamic = False
    if name[0][0].upper() in dynamics:
        dynamic = True
    brch_array = np.array([])
    for brch_tpl in el_to_branch[name[0][0].upper()]:
        brch_array = np.append(brch_array,
                               Branch(name[0]+brch_tpl[2],
                                      [nd[brch_tpl[0]], nd[brch_tpl[1]]],
                                      n, n + 1, val, ctr,
                                      lineal, dynamic))
        n += 1
    return brch_array, 0


def get_branches(cir_el, cir_nd, cir_val, cir_ctr):
    """
    This function takes cir_el and cir_nd arrays, and returns their branches
    list.

    Args:
        cir_el: np array of strings with the elements to parse. size(1,b)
        cir_nd: np array with the nodes to the circuit. size(b,4)
        cir_val: np array with the values of the elements. size 4
        cir_ctr: np array with the node which control the element.
                 If element is not controlled by other node will be 0. size 1

    Returns:
        branch objects np array.

    """

    non_lineal_branch_n = 0
    branch_list = np.array([])
    for i in range(cir_el.size):
        n = len(branch_list)
        (brch_array, non_lineals) = get_el_branches(cir_el[i], cir_nd[i], n,
                                                    cir_val[i], cir_ctr[i],
                                                    )
        branch_list = np.append(branch_list, brch_array)
        non_lineal_branch_n += non_lineals
    return branch_list


def get_aa(nd_list, branch_list):
    nd_map = {}
    i = 0
    for nd in nd_list:
        nd_map[nd] = i
        i += 1
    aa = np.zeros((nd_list.size, branch_list.size), int)
    current_col = 0
    for brch in branch_list:
        aa[nd_map[brch.outcoming_node]][current_col] = 1
        aa[nd_map[brch.incoming_node]][current_col] = -1
        current_col += 1
    return aa


def get_a(aa):
    return aa[1:]


def print_cir_info(nd_list, el_num, branch_list, aa):
    """
    This function takes nd_list, el_num, branch_list and aa. And print the
    information of the current circuit

    Args:
        cir_el: np array of strings with the elements to parse. size(1,b)
        branch_list: np array with the branches of current circuit


    """
    non_lineal_n = 0
    for branch in branch_list:
        if not branch.lineal:
            non_lineal_n += 1
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
    print(aa)


def solve_orders(t, u, orders, branch_list, nd_list, el_num, aa, filename_dc,
                 filename_tr, a):
    """
        This function takes circuit depending info, and use it to identify and
        solve the orders in .cir file.

    Parameters
        t : np.array, circuit T matrix.
        u : np.array, circuit U matrix.
        orders : np.array, containing orders directly parsed from .cir file.
        branch_list : nd.array, containing circuit branches.
        nd_list : nd.array, containing circuit nodes.
        el_num : int, containing element cantity.
        aa : nd.array, circuits Aa matrix.
        filename_dc : String, the filename to save .DC simulations results.
        filename_tr : String, the filename to save .TR simulations results.
        a : np.array, circuits A matrix.

    """
    b = len(branch_list)
    n = len(nd_list)
    cir_lineal = True
    for branch in branch_list:
        cir_lineal *= branch.lineal
    for order in orders:
        order_id = order[0].upper()

        if order_id == ".OP":
            if cir_lineal:
                p2.print_solution(np.linalg.solve(t, u), b, n)
            else:
                p2.print_solution(solve_non_lineal_op(b, n, t=t, u=u,
                                                      branch_list=branch_list,
                                                      a=a),
                                  b, n)

        elif order_id == ".DC":
            solve_dc(order, b, n, cir_lineal, filename_dc, u, branch_list, t,
                     a)

        elif order_id == ".TR":
            solve_tr(order, b, n, cir_lineal, branch_list, filename_tr, a, t)

        elif order_id == ".PR":
            print_cir_info(nd_list, el_num, branch_list, aa)


def solve_dc(order, b, n, cir_lineal, filename_dc, u, branch_list, t, a):
    """
        Solves .DC order.

    Parameters:
        order : nd.array .DC order info
        b : int, branch number.
        n : int, nodes number.
        cir_lineal : boolean, to identify if circuit is lineal.
        filename_dc : String, the filename to save .DC simulations results.
        u : np.array, the U matrix of the circuit.
        branch_list : nd.array, contains the branchs of the circuit.
        t : np.array, the T matrix of the circuit.
        a : np.array, the A matrix of the circuit.

    """
    with open(filename_dc, 'w') as file:
        header = p2.build_csv_header(np.str_.upper(order[8][0]), b, n)
        print(header, file=file)
        values = order[5:8].astype(np.float)
        current_value = values[0]
        step = values[2]
        final_value = values[1]
        while current_value <= final_value:
            ui = u
            index_name = order[8]
            changeable_branch = get_branch_by_name(index_name, branch_list)
            changeable_branch.insert_dc_value(current_value)
            index = changeable_branch.N
            ui[len(ui) - len(branch_list) + index] = current_value
            if cir_lineal:
                solution = np.append(np.array([current_value]),
                                     np.linalg.solve(t, ui))
            else:
                solution = np.append(np.array([current_value]),
                                     solve_non_lineal_op(b, n, t=t, u=ui,
                                                         branch_list=branch_list,
                                                         a=a))
            sol_csv = ','.join(['%.9f' % num for num in solution])
            print(sol_csv, file=file)
            current_value += step


def solve_tr(order, b, n, cir_lineal, branch_list, filename_tr, a, t):
    """
        Solves .TR order.

    Parameters:
        order : nd.array .DC order info
        b : int, branch number.
        n : int, nodes number.
        cir_lineal : boolean, to identify if circuit is lineal.
        filename_tr : String, the filename to save .TR simulations results.
        u : np.array, the U matrix of the circuit.
        branch_list : nd.array, contains the branchs of the circuit.
        a : np.array, the A matrix of the circuit.
        t : np.array, the T matrix of the circuit.

    """
    cir_dynamic = False
    for branch in branch_list:
        if branch.dynamic:
            cir_dynamic = True
    with open(filename_tr, 'w') as file:
        header = p2.build_csv_header("t", b, n)
        print(header, file=file)
        values = order[5:8].astype(np.float)
        current_value = values[0]
        step = values[2]
        final_value = values[1]
        h = order[7]
        h = h.astype(np.float)
        dynamic_values = np.zeros(b)
        dynamic_index = []
        dynamic_names = []
        for branch in branch_list:
            if branch.dynamic:
                # initial V value for NR
                dynamic_index.append(branch.N)
                dynamic_names.append(branch.name)
                dynamic_values[branch.N] = branch.value[1]
            else:
                dynamic_values[branch.N] = None
        while current_value <= final_value:
            if cir_dynamic:
                mi = get_m_matrix(branch_list, dynamic_values=dynamic_values,
                                  h=h)
                ni = get_n_matrix(branch_list, dynamic_values=dynamic_values,
                                  h=h)
                usi = get_us_matrix(branch_list, current_value,
                                    dynamic_values=dynamic_values, h=h)
                ti = get_t_matrix(mi, ni, a)
                ui = get_u_matrix(a, usi)
                if cir_lineal:
                    solution = np.append(np.array([current_value]),
                                         np.linalg.solve(ti, ui))
                else:
                    solution = np.append(
                        np.array([current_value]),
                        solve_non_lineal_op(b, n, t=ti, u=ui,
                                            time=current_value,
                                            branch_list=branch_list, a=a)
                    )
                for index, name in zip(dynamic_index, dynamic_names):
                    if name[0] == "C":
                        dynamic_values[index] = solution[index+n]
                    if name[0] == "L":
                        dynamic_values[index] = solution[index+n+b]
            else:
                usi = get_us_matrix(branch_list, current_value)
                ui = get_u_matrix(a, usi)
                if cir_lineal:
                    solution = np.append(np.array([current_value]),
                                         np.linalg.solve(t, ui))
                else:
                    solution = np.append(np.array([current_value]),
                                         solve_non_lineal_op(b, n, t=t, u=ui,
                                                             time=current_value,
                                                             branch_list=branch_list,
                                                             a=a))
            sol_csv = ','.join(['%.9f' % num for num in solution])
            print(sol_csv, file=file)
            current_value += step


def solve_non_lineal_op(b, n, t, u, branch_list, a, time=0):
    """
        Function to solve the operation point of a non lineal circuit. Uses
        the NR method for it.

    Parameters:
        b : int, the circuits branches cuantity.
        n : int, the circuits nodes cuantity.
        t : np.array, the T matrix of the circuit.
        u : np.array, the U matrix of the circuit.
        branch_list : TYPE
        a : np.array, the A matrix of the circuit.
        time : float, current time if called from .TR analisis. Optional. The
        default is 0.

    Returns:
        current_sol : np.array with the non lineal solution of the operation
        point.

    """
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

        mi = get_m_matrix(branch_list, non_lineal_voltages)
        ni = get_n_matrix(branch_list, non_lineal_voltages)
        usi = get_us_matrix(branch_list,
                            non_lineal_voltages=non_lineal_voltages, t=time)
        ti = get_t_matrix(mi, ni, a)
        ui = get_u_matrix(a, usi)
        prev_sol = current_sol
        current_sol = np.linalg.solve(ti, ui)
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


def solve_circuit(filename):
    """
        This function a path with <filename>.cir document and solve the circuit
        that contains
    Args:
        filename: str with the path+filename of the document to solve.

    """
    # Parse the circuit
    [cir_el, cir_nd, cir_val, cir_ctr, orders] = cir_parser(filename)

    # Get nodes list
    nd_list = get_nd_list(cir_nd)

    # get element quantity
    el_num = get_element_n(cir_el)

    # get branchs list
    branch_list = get_branches(cir_el, cir_nd, cir_val, cir_ctr)

    # check circuit correct
    if not check_file_correct(branch_list):
        sys.exit("file have some error, unable to solve")
        raise ValueError

    # get aa and A
    aa = get_aa(nd_list, branch_list)
    a = get_a(aa)

    # get M N Us T and U arrays
    m = get_m_matrix(branch_list)
    n = get_n_matrix(branch_list)
    us = get_us_matrix(branch_list)
    t = get_t_matrix(m, n, a)
    u = get_u_matrix(a, us)

    # for .DC and .TR
    filename_dc = filename[:-3] + "dc"
    filename_tr = filename[:-3] + "tr"

    # solve orders
    solve_orders(t, u, orders, branch_list, nd_list, el_num, aa, filename_dc,
                 filename_tr, a
                 )
