#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. module:: zlel_main.py
    :synopsis: main module of the program.

.. moduleauthor:: Ander Dokando (anddokan@gmail.com) and Pello Usabiaga
(pellousabiaga@gmail.com).

"""

import zlel.zlel_p1_errore_kontrola as zl1
import zlel.zlel_p2 as zl2
import sys


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        path = "cirs/all/"
        filename = "2_zlel_Q_ezaugarri.cir"
    zl1.solve_circuit(path + filename)
    filenameTR = filename[:-3] + "tr"
    filenameDC = filename[:-3] + "dc"
    zl2.plot_from_cvs(path + filenameDC, "V", "i3", "wololo")
