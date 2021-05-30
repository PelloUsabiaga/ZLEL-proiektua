#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. module:: zlel_main.py
    :synopsis:

.. moduleauthor:: YOUR NAME AND E-MAIL


"""

import zlel.zlel_p1 as zl1
import zlel.zlel_p2 as zl2
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
        filename = "cirs/all/2_zlel_Q_ezaugarri.cir"
    zl1.solveCircuit(filename)
    filenameTR = filename[:-3] + "tr"
    filenameDC = filename[:-3] + "dc"
    zl2.plot_from_cvs(filenameDC, "V", "i3", "wololo")
