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
    [cir_el, cir_nd, cir_val, cir_ctr] = zl1.cir_parser(filename)

#    THIS FUNCTION IS NOT COMPLETE
