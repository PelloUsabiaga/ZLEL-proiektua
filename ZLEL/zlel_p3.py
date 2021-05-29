#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

.. module:: zlel_p3.py
    :synopsis: Put yours

.. moduleauthor:: Put yours

"""

import numpy as np
import sys

if __name__ == "__main__":
    import zlel_p1 as zl1
    import zlel_p2 as zl2
else:
    import zlel.zlel_p1 as zl1
    import zlel.zlel_p2 as zl2



"""
https://stackoverflow.com/questions/419163/what-does-if-name-main-do
"""
if __name__ == "__main__":
    #  start = time.perf_counter()
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "../cirs/all/2_zlel_Q.cir"
        filename = "../cirs/all/2_zlel_1D.cir"


#    end = time.perf_counter()
#    print ("Elapsed time: ")
#    print(end - start) # Time in seconds
