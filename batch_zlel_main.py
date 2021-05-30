#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
"""
 
.. module:: batch_zlel_main.py
    :synopsis: Given a folder with .cir files as argument, this script executes
    zlel_main.py over all of them and redirects STDOUT and STDERR to a file
    with same name as input but with .out extension.
    zlel_main.py must be in the same directory as this script.
 
     positional arguments:
         folder      Folder with .cir files.
 
     optional arguments:
         -h, --help  show this help message and exit
         -o O        Specify the output folder. If the folder does not exist, it
                     creates it.
 
 .. moduleauthor:: Iñigo Arredondo (inigo.arredondo@ehu.eus)
 
 """
 
 
from os import listdir, name, path, makedirs
from subprocess import Popen, PIPE, STDOUT
from sys import stdout, argv
from argparse import ArgumentParser


if __name__ == "__main__":
    if name == 'posix':
        slash = '/'
        python_cmd = 'python3'
    else:
        slash = '\\'
        python_cmd = 'python'

    parser = ArgumentParser(description='Given a folder with .cir ' +
                            'files as argument, this script executes ' +
                            'zlel_main.py over all of them and redirects ' +
                            'STDOUT and STDERR to a file with same name as ' +
                            'input but with .out extension. zlel_main.py ' +
                            'must be in the same directory as this script.')
    parser.add_argument('folder', type=str, help='Folder with .cir files.')
    parser.add_argument('-o', type=str,
                        help='Specify the output folder. If the folder ' +
                        'does not exist, it creates it.')
    args = parser.parse_args()
    cirs = listdir(args.folder)
    if args.o:
        outfolder = args.o
        # Get rid of the slash if any
        if outfolder[-1] == slash:
            outfolder = outfolder[:-1]
        # Create the folder if it doesn exists
        if not path.exists(outfolder):
            makedirs(outfolder)
    else:
        outfolder = args.folder
    for cir in cirs:
        with open(outfolder + slash + cir[:-3] + 'out', "wb") as out:
            cmd = [python_cmd, 'zlel_main.py', args.folder + slash + cir]
            proc = Popen(cmd, stdout=PIPE, stderr=STDOUT)
            for line in proc.stdout:
                stdout.write(line.decode("utf-8"))
                out.write(line)
            proc.wait()
            
