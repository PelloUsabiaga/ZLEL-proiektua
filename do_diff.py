#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

.. module:: do_diff.py
    :synopsis: Compare to files with the same name.
    usage: do_diff_argparse.py [-h] [-d] file1 file2

    Compare two files with the same name line by line.
    If -d is used, it compares the files from 1st argument folder with the
    ones of the 2nd argument folder. 
    Refer to difflib documentation to undertstand the output format.

    positional arguments:
      file1
      file2

    optional arguments:
      -h, --help  show this help message and exit
      -d          Instead of single files with different paths, provide two
                  folders and compare the files from first argument folder that
                  match with the ones of the second argument folder.

.. moduleauthor:: Iñigo Arredondo (inigo.arredondo@ehu.eus)


"""
from os import listdir, name
from difflib import unified_diff
import argparse


def do_diff_folders(infolder, masterfolder, slash):
    """ Compare the files in the infloder with the ones in masterfolder with
    the same name line by line and print the differences.
    If there are no files with the same in both folders it prints a message.

    Args:
        infolder: list of strs with the name (including path) of all the files
        to compare.
        masterfolder: list of strs with the name (including path) of all the
        files to compare.

     """

    for infile in infolder:
        is_compared = False
        infile_name = infile.split(slash)[-1]  # Get only the name 
        with open(infile) as inf:
            in_text = inf.read()
        for masterfile in masterfolder:
            masterfile_name = masterfile.split(slash)[-1]
            if infile_name == masterfile_name:
                is_compared = True
                with open(masterfile) as masterf:
                    master_text = masterf.read()
                print("Comparing " + infile_name + ":")
                for line in unified_diff(in_text, master_text,
                                         fromfile=infile,
                                         tofile=masterfile,
                                         lineterm=''):
                    print(line)
                break
        if is_compared is False:
            print(infile + ": No such file in destination.")


if __name__ == "__main__":
    if name == 'posix':
        slash = '/'
    else:
        slash = '\\'

    parser = argparse.ArgumentParser(description='Compare two files with the' +
                                     ' same name line by line.\n' +
                                     'Refer to difflib documentation to '
                                     'undertstand the output format.')
    parser.add_argument('file1', type=str, help='First file/folder to compare')
    parser.add_argument('file2', type=str, help='Second file/folder to compare')
    parser.add_argument('-d', action='store_const',
                        const=True,
                        help='Instead of single files with different paths, ' +
                        'provide two folders and compare the files from ' +
                        'first argument folder that match with the ones of ' +
                        'the second argument folder.')

    args = parser.parse_args()

    if args.d:
        # Add the path to all the files inside the folder
        path = args.file1
        if path[-1] != slash:  # Ensure that path end with slash
            path = path + slash
        args.file1 = [path + s for s in listdir(args.file1)]
        path = args.file2
        if path[-1] != slash:
            path = path + slash
        args.file2 = [path + s for s in listdir(args.file2)]

    else:
        args.file1 = [args.file1]
        args.file2 = [args.file2]

    do_diff_folders(args.file1, args.file2, slash)
