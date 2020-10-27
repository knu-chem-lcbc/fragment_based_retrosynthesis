#!/usr/bin/python3


import ast

import argparse

parser = argparse.ArgumentParser(description = 'segment name')
parser.add_argument('fnameR', type = str)
parser.add_argument('fnameW', type = str)
args = parser.parse_args()

assigned = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 18, 20, 27, 29, 30, 31, 35, 42, 44, 46, 166, 14, 49, 63, 56, 59, 15, 39, 40, 48, 87, 103, 107, 134, 23, 21, 64, 125, 28, 70]
with open(args.fnameR) as f:
    with open(args.fnameW, 'a') as p:
        for line in f.readlines():
            maccs = line.split('\t')[1].strip()
            y = ast.literal_eval(maccs)
            index = line.split('\t')[0].strip()
            for x in assigned:
                y = [i for i in y if i != x]
            p.write(str(index)+ '\t' + str(y))
            p.write('\n')





