#!/usr/bin/python3



import random
import numpy as np

from rdkit import Chem 
from rdkit.Chem import AllChem 
from rdkit.Chem import MACCSkeys
from rdkit.Chem import DataStructs
import pickle

import argparse


# input file 
#   index       smiles
#   1           CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C
#   2           CC(=O)OC(CC(=O)O)C[N+](C)(C)C
#   3           C1=CC(C(C(=C1)C(=O)O)O)O
#       ...

parser = argparse.ArgumentParser(description = 'segment name')
parser.add_argument('fnameR', type = str)
parser.add_argument('fnameW', type = str)
args = parser.parse_args()

#def smileList(args.fname):

def MACCStoArray(mol):
    maccs_key = MACCSkeys.GenMACCSKeys(mol)
    DataStructs.ConvertToNumpyArray(maccs_key, array)
    NonZeroElements = array.nonzero()
    return NonZeroElements

# convert smiles to nonzero indices/add another list to nested list.
def SmilestoIndices(indices):
    temp = list()
    for i in range(len(indices[0])):
        temp.append(indices[0][i])
    return temp
#

with open(args.fnameR) as f:
    with open(args.fnameW, 'a') as p:
        for line in f.readlines():
            smiles = line.split()[1].strip()
            index = line.split()[0].strip()
            #print(smiles)
            #print(type(smiles))
            array = np.zeros((0, ), dtype=np.int8)
            molR = Chem.MolFromSmiles(smiles)
            if molR is None: continue
            TrueIndices = MACCStoArray(molR)
            p.write(str(index)+ '\t' + str(SmilestoIndices(TrueIndices)))
            p.write('\n')


