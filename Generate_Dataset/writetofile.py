import argparse
from pathlib import Path
import numpy as np
from genrand import genrand

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--o', type=str, action='store')
parser.add_argument('-m','--m', type=int, action='store')
parser.add_argument('-n', '--n', type=int, action='store')
parser.add_argument('-d', "--d", type=float, action='store')

args = parser.parse_args()

if (args.o == ''):
    parser.error("Missing --o flag")
if (args.m == 0):
    parser.error("Invalid m value")
if (args.n == 0):
    parser.error("Invalid n value")
if (args.d <= 0 or args.d > 1):
    parser.error("Invalid d value")

output = args.o
m = args.m
n = args.n
d = args.d

np.set_printoptions(suppress=True)

ppl, f, A, b = genrand(m, n, d)
print(ppl)

np.savetxt(fname=output, X=ppl, delimiter=' ', newline='\n', fmt='%f')













