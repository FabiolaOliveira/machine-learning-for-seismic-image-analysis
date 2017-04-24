#!/usr/bin/env python3
import os, sys, struct

header = sys.stdin.readline()
header = header.split()
table = sys.stdin.read()
table = table.split()

tableout = sys.stdout.buffer

ncols = len(header)
nrows = len(table) // ncols

tableout.write(b'LGATABLE')
tableout.write(struct.pack('i', 4))
tableout.write(struct.pack('i', ncols))
tableout.write(struct.pack('i', nrows))

for h in header:
    tableout.write(h.encode('utf8'))
    tableout.write(b'\0')

for v in table:
    tableout.write(struct.pack('f', float(v)))

