#!/usr/bin/env python3
import os, sys, struct

def writeTable(header, table):

    file = sys.stdout
    tablesep = ' '

    colszs = [len(title) for title in header]
    for i in range(0, len(colszs)):
        colszs[i] = max([colszs[i]]+[len(row[i]) for row in table])

    file.write(
        tablesep.join([i[1].ljust(i[0]) for i in zip(colszs, header)]) + '\n'
    )

    for row in table:
        file.write(
            tablesep.join([i[1].ljust(i[0]) for i in zip(colszs, row)]) + '\n'
        )

table = sys.stdin.buffer

magic = table.read(8)
if magic != b'LGATABLE':
    table.close()
    print('Format exception, magic=%s' % (magic))
    exit(-2)

elem = table.read(4)
elem = struct.unpack('i', elem)[0]
if elem != 4:
    table.close()
    print('Format exception, element size=%d' % (elem))
    exit(-3)

ncols = table.read(4)
ncols = struct.unpack('i', ncols)[0]

nrows = table.read(4)
nrows = struct.unpack('i', nrows)[0]

header = []
data = []

# Read the column headers
for c in range(ncols):
    h = ''
    while True:
        ch = table.read(1)
        if ch == b'\x00':
            break
        h += ch.decode('utf8')
    header.append(h)

for r in range(nrows):
    entry = []
    for c in range(ncols):
        v = table.read(4)
        v = struct.unpack('f', v)[0]
        entry.append(str(v))
    data.append(entry)

table.close()

writeTable(header, data)
