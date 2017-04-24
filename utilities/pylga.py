import struct
from collections import namedtuple

class LGATable(object):
    def __init__(self):
        self.elsz = 0
        self.rows = 0
        self.cols = 0
        self.header = []
        self.data = []

    def delete_row(self, i):
        rows -= 1
        del self.data[i]

    def load(self, stream):
        magic = stream.read(8)
        if magic != b'LGATABLE':
            raise Exception("Invalid magic header")
        elsz = struct.unpack('i', stream.read(4))[0]
        if elsz != 4:
            raise("Invalid element size")
        self.elsz = elsz
        cols = struct.unpack('i', stream.read(4))[0]
        rows = struct.unpack('i', stream.read(4))[0]
        self.rows = rows
        self.cols = cols
        for i in range(cols):
            h = ''
            while True:
                ch = stream.read(1)
                if ch == b'\x00':
                    break
                h += ch.decode('utf8')
            self.header.append(h)
        Entry = namedtuple("Entry", self.header)
        for i in range(rows):
            entry = []
            for j in range(cols):
                v = struct.unpack('f', stream.read(4))[0]
                entry.append(v)
            self.data.append(Entry(*entry))

    def __iter__(self):
        return iter(self.data)
