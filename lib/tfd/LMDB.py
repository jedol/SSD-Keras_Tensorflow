import numpy as np
import lmdb


class Writer(object):
    def __init__(self, path, buffer_size=128):
        ## Input
        ##  path: string = path to create db.
        ##  buffer_size: integer = MB
        self.isopen = True
        self.env = lmdb.open(path, map_size=1024*1024*1024*1024) # 1TB
        self.txn = self.env.begin(write=True, buffers=True)

        self.maximum_buffer_size = buffer_size*1024*1024 # to byte
        self.buffer_size = 0
        self.ind = 0

    def write(self, string):
        key = str(self.ind)
        self.txn.put(key, string)
        self.buffer_size += len(string)
        if self.buffer_size > self.maximum_buffer_size:
            self.txn.commit()
            self.txn = self.env.begin(write=True, buffers=True)
            self.buffer_size = 0
        self.ind += 1

    def close(self):
        if self.isopen:
            if self.buffer_size:
                self.txn.commit()
            self.env.close()
            self.isopen = False

    def __del__(self):
        self.close()


class Reader(object):
    def __init__(self, path):
        ## Input
        ##  path: string = db path to read.
        self.isopen = True
        self.env = lmdb.open(path, readonly=True)
        self.txn = self.env.begin()
        self.cur = self.txn.cursor()
        self.num_data = int(self.txn.stat()['entries'])
    
    def read(self, ind):
        assert ind < self.num_data, 'out of index: {} of {}'.format(ind, self.num_data)
        key = str(ind)
        return self.cur.get(key)

    def close(self):
        if self.isopen:
            self.env.close()
            self.isopen = False

    def __del__(self):
        self.close()