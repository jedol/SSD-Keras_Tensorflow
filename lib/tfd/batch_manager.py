import numpy as np
from multiprocessing import Process,Queue

import lmdb
import cPickle as pickle


class ArrayBatchManager(object):
    def __init__(self, arrays, batch_size, shuffle=True, use_prefetch=True, capacity=32):
        if not isinstance(arrays, list) and not isinstance(arrays, tuple):
            assert isinstance(arrays, np.ndarray)
            arrays = [arrays]

        assert len(arrays)
        assert len(np.unique([len(array) for array in arrays])) == 1

        self.arrays = arrays
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_prefetch = use_prefetch
        self.capacity = capacity
        self.num_data = len(arrays[0])

        self.reset_inds()

        if self.use_prefetch:
            self.batch_queue = Queue(capacity)
            self.proc = Process(target=self._worker)
            self.proc.start()
            def cleanup():
                self.proc.terminate()
                self.proc.join()
            import atexit
            atexit.register(cleanup)

    def reset_inds(self):
        if self.shuffle:
            self.inds = np.random.permutation(self.num_data)
        self.idx = 0

    def next_inds(self):
        if self.shuffle:
            ind = self.inds[self.idx]
        else:
            ind = self.idx
        self.idx += 1
        if self.idx >= self.num_data:
            self.reset_inds()
        return ind

    def _read_batch(self):
        inds = [self.next_inds() for _ in xrange(self.batch_size)]
        return [array[inds] for array in self.arrays]

    def _worker(self):
        while True:
            self.batch_queue.put(self._read_batch())

    def get_batch(self):
        if self.use_prefetch:
            batch = self.batch_queue.get()
        else:
            batch = self._read_batch()
        return batch

    def get_generator(self):
        while True:
            yield self.get_batch()

    def close(self):
        if self.use_prefetch:
            self.proc.terminate()
            self.proc.join()


class LMDBBatchManager(object):
    def __init__(self, source, batch_size, shuffle=True, use_prefetch=True, capacity=32):
        ## open LMDB
        self.env = lmdb.open(source, readonly=True)
        self.txn = self.env.begin()
        self.cur = self.txn.cursor()

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_prefetch = use_prefetch
        self.capacity = capacity
        self.num_data = int(self.txn.stat()['entries'])

        self.reset_inds()

        if self.use_prefetch:
            self.batch_queue = Queue(capacity)
            self.proc = Process(target=self._worker)
            self.proc.start()
            def cleanup():
                self.proc.terminate()
                self.proc.join()
            import atexit
            atexit.register(cleanup)

    def reset_inds(self):
        if self.shuffle:
            self.inds = np.random.permutation(self.num_data)
        self.idx = 0

    def next_inds(self):
        if self.shuffle:
            ind = self.inds[self.idx]
        else:
            ind = self.idx
        self.idx += 1
        if self.idx >= self.num_data:
            self.reset_inds()
        return ind

    def _read_batch(self):
        data_list = list()
        for _ in xrange(self.batch_size):
            string = self.cur.get(str(self.next_inds()))
            # string = self.cur.get('{:08d}'.format(self.next_inds()))

            ## do some decoding here
            ## unpickle is default
            data = pickle.loads(string)

            ## do some data transform here

            ## Note: 'data' should list of array
            data_list.append(data)

        batch = list()
        for i in xrange(len(data_list[0])):
            batch.append(np.array([data[i] for data in data_list]))
        return batch

    def _worker(self):
        while True:
            self.batch_queue.put(self._read_batch())

    def get_batch(self):
        if self.use_prefetch:
            batch = self.batch_queue.get()
        else:
            batch = self._read_batch()
        return batch

    def get_generator(self):
        while True:
            yield self.get_batch()

    def close(self):
        if self.use_prefetch:
            self.proc.terminate()
            self.proc.join()

