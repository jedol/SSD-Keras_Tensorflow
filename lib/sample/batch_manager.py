import numpy as np
import lmdb
import cv2
import cPickle as pickle
from multiprocessing import Process,Queue

from sample.preprocessing import transform_for_train, transform_for_test
from ssd.target import ssd_target_from_sample


def binary_encoder(data):
    ## Input
    ##  data: dict
    ##      'id': sample ID(usually name of image file)
    ##      'image': path to image file
    ##      'objects': dict
    ##          'bbox': bounding box coordinate of object
    ##          'label': label of object
    ## Output
    ##  string: encoded list
    ##      [id, encoded_image, labels, nbboxes]

    ## encode image
    image = cv2.imread(data['image'])
    _, encoded_image = cv2.imencode('.jpg', image)
    data['image'] = encoded_image

    return pickle.dumps(data, pickle.HIGHEST_PROTOCOL)


def binary_decoder(string):
    ## Input
    ##  string: encoded list
    ##      [id, encoded_image, labels, nbboxes]
    ## Output
    ##  data: list
    ##      [id, image, labels, nbboxes]

    data = pickle.loads(string)

    ## decode image
    data['image'] = cv2.imdecode(data['image'], cv2.IMREAD_COLOR)

    return data


class TrainBatchManager(object):
    def __init__(self, source, batch_size, prior_boxes, target, transform=None,
                 shuffle=True, use_prefetch=True, capacity=32):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_prefetch = use_prefetch
        self.capacity = capacity

        self.transform = transform
        self.prior_boxes = prior_boxes
        self.target = target

        ## open LMDB
        self.env = lmdb.open(source, readonly=True)
        self.txn = self.env.begin()
        self.cur = self.txn.cursor()
        self.num_data = int(self.txn.stat()['entries'])

        ## initialize indices
        self.reset_inds()

        ## prefetch thread
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
        items = list()
        for _ in xrange(self.batch_size):
            ## read one binary record
            string = self.cur.get(str(self.next_inds()))

            ## decoding
            data = binary_decoder(string)

            ## ravel objects
            image = data['image']
            labels = np.array([obj['label'] for obj in data['objects']], np.float32)
            bboxes = np.array([obj['bbox'] for obj in data['objects']], np.float32)
            h,w = image.shape[:2]
            nbboxes = bboxes/[w,h,w,h]

            ## transform and smaple data, if needed
            if self.transform is not None:
                image, labels, nbboxes = transform_for_train(image, labels, nbboxes, self.transform)

            ## compute target
            conf_target, loc_target = ssd_target_from_sample(self.prior_boxes, labels, nbboxes, self.target)

            ## Note: 'item' should be list of array
            items.append([image, conf_target, loc_target])

        ## gather items
        batch = list()
        for i in xrange(len(items[0])):
            batch.append(np.array([item[i] for item in items]))

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


class ValidBatchManager(object):
    def __init__(self, source, batch_size, transform=None, shuffle=True, use_prefetch=True, capacity=32):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_prefetch = use_prefetch
        self.capacity = capacity

        self.transform = transform

        ## open LMDB
        self.env = lmdb.open(source, readonly=True)
        self.txn = self.env.begin()
        self.cur = self.txn.cursor()
        self.num_data = int(self.txn.stat()['entries'])

        ## initialize indices
        self.reset_inds()

        ## prefetch thread
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
        batch = list()
        for _ in xrange(self.batch_size):
            ## read one binary record
            string = self.cur.get(str(self.next_inds()))

            ## decoding
            data = binary_decoder(string)

            ## transform and smaple data, if needed
            if self.transform is not None:
                data['image'] = transform_for_test(data['image'], self.transform)

            ## Note: 'item' should be list of array
            batch.append(data)

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

    def get_all(self):
        all_data = list()
        for ind in xrange(self.num_data):
            ## read one binary record
            string = self.cur.get(str(ind))

            ## decoding
            data = binary_decoder(string)

            ## transform and smaple data, if needed
            if self.transform is not None:
                data['image'] = transform_for_test(data['image'], self.transform)

            ## Note: 'item' should be list of array
            all_data.append(data)

        return all_data

    def get_generator(self):
        while True:
            yield self.get_batch()

    def close(self):
        if self.use_prefetch:
            self.proc.terminate()
            self.proc.join()

