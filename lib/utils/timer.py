import timeit as ti


class Timer:
    def __init__(self, start, end, prefix='', print_interval=1.0):
        ## Input
        ##  start: int
        ##  end: int
        ##  prefix: string
        ##  print_interval: float(sec)

        self.start = start
        self.end = end
        self.print_interval = print_interval
        self.prefix = prefix

        self.count = 0
        self.total_count = 0
        
        self.elapsed = 0
        self.total_elapsed = 0
        
        self.t0 = None
        self.t1 = None

    def tic(self):
        self.t0 = ti.default_timer()

    def toc(self):
        if self.t0 == None:
            return
        self.t1 = ti.default_timer()
        
        self.elapsed += self.t1-self.t0
        self.total_elapsed += self.t1-self.t0

        self.count += 1
        self.total_count += 1

        if self.elapsed > self.print_interval:
            processed = float(self.start+self.total_count)/self.end*100
            job_per_sec = self.count/self.elapsed
            remain = int(round(self.total_elapsed*(self.end-self.start-self.total_count)/self.total_count))
            print '{} {}/{} ({:.2f}%) ({:.3f} j/s) (ETA: {})'.format(
                self.prefix, self.start+self.total_count, self.end, processed,
                job_per_sec, self.sec2day(remain))
            self.elapsed = 0
            self.count = 0
        self.t0 = None

    def sec2day(self, sec):
        s = sec
        m = 0
        h = 0
        d = 0
        if sec >= 60:
            m,s = self.div(sec, 60)
            if m >= 60:
                h,m = self.div(m, 60)
                if h >= 60:
                    d,h = self.div(h, 60)
        fmt = str()
        if d > 0:
            fmt += '{:d}day '.format(d)
        fmt += '{:02d}:{:02d}:{:02d}'.format(h,m,s)
        return fmt

    def div(self, a, b):
        q = int(a/b)
        r = a%b
        return q,r


if __name__=='__main__':
    0
    ## for debug
    import numpy as np
    import time
    num_iter = 20
    t = Timer(0, num_iter, prefix='[Test.py]')
    for i in xrange(num_iter):
        t.tic()
        time.sleep(np.random.uniform(0.1, 0.5))
        t.toc()

    
