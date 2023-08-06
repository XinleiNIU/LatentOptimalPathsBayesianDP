from numpy import cumsum
import numpy as np
import bisect

class PMF(object):
    
    def __init__(self, p, v=None, check=True, normalise=False, log=False):
        p = list(map(float, p))
        if normalise:
            if log:
                s = reduce(logaddexp, p)
                p = [_ - s for _ in p]
            else:
                s = sum(p)
                p = [_ / s for _ in p]            
        self.p = list(p)
        self.log = log
        if self.log:
            self.cdf = cumlogaddexp(self.p)
        else:
            self.cdf = cumsum(self.p)
        self.n = len(p)
        if check:
            if self.log:
                assert np.allclose(self.cdf[-1], 0.0), self.cdf
            else:
                assert np.all(np.array(self.p) >= 0), p
                assert np.allclose(self.cdf[-1],1.0)
        self.cdf[-1] = 99
        self.v = v if v is not None else tuple(range(self.n))
        self.vdict = {k: i for i, k in enumerate(self.v)}
        self.uniform = len(set(p)) == 1
    
    def __getitem__(self, v):
        return self.p[self.vdict[v]]
        
    def sample(self, p=False):
        if p:
            if self.n == 1:
                return self.v[0], self.p[0]
            elif self.uniform:
                return random.choice(self.v), self.p[0]
            else:
                u = np.random.rand()
                if self.log:
                    u = np.log(u)
                i = bisect.bisect_left(self.cdf, u, 0, self.n)
                return self.v[i], self.p[i]
        else:
            if self.n == 1:
                return self.v[0]
            elif self.uniform:
                return random.choice(self.v)
            else:
                u = np.random.rand()
                if self.log:
                    u = np.log(u)
                i = bisect.bisect_left(self.cdf, u, 0, self.n)
                return self.v[i]
    
    def to_string(self, indent=0, v2s=str, header=True, prefix=''):
        if header:
            s = ('\t' * indent) + prefix + super(PMF, self).__str__() + '\n'
        else:
            s = ''
        s += ('\t' * indent)
        s += ('\n' + ('\t' * indent)).join([prefix + '%.3i : %.4f = p(%s)' % (i, np.exp(p), v2s(v)) for i, (p, v) in enumerate(zip(self.p, self.v))])
        return s
    
    def __str__(self):
        return self.to_string()
    
    def disp(self, indent=0, v2s=str, header=True):
        print(self.to_string(indent=indent, v2s=v2s, header=header))
    
    def __len__(self):
        return len(self.p)