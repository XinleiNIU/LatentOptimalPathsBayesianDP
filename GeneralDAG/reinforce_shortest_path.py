import numpy as np
import itertools
import random
import time
import collections
import wpack.helpers as wh
from scipy.special import logsumexp
import pdb

def sample_pdf(p):
    x = random.random()
    i = -1
    cdf = 0
    p[-1] = 1
    while x > cdf:
        i += 1
        cdf += p[i]
    return i


class WeightedDAG:

    def __init__(self, conductance, alpha):
        self.conductance = conductance
        self.alpha = alpha
        # self.conductance = tuple(map(collections.OrderedDict, conductance))
        self.n = len(conductance)
        self._reversed_adjacency_unrelabeled = None
        self._node_path_count = None
        self._marginal_q_mus = None
        self._transition_log_probabilities = None
        self.check()

    def check(self):
        assert all(isinstance(_, dict) for _ in self.conductance)
        for i, d in enumerate(self.conductance):
            for j, w in d.items():
                assert j > i and j < self.n and j == int(j), (i, j, type(j), self.n)
        assert self.conductance[-1] == dict()
        assert all((len(d)>0 for d in self.conductance[:-1]))
        assert all([_ > 0 for _ in self.node_path_count]), self.node_path_count

    @classmethod
    def from_matrix(cls, mat, alpha):
        n = mat.shape[0]
        conductance = []
        for i in range(n):
            assert np.all(np.isinf(mat[i,:i]))
            inds = ~np.isinf(mat[i, i:])
            j = np.where(inds)[0].astype(int) + i
            w = mat[i, j]
            conductance.append(dict(zip(j, w)))
        wdag = cls(conductance=conductance, alpha=alpha)
        wdag.check()
        return wdag

    def to_matrix(self):
        mat = np.empty((self.n, self.n), dtype=float)
        mat[:] = np.inf
        for i, d in enumerate(self.conductance):
            if len(d):
                j, w = zip(*d.items())
                mat[i, j] = w
        return mat

    def edgedict_to_matrix(self, e):
        mat = np.empty((self.n, self.n), dtype=float)
        mat[:] = np.nan
        for (u, v), value in e.items():
            mat[u, v] = value
        return mat

    @classmethod
    def random_dag(cls, n, threshold, maxiter, alpha):
        for iter in range(maxiter):
            mat = np.random.randn(2 * n, 2 * n)
            mask = np.random.rand(2 * n, 2 * n) < threshold
            mat[mask] = np.inf
            mat[np.tri(*mat.shape) == 1] = np.inf
            try:
                wdag = WeightedDAG.from_matrix(mat, alpha)
            except:
                continue
            paths = wdag.complete_paths()
            ikeep = sorted(set(itertools.chain(*paths)))
            if len(ikeep) != n:
                continue
            mat = mat[ikeep, :][:, ikeep]
            try:
                wdag = WeightedDAG.from_matrix(mat, alpha)
            except:
                continue
            break
        assert iter < maxiter - 1
        print('created random DAG after iter', iter, 'of',  maxiter)
        return wdag

    @classmethod
    def self_test(cls):

        alpha = np.random.rand() * 2 + 0.05

        # wdag = WeightedDAG(({1: 4, 2: 10}, {2: 4}, {}))
        wdag = WeightedDAG.random_dag(n=7, threshold=0.5, maxiter=9999, alpha=alpha)
        # wdag = WeightedDAG(({1: 1, 2: 1}, {3: 2}, {3:1}, {4:1}, {}))
        # wdag = WeightedDAG.from_matrix(np.array(
        #     [
        #         [-np.inf, 1, 1, -np.inf],
        #         [-np.inf, -np.inf, 1, 1],
        #         [-np.inf, -np.inf, -np.inf, 1],
        #         [-np.inf, -np.inf, -np.inf, -np.inf],
        #     ]
        # ))
        # return
        assert np.all(wdag.to_matrix() == WeightedDAG.from_matrix(mat=wdag.to_matrix(), alpha=alpha).to_matrix())
        print(wdag.to_matrix())
        print('conductance', wdag.conductance)
        print('paths_dfs', WeightedDAG.paths_dfs(wdag, path=[0], paths=[]))
        samples = [wdag.sample_uniform() for _ in range(10)]
        print('sample_uniform empirical_pmf_dict', empirical_pmf_dict(samples))
        spbw, spb = wdag.shortest_path_brute()
        print('shortest_path_brute', spb)
        spdw, spd = wdag.shortest_path_dp()
        print('shortest_path_dp', spd)
        assert spb == spd
        assert np.isclose(spbw, spdw)
        npc = wdag.node_path_count
        npcb = wdag.node_path_count_brute()
        print('complete_paths', wdag.complete_paths())
        print('node_path_count', npc)
        assert np.all(npc == npcb), (npc, npcb)

        n = 1000
        compute_dp_logp = True

        omega_brute = wdag.omega_brute()
        omega = wdag.omega()
        ok_brute, ov_brute = zip(*sorted(omega.items()))
        ok, ov = zip(*sorted(omega_brute.items()))

        assert ok == ok_brute
        assert np.allclose(ov, ov_brute, atol=1e-6), list(zip(ov, ov_brute))

        print('omega matches', list(zip(ov, ov_brute)))

        wdag_r = WeightedDAG.from_matrix(mat=wdag.to_matrix() * np.random.rand(wdag.n, wdag.n), alpha=alpha)
        kld = wdag.kld(wdag_r)
        kld_brute = wdag.kld_brute(wdag_r)

        assert np.isclose(kld, kld_brute, atol=1e-6)

        zeta_brute = wdag.zeta_brute()
        zeta = wdag.zeta()

        assert np.allclose(sorted(zeta), sorted(zeta_brute), atol=1e-6)
        assert np.allclose(zeta, zeta_brute, atol=1e-6), list(zip(zeta, zeta_brute))

        print('omega matches', list(zip(ov, ov_brute)))

        print('kld matches', kld_brute, kld)

        gibbs_distribution = wdag.gibbs_distribution()
        gibbs_paths_brute = wdag.sample_gibbs_brute(n=n)
        gibbs_paths_dp, gibbs_paths_dp_logp = zip(*wdag.sample_gibbs_dp(n=n, compute_logp=compute_dp_logp))
        support = sorted(gibbs_distribution.keys())
        gibbs_paths_dp_pmf = empirical_pmf_dict(gibbs_paths_dp, support=support)
        gibbs_paths_brute_pmf = empirical_pmf_dict(gibbs_paths_brute, support=support)
        print('alpha', alpha)
        print('support', support)
        gibbs_probs_exact = np.array([gibbs_distribution[k] for k in support])
        gibbs_probs_dp = np.array(list(zip(*gibbs_paths_dp_pmf))[1])
        gibbs_probs_brute = np.array(list(zip(*gibbs_paths_brute_pmf))[1])
        print('gibbs_probs exact / dp / brute')
        print(np.vstack([gibbs_probs_exact, gibbs_probs_dp, gibbs_probs_brute]))
        print('gibbs_probs_dp error', np.mean(np.abs(gibbs_probs_exact-gibbs_probs_dp)))
        print('gibbs_probs_brute error', np.mean(np.abs(gibbs_probs_exact-gibbs_probs_brute)))
        seen = set()
        first = True
        if compute_dp_logp:
            for path, logp in zip(gibbs_paths_dp, gibbs_paths_dp_logp):
                if path in seen:
                    continue
                seen.add(path)
                assert np.isclose(logp, np.log(gibbs_distribution[path]), atol=1e-6)
                logp2 = wdag.path_log_probability(path)
                assert np.isclose(logp, logp2, atol=1e-6)
                if first:
                    print('dp logp matches e.g.', logp, logp2, np.log(gibbs_distribution[path]))
                    first = False

        return wdag

    @property
    def reversed_adjacency_unrelabeled(self):
        if self._reversed_adjacency_unrelabeled is None:
            self._reversed_adjacency_unrelabeled = self.reversed_adjacency(relabel=False)
        return self._reversed_adjacency_unrelabeled

    @classmethod
    def paths_dfs(cls, wdag, path, paths):
        datum = path[-1]
        for val in wdag.conductance[datum].keys():
            new_path = path + [val]
            paths = cls.paths_dfs(wdag, new_path, paths)
        else:
            paths += [tuple(path)]
        return paths

    def complete_paths(self):
        paths = type(self).paths_dfs(self, [0], [])
        return [path for path in paths if path[-1] == self.n-1]

    def path_weight(self, path):
        return sum(self.conductance[i][j] for i, j in zip(path[:-1], path[1:]))

    def gibbs_distribution(self):
        paths = self.complete_paths()
        weights = np.array([self.path_weight(path) for path in paths])
        uprobabilities = np.exp(self.alpha*weights)
        probabilities = uprobabilities / uprobabilities.sum()
        return dict(zip(paths, probabilities))

    def sample_uniform(self):
        path = [0]
        while len(self.conductance[path[-1]]):
            j = self.conductance[path[-1]].keys()
            path.append(random.sample(j, 1)[0])
        return tuple(path)

    def shortest_path_brute(self):
        paths = self.complete_paths()
        weights = np.array([self.path_weight(path) for path in paths])
        path = paths[np.argmin(weights)]
        return self.path_weight(path), path

    def shortest_path_dp(self):

        trace = np.zeros(self.n, dtype=int)
        q = np.zeros(self.n)
        q[1:] = np.inf
  
        for i, d in enumerate(self.conductance):
            pdb.set_trace()
            for j, w in d.items():
                if q[i] + w < q[j]:
                    q[j] = q[i] + w
                    trace[j] = i
        j = self.n-1
       
        reversed_path = [j]
        while j > 0:
            j = trace[j]
            reversed_path.append(j)
        path = tuple(reversed(reversed_path))
        return q[-1], path

    def reversed_adjacency(self, relabel):
        adjacency = [[] for _ in range(self.n)]
        for i, d in enumerate(self.conductance):
            for j in d.keys():
                adjacency[j].append(i)
        if relabel:
            relabel = np.array(tuple(range(self.n - 1, -1, -1)), dtype=int)
            adjacency = [tuple(relabel[_]) for _ in reversed(adjacency)]
        return adjacency

    @property
    def node_path_count(self):
        if self._node_path_count is None:
            ra = self.reversed_adjacency(relabel=True)
            itopo = range(self.n)
            c = np.zeros(self.n, dtype=int)
            c[0] = 1
            for j in itopo:
                for i in ra[j]:
                    c[i] += c[j]
            self._node_path_count = list(reversed(c))
        return self._node_path_count

    def node_path_count_brute(self):
        paths = self.complete_paths()
        suffixes = set(itertools.chain(*([[p[i:] for i in range(len(p))] for p in paths])))
        cnt = collections.Counter(s[0] for s in suffixes)
        c = np.array([cnt[i] for i in range(self.n)], dtype=int)
        return c

    def sample_gibbs_brute(self, n):
        paths, probs = zip(*self.gibbs_distribution().items())
        pmf = wh.PMF(probs)
        return tuple(paths[pmf.sample()] for _ in range(n))

    def omega_brute(self):
        paths, probs = zip(*self.gibbs_distribution().items())
        omega = collections.defaultdict(lambda : 0)
        for path, prob in zip(paths, probs):
            for u, v in zip(path[:-1], path[1:]):
                omega[(u,v)] += prob
        return omega

    def zeta_brute(self):
        paths, probs = zip(*self.gibbs_distribution().items())
        zeta = np.zeros(self.n)
        for path, prob in zip(paths, probs):
            for u in path:
                zeta[u] += prob
        return zeta

    def transition_log_probabilities(self):
        if self._transition_log_probabilities is None:
            marginal_q_mus = self.marginal_q_mus()
            self._transition_log_probabilities = [dict() for _ in range(self.n)]
            for iparent, d in enumerate(self.conductance):
                for i in d.keys():
                    self._transition_log_probabilities[iparent][i] = (marginal_q_mus[iparent] + self.alpha * self.conductance[iparent][i]) - marginal_q_mus[i]
        return self._transition_log_probabilities

    def path_log_probability(self, path):
        pi = self.transition_log_probabilities()
        return sum((pi[u][v] for u, v in zip(path[:-1], path[1:])))

    def zeta(self):

        log_pi = self.transition_log_probabilities()

        zeta = np.zeros(self.n)
        zeta[-1] = 1
        ireversetopo = list(reversed(range(self.n)))

        for i in ireversetopo:
            children = self.conductance[i].keys()
            for ichild in children:
                zeta[i] += zeta[ichild] * np.exp(log_pi[i][ichild])

        return zeta

    def omega(self):

        ra = self.reversed_adjacency_unrelabeled
        log_pi = self.transition_log_probabilities()

        lam = np.zeros(self.n)
        lam[0] = 1
        itopo = range(self.n)

        for i in itopo:
            for iparent in ra[i]:
                lam[i] += lam[iparent] * np.exp(log_pi[iparent][i])

        rho = np.zeros(self.n)
        rho[-1] = 1
        ireversetopo = reversed(range(self.n))

        for i in ireversetopo:
            children = self.conductance[i].keys()
            for ichild in children:
                rho[i] += rho[ichild] * np.exp(log_pi[i][ichild])

        omega = collections.defaultdict(lambda: 0)
        for u, d in enumerate(self.conductance):
            for v in d.keys():
                omega[(u, v)] = lam[u] * rho[v] * np.exp(log_pi[u][v])

        return omega

    def marginal_q_mus(self):

        if self._marginal_q_mus is None:

            ra = self.reversed_adjacency_unrelabeled
            self._marginal_q_mus = np.zeros(self.n)

            for i, d in enumerate(self.conductance):

                if len(ra[i]):
                    
                    self._marginal_q_mus[i] = logsumexp(tuple(self._marginal_q_mus[k] + self.alpha * self.conductance[k][i] for k in ra[i]))

        return self._marginal_q_mus

    def sample_gibbs_dp(self, n, compute_logp):

        if n > 1 and n == int(n):
            t0 = time.time()
            rval = tuple(self.sample_gibbs_dp(n=1, compute_logp=compute_logp) for _ in range(n))
            # print('compute_logp', compute_logp, 'sampling rate', wh.pretty_number(n/(time.time()-t0)))
            return rval
        else:
            assert n == 1

        ra = self.reversed_adjacency_unrelabeled
        log_pi = self.transition_log_probabilities()

        reversed_path = [self.n - 1]
        logp = 0
        while reversed_path[-1] > 0:

            this_logps = [log_pi[k][reversed_path[-1]] for k in ra[reversed_path[-1]]]
            k = sample_pdf(np.exp(this_logps))
            reversed_path.append(ra[reversed_path[-1]][k])
            logp += this_logps[k]

        path = tuple(reversed(reversed_path))

        return path, logp

    def kld_brute(self, other):
        rval = 0
        for path in self.complete_paths():
            logp = self.path_log_probability(path)
            logpr = other.path_log_probability(path)
            rval += np.exp(logp) * (logp-logpr)
        return rval

    def kld(self, other):
        assert self.alpha == other.alpha
        rval = other.marginal_q_mus()[-1] - self.marginal_q_mus()[-1]
        omega_dict = self.omega()
        rval += sum([omega * self.alpha * (self.conductance[u][v] - other.conductance[u][v]) for (u, v), omega in omega_dict.items()])
        return rval


def empirical_pmf_dict(x, support=None):
    c = collections.Counter(x)
    return [(k, c[k]/len(x)) for k in (sorted(c.keys()) if support is None else support)]


if __name__ == '__main__':

    # test_shortest_monotonic_path()

    wdag = WeightedDAG.self_test()
