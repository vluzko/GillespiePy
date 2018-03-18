# GillespiePy

This *was* originally just a fork of StochPy, with the goal of improving performance. However the StochPy code base is almost unsalvageable, so it has been rewritten from scratch.

GillespiePy is a Python library for simulating stochastic systems. Currently it is simply an implementation of the Gillespie algorithm [[Gillespie 1977](http://pubs.acs.org/doi/abs/10.1021/j100540a008)]. There are plans to expand to more modern implementations of the same algorithm.

## Performance
The Gillespie algorithm is relatively slow, and Python isn't exactly known for performance. To get around this I have done some *extremely* hacky things involving Numba and run time code generation. This does lead to significant performance increases, but Here Be Dirty Code.

## Algorithms

### Implemented
* Direct stochastic simulation, [Gillespie 1977](http://pubs.acs.org/doi/abs/10.1021/j100540a008)

### Planned
* Next reaction [Gibson et al 2000](http://pubs.acs.org/doi/abs/10.1021/jp993732q)
* Delayed next reaction [Anderson 2007](https://arxiv.org/abs/0708.0370)
* Tau leaping? It isn't exact which is kind of a problem.
* GPU support? [Komarov et al 2012](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0046693)
* Non Markovian processes? [Boguna et al 2013](https://arxiv.org/abs/1310.0926)