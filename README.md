# goESHyperNEAT 🇺🇦 [![Made in Ukraine](https://img.shields.io/badge/made_in-ukraine-ffd700.svg?labelColor=0057b7)](https://u24.gov.ua)

[![Build Status](https://travis-ci.org/yaricom/goESHyperNEAT.svg?branch=master)](https://travis-ci.org/yaricom/goESHyperNEAT) [![GoDoc](https://godoc.org/github.com/yaricom/goESHyperNEAT?status.svg)](https://godoc.org/github.com/yaricom/goESHyperNEAT)

## Overview
The implementation of Evolvable-Substrate HyperNEAT (ES-HyperNEAT) algorithm in GO language. The [ES-HyperNEAT][5] is an extension of the original
[HyperNEAT][4] method for evolving large-scale artificial neural networks using method of [NeuroEvolution of Augmenting Topologies][6].

The **HyperNEAT** is hypercube-based extension of NEAT allowing to encode ANNs in the substrate with specific geometric topology and with significant
number of neural units. In this respect it is similar to it's biological equivalent (brain) which also has defined topological
structure with groups of neural units in different regions performing different cognitive tasks. Another definitive trait
of HyperNEAT is usage of [Compositional Pattern Producing Network][7] (**CPPN**) to generate patterns of weights between network nodes
which allows to compactly encode huge neural network structures.

With all the power of **HyperNEAT** algorithm is has major drawback that neural nodes must be manually placed into substrate
by human before algorithm execution to reflect inherent geometrical topology of the task in hand. And with increased number
of hidden nodes in the network this leads to the reduction of algorithm efficiency due to lack of ability to estimate where
CPPN generated patterns will have intersection with manually seeded nodes.

This drawback is addressed by **Evolvable-Substrate HyperNEAT** method which allows to encode hidden nodes position
in the substrate in the CPPN generated patterns of weights. As additional benefit of this the substrate is able to evolve it's
geometrical topology during training, producing regions with varying neural density, thereby providing a kind of scaffolding
for situating cognitive structures in the biological brains.


## References

1. The original C++ NEAT implementation created by Kenneth O. Stanley, see: [NEAT][1]
2. Other NEAT implementations can be found at [NEAT Software Catalog][2]
3. [The ES-HyperNEAT Users Page][3]
4. Kenneth O. Stanley, David D’Ambrosio and Jason Gauci, [A Hypercube-Based Indirect Encoding for Evolving Large-Scale Neural Networks][4], Artificial Life journal, Cambridge, MA: MIT Press, 2009, vol. 15, no. 2, pp. 185-212 
5. Sebastian Risi, Kenneth O. Stanley, [An Enhanced Hypercube-Based Encoding for Evolving the Placement, Density and Connectivity of Neurons][5], Artificial Life journal, Cambridge, MA: MIT Press, 2012, vol. 18, no. 4, pp. 331-363
6. Kenneth O. Stanley, [Ph.D. Dissertation: Efficient Evolution of Neural Networks through Complexification][6], Department of Computer Sciences, The University of Texas at Austin, Technical Report~AI-TR-04–39, August 2004
7. Kenneth O. Stanley, [Compositional Pattern Producing Networks: A Novel Abstraction of Development][7], Genetic Programming and Evolvable Machines, Special Issue on Developmental Systems, New York, NY: Springer, 2007, vol. 8, pp. 131-162
8. Omelianenko, Iaroslav, [The GoLang implementation of NeuroEvolution of Augmented Topologies (NEAT) algorithm][8], GitHub, Computer software

This source code maintained and managed by [Iaroslav Omelianenko][9]


[1]:http://www.cs.ucf.edu/~kstanley/neat.html
[2]:http://eplex.cs.ucf.edu/neat_software/
[3]:http://eplex.cs.ucf.edu/hyperNEATpage/HyperNEAT.html
[4]:https://doi.org/10.1162/artl.2009.15.2.15202
[5]:https://doi.org/10.1162/ARTL_a_00071
[6]:https://nn.cs.utexas.edu/downloads/papers/stanley.phd04.pdf
[7]:https://doi.org/10.1007/s10710-007-9028-8
[8]:https://doi.org/10.5281/zenodo.13628842
[9]:https://io42.space