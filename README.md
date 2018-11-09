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

This drawback is addressed by **Evolved-Substrate HyperNEAT** method which allows to encode hidden nodes position
in the substrate in the CPPN generated patterns of weights. As additional benefit of this the substrate is able to evolve it's
geometrical topology during training, producing regions with varying neural density, thereby providing a kind of scaffolding
for situating cognitive structures in the biological brains.


## References:

1. The original C++ NEAT implementation created by Kenneth O. Stanley, see: [NEAT][1]
2. Other NEAT implementations can be found at [NEAT Software Catalog][2]
3. [The ES-HyperNEAT Users Page][3]
4. Kenneth O. Stanley, David D’Ambrosio and Jason Gauci, [A Hypercube-Based Indirect Encoding for Evolving Large-Scale Neural Networks][4], Artificial Life journal 15(2), Cambridge, MA: MIT Press, 2009
5. Sebastian Risi, Kenneth O. Stanley, [An Enhanced Hypercube-Based Encoding for Evolving the Placement, Density and Connectivity of Neurons][5], Artificial Life journal, Cambridge, MA: MIT Press, 2012
6. Kenneth O. Stanley, [Ph.D. Dissertation: EFFICIENT EVOLUTION OF NEURAL NETWORKS THROUGH COMPLEXIFICATION][6], Department of Computer Sciences, The University of Texas at Austin, Technical Report~AI-TR-04–39, August 2004
7. Kenneth O. Stanley, [Compositional Pattern Producing Networks: A Novel Abstraction of Development][7], Genetic Programming and Evolvable Machines, Special Issue on Developmental Systems, New York, NY: Springer, 2007
8. Iaroslav Omelianenko, [The GoLang NEAT implementation][8], GitHub, 2018

This source code maintained and managed by [Iaroslav Omelianenko][9]


[1]:http://www.cs.ucf.edu/~kstanley/neat.html
[2]:http://eplex.cs.ucf.edu/neat_software/
[3]:http://eplex.cs.ucf.edu/hyperNEATpage/HyperNEAT.html
[4]:http://eplex.cs.ucf.edu/papers/stanley_alife09.pdf
[5]:https://www.mitpressjournals.org/doi/pdfplus/10.1162/ARTL_a_00071
[6]:http://nn.cs.utexas.edu/keyword?stanley:phd04
[7]:http://eplex.cs.ucf.edu/papers/stanley_gpem07.pdf
[8]:https://github.com/yaricom/goNEAT
[9]:https://io42.space