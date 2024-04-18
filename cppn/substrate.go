package cppn

import (
	"errors"
	"fmt"
	"github.com/yaricom/goESHyperNEAT/v2/hyperneat"
	neatmath "github.com/yaricom/goNEAT/v4/neat/math"
	"github.com/yaricom/goNEAT/v4/neat/network"
	"math"
)

// Substrate represents substrate holding configuration of ANN with weights produced by CPPN. According to HyperNEAT method
// the ANN neurons are encoded as coordinates in hypercube presented by this substrate.
// By default, neurons will be placed into substrate within grid layout
type Substrate struct {
	// The layout of neuron nodes in this substrate
	Layout SubstrateLayout

	// The activation function's type for neurons encoded
	NodesActivation neatmath.NodeActivationType
}

// NewSubstrate creates new instance of substrate.
func NewSubstrate(layout SubstrateLayout, nodesActivation neatmath.NodeActivationType) *Substrate {
	substr := Substrate{
		Layout:          layout,
		NodesActivation: nodesActivation,
	}
	return &substr
}

// CreateNetworkSolver creates network solver based on current substrate layout and provided Compositional Pattern Producing Network which
// used to define connections between network nodes. Optional graph_builder can be provided to collect graph nodes and edges
// of created network solver. With graph builder it is possible to save/load network configuration as well as visualize it.
// If the useLeo is True thar Link Expression Output extension to the HyperNEAT will be used instead of standard weight threshold
// technique of HyperNEAT to determine whether to express link between two nodes or not. With LEO the link expressed based
// on value of additional output of the CPPN (if > 0 then expressed)
func (s *Substrate) CreateNetworkSolver(cppn network.Solver, useLeo bool, graphBuilder SubstrateGraphBuilder, options *hyperneat.Options) (network.Solver, error) {
	// check conditions
	if s.Layout.BiasCount() > 1 {
		return nil, errors.New("SUBSTRATE: maximum one BIAS node per network supported")
	}

	// the network layers will be collected in order: bias, input, output, hidden
	firstBias := 0
	firstInput := s.Layout.BiasCount()
	firstOutput := firstInput + s.Layout.InputCount()
	firstHidden := firstOutput + s.Layout.OutputCount()
	lastHidden := firstHidden + s.Layout.HiddenCount()

	totalNeuronCount := lastHidden

	links := make([]*network.FastNetworkLink, 0)
	biasList := make([]float64, totalNeuronCount)

	// inline function to find activation type for a given neuron
	activationForNeuron := func(nodeIndex int) neatmath.NodeActivationType {
		if nodeIndex < firstOutput {
			// all bias and input neurons has null activation function associated because they actually have
			// no inputs to be activated upon
			return neatmath.NullActivation
		} else {
			return s.NodesActivation
		}
	}

	// give bias inputs to all hidden and output nodes.
	coordinates := make([]float64, 6)
	for bi := firstBias; bi < firstInput; bi++ {

		// the bias coordinates
		if biasPosition, err := s.Layout.NodePosition(bi-firstBias, network.BiasNeuron); err != nil {
			return nil, err
		} else {
			coordinates[0] = biasPosition.X
			coordinates[1] = biasPosition.Y
			coordinates[2] = biasPosition.Z

			// add bias node to builder
			if _, err = addNodeToBuilder(graphBuilder, bi, network.BiasNeuron, activationForNeuron(bi), biasPosition); err != nil {
				return nil, err
			}
		}

		// link the bias to all hidden nodes.
		for hi := firstHidden; hi < lastHidden; hi++ {
			// get hidden neuron coordinates
			if hiddenPosition, err := s.Layout.NodePosition(hi-firstHidden, network.HiddenNeuron); err != nil {
				return nil, err
			} else {
				coordinates[2] = hiddenPosition.X
				coordinates[3] = hiddenPosition.Y
				coordinates[4] = hiddenPosition.Z

				// add node to graph
				if _, err = addNodeToBuilder(graphBuilder, hi, network.HiddenNeuron, activationForNeuron(hi), hiddenPosition); err != nil {
					return nil, err
				}
			}
			// find connection weight
			if link, err := fastNetworkLink(coordinates, cppn, useLeo, bi, hi, options); err != nil {
				return nil, err
			} else if link != nil {
				biasList[hi] = link.Weight
				// add node and edge to graph
				if _, err := addEdgeToBuilder(graphBuilder, bi, hi, link.Weight); err != nil {
					return nil, err
				}
			}
		}

		// link the bias to all output nodes
		for oi := firstOutput; oi < firstHidden; oi++ {
			// get output neuron coordinates
			if outputPosition, err := s.Layout.NodePosition(oi-firstOutput, network.OutputNeuron); err != nil {
				return nil, err
			} else {
				coordinates[2] = outputPosition.X
				coordinates[3] = outputPosition.Y
				coordinates[4] = outputPosition.Z

				// add node to graph
				if _, err = addNodeToBuilder(graphBuilder, oi, network.OutputNeuron, activationForNeuron(oi), outputPosition); err != nil {
					return nil, err
				}
			}
			// find connection weight
			if link, err := fastNetworkLink(coordinates, cppn, useLeo, bi, oi, options); err != nil {
				return nil, err
			} else if link != nil {
				biasList[oi] = link.Weight
				// add node and edge to graph
				if _, err := addEdgeToBuilder(graphBuilder, bi, oi, link.Weight); err != nil {
					return nil, err
				}
			}
		}
	}

	if s.Layout.HiddenCount() > 0 {
		// link input nodes to hidden ones
		for in := firstInput; in < firstOutput; in++ {
			// get coordinates of input node
			if inputPosition, err := s.Layout.NodePosition(in-firstInput, network.InputNeuron); err != nil {
				return nil, err
			} else {
				coordinates[0] = inputPosition.X
				coordinates[1] = inputPosition.Y
				coordinates[2] = inputPosition.Z

				// add node to graph
				if _, err = addNodeToBuilder(graphBuilder, in, network.InputNeuron, activationForNeuron(in), inputPosition); err != nil {
					return nil, err
				}
			}
			for hi := firstHidden; hi < lastHidden; hi++ {
				// get hidden neuron coordinates
				if hiddenPosition, err := s.Layout.NodePosition(hi-firstHidden, network.HiddenNeuron); err != nil {
					return nil, err
				} else {
					coordinates[2] = hiddenPosition.X
					coordinates[3] = hiddenPosition.Y
					coordinates[4] = hiddenPosition.Z
				}
				// find connection weight
				if link, err := fastNetworkLink(coordinates, cppn, useLeo, in, hi, options); err != nil {
					return nil, err
				} else if link != nil {
					links = append(links, link)
					// add node and edge to graph
					if _, err := addEdgeToBuilder(graphBuilder, in, hi, link.Weight); err != nil {
						return nil, err
					}
				}
			}
		}

		// link all hidden nodes to all output nodes.
		for hi := firstHidden; hi < lastHidden; hi++ {
			if hiddenPosition, err := s.Layout.NodePosition(hi-firstHidden, network.HiddenNeuron); err != nil {
				return nil, err
			} else {
				coordinates[0] = hiddenPosition.X
				coordinates[1] = hiddenPosition.Y
				coordinates[2] = hiddenPosition.Z
			}
			for oi := firstOutput; oi < firstHidden; oi++ {
				// get output neuron coordinates
				if outputPosition, err := s.Layout.NodePosition(oi-firstOutput, network.OutputNeuron); err != nil {
					return nil, err
				} else {
					coordinates[2] = outputPosition.X
					coordinates[3] = outputPosition.Y
					coordinates[4] = outputPosition.Z
				}
				// find connection weight
				if link, err := fastNetworkLink(coordinates, cppn, useLeo, hi, oi, options); err != nil {
					return nil, err
				} else if link != nil {
					links = append(links, link)
					// add node and edge to graph
					if _, err := addEdgeToBuilder(graphBuilder, hi, oi, link.Weight); err != nil {
						return nil, err
					}
				}
			}
		}
	} else {
		// connect all input nodes directly to all output nodes
		for in := firstInput; in < firstOutput; in++ {
			// get coordinates of input node
			if inputPosition, err := s.Layout.NodePosition(in-firstInput, network.InputNeuron); err != nil {
				return nil, err
			} else {
				coordinates[0] = inputPosition.X
				coordinates[1] = inputPosition.Y
				coordinates[2] = inputPosition.Z

				// add node to graph
				if _, err = addNodeToBuilder(graphBuilder, in, network.InputNeuron, activationForNeuron(in), inputPosition); err != nil {
					return nil, err
				}
			}
			for oi := firstOutput; oi < firstHidden; oi++ {
				// get output neuron coordinates
				if outputPosition, err := s.Layout.NodePosition(oi-firstOutput, network.OutputNeuron); err != nil {
					return nil, err
				} else {
					coordinates[2] = outputPosition.X
					coordinates[3] = outputPosition.Y
					coordinates[4] = outputPosition.Z
				}
				// find connection weight
				if link, err := fastNetworkLink(coordinates, cppn, useLeo, in, oi, options); err != nil {
					return nil, err
				} else if link != nil {
					links = append(links, link)
					// add node and edge to graph
					if _, err := addEdgeToBuilder(graphBuilder, in, oi, link.Weight); err != nil {
						return nil, err
					}
				}
			}
		}
	}

	// build activations
	activations := make([]neatmath.NodeActivationType, totalNeuronCount)
	for i := 0; i < totalNeuronCount; i++ {
		activations[i] = activationForNeuron(i)
	}

	if totalNeuronCount == 0 || len(links) == 0 || len(activations) != totalNeuronCount {
		message := fmt.Sprintf("failed to create network solver: links [%d], nodes [%d], activations [%d]",
			len(links), totalNeuronCount, len(activations))
		return nil, errors.New(message)
	}

	// create fast network solver
	solver := network.NewFastModularNetworkSolver(
		s.Layout.BiasCount(), s.Layout.InputCount(), s.Layout.OutputCount(), totalNeuronCount,
		activations, links, biasList, nil)
	return solver, nil
}

func fastNetworkLink(coordinates []float64, cppn network.Solver, useLeo bool, source, target int, options *hyperneat.Options) (*network.FastNetworkLink, error) {
	if outs, err := queryCPPN(coordinates, cppn); err != nil {
		return nil, err
	} else if useLeo && outs[1] > 0 {
		// add links only when CPPN LEO output signals to
		return createLink(outs[0], source, target, options.WeightRange), nil
	} else if !useLeo && math.Abs(outs[0]) >= options.LinkThreshold {
		// add only links with signal exceeding provided threshold
		return createThresholdNormalizedLink(outs[0], source, target, options.LinkThreshold, options.WeightRange), nil
	}
	return nil, nil
}
