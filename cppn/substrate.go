package cppn

import (
	"errors"
	"math"

	"github.com/yaricom/goNEAT/neat/network"
	"github.com/yaricom/goNEAT/neat/utils"
	"github.com/yaricom/goESHyperNEAT/hyperneat"
)

// Represents substrate holding configuration of ANN with weights produced by CPPN. According to HyperNEAT method
// the ANN neurons are encoded as coordinates in hypercube presented by this substrate.
// By default neurons will be placed into substrate within grid layout
type Substrate struct {
	// The layout of neuron nodes in this substrate
	Layout          SubstrateLayout

	// The activation function's type for neurons encoded
	NodesActivation utils.NodeActivationType
}

// Creates new instance
func NewSubstrate(layout SubstrateLayout, nodesActivation utils.NodeActivationType) *Substrate {
	substr := Substrate{
		Layout:layout,
		NodesActivation:nodesActivation,
	}
	return &substr
}

// Creates network solver based on current substrate layout and provided Compositional Pattern Producing Network which
// used to define connections between network nodes. Optional graph_builder can be provided to collect graph nodes and edges
// of created network solver. With graph builder it is possible to save/load network configuration as well as visualize it.
// If use_leo is True thar Link Expression Output extension to the HyperNEAT will be used instead of standard weight threshold
// technique of HyperNEAT to determine whether to express link between two nodes or not. With LEO the link expressed based
// on value of additional output of the CPPN (if > 0 then expressed)
func (s *Substrate) CreateNetworkSolver(cppn network.NetworkSolver, use_leo bool, graph_builder GraphBuilder, context *hyperneat.HyperNEATContext) (network.NetworkSolver, error) {
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

	connections := make([]*network.FastNetworkLink, 0)
	biasList := make([]float64, totalNeuronCount)

	// inline function to find activation type for a given neuron
	activationForNeuron := func(n_index int) utils.NodeActivationType {
		if n_index < firstOutput {
			// all bias and input neurons has null activation function associated because they actually has
			// no inputs to be activated upon
			return utils.NullActivation
		} else {
			return s.NodesActivation
		}
	}

	// give bias inputs to all hidden and output nodes.
	var link *network.FastNetworkLink
	coord := make([]float64, 4)
	for bi := firstBias; bi < firstInput; bi++ {

		// the bias coordinates
		if b_coord, err := s.Layout.NodePosition(bi - firstBias, network.BiasNeuron); err != nil {
			return nil, err
		} else {
			coord[0] = b_coord.X
			coord[1] = b_coord.Y

			// add bias node to builder
			if _, err := addNodeToBuilder(graph_builder, bi, network.BiasNeuron, activationForNeuron(bi), b_coord); err != nil {
				return nil, err
			}
		}


		// link the bias to all hidden nodes.
		for hi := firstHidden; hi < lastHidden; hi++ {
			// get hidden neuron coordinates
			if h_coord, err := s.Layout.NodePosition(hi - firstHidden, network.HiddenNeuron); err != nil {
				return nil, err
			} else {
				coord[2] = h_coord.X
				coord[3] = h_coord.Y

				// add node to graph
				if _, err := addNodeToBuilder(graph_builder, hi, network.HiddenNeuron, activationForNeuron(hi), h_coord); err != nil {
					return nil, err
				}
			}
			// find connection weight
			link = nil
			if outs, err := queryCPPN(coord, cppn); err != nil {
				return nil, err
			} else if use_leo && outs[1] > 0 {
				// add links only when CPPN's LEO output signals to
				link = createLink(outs[0], bi, hi, context.WeightRange)
			} else if !use_leo && math.Abs(outs[0]) > context.LinkThershold {
				// add only connections with signal exceeding provided threshold
				link = createThreshlodNormalizedLink(outs[0], bi, hi, context.LinkThershold, context.WeightRange)
			}
			if link != nil {
				biasList[hi] = link.Weight
				// add node and edge to graph
				if _, err := addEdgeToBuilder(graph_builder, bi, hi, link.Weight); err != nil {
					return nil, err
				}
			}
		}

		// link the bias to all output nodes
		for oi := firstOutput; oi < firstHidden; oi++ {
			// get output neuron coordinates
			if o_coord, err := s.Layout.NodePosition(oi - firstOutput, network.OutputNeuron); err != nil {
				return nil, err
			} else {
				coord[2] = o_coord.X
				coord[3] = o_coord.Y

				// add node to graph
				if _, err := addNodeToBuilder(graph_builder, oi, network.OutputNeuron, activationForNeuron(oi), o_coord); err != nil {
					return nil, err
				}
			}
			// find connection weight
			link = nil
			if outs, err := queryCPPN(coord, cppn); err != nil {
				return nil, err
			} else if use_leo && outs[1] > 0 {
				// add links only when CPPN's LEO output signals to
				link = createLink(outs[0], bi, oi, context.WeightRange)
			} else if !use_leo && math.Abs(outs[0]) > context.LinkThershold {
				// add only connections with signal exceeding provided threshold
				link = createThreshlodNormalizedLink(outs[0], bi, oi, context.LinkThershold, context.WeightRange)
			}
			if link != nil {
				biasList[oi] = link.Weight
				// add node and edge to graph
				if _, err := addEdgeToBuilder(graph_builder, bi, oi, link.Weight); err != nil {
					return nil, err
				}
			}
		}
	}

	if s.Layout.HiddenCount() > 0 {
		// link input nodes to hidden ones
		for in := firstInput; in < firstOutput; in++ {
			// get coordinates of input node
			if i_coord, err := s.Layout.NodePosition(in - firstInput, network.InputNeuron); err != nil {
				return nil, err
			} else {
				coord[0] = i_coord.X
				coord[1] = i_coord.Y

				// add node to graph
				if _, err := addNodeToBuilder(graph_builder, in, network.InputNeuron, activationForNeuron(in), i_coord); err != nil {
					return nil, err
				}
			}
			for hi := firstHidden; hi < lastHidden; hi++ {
				// get hidden neuron coordinates
				if h_coord, err := s.Layout.NodePosition(hi - firstHidden, network.HiddenNeuron); err != nil {
					return nil, err
				} else {
					coord[2] = h_coord.X
					coord[3] = h_coord.Y
				}
				// find connection weight
				link = nil
				if outs, err := queryCPPN(coord, cppn); err != nil {
					return nil, err
				} else if use_leo && outs[1] > 0 {
					// add links only when CPPN's LEO output signals to
					link = createLink(outs[0], in, hi, context.WeightRange)
				} else if !use_leo && math.Abs(outs[0]) > context.LinkThershold {
					// add only connections with signal exceeding provided threshold
					link = createThreshlodNormalizedLink(outs[0], in, hi, context.LinkThershold, context.WeightRange)

				}
				if link != nil {
					connections = append(connections, link)
					// add node and edge to graph
					if _, err := addEdgeToBuilder(graph_builder, in, hi, link.Weight); err != nil {
						return nil, err
					}
				}
			}
		}

		// link all hidden nodes to all output nodes.
		for hi := firstHidden; hi < lastHidden; hi++ {
			if h_coord, err := s.Layout.NodePosition(hi - firstHidden, network.HiddenNeuron); err != nil {
				return nil, err
			} else {
				coord[0] = h_coord.X
				coord[1] = h_coord.Y
			}
			for oi := firstOutput; oi < firstHidden; oi++ {
				// get output neuron coordinates
				if o_coord, err := s.Layout.NodePosition(oi - firstOutput, network.OutputNeuron); err != nil {
					return nil, err
				} else {
					coord[2] = o_coord.X
					coord[3] = o_coord.Y
				}
				// find connection weight
				link = nil
				if outs, err := queryCPPN(coord, cppn); err != nil {
					return nil, err
				} else if use_leo && outs[1] > 0 {
					// add links only when CPPN's LEO output signals to
					link = createLink(outs[0], hi, oi, context.WeightRange)
				} else if !use_leo && math.Abs(outs[0]) > context.LinkThershold {
					// add only connections with signal exceeding provided threshold
					link = createThreshlodNormalizedLink(outs[0], hi, oi, context.LinkThershold, context.WeightRange)
				}
				if link != nil {
					connections = append(connections, link)
					// add node and edge to graph
					if _, err := addEdgeToBuilder(graph_builder, hi, oi, link.Weight); err != nil {
						return nil, err
					}
				}
			}
		}
	} else {
		// connect all input nodes directly to all output nodes
		for in := firstInput; in < firstOutput; in++ {
			// get coordinates of input node
			if i_coord, err := s.Layout.NodePosition(in - firstInput, network.InputNeuron); err != nil {
				return nil, err
			} else {
				coord[0] = i_coord.X
				coord[1] = i_coord.Y

				// add node to graph
				if _, err := addNodeToBuilder(graph_builder, in, network.InputNeuron, activationForNeuron(in), i_coord); err != nil {
					return nil, err
				}
			}
			for oi := firstOutput; oi < firstHidden; oi++ {
				// get output neuron coordinates
				if o_coord, err := s.Layout.NodePosition(oi - firstOutput, network.OutputNeuron); err != nil {
					return nil, err
				} else {
					coord[2] = o_coord.X
					coord[3] = o_coord.Y
				}
				// find connection weight
				link = nil
				if outs, err := queryCPPN(coord, cppn); err != nil {
					return nil, err
				} else if use_leo && outs[1] > 0 {
					// add links only when CPPN's LEO output signals to
					link = createLink(outs[0], in, oi, context.WeightRange)
				} else if !use_leo && math.Abs(outs[0]) > context.LinkThershold {
					// add only connections with signal exceeding provided threshold
					link = createThreshlodNormalizedLink(outs[0], in, oi, context.LinkThershold, context.WeightRange)

				}
				if link != nil {
					connections = append(connections, link)
					// add node and edge to graph
					if _, err := addEdgeToBuilder(graph_builder, in, oi, link.Weight); err != nil {
						return nil, err
					}
				}
			}
		}
	}

	// build activations
	activations := make([]utils.NodeActivationType, totalNeuronCount)
	for i := 0; i < totalNeuronCount; i++ {
		activations[i] = activationForNeuron(i)
	}

	// create fast network solver
	solver := network.NewFastModularNetworkSolver(
		s.Layout.BiasCount(), s.Layout.InputCount(), s.Layout.OutputCount(), totalNeuronCount,
		activations, connections, biasList, nil)
	return solver, nil
}
