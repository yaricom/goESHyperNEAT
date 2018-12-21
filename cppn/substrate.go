package cppn

import (
	"github.com/yaricom/goNEAT/neat/network"
	"github.com/yaricom/goESHyperNEAT/hyperneat"
	"math"
	"errors"
)

// Represents substrate holding configuration of ANN with weights produced by CPPN. According to HyperNEAT method
// the ANN neurons are encoded as coordinates in hypercube presented by this substrate.
// By default neurons will be placed into substrate within grid layout
type Substrate struct {
	// The layout of neuron nodes in this substrate
	Layout          SubstrateLayout

	// The activation function's type for neurons encoded
	NodesActivation network.NodeActivationType
}

// Creates new instance
func NewSubstrate(layout *SubstrateLayout, nodesActivation network.NodeActivationType) *Substrate {
	substr := Substrate{
		Layout:layout,
		NodesActivation:nodesActivation,
	}
	return &substr
}

// Creates network solver based on current substrate layout and provided Compositional Pattern Producing Network which
// used to define connections between network nodes.
func (s *Substrate) CreateNetworkSolver(cppn network.NetworkSolver, context *hyperneat.HyperNEATContext) (network.NetworkSolver, error) {
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

	// give bias inputs to all hidden and output nodes.
	coord := make([]float64, 4)
	for bi := firstBias; bi < firstInput; bi++ {

		// link the bias to all hidden nodes.
		for hi := firstHidden; hi < lastHidden; hi++ {
			// get hidden neuron coordinates
			if h_coord, err := s.Layout.NodePosition(0, network.HiddenNeuron); err != nil {
				return nil, err
			} else {
				coord[2] = h_coord.X
				coord[3] = h_coord.Y
			}
			// find connection weight
			if outs, err := queryCPPN(coord, cppn); err != nil {
				return nil, err
			} else if math.Abs(outs[0]) > context.LinkThershold {
				// add only connections with signal exceeding provided threshold
				link := createLink(outs[0], bi, hi, context)
				biasList[hi] = link.Weight
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
			}
			// find connection weight
			if outs, err := queryCPPN(coord, cppn); err != nil {
				return nil, err
			} else if math.Abs(outs[0]) > context.LinkThershold {
				// add only connections with signal exceeding provided threshold
				link := createLink(outs[0], bi, oi, context)
				biasList[oi] = link.Weight
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
			}
			for hi := firstHidden; hi < lastHidden; hi++ {
				// get hidden neuron coordinates
				if h_coord, err := s.Layout.NodePosition(0, network.HiddenNeuron); err != nil {
					return nil, err
				} else {
					coord[2] = h_coord.X
					coord[3] = h_coord.Y
				}
				// find connection weight
				if outs, err := queryCPPN(coord, cppn); err != nil {
					return nil, err
				} else if math.Abs(outs[0]) > context.LinkThershold {
					// add only connections with signal exceeding provided threshold
					link := createLink(outs[0], in, hi, context)
					connections = append(connections, link)
				}
			}
		}

		// link all hidden nodes to all output nodes.
		for hi := firstHidden; hi < lastHidden; hi++ {
			if h_coord, err := s.Layout.NodePosition(0, network.HiddenNeuron); err != nil {
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
				if outs, err := queryCPPN(coord, cppn); err != nil {
					return nil, err
				} else if math.Abs(outs[0]) > context.LinkThershold {
					// add only connections with signal exceeding provided threshold
					link := createLink(outs[0], hi, oi, context)
					connections = append(connections, link)
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
				if outs, err := queryCPPN(coord, cppn); err != nil {
					return nil, err
				} else if math.Abs(outs[0]) > context.LinkThershold {
					// add only connections with signal exceeding provided threshold
					link := createLink(outs[0], in, oi, context)
					connections = append(connections, link)
				}
			}
		}
	}

	// build activations
	activations := make([]network.NodeActivationType, totalNeuronCount)
	for i := 0; i < totalNeuronCount; i++ {
		if i < firstOutput {
			activations[i] = network.NullActivation
		} else {
			activations[i] = s.NodesActivation
		}
	}

	// create fast network solver
	solver := network.NewFastModularNetworkSolver(
		s.Layout.BiasCount(), s.Layout.InputCount(), s.Layout.OutputCount(), totalNeuronCount,
		activations, connections, biasList, nil)
	return solver
}


