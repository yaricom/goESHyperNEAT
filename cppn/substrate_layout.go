package cppn

import "github.com/yaricom/goNEAT/neat/network"


// Defines layout of neurons in the substrate
type SubtrateLayout interface {
	// Returns coordinates of next neuron with specified type
	NextNode(index int, n_type network.NodeType) PointF

	// Returns number of BIAS neurons in the layout
	BiasCount() int
	// Returns number of INPUT neurons in the layout
	InputCount() int
	// Returns number of HIDDEN neurons in the layout
	HiddenCount() int
	// Returns number of OUTPUT neurons in the layout
	OutputCount() int
}

// Defines grid substrate layout
type GridSubstrateLayout struct {
	// The number of bias nodes encoded in this substrate
	biasCount      int
	// The number of input nodes encoded in this substrate
	inputCount     int
	// The number of hidden nodes encoded in this substrate
	hiddenCount    int
	// The number of output nodes encoded in this substrate
	outputCount    int

	// The input coordinates increment
	inputDelta  float64
	// The hidden coordinates increment
	hiddenDelta float64
	// The output coordinates increment
	outputDelta float64
}

// Creates new instance with specified number of nodes to create layout for
func NewGridSubstrateLayout(biasCount, inputCount, outputCount, hiddenCount int) *GridSubstrateLayout {
	s := GridSubstrateLayout{biasCount:biasCount, inputCount:inputCount, outputCount:outputCount,hiddenCount:hiddenCount}

	if s.inputCount != 0 {
		s.inputDelta = 2.0 / float64(inputCount)
	}

	if s.hiddenCount != 0 {
		s.hiddenDelta = 2.0 / float64(hiddenCount)
	}

	if s.outputCount != 0 {
		s.outputDelta = 2.0 / float64(outputCount)
	}
	return &s
}

func (g *GridSubstrateLayout) BiasCount() int {
	return g.biasCount
}

func (g *GridSubstrateLayout) InputCount() int {
	return g.inputCount
}

func (g *GridSubstrateLayout) HiddenCount() int {
	return g.hiddenCount
}

func (g *GridSubstrateLayout) OutputCount() int {
	return g.outputCount
}
