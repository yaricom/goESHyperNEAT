package cppn

import (
	"github.com/yaricom/goNEAT/neat/network"
	"errors"
)


// Defines layout of neurons in the substrate
type SubtrateLayout interface {
	// Returns coordinates of the neuron with specified index [0; count) and type
	NodePosition(index int, n_type network.NodeNeuronType) (*PointF, error)

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

func (g *GridSubstrateLayout) NodePosition(index int, n_type network.NodeNeuronType) (*PointF, error) {
	if index < 0 {
		return nil, errors.New("The neuron index can not be negative")
	}
	point := PointF{X:0.0, Y:0.0}
	delta := 0.0
	count := 0
	switch n_type {
	case network.BiasNeuron:
		count = g.biasCount

	case network.HiddenNeuron:
		delta = g.hiddenDelta
		count = g.hiddenCount

	case network.InputNeuron:
		delta = g.inputDelta
		count = g.inputCount
		point.Y = -1.0

	case network.OutputNeuron:
		delta = g.outputDelta
		count = g.outputCount
		point.Y = 1.0
	}

	if index >= count {
		return nil, errors.New("The neuron's index is out of range")
	} else if n_type == network.BiasNeuron {
		return &point, nil
	}
	// calculate X position
	point.X = -1.0 + delta / 2.0 // the initial position with half delta shift
	point.X += float64(index) * delta

	return &point, nil
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
