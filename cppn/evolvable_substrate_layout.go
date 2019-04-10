package cppn

import (
	"github.com/yaricom/goNEAT/neat/network"
	"errors"
	"fmt"
)

// Defines layout of neurons in the substrate
type EvolvableSubstrateLayout interface {
	// Returns coordinates of the neuron with specified index [0; count) and type
	NodePosition(index int, n_type network.NodeNeuronType) (*PointF, error)

	// Adds new hidden node to the substrate
	// Returns the index of added hidden neuron or error if failed.
	AddHiddenNode(position *PointF) (int, error)
	// Returns index of hidden node at specified position or -1 if not fund
	IndexOfHidden(position *PointF) int

	// Returns number of INPUT neurons in the layout
	InputCount() int
	// Returns number of HIDDEN neurons in the layout
	HiddenCount() int
	// Returns number of OUTPUT neurons in the layout
	OutputCount() int
}

// Creates new instance with given input and output neurons count
func NewMappedEvolvableSubstrateLayout(inputCount, outputCount int) (*MappedEvolvableSubstrateLayout, error) {
	if inputCount == 0 {
		return nil, errors.New("The number of input neurons can not be ZERO")
	}
	if outputCount == 0 {
		return nil, errors.New("The number of output neurons can not be ZERO")
	}

	l := MappedEvolvableSubstrateLayout{
		hNodesMap:make(map[PointF]int),
		hNodesList:make([]*PointF, 0),
		inputCount:inputCount,
		outputCount:outputCount,
	}

	l.inputDelta = 2.0 / float64(inputCount)
	l.outputDelta = 2.0 / float64(outputCount)

	return &l, nil
}

// The EvolvableSubstrateLayout implementation using map for binding between hidden node and its index
type MappedEvolvableSubstrateLayout struct {
	// The map to hold binding between hidden node and its index for fast search
	hNodesMap   map[PointF]int
	// The list of all known hidden nodes in specific order
	hNodesList  []*PointF

	// The number of input nodes encoded in this substrate
	inputCount  int
	// The number of output nodes encoded in this substrate
	outputCount int

	// The input coordinates increment
	inputDelta  float64
	// The output coordinates increment
	outputDelta float64
}

func (l *MappedEvolvableSubstrateLayout) NodePosition(index int, n_type network.NodeNeuronType) (*PointF, error) {
	if index < 0 {
		return nil, errors.New("The neuron index can not be negative")
	}
	point := PointF{X:0.0, Y:0.0}
	delta := 0.0
	count := 0
	switch n_type {
	case network.BiasNeuron:
		return nil, errors.New("The BIAS neurons is not supported by Evolvable Substrate")

	case network.HiddenNeuron:
		count = len(l.hNodesList)

	case network.InputNeuron:
		delta = l.inputDelta
		count = l.inputCount
		point.Y = -1.0

	case network.OutputNeuron:
		delta = l.outputDelta
		count = l.outputCount
		point.Y = 1.0
	}

	if index >= count {
		return nil, errors.New("The neuron's index is out of range")
	} else if n_type == network.HiddenNeuron {
		// return stored hidden neuron position
		return l.hNodesList[index], nil
	}

	// calculate X position
	point.X = -1.0 + delta / 2.0 // the initial position with half delta shift
	point.X += float64(index) * delta

	return &point, nil
}

func (l *MappedEvolvableSubstrateLayout) AddHiddenNode(position *PointF) (int, error) {
	// check if given hidden node already exists
	if l.IndexOfHidden(position) != -1 {
		return -1, errors.New(fmt.Sprintf("Hidden node already exists at the position: %s", position))
	}
	// add to the list and map it
	l.hNodesList = append(l.hNodesList, position)
	index := len(l.hNodesList) - 1
	l.hNodesMap[*position] = index
	return index, nil
}

func (l *MappedEvolvableSubstrateLayout) IndexOfHidden(position *PointF) int {
	if index, ok := l.hNodesMap[*position]; ok {
		return -1
	} else {
		return index
	}
}

func (l *MappedEvolvableSubstrateLayout) InputCount() int {
	return l.inputCount
}

func (l *MappedEvolvableSubstrateLayout) HiddenCount() int {
	return len(l.hNodesList)
}

func (l *MappedEvolvableSubstrateLayout) OutputCount() int {
	return l.outputCount
}

func (l *MappedEvolvableSubstrateLayout) String() string {
	str := fmt.Sprintf("MappedEvolvableSubstrateLayout:\n\tINPT: %d\n\tHIDN: %d\n\tOUTP: %d",
		l.InputCount(), l.HiddenCount(), l.OutputCount())
	return str
}