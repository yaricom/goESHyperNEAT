package cppn

import (
	"fmt"
	"github.com/pkg/errors"
	"github.com/yaricom/goNEAT/v3/neat/network"
)

// EvolvableSubstrateLayout Defines layout of neurons in the substrate
type EvolvableSubstrateLayout interface {
	// NodePosition Returns coordinates of the neuron with specified index [0; count) and type
	NodePosition(index int, nType network.NodeNeuronType) (*PointF, error)

	// AddHiddenNode Adds new hidden node to the substrate
	// Returns the index of added hidden neuron or error if failed.
	AddHiddenNode(position *PointF) (int, error)
	// IndexOfHidden Returns index of hidden node at specified position or -1 if not fund
	IndexOfHidden(position *PointF) int

	// InputCount Returns number of INPUT neurons in the layout
	InputCount() int
	// HiddenCount Returns number of HIDDEN neurons in the layout
	HiddenCount() int
	// OutputCount Returns number of OUTPUT neurons in the layout
	OutputCount() int
}

// NewMappedEvolvableSubstrateLayout Creates new instance with given input and output neurons count
func NewMappedEvolvableSubstrateLayout(inputCount, outputCount int) (*MappedEvolvableSubstrateLayout, error) {
	if inputCount == 0 {
		return nil, errors.New("the number of input neurons can not be ZERO")
	}
	if outputCount == 0 {
		return nil, errors.New("the number of output neurons can not be ZERO")
	}

	l := &MappedEvolvableSubstrateLayout{
		hNodesMap:   make(map[PointF]int),
		hNodesList:  make([]*PointF, 0),
		inputCount:  inputCount,
		outputCount: outputCount,
		inputDelta:  2.0 / float64(inputCount),
		outputDelta: 2.0 / float64(outputCount),
	}
	return l, nil
}

// MappedEvolvableSubstrateLayout the EvolvableSubstrateLayout implementation using map for binding between hidden
// node and its index
type MappedEvolvableSubstrateLayout struct {
	// The map to hold binding between hidden node and its index for fast search
	hNodesMap map[PointF]int
	// The list of all known hidden nodes in specific order
	hNodesList []*PointF

	// The number of input nodes encoded in this substrate
	inputCount int
	// The number of output nodes encoded in this substrate
	outputCount int

	// The input coordinates increment
	inputDelta float64
	// The output coordinates increment
	outputDelta float64
}

func (m *MappedEvolvableSubstrateLayout) NodePosition(index int, nType network.NodeNeuronType) (*PointF, error) {
	if index < 0 {
		return nil, errors.New("neuron index can not be negative")
	}
	point := PointF{X: 0.0, Y: 0.0}
	delta := 0.0
	count := 0
	switch nType {
	case network.BiasNeuron:
		return nil, errors.New("the BIAS neurons is not supported by Evolvable Substrate")

	case network.HiddenNeuron:
		count = len(m.hNodesList)

	case network.InputNeuron:
		delta = m.inputDelta
		count = m.inputCount
		point.Y = -1.0

	case network.OutputNeuron:
		delta = m.outputDelta
		count = m.outputCount
		point.Y = 1.0
	}

	if index >= count {
		return nil, errors.New("neuron index is out of range")
	} else if nType == network.HiddenNeuron {
		// return stored hidden neuron position
		return m.hNodesList[index], nil
	}

	// calculate X position
	point.X = -1.0 + delta/2.0 // the initial position with half delta shift
	point.X += float64(index) * delta

	return &point, nil
}

func (m *MappedEvolvableSubstrateLayout) AddHiddenNode(position *PointF) (int, error) {
	// check if given hidden node already exists
	if m.IndexOfHidden(position) != -1 {
		return -1, errors.Errorf("hidden node already exists at the position: %s", position)
	}
	// add to the list and map it
	m.hNodesList = append(m.hNodesList, position)
	index := len(m.hNodesList) - 1
	m.hNodesMap[*position] = index
	return index, nil
}

func (m *MappedEvolvableSubstrateLayout) IndexOfHidden(position *PointF) int {
	if index, ok := m.hNodesMap[*position]; ok {
		return index
	} else {
		return -1
	}
}

func (m *MappedEvolvableSubstrateLayout) BiasCount() int {
	// No BIAS nodes
	return 0
}

func (m *MappedEvolvableSubstrateLayout) InputCount() int {
	return m.inputCount
}

func (m *MappedEvolvableSubstrateLayout) HiddenCount() int {
	return len(m.hNodesList)
}

func (m *MappedEvolvableSubstrateLayout) OutputCount() int {
	return m.outputCount
}

func (m *MappedEvolvableSubstrateLayout) String() string {
	str := fmt.Sprintf("MappedEvolvableSubstrateLayout:\n\tINPT: %d\n\tHIDN: %d\n\tOUTP: %d",
		m.InputCount(), m.HiddenCount(), m.OutputCount())
	return str
}
