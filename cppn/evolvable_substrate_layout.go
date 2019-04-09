package cppn

import (
	"github.com/yaricom/goNEAT/neat/network"
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
