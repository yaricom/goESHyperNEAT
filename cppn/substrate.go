package cppn

import "github.com/yaricom/goNEAT/neat/network"

// Represents substrate holding configuration of ANN with weights produced by CPPN. According to HyperNEAT method
// the ANN neurons are encoded as coordinates in hypercube presented by this substrate.
// By default neurons will be placed into substrate within grid layout
type Substrate struct {
	// The number of bias nodes encoded in this substrate
	BiasCount      int
	// The number of input nodes encoded in this substrate
	InputCount     int
	// The number of hidden nodes encoded in this substrate
	HiddenCount    int
	// The number of output nodes encoded in this substrate
	OutputCount    int

	// The activation function's type for neurons encoded
	NodeActivation network.NodeActivationType
}
