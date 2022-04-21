package retina

import (
	"fmt"

	"github.com/yaricom/goESHyperNEAT/v2/cppn"
	"github.com/yaricom/goESHyperNEAT/v2/eshyperneat"
	"github.com/yaricom/goNEAT/neat/genetics"
	"github.com/yaricom/goNEAT/neat/network"
	"github.com/yaricom/goNEAT/neat/utils"
)

const debug = true

// CPPNNetworkBuilder handles the building of Network's using CPPN network's to query connections/topology by the esHyperNEAT algorithm
type CPPNNetworkBuilder struct {
	UseLEO          bool
	SubstrateLayout cppn.EvolvableSubstrateLayout
	NodesActivation utils.NodeActivationType
	Context         *eshyperneat.ESHyperNEATContext

	// New Graph Builder generator function.
	// Note: Graph Builder is used to store graphs in a user-friendly format
	NewGraphBuilder func() *cppn.SubstrateGraphBuilder
}

//CreateANNFromCPPNOrganism creates a NetworkSolver (ANN) by querying the Organism.Phenotype cppnNetwork by the ESHyperNeat Algorithm
func (builder *CPPNNetworkBuilder) CreateANNFromCPPNOrganism(cppnOrganism *genetics.Organism) (network.NetworkSolver, error) {
	return builder.CreateANNFromCPPN(cppnOrganism.Phenotype)
}

//CreateANNFromCPPN creates a NetworkSolver (ANN) by querying the cppnNetwork by the ESHyperNeat Algorithm
func (builder *CPPNNetworkBuilder) CreateANNFromCPPN(cppnNetwork *network.Network) (network.NetworkSolver, error) {
	// create substrate which will be connected to form the network
	substrate := cppn.NewEvolvableSubstrate(builder.SubstrateLayout, builder.NodesActivation)
	cppnFastNetwork, err := cppnNetwork.FastNetworkSolver()
	if err != nil {
		return nil, err
	}

	// create a new graphBuilder to store the graph in a graphable and user-friendly format - graphml xml
	graphBuilder := builder.NewGraphBuilder()

	// unwrap graphBuilder. graphBuilder may not be provided (its nil)
	var annSolver network.NetworkSolver
	if graphBuilder == nil {
		// create our ann by querying the cppn network and applying the eshypernet quadtree/pruning/banding algorithm
		annSolver, err = substrate.CreateNetworkSolver(
			cppnFastNetwork, nil, builder.Context,
		)
	} else {
		// create our ann by querying the cppn network and applying the eshypernet quadtree/pruning/banding algorithm
		annSolver, err = substrate.CreateNetworkSolver(
			cppnFastNetwork, *graphBuilder, builder.Context,
		)
	}
	if err != nil {
		return nil, err
	}

	if debug {
		fmt.Print("CPPN: ", cppnFastNetwork, "\n")
		fmt.Print("Generated ANN: ", annSolver, "\n\n")
	}

	return annSolver, nil
}
