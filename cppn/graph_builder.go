package cppn

import "github.com/yaricom/goNEAT/neat/network"

// The graph builder able to build weighted directed graphs representing substrate networks
type GraphBuilder interface {

	// Adds specified node to the graph with provided position
	AddNode(nodeId int, nodeType network.NodeActivationType, position PointF) error
	// Adds edge between two graph nodes
	AddWeightedEdge(sourceId, targetId int, weight float64) error

}
