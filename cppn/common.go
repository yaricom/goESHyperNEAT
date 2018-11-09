// The package CPPN provides implementation of Compositional Pattern Producing Network which is a part of Hypercube-based
// NEAT algorithm implementation
package cppn

// Defines the quad-point in the 4 dimensional hypercube
type QuadPoint struct {
	// The associated coordinates
	X1, X2, Y1, Y2 float64
	// The value for this point
	Value          float64
}

// Defines quad-tree node to model 4 dimensional hypercube
type QuadNode struct {
	// The coordinates of center of this quad-tree node's square
	X, Y   float64
	// The width of this quad-tree node's square
	Width  float64

	// The weight of link encoded by this node
	Weight float64
	// The level of this node in the quad-tree
	Level  int

	// The children of this node
	Nodes  []*QuadNode
}

// Creates new quad-node with given parameters
func NewQuadNode(x, y, width float64, level int) *QuadNode {
	node := QuadNode{
		X:x,
		Y:y,
		Width:width,
		Weight:0.0,
		Level:level,
		Nodes:make([]*QuadNode, 0),
	}
	return &node
}
