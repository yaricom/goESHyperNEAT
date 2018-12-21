// The package CPPN provides implementation of Compositional Pattern Producing Network which is a part of Hypercube-based
// NEAT algorithm implementation
package cppn

import (
	"fmt"
	"github.com/yaricom/goESHyperNEAT/hyperneat"
	"github.com/yaricom/goNEAT/neat/network"
	"math"
	"errors"
)

// Defines point with float precision coordinates
type PointF struct {
	X, Y float64
}

func (p *PointF) String() string {
	return fmt.Sprintf("(%f, %f)", p.X, p.Y)
}

// Defines the quad-point in the 4 dimensional hypercube
type QuadPoint struct {
	// The associated coordinates
	X1, X2, Y1, Y2 float64
	// The value for this point
	Value          float64
}

func (q *QuadPoint) String() string {
	return fmt.Sprintf("((%f, %f),(%f, %f)) = %f", q.X1, q.Y1, q.X2, q.Y2, q.Value)
}

// Defines quad-tree node to model 4 dimensional hypercube
type QuadNode struct {
	// The coordinates of center of this quad-tree node's square
	X, Y  float64
	// The width of this quad-tree node's square
	Width float64

	// The CPPN activation level for this node
	W     float64
	// The level of this node in the quad-tree
	Level int

	// The children of this node
	Nodes []*QuadNode
}

func (q *QuadNode) String() string {
	return fmt.Sprintf("((%f, %f), %f) = %f at %d", q.X, q.Y, q.Width, q.W, q.Level)
}

// Creates new quad-node with given parameters
func NewQuadNode(x, y, width float64, level int) *QuadNode {
	node := QuadNode{
		X:x,
		Y:y,
		Width:width,
		W:0.0,
		Level:level,
		Nodes:make([]*QuadNode, 0),
	}
	return &node
}

// Creates link between source and target nodes, given calculated CPPN output for their coordinates
func createLink(cppnOutput float64, srcIndx, dstIndx int, context *hyperneat.HyperNEATContext) network.FastNetworkLink {
	weight := (math.Abs(cppnOutput) - context.LinkThershold) / (1 - context.LinkThershold) // normalize [0, 1]
	weight *= context.WeightRange // scale to fit given weight range
	if math.Signbit(cppnOutput) {
		weight *= -1 // restore sign
	}
	link := network.FastNetworkLink{
		Weight:weight,
		SourceIndx:srcIndx,
		TargetIndx:dstIndx,
	}
	return &link
}

// Calculates outputs of provided CPPN network solver with given hypercube coordinates.
func queryCPPN(coordinates[]float64, cppn network.NetworkSolver) ([]float64, error) {
	// flush networks activation from previous run
	if res, err := cppn.Flush(); err != nil {
		return nil, err
	} else if !res {
		return nil, errors.New("failed to flush CPPN network")
	}
	// load inputs
	if err := cppn.LoadSensors(coordinates); err != nil {
		return nil, err
	}
	// do activations
	if res, err := cppn.RecursiveSteps(); err != nil {
		return nil, err
	} else if !res {
		return nil, errors.New("failed to relax CPPN network recursively")
	}

	return cppn.ReadOutputs(), nil
}
