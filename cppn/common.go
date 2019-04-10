// The package CPPN provides implementation of Compositional Pattern Producing Network which is a part of Hypercube-based
// NEAT algorithm implementation
package cppn

import (
	"fmt"
	"github.com/yaricom/goNEAT/neat/network"
	"math"
	"errors"
	"github.com/yaricom/goNEAT/neat/genetics"
	"os"
)

// Defines point with float precision coordinates
type PointF struct {
	X, Y float64
}

func NewPointF(x, y float64) *PointF {
	return &PointF{X:x, Y:y}
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

// Creates new quad point
func NewQuadPoint(x1, y1, x2, y2, value float64) *QuadPoint {
	return &QuadPoint{X1:x1, Y1:y1, X2:x2, Y2:y2, Value:value}
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
	}
	return &node
}

// Reads CPPN from specified genome and creates network solver
func ReadCPPNfromGenomeFile(genomePath string) (network.NetworkSolver, error) {
	if genomeFile, err := os.Open(genomePath); err != nil {
		return nil, err
	} else if r, err := genetics.NewGenomeReader(genomeFile, genetics.YAMLGenomeEncoding); err != nil {
		return nil, err
	} else if genome, err := r.Read(); err != nil {
		return nil, err
	} else if netw, err := genome.Genesis(genome.Id); err != nil {
		return nil, err
	} else {
		return netw.FastNetworkSolver()
	}
}

// Creates link between source and target nodes, given calculated CPPN output for their coordinates
func createLink(cppnOutput float64, srcIndx, dstIndx int, linkThreshold, weightRange float64) *network.FastNetworkLink {
	weight := (math.Abs(cppnOutput) - linkThreshold) / (1 - linkThreshold) // normalize [0, 1]
	weight *= weightRange // scale to fit given weight range
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

// Determines variance among CPPN values for certain hypercube region around specified node.
// This variance is a heuristic indicator of the heterogeneity (i.e. presence of information) of a region.
func nodeVariance(node *QuadNode) float64 {
	// quick check
	if len(node.Nodes) == 0 {
		return 0.0
	}

	cppn_vals := nodeCPPNValues(node)
	// calculate median and variance
	m, v := 0.0, 0.0
	for _, f := range cppn_vals {
		m += f
	}
	m /= float64(len(cppn_vals))

	for _, f := range cppn_vals {
		v += math.Pow(f - m, 2)
	}
	v /= float64(len(cppn_vals))

	return v
}

// Collects the CPPN values stored in a given quadtree node
// Used to estimate the variance in a certain region of space around node
func nodeCPPNValues(n *QuadNode) []float64 {
	if len(n.Nodes) > 0 {
		accumulator := make([]float64, 0)
		for _, p := range n.Nodes {
			// go into child nodes
			p_vals := nodeCPPNValues(p)
			accumulator = append(accumulator, p_vals...)
		}
		return accumulator
	} else {
		return []float64{n.W}
	}
}