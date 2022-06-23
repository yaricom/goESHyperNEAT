package cppn

import "fmt"

// PointF Defines point with float precision coordinates
type PointF struct {
	X, Y float64
}

func NewPointF(x, y float64) *PointF {
	return &PointF{X: x, Y: y}
}

func (p *PointF) String() string {
	return fmt.Sprintf("(%f, %f)", p.X, p.Y)
}

// QuadPoint Defines the quad-point in the 4 dimensional hypercube
type QuadPoint struct {
	// The associated coordinates
	X1, X2, Y1, Y2 float64
	// The CPPN outputs for this point
	CppnOut []float64
}

func (q *QuadPoint) Weight() float64 {
	return q.CppnOut[0]
}

func (q *QuadPoint) String() string {
	str := fmt.Sprintf("((%f, %f),(%f, %f)) = %f", q.X1, q.Y1, q.X2, q.Y2, q.Weight())
	if len(q.CppnOut) > 1 {
		var status string
		if q.CppnOut[1] > 0 {
			status = "enabled"
		} else {
			status = "disabled"
		}
		str = fmt.Sprintf("%s [%s]", str, status)
	}
	return str
}

// NewQuadPoint Creates new quad point
func NewQuadPoint(x1, y1, x2, y2 float64, node *QuadNode) *QuadPoint {
	outs := make([]float64, len(node.CppnOut))
	copy(outs, node.CppnOut)
	return &QuadPoint{X1: x1, Y1: y1, X2: x2, Y2: y2, CppnOut: outs}
}

// QuadNode Defines quad-tree node to model 4 dimensional hypercube
type QuadNode struct {
	// The coordinates of center of this quad-tree node's square
	X, Y float64
	// The width of this quad-tree node's square
	Width float64

	// The CPPN outputs for this node
	CppnOut []float64
	// The level of this node in the quad-tree
	Level int

	// The children of this node
	Nodes []*QuadNode
}

func (q *QuadNode) Weight() float64 {
	return q.CppnOut[0]
}

func (q *QuadNode) String() string {
	return fmt.Sprintf("((%f, %f), %f) = %f at %d", q.X, q.Y, q.Width, q.CppnOut, q.Level)
}

// NewQuadNode Creates new quad-node with given parameters
func NewQuadNode(x, y, width float64, level int) *QuadNode {
	node := QuadNode{
		X:       x,
		Y:       y,
		Width:   width,
		CppnOut: []float64{0.0},
		Level:   level,
	}
	return &node
}
