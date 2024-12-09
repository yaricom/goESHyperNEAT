package cppn

import "fmt"

// PointF Defines point with float precision coordinates
type PointF struct {
	X, Y, Z float64
}

func NewPointF(x, y float64) *PointF {
	return &PointF{X: x, Y: y}
}

func (p *PointF) String() string {
	return fmt.Sprintf("(%f, %f, %f)", p.X, p.Y, p.Z)
}

// QuadPoint Defines the quad-point in the 6 dimensional hypercube
type QuadPoint struct {
	// The associated coordinates
	X1, X2, Y1, Y2, Z1, Z2 float64
	// Weight
	Weight float64
	// Leo
	Leo float64
}

func (q *QuadPoint) String() string {
	str := fmt.Sprintf("((%f, %f, %f),(%f, %f, %f)) = %f", q.X1, q.Y1, q.Z1, q.X2, q.Y2, q.Z2, q.Weight)
	if q.Leo >= 0 {
		var status string
		if q.Leo > 0 {
			status = "enabled"
		} else {
			status = "disabled"
		}
		str = fmt.Sprintf("%s [%s]", str, status)
	}
	return str
}

// NewQuadPoint Creates new quad point
func NewQuadPoint(x1, y1, z1, x2, y2, z2 float64, node *QuadNode) *QuadPoint {
	return &QuadPoint{X1: x1, Y1: y1, Z1: z1, X2: x2, Y2: y2, Z2: z2, Weight: node.Weight(), Leo: node.Leo()}
}

// QuadNode Defines quad-tree node to model 4 dimensional hypercube
type QuadNode struct {
	// The coordinates of center of this quad-tree node's square
	X, Y, Z float64
	// The width of this quad-tree node's square
	Width float64
	// The height of this quad-tree node's square
	Height float64

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

func (q *QuadNode) Leo() float64 {
	if len(q.CppnOut) > 1 {
		return q.CppnOut[1]
	}
	// if no LEO output - enable link unconditionally
	return 1.0
}

func (q *QuadNode) HasLeo() bool {
	return len(q.CppnOut) > 1
}

func (q *QuadNode) String() string {
	return fmt.Sprintf("((%f, %f, %f), %f x %f) = %f at %d", q.X, q.Y, q.Z, q.Width, q.Height, q.CppnOut, q.Level)
}

// NewQuadNode Creates new quad-node with given parameters
func NewQuadNode(x, y, width, height float64, level int) *QuadNode {
	node := QuadNode{
		X:       x,
		Y:       y,
		Width:   width,
		Height:  height,
		CppnOut: []float64{0.0},
		Level:   level,
	}
	return &node
}

// NewQuadNodeZ Creates new quad-node with given parameters with Z coordinate value
func NewQuadNodeZ(x, y, z, width, height float64, level int) *QuadNode {
	node := QuadNode{
		X:       x,
		Y:       y,
		Z:       z,
		Width:   width,
		Height:  height,
		CppnOut: []float64{0.0},
		Level:   level,
	}
	return &node
}
