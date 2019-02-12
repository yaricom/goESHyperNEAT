package cppn

import (
	"math"
	"container/list"

	"github.com/yaricom/goNEAT/neat/network"
	"github.com/yaricom/goESHyperNEAT/hyperneat"
)

// The evolvable substrate holds configuration of ANN produced by CPPN within hypecube where each 4-dimensional point
// mark connection weight between two ANN units. The topology of ANN is not rigid as in plain substrate and can be evolved
// by introducing novel nodes to the ANN. This provides extra benefits that the topology of ANN should not be handcrafted
// by human, but produced during substrate generation from controlling CPPN and nodes locations may be arbitrary that suits
// the best for the task at hand.
type EvolvableSubstrate struct {
	// The CPPN network solver to describe geometry of substrate
	cppn  network.NetworkSolver

	// The reusable coordinates bufer
	coord []float64
}

// Creates new instance of evolvable substrate
func NewEvolvableSubstrate() *EvolvableSubstrate {
	es := EvolvableSubstrate{
		coord:make([]float64, 4),
	}
	return &es
}

// Creates network solver based on current substrate layout and provided Compositional Pattern Producing Network which
// used to define connections between network nodes. Optional graph_builder can be provided to collect graph nodes and edges
// of created network solver. With graph builder it is possible to save/load network configuration as well as visualize it.
func (es *EvolvableSubstrate) CreateNetworkSolver(cppn network.NetworkSolver, graph_builder GraphBuilder, context *hyperneat.HyperNEATContext) (network.NetworkSolver, error) {
	es.cppn = cppn

	return nil, nil
}

// Divides and initialize the quadtree from provided coordinates of source (outgoing = true) or target node (outgoing = false) at (a, b).
// Returns quadtree, in which each quadnode at (x,y) stores CPPN activation level for its position. The initialized
// quadtree is used in the PruningAndExtraction phase to generate the actual ANN connections.
func (es *EvolvableSubstrate) quadTreeDivideAndInit(a, b float64, outgoing bool, context *hyperneat.ESHyperNEATContext) (root *QuadNode, err error) {
	root = NewQuadNode(0.0, 0.0, 1.0, 1)

	queue := list.New()
	queue.PushBack(root)

	for queue.Len() > 0 {
		// de-queue
		p := queue.Remove(queue.Front()).(*QuadNode)

		// Divide into sub-regions and assign children to parent
		p.Nodes = []*QuadNode{
			NewQuadNode(p.X - p.Width / 2.0, p.Y - p.Width / 2.0, p.Width / 2.0, p.Level + 1),
			NewQuadNode(p.X - p.Width / 2.0, p.Y + p.Width / 2.0, p.Width / 2.0, p.Level + 1),
			NewQuadNode(p.X + p.Width / 2.0, p.Y - p.Width / 2.0, p.Width / 2.0, p.Level + 1),
			NewQuadNode(p.X + p.Width / 2.0, p.Y + p.Width / 2.0, p.Width / 2.0, p.Level + 1),
		}

		for _, c := range p.Nodes {
			if outgoing {
				// Querying connection from input or hidden node (Outgoing connectivity pattern)
				if c.W, err = es.queryCPPN(a, b, c.X, c.Y); err != nil {
					return nil, err
				}
			} else {
				// Querying connection to output node (Incoming connectivity pattern)
				if c.W, err = es.queryCPPN(c.X, c.Y, a, b); err != nil {
					return nil, err
				}
			}

		}

		// Divide until initial resolution or if variance is still high
		if p.Level < context.InitialDepth || (p.Level < context.MaximalDepth && NodeVariance(p) > context.DivisionThreshold) {
			for _, c := range p.Nodes {
				queue.PushBack(c)
			}
		}
	}
	return root, nil
}

// Decides what regions should have higher neuron density based on variation and express new neurons and connections into
// these regions.
// Receives coordinates of source (outgoing = true) or target node (outgoing = false) at (a, b) and initialized quadtree node.
// Adds the connections that are in bands of the two-dimensional cross-section of the  hypercube containing the source
// or target node to the connections list and return modified list.
func (es *EvolvableSubstrate) pruneAndExpress(a, b float64, connections[]*QuadPoint, node *QuadNode, outgoing bool, context *hyperneat.ESHyperNEATContext) ([]*QuadPoint, error) {
	// fast check
	if len(node.Nodes) == 0 {
		return connections, nil
	}

	// Traverse quadtree depth-first until the current node’s variance is smaller than the variance threshold or
	// until the node has no children (which means that the variance is zero).
	left, right, top, bottom := 0.0, 0.0, 0.0, 0.0
	for _, c := range node.Nodes {
		childVariance := NodeVariance(c)

		if childVariance >= context.VarianceThreshold {
			if conn, err := es.pruneAndExpress(a, b, connections, c, outgoing, context); err != nil {
				return nil, err
			} else {
				connections = append(connections, conn...)
			}
		} else {
			// Determine if point is in a band by checking neighbor CPPN values
			if outgoing {
				left = math.Abs(c.W - es.queryCPPN(a, b, c.X - node.Width, c.Y))
				right = math.Abs(c.W - es.queryCPPN(a, b, c.X + node.Width, c.Y))
				top = math.Abs(c.W - es.queryCPPN(a, b, c.X, c.Y - node.Width))
				bottom = math.Abs(c.W - es.queryCPPN(a, b, c.X, c.Y + node.Width))
			} else {
				left = math.Abs(c.W - es.queryCPPN(c.X - node.Width, c.Y, a, b))
				right = math.Abs(c.W - es.queryCPPN(c.X + node.Width, c.Y, a, b))
				top = math.Abs(c.W - es.queryCPPN(c.X, c.Y - node.Width, a, b))
				bottom = math.Abs(c.W - es.queryCPPN(c.X, c.Y + node.Width, a, b))
			}

			if math.Max(math.Min(top, bottom), math.Min(left, right)) > context.BandingThreshold {
				// Create new connection specified by QuadPoint(x1,y1,x2,y2,weight) in 4D hypercube
				var conn *QuadPoint
				if outgoing {
					conn = NewQuadPoint(a, b, c.X, c.Y, c.W)
				} else {
					conn = NewQuadPoint(c.X, c.Y, a, b, c.W)
				}

				connections = append(connections, conn)
			}
		}
	}

	return connections
}

// Query CPPN associated with this substrate for specified Hypercube coordinate and returns value produced or error if
// operation failed
func (es *EvolvableSubstrate) queryCPPN(x1, y1, x2, y2 float64) (float64, error) {
	es.coord[0] = x1
	es.coord[1] = y1
	es.coord[2] = x2
	es.coord[3] = y2

	if outs, err := queryCPPN(es.coord, es.cppn); err != nil {
		return math.MaxFloat64, err
	} else {
		return outs[0], nil
	}
}
