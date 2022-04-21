package cppn

import (
	"container/list"
	"github.com/yaricom/goESHyperNEAT/v2/eshyperneat"
	"math"

	"fmt"
	"github.com/yaricom/goNEAT/neat/network"
	"github.com/yaricom/goNEAT/neat/utils"
)

// EvolvableSubstrate The evolvable substrate holds configuration of ANN produced by CPPN within hypecube where each 4-dimensional point
// mark connection weight between two ANN units. The topology of ANN is not rigid as in plain substrate and can be evolved
// by introducing novel nodes to the ANN. This provides extra benefits that the topology of ANN should not be handcrafted
// by human, but produced during substrate generation from controlling CPPN and nodes locations may be arbitrary that suits
// the best for the task at hand.
type EvolvableSubstrate struct {
	// The layout of neuron nodes in this substrate
	Layout EvolvableSubstrateLayout
	// The activation function's type for neurons encoded
	NodesActivation utils.NodeActivationType

	// The CPPN network solver to describe geometry of substrate
	cppn network.NetworkSolver
	// The reusable coordinates buffer
	coords []float64
}

// NewEvolvableSubstrate Creates new instance of evolvable substrate
func NewEvolvableSubstrate(layout EvolvableSubstrateLayout, nodesActivation utils.NodeActivationType) *EvolvableSubstrate {
	return &EvolvableSubstrate{
		coords:          make([]float64, 4),
		Layout:          layout,
		NodesActivation: nodesActivation,
	}
}

// CreateNetworkSolver Creates network solver based on current substrate layout and provided Compositional Pattern Producing Network which
// used to define connections between network nodes. Optional graph_builder can be provided to collect graph nodes and edges
// of created network solver. With graph builder it is possible to save/load network configuration as well as visualize it.
func (es *EvolvableSubstrate) CreateNetworkSolver(cppn network.NetworkSolver, graphBuilder SubstrateGraphBuilder, context *eshyperneat.ESHyperNEATContext) (network.NetworkSolver, error) {
	es.cppn = cppn

	// the network layers will be collected in order: bias, input, output, hidden
	firstInput := 0
	firstOutput := firstInput + es.Layout.InputCount()
	firstHidden := firstOutput + es.Layout.OutputCount()

	connections := make([]*network.FastNetworkLink, 0)
	// The map to hold already created connections
	connMap := make(map[string]*network.FastNetworkLink)

	// The function to add new connection if appropriate
	addConnection := func(weight float64, source, target int) (*network.FastNetworkLink, bool) {
		key := fmt.Sprintf("%d_%d", source, target)
		if _, ok := connMap[key]; ok {
			// connection already excists
			return nil, false
		}
		link := createLink(weight, source, target, context.WeightRange)
		connections = append(connections, link)
		connMap[key] = link
		return link, true
	}

	// Build connections from input nodes to the hidden nodes
	var root *QuadNode
	for in := firstInput; in < firstOutput; in++ {
		// Analyse outgoing connectivity pattern from this input
		input, err := es.Layout.NodePosition(in-firstInput, network.InputNeuron)
		if err != nil {
			return nil, err
		}
		// add input node to graph
		if _, err := addNodeToBuilder(graphBuilder, in, network.InputNeuron, utils.NullActivation, input); err != nil {
			return nil, err
		}

		if root, err = es.quadTreeDivideAndInit(input.X, input.Y, true, context); err != nil {
			return nil, err
		}
		qPoints := make([]*QuadPoint, 0)
		if qPoints, err = es.pruneAndExpress(input.X, input.Y, qPoints, root, true, context); err != nil {
			return nil, err
		}
		// iterate over quad points and add nodes/connections
		for _, qp := range qPoints {
			nodePoint := NewPointF(qp.X2, qp.Y2)
			targetIndex := es.Layout.IndexOfHidden(nodePoint)
			if targetIndex == -1 {
				// add hidden node to the substrate layout
				if targetIndex, err = es.Layout.AddHiddenNode(nodePoint); err != nil {
					return nil, err
				}

				targetIndex += firstHidden // adjust index to the global indexes space
				// add a node to the graph
				if _, err := addNodeToBuilder(graphBuilder, targetIndex, network.HiddenNeuron, es.NodesActivation, nodePoint); err != nil {
					return nil, err
				}

			} else {
				// adjust index to the global indexes space
				targetIndex += firstHidden
			}
			// add connection
			if link, ok := addConnection(qp.Value, in, targetIndex); ok {
				// add an edge to the graph
				if _, err := addEdgeToBuilder(graphBuilder, in, targetIndex, link.Weight); err != nil {
					return nil, err
				}
			}
		}
	}

	// Build more hidden nodes into unexplored area through a number of iterations
	firstHiddenIter := firstHidden
	lastHidden := firstHiddenIter + es.Layout.HiddenCount()
	for step := 0; step < context.ESIterations; step++ {
		for hi := firstHiddenIter; hi < lastHidden; hi++ {
			// Analyse outgoing connectivity pattern from this hidden node
			hidden, err := es.Layout.NodePosition(hi-firstHidden, network.HiddenNeuron)
			if err != nil {
				return nil, err
			}
			if root, err = es.quadTreeDivideAndInit(hidden.X, hidden.Y, true, context); err != nil {
				return nil, err
			}
			qPoints := make([]*QuadPoint, 0)
			if qPoints, err = es.pruneAndExpress(hidden.X, hidden.Y, qPoints, root, true, context); err != nil {
				return nil, err
			}
			// iterate over quad points and add nodes/connections
			for _, qp := range qPoints {
				nodePoint := NewPointF(qp.X2, qp.Y2)
				targetIndex := es.Layout.IndexOfHidden(nodePoint)
				if targetIndex == -1 {
					// add hidden node to the substrate layout
					if targetIndex, err = es.Layout.AddHiddenNode(nodePoint); err != nil {
						return nil, err
					}

					targetIndex += firstHidden // adjust index to the global indexes space
					// add a node to the graph
					if _, err := addNodeToBuilder(graphBuilder, targetIndex, network.HiddenNeuron, es.NodesActivation, nodePoint); err != nil {
						return nil, err
					}
				} else {
					// adjust index to the global indexes space
					targetIndex += firstHidden
				}
				// add connection
				if link, ok := addConnection(qp.Value, hi, targetIndex); ok {
					// add an edge to the graph
					if _, err := addEdgeToBuilder(graphBuilder, hi, targetIndex, link.Weight); err != nil {
						return nil, err
					}
				}
			}
		}

		// move to the next window
		firstHiddenIter = lastHidden
		lastHidden = lastHidden + (es.Layout.HiddenCount() - lastHidden)
	}

	// Connect hidden nodes to the output
	for oi := firstOutput; oi < firstHidden; oi++ {
		// Analyse incoming connectivity pattern
		output, err := es.Layout.NodePosition(oi-firstOutput, network.OutputNeuron)
		if err != nil {
			return nil, err
		}
		// add output node to graph
		if _, err := addNodeToBuilder(graphBuilder, oi, network.OutputNeuron, es.NodesActivation, output); err != nil {
			return nil, err
		}

		if root, err = es.quadTreeDivideAndInit(output.X, output.Y, false, context); err != nil {
			return nil, err
		}
		qPoints := make([]*QuadPoint, 0)
		if qPoints, err = es.pruneAndExpress(output.X, output.Y, qPoints, root, false, context); err != nil {
			return nil, err
		}

		// iterate over quad points and add nodes/connections where appropriate
		for _, qp := range qPoints {
			nodePoint := NewPointF(qp.X1, qp.Y1)
			sourceIndex := es.Layout.IndexOfHidden(nodePoint)
			if sourceIndex != -1 {
				// only connect to the hidden nodes that already exists and connected to the input/hidden nodes
				sourceIndex += firstHidden // adjust index to the global indexes space

				// add connection
				if link, ok := addConnection(qp.Value, sourceIndex, oi); ok {
					// add an edge to the graph
					if _, err := addEdgeToBuilder(graphBuilder, sourceIndex, oi, link.Weight); err != nil {
						return nil, err
					}
				}
			}
		}
	}

	totalNeuronCount := es.Layout.InputCount() + es.Layout.OutputCount() + es.Layout.HiddenCount()

	// build activations
	activations := make([]utils.NodeActivationType, totalNeuronCount)
	for i := 0; i < totalNeuronCount; i++ {
		if i < firstOutput {
			// input nodes - NULL activation
			activations[i] = utils.NullActivation
		} else {
			// hidden/output nodes - defined activation
			activations[i] = es.NodesActivation
		}

	}

	// create fast network solver
	solver := network.NewFastModularNetworkSolver(
		0, es.Layout.InputCount(), es.Layout.OutputCount(), totalNeuronCount,
		activations, connections, nil, nil) // No BIAS
	return solver, nil
}

// Divides and initialize the quadtree from provided coordinates of source (outgoing = true) or target node (outgoing = false) at (a, b).
// Returns quadtree, in which each quadnode at (x,y) stores CPPN activation level for its position. The initialized
// quadtree is used in the PruningAndExtraction phase to generate the actual ANN connections.
func (es *EvolvableSubstrate) quadTreeDivideAndInit(a, b float64, outgoing bool, context *eshyperneat.ESHyperNEATContext) (root *QuadNode, err error) {
	root = NewQuadNode(0.0, 0.0, 1.0, 1)

	queue := list.New()
	queue.PushBack(root)

	for queue.Len() > 0 {
		// de-queue
		p := queue.Remove(queue.Front()).(*QuadNode)

		// Divide into sub-regions and assign children to parent
		p.Nodes = []*QuadNode{
			NewQuadNode(p.X-p.Width/2.0, p.Y-p.Width/2.0, p.Width/2.0, p.Level+1),
			NewQuadNode(p.X-p.Width/2.0, p.Y+p.Width/2.0, p.Width/2.0, p.Level+1),
			NewQuadNode(p.X+p.Width/2.0, p.Y-p.Width/2.0, p.Width/2.0, p.Level+1),
			NewQuadNode(p.X+p.Width/2.0, p.Y+p.Width/2.0, p.Width/2.0, p.Level+1),
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
		if p.Level < context.InitialDepth || (p.Level < context.MaximalDepth && nodeVariance(p) > context.DivisionThreshold) {
			for _, c := range p.Nodes {
				queue.PushBack(c)
			}
		}
	}
	return root, nil
}

// Decides what regions should have higher neuron density based on variation and express new neurons and connections into
// these regions.
// Receive coordinates of source (outgoing = true) or target node (outgoing = false) at (a, b) and initialized quadtree node.
// Adds the connections that are in bands of the two-dimensional cross-section of the  hypercube containing the source
// or target node to the connections list and return modified list.
func (es *EvolvableSubstrate) pruneAndExpress(a, b float64, connections []*QuadPoint, node *QuadNode, outgoing bool, context *eshyperneat.ESHyperNEATContext) ([]*QuadPoint, error) {
	// fast check
	if len(node.Nodes) == 0 {
		return connections, nil
	}

	// Traverse quadtree depth-first until the current node’s variance is smaller than the variance threshold or
	// until the node has no children (which means that the variance is zero).
	left, right, top, bottom := 0.0, 0.0, 0.0, 0.0
	for _, c := range node.Nodes {
		childVariance := nodeVariance(c)

		if childVariance >= context.VarianceThreshold {
			if conn, err := es.pruneAndExpress(a, b, connections, c, outgoing, context); err != nil {
				return nil, err
			} else {
				connections = append(connections, conn...)
			}
		} else {
			// Determine if point is in a band by checking neighbor CPPN values
			if outgoing {
				if l, err := es.queryCPPN(a, b, c.X-node.Width, c.Y); err != nil {
					return nil, err
				} else {
					left = math.Abs(c.W - l)
				}
				if r, err := es.queryCPPN(a, b, c.X+node.Width, c.Y); err != nil {
					return nil, err
				} else {
					right = math.Abs(c.W - r)
				}
				if t, err := es.queryCPPN(a, b, c.X, c.Y-node.Width); err != nil {
					return nil, err
				} else {
					top = math.Abs(c.W - t)
				}
				if b, err := es.queryCPPN(a, b, c.X, c.Y+node.Width); err != nil {
					return nil, err
				} else {
					bottom = math.Abs(c.W - b)
				}
			} else {
				if l, err := es.queryCPPN(c.X-node.Width, c.Y, a, b); err != nil {
					return nil, err
				} else {
					left = math.Abs(c.W - l)
				}
				if r, err := es.queryCPPN(c.X+node.Width, c.Y, a, b); err != nil {
					return nil, err
				} else {
					right = math.Abs(c.W - r)
				}
				if t, err := es.queryCPPN(c.X, c.Y-node.Width, a, b); err != nil {
					return nil, err
				} else {
					top = math.Abs(c.W - t)
				}
				if b, err := es.queryCPPN(c.X, c.Y+node.Width, a, b); err != nil {
					return nil, err
				} else {
					bottom = math.Abs(c.W - b)
				}
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

	return connections, nil
}

// Query CPPN associated with this substrate for specified Hypercube coordinate and returns value produced or error if
// operation failed
func (es *EvolvableSubstrate) queryCPPN(x1, y1, x2, y2 float64) (float64, error) {
	es.coords[0] = x1
	es.coords[1] = y1
	es.coords[2] = x2
	es.coords[3] = y2

	if outs, err := queryCPPN(es.coords, es.cppn); err != nil {
		return math.MaxFloat64, err
	} else {
		return outs[0], nil
	}
}
