package cppn

import (
	"container/list"
	"errors"
	"fmt"
	"github.com/yaricom/goESHyperNEAT/v2/eshyperneat"
	neatmath "github.com/yaricom/goNEAT/v4/neat/math"
	"github.com/yaricom/goNEAT/v4/neat/network"
	"math"
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
	NodesActivation neatmath.NodeActivationType

	// The CPPN network solver to describe geometry of substrate
	cppn network.Solver
	// The reusable coordinates buffer
	coords []float64
}

// NewEvolvableSubstrate Creates new instance of evolvable substrate
func NewEvolvableSubstrate(layout EvolvableSubstrateLayout, nodesActivation neatmath.NodeActivationType) *EvolvableSubstrate {
	return &EvolvableSubstrate{
		coords:          make([]float64, 4),
		Layout:          layout,
		NodesActivation: nodesActivation,
	}
}

// NewEvolvableSubstrateWithBias creates new instance of evolvable substrate with defined cppnBias value.
// The cppnBias will be provided as first value of the CPPN inputs array.
func NewEvolvableSubstrateWithBias(layout EvolvableSubstrateLayout, nodesActivation neatmath.NodeActivationType, cppnBias float64) *EvolvableSubstrate {
	coords := make([]float64, 5)
	coords[0] = cppnBias
	return &EvolvableSubstrate{
		coords:          coords,
		Layout:          layout,
		NodesActivation: nodesActivation,
	}
}

// CreateNetworkSolver Creates network solver based on current substrate layout and provided Compositional Pattern Producing Network which
// used to define connections between network nodes. Optional graph_builder can be provided to collect graph nodes and edges
// of created network solver. With graph builder it is possible to save/load network configuration as well as visualize it.
func (es *EvolvableSubstrate) CreateNetworkSolver(cppn network.Solver, useLeo bool, graphBuilder SubstrateGraphBuilder, options *eshyperneat.Options) (network.Solver, error) {
	es.cppn = cppn

	// the network layers will be collected in order: bias, input, output, hidden
	firstInput := 0
	firstOutput := firstInput + es.Layout.InputCount()
	firstHidden := firstOutput + es.Layout.OutputCount()

	links := make([]*network.FastNetworkLink, 0)
	// The map to hold already created links
	connMap := make(map[string]*network.FastNetworkLink)

	// The function to add new link to the network if appropriate
	addLink := func(qp *QuadPoint, source, target int) (*network.FastNetworkLink, bool) {
		key := fmt.Sprintf("%d_%d", source, target)
		if _, ok := connMap[key]; ok {
			// connection already exists
			return nil, false
		}
		var link *network.FastNetworkLink
		if useLeo && qp.CppnOut[1] > 0 {
			link = createLink(qp.Weight(), source, target, options.WeightRange)
		} else if !useLeo && math.Abs(qp.Weight()) > options.LinkThreshold {
			// add only connections with signal exceeding provided threshold
			link = createThresholdNormalizedLink(qp.Weight(), source, target, options.LinkThreshold, options.WeightRange)
		}
		if link != nil {
			links = append(links, link)
			connMap[key] = link
			return link, true
		} else {
			return nil, false
		}
	}

	// Build links from input nodes to the hidden nodes
	var root *QuadNode
	for in := firstInput; in < firstOutput; in++ {
		// Analyse outgoing connectivity pattern from this input
		input, err := es.Layout.NodePosition(in-firstInput, network.InputNeuron)
		if err != nil {
			return nil, err
		}
		// add input node to graph
		if _, err = addNodeToBuilder(graphBuilder, in, network.InputNeuron, neatmath.NullActivation, input); err != nil {
			return nil, err
		}

		if root, err = es.quadTreeDivideAndInit(input.X, input.Y, true, options); err != nil {
			return nil, err
		}
		qPoints := make([]*QuadPoint, 0)
		if qPoints, err = es.pruneAndExpress(input.X, input.Y, qPoints, root, true, options); err != nil {
			return nil, err
		}
		// iterate over quad points and add nodes/links
		for _, qp := range qPoints {
			// add hidden node to the substrate layout if needed
			targetIndex, err := es.addHiddenNode(qp, firstHidden, graphBuilder)
			if err != nil {
				return nil, err
			}
			// add connection
			if link, ok := addLink(qp, in, targetIndex); ok {
				// add an edge to the graph
				if _, err = addEdgeToBuilder(graphBuilder, in, targetIndex, link.Weight); err != nil {
					return nil, err
				}
			}
		}
	}

	// Build more hidden nodes into unexplored area through a number of iterations
	firstHiddenIter := firstHidden
	lastHidden := firstHiddenIter + es.Layout.HiddenCount()
	for step := 0; step < options.ESIterations; step++ {
		for hi := firstHiddenIter; hi < lastHidden; hi++ {
			// Analyse outgoing connectivity pattern from this hidden node
			hidden, err := es.Layout.NodePosition(hi-firstHidden, network.HiddenNeuron)
			if err != nil {
				return nil, err
			}
			if root, err = es.quadTreeDivideAndInit(hidden.X, hidden.Y, true, options); err != nil {
				return nil, err
			}
			qPoints := make([]*QuadPoint, 0)
			if qPoints, err = es.pruneAndExpress(hidden.X, hidden.Y, qPoints, root, true, options); err != nil {
				return nil, err
			}
			// iterate over quad points and add nodes/links
			for _, qp := range qPoints {
				// add hidden node to the substrate layout if needed
				targetIndex, err := es.addHiddenNode(qp, firstHidden, graphBuilder)
				if err != nil {
					return nil, err
				}
				// add connection
				if link, ok := addLink(qp, hi, targetIndex); ok {
					// add an edge to the graph
					if _, err = addEdgeToBuilder(graphBuilder, hi, targetIndex, link.Weight); err != nil {
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
		if _, err = addNodeToBuilder(graphBuilder, oi, network.OutputNeuron, es.NodesActivation, output); err != nil {
			return nil, err
		}

		if root, err = es.quadTreeDivideAndInit(output.X, output.Y, false, options); err != nil {
			return nil, err
		}
		qPoints := make([]*QuadPoint, 0)
		if qPoints, err = es.pruneAndExpress(output.X, output.Y, qPoints, root, false, options); err != nil {
			return nil, err
		}

		// iterate over quad points and add nodes/links where appropriate
		for _, qp := range qPoints {
			nodePoint := NewPointF(qp.X1, qp.Y1)
			sourceIndex := es.Layout.IndexOfHidden(nodePoint)
			if sourceIndex != -1 {
				// only connect to the hidden nodes that already exists and connected to the input/hidden nodes
				sourceIndex += firstHidden // adjust index to the global indexes space

				// add connection
				if link, ok := addLink(qp, sourceIndex, oi); ok {
					// add an edge to the graph
					if _, err = addEdgeToBuilder(graphBuilder, sourceIndex, oi, link.Weight); err != nil {
						return nil, err
					}
				}
			}
		}
	}

	totalNeuronCount := es.Layout.InputCount() + es.Layout.OutputCount() + es.Layout.HiddenCount()

	// build activations
	activations := make([]neatmath.NodeActivationType, totalNeuronCount)
	for i := 0; i < totalNeuronCount; i++ {
		if i < firstOutput {
			// input nodes - NULL activation
			activations[i] = neatmath.NullActivation
		} else {
			// hidden/output nodes - defined activation
			activations[i] = es.NodesActivation
		}

	}

	// create fast network solver
	if totalNeuronCount == 0 || (!useLeo && len(links) == 0) || len(activations) != totalNeuronCount {
		message := fmt.Sprintf("failed to create network solver: links [%d], nodes [%d], activations [%d], LEO [%t]",
			len(links), totalNeuronCount, len(activations), useLeo)
		return nil, errors.New(message)
	}
	solver := network.NewFastModularNetworkSolver(
		0, es.Layout.InputCount(), es.Layout.OutputCount(), totalNeuronCount,
		activations, links, nil, nil) // No BIAS
	return solver, nil
}

func (es *EvolvableSubstrate) addHiddenNode(qp *QuadPoint, firstHidden int, graphBuilder SubstrateGraphBuilder) (targetIndex int, err error) {
	nodePoint := NewPointF(qp.X2, qp.Y2)
	targetIndex = es.Layout.IndexOfHidden(nodePoint)
	if targetIndex == -1 {
		// add hidden node to the substrate layout
		if targetIndex, err = es.Layout.AddHiddenNode(nodePoint); err != nil {
			return -1, err
		}

		targetIndex += firstHidden // adjust index to the global indexes space
		// add a node to the graph
		if _, err = addNodeToBuilder(graphBuilder, targetIndex, network.HiddenNeuron, es.NodesActivation, nodePoint); err != nil {
			return -1, err
		}
	} else {
		// adjust index to the global indexes space
		targetIndex += firstHidden
	}
	return targetIndex, nil
}

// Divides and initialize the quadtree from provided coordinates of source (outgoing = true) or target node (outgoing = false) at (a, b).
// Returns quadtree, in which each quadnode at (x,y) stores CPPN activation level for its position. The initialized
// quadtree is used in the PruningAndExtraction phase to generate the actual ANN connections.
func (es *EvolvableSubstrate) quadTreeDivideAndInit(a, b float64, outgoing bool, options *eshyperneat.Options) (root *QuadNode, err error) {
	root = NewQuadNode(0.0, 0.0, 1.0, 1)

	queue := list.New()
	queue.PushBack(root)

	for queue.Len() > 0 {
		// de-queue
		p := queue.Remove(queue.Front()).(*QuadNode)

		// Divide into subregions and assign children to parent
		p.Nodes = []*QuadNode{
			NewQuadNode(p.X-p.Width/2.0, p.Y-p.Width/2.0, p.Width/2.0, p.Level+1),
			NewQuadNode(p.X-p.Width/2.0, p.Y+p.Width/2.0, p.Width/2.0, p.Level+1),
			NewQuadNode(p.X+p.Width/2.0, p.Y-p.Width/2.0, p.Width/2.0, p.Level+1),
			NewQuadNode(p.X+p.Width/2.0, p.Y+p.Width/2.0, p.Width/2.0, p.Level+1),
		}

		for _, node := range p.Nodes {
			if outgoing {
				// Querying connection from input or hidden node (Outgoing connectivity pattern)
				if node.CppnOut, err = es.queryCPPN(a, b, node.X, node.Y); err != nil {
					return nil, err
				}
			} else {
				// Querying connection to output node (Incoming connectivity pattern)
				if node.CppnOut, err = es.queryCPPN(node.X, node.Y, a, b); err != nil {
					return nil, err
				}
			}

		}

		// Divide until initial resolution or if variance is still high
		if p.Level < options.InitialDepth || (p.Level < options.MaximalDepth && nodeVariance(p) > options.DivisionThreshold) {
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
func (es *EvolvableSubstrate) pruneAndExpress(a, b float64, connections []*QuadPoint, node *QuadNode, outgoing bool, options *eshyperneat.Options) ([]*QuadPoint, error) {
	// fast check
	if len(node.Nodes) == 0 {
		return connections, nil
	}

	// Traverse quadtree depth-first until the current node’s variance is smaller than the variance threshold or
	// until the node has no children (which means that the variance is zero).
	left, right, top, bottom := 0.0, 0.0, 0.0, 0.0
	for _, quadNode := range node.Nodes {
		childVariance := nodeVariance(quadNode)

		if childVariance >= options.VarianceThreshold {
			if conn, err := es.pruneAndExpress(a, b, connections, quadNode, outgoing, options); err != nil {
				return nil, err
			} else {
				connections = append(connections, conn...)
			}
		} else {
			// Determine if point is in a band by checking neighbor CPPN values
			if outgoing {
				if l, err := es.queryCPPN(a, b, quadNode.X-node.Width, quadNode.Y); err != nil {
					return nil, err
				} else {
					left = math.Abs(quadNode.Weight() - l[0])
				}
				if r, err := es.queryCPPN(a, b, quadNode.X+node.Width, quadNode.Y); err != nil {
					return nil, err
				} else {
					right = math.Abs(quadNode.Weight() - r[0])
				}
				if t, err := es.queryCPPN(a, b, quadNode.X, quadNode.Y-node.Width); err != nil {
					return nil, err
				} else {
					top = math.Abs(quadNode.Weight() - t[0])
				}
				if b, err := es.queryCPPN(a, b, quadNode.X, quadNode.Y+node.Width); err != nil {
					return nil, err
				} else {
					bottom = math.Abs(quadNode.Weight() - b[0])
				}
			} else {
				if l, err := es.queryCPPN(quadNode.X-node.Width, quadNode.Y, a, b); err != nil {
					return nil, err
				} else {
					left = math.Abs(quadNode.Weight() - l[0])
				}
				if r, err := es.queryCPPN(quadNode.X+node.Width, quadNode.Y, a, b); err != nil {
					return nil, err
				} else {
					right = math.Abs(quadNode.Weight() - r[0])
				}
				if t, err := es.queryCPPN(quadNode.X, quadNode.Y-node.Width, a, b); err != nil {
					return nil, err
				} else {
					top = math.Abs(quadNode.Weight() - t[0])
				}
				if b, err := es.queryCPPN(quadNode.X, quadNode.Y+node.Width, a, b); err != nil {
					return nil, err
				} else {
					bottom = math.Abs(quadNode.Weight() - b[0])
				}
			}

			if math.Max(math.Min(top, bottom), math.Min(left, right)) > options.BandingThreshold {
				// Create new connection specified by QuadPoint(x1,y1,x2,y2,weight) in 4D hypercube
				var conn *QuadPoint
				if outgoing {
					conn = NewQuadPoint(a, b, quadNode.X, quadNode.Y, quadNode)
				} else {
					conn = NewQuadPoint(quadNode.X, quadNode.Y, a, b, quadNode)
				}

				connections = append(connections, conn)
			}
		}
	}

	return connections, nil
}

// Query CPPN associated with this substrate for specified Hypercube coordinate and returns value produced or error if
// operation failed
func (es *EvolvableSubstrate) queryCPPN(x1, y1, x2, y2 float64) ([]float64, error) {
	offset := 0
	if len(es.coords) == 5 {
		// CPPN bias defined
		offset = 1
	}
	es.coords[offset] = x1
	es.coords[offset+1] = y1
	es.coords[offset+2] = x2
	es.coords[offset+3] = y2

	if outs, err := queryCPPN(es.coords, es.cppn); err != nil {
		return nil, err
	} else {
		return outs, nil
	}
}
