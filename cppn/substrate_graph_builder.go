package cppn

import (
	"errors"
	"io"

	"github.com/yaricom/goGraphML/graphml"
	"github.com/yaricom/goNEAT/neat/network"
	"github.com/yaricom/goNEAT/neat/utils"
)

const (
	nodeAttrID                 = "id"
	nodeAttrNodeNeuronType     = "NodeNeuronType"
	nodeAttrNodeActivationType = "NodeActivationType"
	nodeAttrX                  = "X"
	nodeAttrY                  = "Y"
	edgeAttrWeight             = "weight"
	edgeAttrSourceId           = "sourceId"
	edgeAttrTargetId           = "targetId"
)

// The graph builder able to build weighted directed graphs representing substrate networks
type SubstrateGraphBuilder interface {
	// Adds specified node to the graph with provided position
	AddNode(nodeId int, nodeNeuronType network.NodeNeuronType, nodeActivation utils.NodeActivationType, position *PointF) error
	// Adds edge between two graph nodes
	AddWeightedEdge(sourceId, targetId int, weight float64) error

	// Returns the number of nodes in the graph
	NodesCount() (int, error)
	// Returns the number of edges in the graph
	EdgesCount() (int, error)

	// Marshal graph to the provided writer
	Marshal(w io.Writer) error
	// Unmarshal graph from the provided reader
	UnMarshal(r io.Reader) error
}

// The graph builder based on GraphML specification
type graphMLBuilder struct {
	// The GraphML instance
	graphML *graphml.GraphML

	// The flag to indicate whether marshal should generate compact graph presentation
	compact bool

	// The map to hold already added nodes
	nodesMap map[int]*graphml.Node
}

// Creates new instance with specified description to be included into serialized graph. If compact is true than graph
// will be marshaled into compact form
func NewSubstrateGraphMLBuilder(description string, compact bool) SubstrateGraphBuilder {
	return &graphMLBuilder{
		nodesMap: make(map[int]*graphml.Node),
		compact:  compact,
		graphML:  graphml.NewGraphML(description),
	}
}

func (b *graphMLBuilder) AddNode(nodeId int, nodeNeuronType network.NodeNeuronType, nodeActivation utils.NodeActivationType, position *PointF) (err error) {
	// create attributes map
	nodeAttr := make(map[string]interface{})
	nodeAttr[nodeAttrID] = nodeId
	nodeAttr[nodeAttrNodeNeuronType] = network.NeuronTypeName(nodeNeuronType)
	if nodeAttr[nodeAttrNodeActivationType], err = utils.NodeActivators.ActivationNameFromType(nodeActivation); err != nil {
		return err
	}
	nodeAttr[nodeAttrX] = position.X
	nodeAttr[nodeAttrY] = position.Y

	// add node to the graph
	if graph, err := b.graph(); err != nil {
		return err
	} else if node, err := graph.AddNode(nodeAttr, ""); err != nil {
		return err
	} else {
		// store node
		b.nodesMap[nodeId] = node
	}
	return nil
}

func (b *graphMLBuilder) AddWeightedEdge(sourceId, targetId int, weight float64) (err error) {
	// create attributes map
	edgeAttr := make(map[string]interface{})
	edgeAttr[edgeAttrWeight] = weight
	edgeAttr[edgeAttrSourceId] = sourceId
	edgeAttr[edgeAttrTargetId] = targetId

	// add edge to graph
	var source, target *graphml.Node
	ok := false
	if source, ok = b.nodesMap[sourceId]; !ok {
		return errors.New("source node not found")
	}
	if target, ok = b.nodesMap[targetId]; !ok {
		return errors.New("target node not found")
	}

	if graph, err := b.graph(); err != nil {
		return err
	} else {
		_, err = graph.AddEdge(source, target, edgeAttr, graphml.EdgeDirectionDefault, "")
	}
	return err
}

func (b *graphMLBuilder) NodesCount() (int, error) {
	if graph, err := b.graph(); err != nil {
		return -1, err
	} else {
		return len(graph.Nodes), nil
	}
}

func (b *graphMLBuilder) EdgesCount() (int, error) {
	if graph, err := b.graph(); err != nil {
		return -1, err
	} else {
		return len(graph.Edges), nil
	}
}

func (b *graphMLBuilder) Marshal(w io.Writer) error {
	return b.graphML.Encode(w, !b.compact)
}

func (b *graphMLBuilder) UnMarshal(r io.Reader) error {
	if err := b.graphML.Decode(r); err != nil {
		return err
	} else if len(b.graphML.Graphs) != 1 {
		return errors.New("none or more than one graph detected")
	}
	return nil
}

// returns graph associated with this builder
func (b *graphMLBuilder) graph() (*graphml.Graph, error) {
	if len(b.graphML.Graphs) == 0 {
		// add new graph
		if graph, err := b.graphML.AddGraph("", graphml.EdgeDirectionDirected, nil); err != nil {
			return nil, err
		} else {
			return graph, nil
		}
	}
	return b.graphML.Graphs[0], nil
}

func addNodeToBuilder(builder SubstrateGraphBuilder, nodeId int, nodeType network.NodeNeuronType, nodeActivation utils.NodeActivationType, position *PointF) (bool, error) {
	if builder == nil {
		return false, nil
	} else if err := builder.AddNode(nodeId, nodeType, nodeActivation, position); err != nil {
		return false, err
	}
	return true, nil
}

func addEdgeToBuilder(builder SubstrateGraphBuilder, sourceId, targetId int, weight float64) (bool, error) {
	if builder == nil {
		return false, nil
	} else if err := builder.AddWeightedEdge(sourceId, targetId, weight); err != nil {
		return false, err
	}
	return true, nil
}
