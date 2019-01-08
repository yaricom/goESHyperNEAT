package cppn

import (
	"github.com/yaricom/goNEAT/neat/network"
	"github.com/yaricom/goGraphML/graphml"
	"errors"
	"io"
)

// The graph builder able to build weighted directed graphs representing substrate networks
type GraphBuilder interface {
	// Adds specified node to the graph with provided position
	AddNode(nodeId int, nodeNeuronType network.NodeNeuronType, nodeActivation network.NodeActivationType, position *PointF) error
	// Adds edge between two graph nodes
	AddWeightedEdge(sourceId, targetId int, weight float64) error

	// Serialize built graph to the provided writer
	Marshall(w io.Writer) error
}

// The graph builder based on GraphML specification
type GraphMLBuilder struct {
	// The GraphML instance
	graphML  *graphml.GraphML
	// The current graph instance
	graph    *graphml.Graph

	// The map to hold already added nodes
	nodesMap map[int]*graphml.Node
}

// Creates new instance with specified description to be included into serialized graph
func NewGraphMLBuilder(description string) (*GraphMLBuilder, error) {
	graph_builder := &GraphMLBuilder{
		nodesMap:make(map[int]*graphml.Node),
	}

	// create GraphML
	graph_builder.graphML = graphml.NewGraphML(description)
	if graph, err := graph_builder.graphML.AddGraph("", graphml.EdgeDirectionDirected, nil); err != nil {
		return nil, err
	} else {
		graph_builder.graph = graph
	}

	return graph_builder, nil
}

func (gml *GraphMLBuilder) AddNode(nodeId int, nodeNeuronType network.NodeNeuronType, nodeActivation network.NodeActivationType, position *PointF) (err error) {
	// create attributes map
	n_attr := make(map[string]interface{})
	n_attr["id"] = nodeId
	n_attr["NodeNeuronType"] = network.NeuronTypeName(nodeNeuronType)
	if n_attr["NodeActivationType"], err = network.NodeActivators.ActivationNameFromType(nodeActivation); err != nil {
		return err
	}
	n_attr["X"] = position.X
	n_attr["Y"] = position.Y

	// add node to the graph
	if node, err := gml.graph.AddNode(n_attr, ""); err != nil {
		return err
	} else {
		// store node
		gml.nodesMap[nodeId] = node
	}
	return nil
}

func (gml *GraphMLBuilder) AddWeightedEdge(sourceId, targetId int, weight float64) error {
	// create attributes map
	e_attr := make(map[string]interface{})
	e_attr["weight"] = weight
	e_attr["sourceId"] = sourceId
	e_attr["targetId"] = targetId

	// add edge to graph
	var source, target *graphml.Node
	ok := false
	if source, ok = gml.nodesMap[sourceId]; !ok {
		return errors.New("source node not found")
	}
	if target, ok = gml.nodesMap[targetId]; !ok {
		return errors.New("target node not found")
	}
	_, err := gml.graph.AddEdge(source, target, e_attr, graphml.EdgeDirectionDefault, "")
	return err
}
