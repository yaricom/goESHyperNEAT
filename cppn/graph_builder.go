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

	// Marshal graph to the provided writer
	Marshal(w io.Writer) error
	// Unmarshal graph from the provided reader
	UnMarshal(r io.Reader) error
}

// The graph builder based on GraphML specification
type GraphMLBuilder struct {
	// The GraphML instance
	graphML  *graphml.GraphML

	// The flag to indicate whether marshal should generate compact graph presentation
	compact  bool

	// The map to hold already added nodes
	nodesMap map[int]*graphml.Node
}

// Creates new instance with specified description to be included into serialized graph. If compact is true than graph
// will be marshaled into compact form
func NewGraphMLBuilder(description string, compact bool) (*GraphMLBuilder, error) {
	graph_builder := &GraphMLBuilder{
		nodesMap:make(map[int]*graphml.Node),
		compact:compact,
	}

	// create GraphML
	graph_builder.graphML = graphml.NewGraphML(description)

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
	if graph, err := gml.graph(); err != nil {
		return err
	} else if node, err := graph.AddNode(n_attr, ""); err != nil {
		return err
	} else {
		// store node
		gml.nodesMap[nodeId] = node
	}
	return nil
}

func (gml *GraphMLBuilder) AddWeightedEdge(sourceId, targetId int, weight float64) (err error) {
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

	if graph, err := gml.graph(); err != nil {
		return err
	} else {
		_, err = graph.AddEdge(source, target, e_attr, graphml.EdgeDirectionDefault, "")
	}
	return err
}

func (gml *GraphMLBuilder) Marshal(w io.Writer) error {
	return gml.graphML.Encode(w, !gml.compact)
}

func (gml *GraphMLBuilder) UnMarshal(r io.Reader) error {
	if err := gml.graphML.Decode(r); err != nil {
		return err
	} else if len(gml.graphML.Graphs) != 1 {
		return errors.New("none or more than one graph detected")
	}
	return nil
}

// returns graph associated with this builder
func (gml *GraphMLBuilder) graph() (*graphml.Graph, error) {
	if len(gml.graphML.Graphs) == 0 {
		// add new graph
		if graph, err := gml.graphML.AddGraph("", graphml.EdgeDirectionDirected, nil); err != nil {
			return nil, err
		} else {
			return graph, nil
		}
	}
	return gml.graphML.Graphs[0], nil
}
