package cppn

import (
	"testing"
	"github.com/yaricom/goNEAT/neat/network"
	"bytes"
	"strings"
)

func TestNewGraphMLBuilder(t *testing.T) {
	description := "test graph"

	builder, err := NewGraphMLBuilder(description, false)
	if err != nil {
		t.Error(err)
	}
	if builder == nil {
		t.Error("builder == nil")
		return
	}

	// check auxiliary components initialized
	if builder.graphML == nil {
		t.Error("builder.graphML == nil")
	}
	if builder.graph == nil {
		t.Error("builder.graph == nil")
	}
}

func TestGraphMLBuilder_AddNode(t *testing.T) {
	description := "test add node graph"

	builder, err := NewGraphMLBuilder(description, false)
	if err != nil {
		t.Error(err)
	}
	if builder == nil {
		t.Error("builder == nil")
		return
	}

	// add test nodes
	nodes := createTestNodes()
	for _, node := range nodes {
		position := &PointF{X:node["X"].(float64), Y:node["Y"].(float64)}
		if err := builder.AddNode(node["id"].(int), node["NodeNeuronType"].(network.NodeNeuronType),
			node["NodeActivationType"].(network.NodeActivationType), position); err != nil {
			t.Error(err)
			return
		}
	}

	// test results
	graph, err := builder.graph()
	if err != nil {
		t.Error(err)
		return
	}
	if len(graph.Nodes) != len(nodes) {
		t.Error("len(graph.Nodes) != len(nodes)")
	}

	for i, g_node := range graph.Nodes {
		gn_attr, err := g_node.GetAttributes()
		if err != nil {
			t.Error(err)
			return
		}
		if len(gn_attr) != len(nodes[i]) {
			t.Error("len(gn_attr) != len(nodes[i]) at:", i, len(gn_attr), len(nodes[i]))
		}
		for k, v := range nodes[i] {
			if k == "NodeNeuronType" {
				nnt := network.NeuronTypeName(v.(network.NodeNeuronType))
				if gn_attr[k] != nnt {
					t.Error("gn_attr[k] != nnt", gn_attr[k], nnt)
				}
			} else if k == "NodeActivationType" {
				nat, err := network.NodeActivators.ActivationNameFromType(v.(network.NodeActivationType))
				if err != nil {
					t.Error(err)
					return
				}
				if gn_attr[k] != nat {
					t.Error("gn_attr[k] != nat", gn_attr[k], nat)
				}
			} else if gn_attr[k] != v {
				t.Error("wrong node data attribute with key:", k, gn_attr[k], v)
			}
		}
	}
}

func TestGraphMLBuilder_AddWeightedEdge(t *testing.T) {
	description := "test add edge graph"

	builder, err := NewGraphMLBuilder(description, false)
	if err != nil {
		t.Error(err)
	}
	if builder == nil {
		t.Error("builder == nil")
		return
	}

	// add nodes
	nodes := createTestNodes()
	for _, node := range nodes {
		position := &PointF{X:node["X"].(float64), Y:node["Y"].(float64)}
		if err := builder.AddNode(node["id"].(int), node["NodeNeuronType"].(network.NodeNeuronType),
			node["NodeActivationType"].(network.NodeActivationType), position); err != nil {
			t.Error(err)
			return
		}
	}

	// add edges
	edges := createTestEdges()
	for _, e := range edges {
		err := builder.AddWeightedEdge(e["sourceId"].(int), e["targetId"].(int), e["weight"].(float64))
		if err != nil {
			t.Error(err)
			return
		}
	}

	// check results
	graph, err := builder.graph()
	if err != nil {
		t.Error(err)
		return
	}
	if len(graph.Edges) != len(edges) {
		t.Error("len(graph.Edges) != len(edges)", len(graph.Edges), len(edges))
		return
	}

	for i, g_edge := range graph.Edges {
		e_attr, err := g_edge.GetAttributes()
		if err != nil {
			t.Error(err)
			return
		}
		for k, v := range edges[i] {
			if e_attr[k] != v {
				t.Error("Wrong edge attribute for key: ", k, e_attr[k], v)
			}
		}
	}
}

func TestGraphMLBuilder_Marshal(t *testing.T) {
	description := "test marshal graph"

	builder, err := NewGraphMLBuilder(description, true)
	if err != nil {
		t.Error(err)
	}
	if builder == nil {
		t.Error("builder == nil")
		return
	}

	// add nodes
	nodes := createTestNodes()
	for _, node := range nodes {
		position := &PointF{X:node["X"].(float64), Y:node["Y"].(float64)}
		if err := builder.AddNode(node["id"].(int), node["NodeNeuronType"].(network.NodeNeuronType),
			node["NodeActivationType"].(network.NodeActivationType), position); err != nil {
			t.Error(err)
			return
		}
	}

	// add edges
	edges := createTestEdges()
	for _, e := range edges {
		err := builder.AddWeightedEdge(e["sourceId"].(int), e["targetId"].(int), e["weight"].(float64))
		if err != nil {
			t.Error(err)
			return
		}
	}

	// test Marshal
	var buf bytes.Buffer
	err = builder.Marshal(&buf)
	if err != nil {
		t.Error(err)
		return
	}
	if buf.Len() != len(graph_xml) {
		t.Error("buf.Len() == len(graph_xml)", buf.Len(), len(graph_xml))
	}
}

func TestGraphMLBuilder_UnMarshal(t *testing.T) {
	builder, err := NewGraphMLBuilder("", true)
	if err != nil {
		t.Error(err)
		return
	}

	err = builder.UnMarshal(strings.NewReader(graph_xml))
	if err != nil {
		t.Error(err)
		return
	}

	// check results
	graph, err := builder.graph()
	if err != nil {
		t.Error(err)
		return
	}
	if len(graph.Nodes) != 5 {
		t.Error("len(graph.Nodes) != 5", len(graph.Nodes))
	}
	if len(graph.Edges) != 6 {
		t.Error("len(graph.Edges) != 6", len(graph.Edges))
	}
}

func createTestEdges() []map[string]interface{} {
	return []map[string]interface{}{
		{"sourceId":1, "targetId": 3, "weight":-1.0},
		{"sourceId":1, "targetId": 4, "weight": 0.5},
		{"sourceId":2, "targetId": 3, "weight": 1.5},
		{"sourceId":2, "targetId": 4, "weight":-0.5},
		{"sourceId":3, "targetId": 5, "weight": 0.5},
		{"sourceId":4, "targetId": 5, "weight": 0.5},
	}
}

func createTestNodes() []map[string]interface{} {
	return []map[string]interface{}{
		{"id":1, "X":-0.5, "Y":-1.0, "NodeNeuronType":network.InputNeuron, "NodeActivationType":network.NullActivation},
		{"id":2, "X": 0.5, "Y":-1.0, "NodeNeuronType":network.InputNeuron, "NodeActivationType":network.NullActivation},
		{"id":3, "X": 0.0, "Y": 0.0, "NodeNeuronType":network.HiddenNeuron, "NodeActivationType":network.SigmoidSteepenedActivation},
		{"id":4, "X": 0.0, "Y": 0.0, "NodeNeuronType":network.HiddenNeuron, "NodeActivationType":network.SigmoidSteepenedActivation},
		{"id":5, "X": 0.0, "Y": 1.0, "NodeNeuronType":network.OutputNeuron, "NodeActivationType":network.LinearActivation},
	}
}

const graph_xml = "<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\"><desc>test marshal graph</desc><key id=\"d0\" for=\"node\" attr.name=\"id\" attr.type=\"int\"></key><key id=\"d1\" for=\"node\" attr.name=\"NodeNeuronType\" attr.type=\"string\"></key><key id=\"d2\" for=\"node\" attr.name=\"NodeActivationType\" attr.type=\"string\"></key><key id=\"d3\" for=\"node\" attr.name=\"X\" attr.type=\"double\"></key><key id=\"d4\" for=\"node\" attr.name=\"Y\" attr.type=\"double\"></key><key id=\"d5\" for=\"edge\" attr.name=\"weight\" attr.type=\"double\"></key><key id=\"d6\" for=\"edge\" attr.name=\"sourceId\" attr.type=\"int\"></key><key id=\"d7\" for=\"edge\" attr.name=\"targetId\" attr.type=\"int\"></key><graph id=\"g0\" edgedefault=\"directed\"><node id=\"n0\"><data key=\"d0\">1</data><data key=\"d1\">INPT</data><data key=\"d2\">NullActivation</data><data key=\"d3\">-0.5</data><data key=\"d4\">-1</data></node><node id=\"n1\"><data key=\"d0\">2</data><data key=\"d1\">INPT</data><data key=\"d2\">NullActivation</data><data key=\"d3\">0.5</data><data key=\"d4\">-1</data></node><node id=\"n2\"><data key=\"d2\">SigmoidSteepenedActivation</data><data key=\"d3\">0</data><data key=\"d4\">0</data><data key=\"d0\">3</data><data key=\"d1\">HIDN</data></node><node id=\"n3\"><data key=\"d1\">HIDN</data><data key=\"d2\">SigmoidSteepenedActivation</data><data key=\"d3\">0</data><data key=\"d4\">0</data><data key=\"d0\">4</data></node><node id=\"n4\"><data key=\"d3\">0</data><data key=\"d4\">1</data><data key=\"d0\">5</data><data key=\"d1\">OUTP</data><data key=\"d2\">LinearActivation</data></node><edge id=\"e0\" source=\"n0\" target=\"n2\"><data key=\"d5\">-1</data><data key=\"d6\">1</data><data key=\"d7\">3</data></edge><edge id=\"e1\" source=\"n0\" target=\"n3\"><data key=\"d5\">0.5</data><data key=\"d6\">1</data><data key=\"d7\">4</data></edge><edge id=\"e2\" source=\"n1\" target=\"n2\"><data key=\"d5\">1.5</data><data key=\"d6\">2</data><data key=\"d7\">3</data></edge><edge id=\"e3\" source=\"n1\" target=\"n3\"><data key=\"d5\">-0.5</data><data key=\"d6\">2</data><data key=\"d7\">4</data></edge><edge id=\"e4\" source=\"n2\" target=\"n4\"><data key=\"d7\">5</data><data key=\"d5\">0.5</data><data key=\"d6\">3</data></edge><edge id=\"e5\" source=\"n3\" target=\"n4\"><data key=\"d5\">0.5</data><data key=\"d6\">4</data><data key=\"d7\">5</data></edge></graph></graphml>"