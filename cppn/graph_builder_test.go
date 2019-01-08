package cppn

import (
	"testing"
	"github.com/yaricom/goNEAT/neat/network"
)

func TestNewGraphMLBuilder(t *testing.T) {
	description := "test graph"

	builder, err := NewGraphMLBuilder(description)
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

	builder, err := NewGraphMLBuilder(description)
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
	if len(builder.graph.Nodes) != len(nodes) {
		t.Error("len(builder.graph.Nodes) != len(nodes)")
	}

	for i, g_node := range builder.graph.Nodes {
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
			} else  if gn_attr[k] != v {
				t.Error("wrong node data attribute with key:", k, gn_attr[k], v)
			}
		}
	}
}

func TestGraphMLBuilder_AddWeightedEdge(t *testing.T) {
	description := "test add edge graph"

	builder, err := NewGraphMLBuilder(description)
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
	edges := []map[string]interface{}{
		{"sourceId":1, "targetId": 3, "weight":-1.0},
		{"sourceId":1, "targetId": 4, "weight": 0.5},
		{"sourceId":2, "targetId": 3, "weight": 1.5},
		{"sourceId":2, "targetId": 4, "weight":-0.5},
		{"sourceId":3, "targetId": 5, "weight": 0.5},
		{"sourceId":4, "targetId": 5, "weight": 0.5},
	}
	for _, e := range edges {
		err := builder.AddWeightedEdge(e["sourceId"].(int), e["targetId"].(int), e["weight"].(float64))
		if err != nil {
			t.Error(err)
			return
		}
	}

	// check results
	if len(builder.graph.Edges) != len(edges) {
		t.Error("len(builder.graph.Edges) != len(edges)", len(builder.graph.Edges), len(edges))
		return
	}

	for i, g_edge := range builder.graph.Edges {
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

func createTestNodes() []map[string]interface{} {
	return []map[string]interface{}{
		{"id":1, "X":-0.5, "Y":-1.0, "NodeNeuronType":network.InputNeuron, "NodeActivationType":network.NullActivation},
		{"id":2, "X": 0.5, "Y":-1.0, "NodeNeuronType":network.InputNeuron, "NodeActivationType":network.NullActivation},
		{"id":3, "X": 0.0, "Y": 0.0, "NodeNeuronType":network.HiddenNeuron, "NodeActivationType":network.SigmoidSteepenedActivation},
		{"id":4, "X": 0.0, "Y": 0.0, "NodeNeuronType":network.HiddenNeuron, "NodeActivationType":network.SigmoidSteepenedActivation},
		{"id":5, "X": 0.0, "Y": 1.0, "NodeNeuronType":network.OutputNeuron, "NodeActivationType":network.LinearActivation},
	}
}
