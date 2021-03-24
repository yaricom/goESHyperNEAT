package cppn

import (
	"bytes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"strings"
	"testing"

	"github.com/yaricom/goNEAT/neat/network"
	"github.com/yaricom/goNEAT/neat/utils"
)

func TestNewGraphMLBuilder(t *testing.T) {
	description := "test graph"
	builder := NewSubstrateGraphMLBuilder(description, false).(*graphMLBuilder)

	// check auxiliary components initialized
	assert.NotNil(t, builder.graphML)
	graph, err := builder.graph()
	assert.NoError(t, err, "failed to get graph")
	assert.NotNil(t, graph, "graph is expected")
}

func TestGraphMLBuilder_AddNode(t *testing.T) {
	description := "test add node graph"
	builder := NewSubstrateGraphMLBuilder(description, false).(*graphMLBuilder)

	// add test nodes
	nodes := createTestNodes()
	for _, node := range nodes {
		position := &PointF{X: node[nodeAttrX].(float64), Y: node[nodeAttrY].(float64)}
		err := builder.AddNode(
			node[nodeAttrID].(int),
			node[nodeAttrNodeNeuronType].(network.NodeNeuronType),
			node[nodeAttrNodeActivationType].(utils.NodeActivationType),
			position)
		require.NoError(t, err, "failed to add node")
	}

	// test results
	graph, err := builder.graph()
	require.NoError(t, err, "failed to build graph")
	assert.Len(t, graph.Nodes, len(nodes), "wrong number of nodes in the graph")

	for i, gNode := range graph.Nodes {
		gnAttr, err := gNode.GetAttributes()
		require.NoError(t, err, "failed to get node attributes")
		require.Len(t, gnAttr, len(nodes[i]), "wrong number of node attributes for node: %v", gNode)

		for k, v := range nodes[i] {
			if k == nodeAttrNodeNeuronType {
				nnt := network.NeuronTypeName(v.(network.NodeNeuronType))
				assert.Equal(t, nnt, gnAttr[k], "wrong neuron type")
			} else if k == nodeAttrNodeActivationType {
				nat, err := utils.NodeActivators.ActivationNameFromType(v.(utils.NodeActivationType))
				assert.NoError(t, err, "failed to get activation name from type: %v", v)
				assert.Equal(t, nat, gnAttr[k], "wrong activation type")
			} else if gnAttr[k] != v {
				t.Error("wrong node data attribute with key:", k, gnAttr[k], v)
			}
		}
	}
}

func TestGraphMLBuilder_AddWeightedEdge(t *testing.T) {
	description := "test add edge graph"
	builder := NewSubstrateGraphMLBuilder(description, false).(*graphMLBuilder)

	// add nodes
	nodes := createTestNodes()
	for _, node := range nodes {
		position := &PointF{X: node[nodeAttrX].(float64), Y: node[nodeAttrY].(float64)}
		err := builder.AddNode(
			node[nodeAttrID].(int),
			node[nodeAttrNodeNeuronType].(network.NodeNeuronType),
			node[nodeAttrNodeActivationType].(utils.NodeActivationType),
			position)
		require.NoError(t, err, "failed to add node: %v", node)
	}

	// add edges
	edges := createTestEdges()
	for _, e := range edges {
		err := builder.AddWeightedEdge(e[edgeAttrSourceId].(int), e[edgeAttrTargetId].(int), e[edgeAttrWeight].(float64))
		require.NoError(t, err, "failed to add edge: %v", e)
	}

	// check results
	graph, err := builder.graph()
	require.NoError(t, err, "failed to build graph")
	require.Len(t, graph.Edges, len(edges), "wrong number of graph edges")

	for i, gEdge := range graph.Edges {
		eAttr, err := gEdge.GetAttributes()
		require.NoError(t, err, "failed to get edge attributes: %v", gEdge)
		for k, v := range edges[i] {
			assert.Equal(t, v, eAttr[k], "wrong edge attribute for key: %s", k)
		}
	}
}

func TestGraphMLBuilder_Marshal(t *testing.T) {
	description := "test marshal graph"
	builder := NewSubstrateGraphMLBuilder(description, true)

	// add nodes
	nodes := createTestNodes()
	for _, node := range nodes {
		position := &PointF{X: node[nodeAttrX].(float64), Y: node[nodeAttrY].(float64)}
		err := builder.AddNode(
			node[nodeAttrID].(int),
			node[nodeAttrNodeNeuronType].(network.NodeNeuronType),
			node[nodeAttrNodeActivationType].(utils.NodeActivationType),
			position)
		require.NoError(t, err, "failed to add node: %v", node)
	}

	// add edges
	edges := createTestEdges()
	for _, e := range edges {
		err := builder.AddWeightedEdge(e[edgeAttrSourceId].(int), e[edgeAttrTargetId].(int), e[edgeAttrWeight].(float64))
		require.NoError(t, err, "failed to add edge: %v", e)
	}

	// test Marshal
	var buf bytes.Buffer
	err := builder.Marshal(&buf)
	require.NoError(t, err, "failed to marshal")
	assert.Equal(t, len(graphXml), len(buf.String()), "wrong length of marshalled string")
}

func TestGraphMLBuilder_UnMarshal(t *testing.T) {
	builder := NewSubstrateGraphMLBuilder("", true).(*graphMLBuilder)

	err := builder.UnMarshal(strings.NewReader(graphXml))
	require.NoError(t, err, "failed to unmarshal")

	// check results
	graph, err := builder.graph()
	require.NoError(t, err, "failed to build graph")
	assert.Len(t, graph.Nodes, 5, "wrong nodes number")
	assert.Len(t, graph.Edges, 6, "wrong edges number")
}

func createTestEdges() []map[string]interface{} {
	return []map[string]interface{}{
		{"sourceId": 1, "targetId": 3, "weight": -1.0},
		{"sourceId": 1, "targetId": 4, "weight": 0.5},
		{"sourceId": 2, "targetId": 3, "weight": 1.5},
		{"sourceId": 2, "targetId": 4, "weight": -0.5},
		{"sourceId": 3, "targetId": 5, "weight": 0.5},
		{"sourceId": 4, "targetId": 5, "weight": 0.5},
	}
}

func createTestNodes() []map[string]interface{} {
	return []map[string]interface{}{
		{"id": 1, "X": -0.5, "Y": -1.0, "NodeNeuronType": network.InputNeuron, "NodeActivationType": utils.NullActivation},
		{"id": 2, "X": 0.5, "Y": -1.0, "NodeNeuronType": network.InputNeuron, "NodeActivationType": utils.NullActivation},
		{"id": 3, "X": 0.0, "Y": 0.0, "NodeNeuronType": network.HiddenNeuron, "NodeActivationType": utils.SigmoidSteepenedActivation},
		{"id": 4, "X": 0.0, "Y": 0.0, "NodeNeuronType": network.HiddenNeuron, "NodeActivationType": utils.SigmoidSteepenedActivation},
		{"id": 5, "X": 0.0, "Y": 1.0, "NodeNeuronType": network.OutputNeuron, "NodeActivationType": utils.LinearActivation},
	}
}

const graphXml = "<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\"><desc>test marshal graph</desc><key id=\"d0\" for=\"node\" attr.name=\"id\" attr.type=\"int\"></key><key id=\"d1\" for=\"node\" attr.name=\"NodeNeuronType\" attr.type=\"string\"></key><key id=\"d2\" for=\"node\" attr.name=\"NodeActivationType\" attr.type=\"string\"></key><key id=\"d3\" for=\"node\" attr.name=\"X\" attr.type=\"double\"></key><key id=\"d4\" for=\"node\" attr.name=\"Y\" attr.type=\"double\"></key><key id=\"d5\" for=\"edge\" attr.name=\"weight\" attr.type=\"double\"></key><key id=\"d6\" for=\"edge\" attr.name=\"sourceId\" attr.type=\"int\"></key><key id=\"d7\" for=\"edge\" attr.name=\"targetId\" attr.type=\"int\"></key><graph id=\"g0\" edgedefault=\"directed\"><node id=\"n0\"><data key=\"d0\">1</data><data key=\"d1\">INPT</data><data key=\"d2\">NullActivation</data><data key=\"d3\">-0.5</data><data key=\"d4\">-1</data></node><node id=\"n1\"><data key=\"d0\">2</data><data key=\"d1\">INPT</data><data key=\"d2\">NullActivation</data><data key=\"d3\">0.5</data><data key=\"d4\">-1</data></node><node id=\"n2\"><data key=\"d2\">SigmoidSteepenedActivation</data><data key=\"d3\">0</data><data key=\"d4\">0</data><data key=\"d0\">3</data><data key=\"d1\">HIDN</data></node><node id=\"n3\"><data key=\"d1\">HIDN</data><data key=\"d2\">SigmoidSteepenedActivation</data><data key=\"d3\">0</data><data key=\"d4\">0</data><data key=\"d0\">4</data></node><node id=\"n4\"><data key=\"d3\">0</data><data key=\"d4\">1</data><data key=\"d0\">5</data><data key=\"d1\">OUTP</data><data key=\"d2\">LinearActivation</data></node><edge id=\"e0\" source=\"n0\" target=\"n2\"><data key=\"d5\">-1</data><data key=\"d6\">1</data><data key=\"d7\">3</data></edge><edge id=\"e1\" source=\"n0\" target=\"n3\"><data key=\"d5\">0.5</data><data key=\"d6\">1</data><data key=\"d7\">4</data></edge><edge id=\"e2\" source=\"n1\" target=\"n2\"><data key=\"d5\">1.5</data><data key=\"d6\">2</data><data key=\"d7\">3</data></edge><edge id=\"e3\" source=\"n1\" target=\"n3\"><data key=\"d5\">-0.5</data><data key=\"d6\">2</data><data key=\"d7\">4</data></edge><edge id=\"e4\" source=\"n2\" target=\"n4\"><data key=\"d7\">5</data><data key=\"d5\">0.5</data><data key=\"d6\">3</data></edge><edge id=\"e5\" source=\"n3\" target=\"n4\"><data key=\"d5\">0.5</data><data key=\"d6\">4</data><data key=\"d7\">5</data></edge></graph></graphml>"
