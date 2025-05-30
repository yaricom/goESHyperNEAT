package cppn

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

const (
	cppnHyperNEATTestGenomePath    = "../data/test/test_cppn_hyperneat_genome.yml"
	cppnLeoHyperNEATTestGenomePath = "../data/test/test_cppn_leo_hyperneat_genome.yml"
)

func TestQuadNode_NodeVariance(t *testing.T) {
	root := buildTree()

	// get variance and check results
	variance := nodeVariance(root)
	assert.InDelta(t, 1.6666666666666667, variance, 1e-16)
}

func TestQuadNode_nodeCPPNValues(t *testing.T) {
	root := buildTree()

	// get CPPN values and test results
	vals := nodeCPPNValues(root)
	require.Len(t, vals, 7, "wrong node values length")

	expected := []float64{0, 1, 2, 3, 1, 2, 3}
	assert.ElementsMatch(t, expected, vals)
}

func TestFastSolverFromGenomeFile(t *testing.T) {
	cppn, err := FastSolverFromGenomeFile(cppnHyperNEATTestGenomePath)
	require.NoError(t, err, "failed to read genome file")
	require.NotNil(t, cppn, "CPPN expected")
	require.Equal(t, 9, cppn.NodeCount(), "wrong nodes number")
	require.Equal(t, 8, cppn.LinkCount(), "wrong links number")

	// test query
	coords := []float64{0.0, 0.0, 0.0, 0.5, 0.5, 0.0}
	outs, err := queryCPPN(coords, cppn)
	require.NoError(t, err, "failed to query CPPN")
	require.NotNil(t, outs, "output expected")
	require.Len(t, outs, 1)
	assert.InDelta(t, 0.566, outs[0], 1e-3, "wrong output value")
}

func TestFastSolverFromGenomeFile_LEO(t *testing.T) {
	cppn, err := FastSolverFromGenomeFile(cppnLeoHyperNEATTestGenomePath)
	require.NoError(t, err, "failed to read genome file")
	require.NotNil(t, cppn, "CPPN expected")
	require.Equal(t, 11, cppn.NodeCount(), "wrong nodes number")
	require.Equal(t, 15, cppn.LinkCount(), "wrong links number")

	// test query
	coords := []float64{0.0, 0.0, 0.0, 0.5, 0.5, 0.0}
	outs, err := queryCPPN(coords, cppn)
	require.NoError(t, err, "failed to query CPPN")
	require.NotNil(t, outs, "output expected")
	require.Len(t, outs, 2)
	t.Log(outs)
	assert.InDelta(t, 0.048, outs[0], 1e-3, "wrong output value")
	assert.Equal(t, 1.0, outs[1], "wrong LEO value")
}

func buildTree() *QuadNode {
	root := NewQuadNode(0, 0, 1, 1, 1)
	root.Nodes = []*QuadNode{
		NewQuadNode(-1, 1, 0.5, 0.5, 2),
		NewQuadNode(-1, -1, 0.5, 0.5, 2),
		NewQuadNode(1, 1, 0.5, 0.5, 2),
		NewQuadNode(1, -1, 0.5, 0.5, 2),
	}
	fillW(root.Nodes, 1.0)

	root.Nodes[0].Nodes = []*QuadNode{
		NewQuadNode(-1, 1, 0.5, 0.5, 3),
		NewQuadNode(-1, -1, 0.5, 0.5, 3),
		NewQuadNode(1, 1, 0.5, 0.5, 3),
		NewQuadNode(1, -1, 0.5, 0.5, 3),
	}
	fillW(root.Nodes[0].Nodes, 1.0)
	return root
}

func fillW(nodes []*QuadNode, factor float64) {
	for i, n := range nodes {
		n.CppnOut = []float64{float64(i) * factor}
	}
}
