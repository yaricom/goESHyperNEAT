package cppn

import "testing"

func TestQuadNode_NodeVariance(t *testing.T) {
	root := buildTree()

	// get variance and check results
	variance := NodeVariance(root)
	if variance != 3.3877551020408165 {
		t.Error("variance != 3.3877551020408165", variance)
	}
}

func TestQuadNode_nodeCPPNValues(t *testing.T) {
	root := buildTree()

	// get CPPN values and test results
	vals := nodeCPPNValues(root)

	if len(vals) != 7 {
		t.Error("len(vals) != 7", len(vals))
		return
	}

	exp_vals := []float64{0, 1, 2, 3, 2, 4, 6}
	for i, v := range exp_vals {
		if vals[i] != v {
			t.Error("vals[i] != v, at:", i, vals[i], v)
		}
	}
}

func buildTree() *QuadNode {
	root := NewQuadNode(0, 0, 1, 1)
	root.Nodes = []*QuadNode{
		NewQuadNode(-1, 1, 0.5, 2),
		NewQuadNode(-1, -1, 0.5, 2),
		NewQuadNode(1, 1, 0.5, 2),
		NewQuadNode(1, -1, 0.5, 2),
	}
	fillW(root.Nodes, 2.0)

	root.Nodes[0].Nodes = []*QuadNode{
		NewQuadNode(-1, 1, 0.5, 3),
		NewQuadNode(-1, -1, 0.5, 3),
		NewQuadNode(1, 1, 0.5, 3),
		NewQuadNode(1, -1, 0.5, 3),
	}
	fillW(root.Nodes[0].Nodes, 1.0)
	return root
}

func fillW(nodes []*QuadNode, factor float64) {
	for i, n := range nodes {
		n.W = float64(i) * factor
	}
}
