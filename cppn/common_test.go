package cppn

import "testing"

const (
	cppn_hyperneat_test_genome_path = "../data/test_cppn_hyperneat_genome.yml"
	cppn_leo_hyperneat_test_genome_path = "../data/test_cppn_leo_hyperneat_genome.yml"
)

func TestQuadNode_NodeVariance(t *testing.T) {
	root := buildTree()

	// get variance and check results
	variance := nodeVariance(root)
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

func TestReadCPPNfromGenomeFile(t *testing.T) {
	cppn, err := ReadCPPNfromGenomeFile(cppn_hyperneat_test_genome_path)
	if err != nil {
		t.Error(err)
		return
	}
	if cppn == nil {
		t.Error("cppn == nil")
		return
	}
	if cppn.NodeCount() != 7 {
		t.Error("cppn.NodeCount() != 7", cppn.NodeCount())
	}
	if cppn.LinkCount() != 7 {
		t.Error("cppn.LinkCount() != 7", cppn.LinkCount())
	}
	// test query
	coords := []float64{0.0, 0.0, 0.5, 0.5}
	outs, err := queryCPPN(coords, cppn)
	if err != nil {
		t.Error(err)
		return
	}
	if outs == nil {
		t.Error("outs == nil")
		return
	}
	if len(outs) != 1 {
		t.Error("len(outs) != 1", len(outs))
	}
	if outs[0] != 0.4864161653290716 {
		t.Error("uts[0] != 0.4864161653290716", outs[0])
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
