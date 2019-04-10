package cppn

import (
	"os"
	"testing"
	"bytes"

	"github.com/yaricom/goNEAT/neat/utils"
	"github.com/yaricom/goESHyperNEAT/hyperneat"
)

func TestNewSubstrate(t *testing.T) {
	biasCount, inputCount, hiddenCount, outputCount := 1, 4, 2, 2
	layout := NewGridSubstrateLayout(biasCount, inputCount, outputCount, hiddenCount)

	// create new substrate
	substr := NewSubstrate(layout, utils.SigmoidSteepenedActivation)
	if substr == nil {
		t.Error("substr == nil")
	}
	if substr.NodesActivation != utils.SigmoidSteepenedActivation {
		t.Error("substr.NodesActivation != network.SigmoidSteepenedActivation", substr.NodesActivation)
	}
}

func TestSubstrate_CreateNetworkSolver(t *testing.T) {
	biasCount, inputCount, hiddenCount, outputCount := 1, 4, 2, 2
	layout := NewGridSubstrateLayout(biasCount, inputCount, outputCount, hiddenCount)

	// create new substrate
	substr := NewSubstrate(layout, utils.SigmoidSteepenedActivation)
	if substr == nil {
		t.Error("substr == nil")
	}
	if substr.NodesActivation != utils.SigmoidSteepenedActivation {
		t.Error("substr.NodesActivation != network.SigmoidSteepenedActivation", substr.NodesActivation)
	}

	// create solver from substrate
	cppn, err := ReadCPPNfromGenomeFile(cppn_hyperneat_test_genome_path)
	if err != nil {
		t.Error(err)
		return
	}
	context, err := loadHyperNeatContext("../data/test_hyper.neat.yml")
	if err != nil {
		t.Error(err)
		return
	}

	solver, err := substr.CreateNetworkSolver(cppn, nil, context)
	if err != nil {
		t.Error(err)
		return
	}

	// test solver
	totalNodeCount := biasCount + inputCount + hiddenCount + outputCount
	if solver.NodeCount() != totalNodeCount {
		t.Error("solver.NodeCount() != totalNodeCount", solver.NodeCount(), totalNodeCount)
	}
	t.Log(solver)
	totalLinkCount := 14//biasCount * (hiddenCount + outputCount)
	if solver.LinkCount() != totalLinkCount {
		t.Error("Wrong link count", solver.LinkCount())
	}
	//t.Logf("Links: %d\n", solver.LinkCount())

	// test outputs
	signals := []float64{0.9, 5.2, 1.2, 0.6}
	if err := solver.LoadSensors(signals); err != nil {
		t.Error(err)
	}

	if res, err := solver.RecursiveSteps(); err != nil {
		t.Error(err)
	} else if !res {
		t.Error("failed to relax network")
	} else {
		outs := solver.ReadOutputs()
		for i, out := range outs {
			if out != 0.5 {
				t.Error("out != 0.5", out, i)
			}
		}

	}
	t.Log(solver)
}

func TestSubstrate_CreateNetworkSolverWithGraphBuilder(t *testing.T) {
	biasCount, inputCount, hiddenCount, outputCount := 1, 4, 2, 2
	layout := NewGridSubstrateLayout(biasCount, inputCount, outputCount, hiddenCount)

	// create graph builder
	builder, err := NewGraphMLBuilder("", false)
	if err != nil {
		t.Error(err)
		return
	}

	// create new substrate
	substr := NewSubstrate(layout, utils.SigmoidSteepenedActivation)
	if substr == nil {
		t.Error("substr == nil")
		return
	}
	if substr.NodesActivation != utils.SigmoidSteepenedActivation {
		t.Error("substr.NodesActivation != network.SigmoidSteepenedActivation", substr.NodesActivation)
	}

	// create solver from substrate
	cppn, err := ReadCPPNfromGenomeFile(cppn_hyperneat_test_genome_path)
	if err != nil {
		t.Error(err)
		return
	}
	context, err := loadHyperNeatContext("../data/test_hyper.neat.yml")
	if err != nil {
		t.Error(err)
		return
	}
	solver, err := substr.CreateNetworkSolver(cppn, builder, context)
	if err != nil {
		t.Error(err)
		return
	}
	if solver == nil {
		t.Error("solver == nil")
	}

	// test builder
	graph, err := builder.graph()
	if err != nil {
		t.Error(err)
		return
	}
	s_nodes := biasCount + inputCount + hiddenCount + outputCount
	if len(graph.Nodes) != s_nodes {
		t.Error("len(graph.Nodes) != s_nodes", len(graph.Nodes), s_nodes)
	}
	s_edges := 14
	if len(graph.Edges) != s_edges {
		t.Error("len(graph.Edges) != s_edges", len(graph.Edges), s_edges)
	}

	var buf bytes.Buffer
	err = builder.Marshal(&buf)
	if err != nil {
		t.Error(err)
		return
	}
	str_out := buf.String()
	if len(str_out) != 6000 {
		t.Error("len(str_out) != 6000", len(str_out))
	}
	//t.Log(len(str_out))
	//t.Log(str_out)
}

// Loads HyperNeat context from provided config file's path
func loadHyperNeatContext(configPath string) (*hyperneat.HyperNEATContext, error) {
	if r, err := os.Open(configPath); err != nil {
		return nil, err
	} else {
		context := &hyperneat.HyperNEATContext{}
		if err := context.LoadContext(r); err != nil {
			return nil, err
		} else {
			return context, nil
		}
	}
}


