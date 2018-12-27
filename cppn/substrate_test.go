package cppn

import (
	"github.com/yaricom/goNEAT/neat/network"
	"os"
	"github.com/yaricom/goNEAT/neat/genetics"
	"testing"
	"github.com/yaricom/goESHyperNEAT/hyperneat"
)

const (
	cppn_hyperneat_test_genome_path = "../data/test_cppn_hyperneat_genome.yml"
)

func TestNewSubstrate(t *testing.T) {
	biasCount, inputCount, hiddenCount, outputCount := 1, 4, 2, 2
	layout := NewGridSubstrateLayout(biasCount, inputCount, outputCount, hiddenCount)

	// create new substrate
	substr := NewSubstrate(layout, network.SigmoidSteepenedActivation)
	if substr == nil {
		t.Error("substr == nil")
	}
	if substr.NodesActivation != network.SigmoidSteepenedActivation {
		t.Error("substr.NodesActivation != network.SigmoidSteepenedActivation", substr.NodesActivation)
	}
}

func TestSubstrate_CreateNetworkSolver(t *testing.T) {
	biasCount, inputCount, hiddenCount, outputCount := 1, 4, 2, 2
	layout := NewGridSubstrateLayout(biasCount, inputCount, outputCount, hiddenCount)

	// create new substrate
	substr := NewSubstrate(layout, network.SigmoidSteepenedActivation)
	if substr == nil {
		t.Error("substr == nil")
	}
	if substr.NodesActivation != network.SigmoidSteepenedActivation {
		t.Error("substr.NodesActivation != network.SigmoidSteepenedActivation", substr.NodesActivation)
	}

	// create solver from substrate
	cppn, err := buildCPPNfromGenome(cppn_hyperneat_test_genome_path)
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

// Builds test CPPN from specified genome
func buildCPPNfromGenome(genomePath string) (network.NetworkSolver, error) {
	if genomeFile, err := os.Open(genomePath); err != nil {
		return nil, err
	} else if r, err := genetics.NewGenomeReader(genomeFile, genetics.YAMLGenomeEncoding); err != nil {
		return nil, err
	} else if genome, err := r.Read(); err != nil {
		return nil, err
	} else if netw, err := genome.Genesis(genome.Id); err != nil {
		return nil, err
	} else {
		return netw.FastNetworkSolver()
	}
}


