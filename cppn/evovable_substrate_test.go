package cppn

import (
	"testing"
	"github.com/yaricom/goNEAT/neat/utils"
	"github.com/yaricom/goESHyperNEAT/hyperneat"
	"os"
)

func TestNewEvolvableSubstrate(t *testing.T) {
	inputCount, outputCount := 4, 2
	layout, err := NewMappedEvolvableSubstrateLayout(inputCount, outputCount)
	if err != nil {
		t.Error(err)
		return
	}

	substr := NewEvolvableSubstrate(layout, utils.TanhActivation)
	if substr == nil {
		t.Error("substr == nil")
	}
}

func TestEvolvableSubstrate_CreateNetworkSolver(t *testing.T) {
	inputCount, outputCount := 4, 2
	layout, err := NewMappedEvolvableSubstrateLayout(inputCount, outputCount)
	if err != nil {
		t.Error(err)
		return
	}

	substr := NewEvolvableSubstrate(layout, utils.SigmoidSteepenedActivation)
	if substr == nil {
		t.Error("substr == nil")
	}

	cppn, err := ReadCPPNfromGenomeFile(cppn_hyperneat_test_genome_path)
	if err != nil {
		t.Error(err)
		return
	}
	context, err := loadESHyperNeatContext("../data/test_es_hyper.neat.yml")
	if err != nil {
		t.Error(err)
		return
	}

	// test solver creation
	graph, err := NewGraphMLBuilder("", false)

	solver, err := substr.CreateNetworkSolver(cppn, graph, context)
	if err != nil {
		t.Error(err)
		return
	}

	//var buf bytes.Buffer
	//err = graph.Marshal(&buf)
	//t.Log(buf.String())

	totalNodeCount := inputCount + outputCount + layout.HiddenCount()
	if solver.NodeCount() != totalNodeCount {
		t.Error("solver.NodeCount() != totalNodeCount", solver.NodeCount(), totalNodeCount)
	}
	totalLinkCount := 4
	if solver.LinkCount() != totalLinkCount {
		t.Error("Wrong link count", solver.LinkCount(), totalLinkCount)
	}

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
		out_expected := []float64{0.5000000095847159, 0.5}
		for i, out := range outs {
			if out != out_expected[i] {
				t.Error("out != out_expected", out, out_expected[i], i)
			}
		}

	}
}

// Loads ES-HyperNeat context from provided config file's path
func loadESHyperNeatContext(configPath string) (*hyperneat.ESHyperNEATContext, error) {
	if r, err := os.Open(configPath); err != nil {
		return nil, err
	} else {
		context := &hyperneat.ESHyperNEATContext{}
		if err := context.LoadFullContext(r); err != nil {
			return nil, err
		} else {
			return context, nil
		}
	}
}
