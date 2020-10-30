package cppn

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goESHyperNEAT/eshyperneat"
	"github.com/yaricom/goNEAT/neat/utils"
	"os"
	"testing"
)

func TestEvolvableSubstrate_CreateNetworkSolver(t *testing.T) {
	inputCount, outputCount := 4, 2
	layout, err := NewMappedEvolvableSubstrateLayout(inputCount, outputCount)
	require.NoError(t, err, "failed to create layout")

	substr := NewEvolvableSubstrate(layout, utils.SigmoidSteepenedActivation)

	cppn, err := ReadCPPNfromGenomeFile(cppn_hyperneat_test_genome_path)
	require.NoError(t, err, "failed to read CPPN")
	context, err := loadESHyperNeatContext("../data/test_es_hyper.neat.yml")
	require.NoError(t, err, "failed to read ESHyperNEAT context")

	// test solver creation
	graph := NewGraphMLBuilder("", false)
	solver, err := substr.CreateNetworkSolver(cppn, graph, context)
	require.NoError(t, err, "failed to create solver")

	//var buf bytes.Buffer
	//err = graph.Marshal(&buf)
	//t.Log(buf.String())

	totalNodeCount := inputCount + outputCount + layout.HiddenCount()
	assert.Equal(t, totalNodeCount, solver.NodeCount(), "wrong total node count")
	assert.Equal(t, 10, solver.LinkCount(), "wrong link number")

	// test outputs
	signals := []float64{0.9, 5.2, 1.2, 0.6}
	err = solver.LoadSensors(signals)
	assert.NoError(t, err, "failed to load sensors")

	res, err := solver.RecursiveSteps()
	require.NoError(t, err, "failed to propagate recursive activation")
	require.True(t, res, "failed to relax network")

	outs := solver.ReadOutputs()
	outExpected := []float64{0.5, 0.5}
	delta := 0.0000000001
	for i, out := range outs {
		assert.InDelta(t, outExpected[i], out, delta, "unexpected output: %v at: %d", out, i)
	}
}

// Loads ES-HyperNeat context from provided config file's path
func loadESHyperNeatContext(configPath string) (*eshyperneat.ESHyperNEATContext, error) {
	if r, err := os.Open(configPath); err != nil {
		return nil, err
	} else if ctx, err := eshyperneat.Load(r); err != nil {
		return nil, err
	} else {
		return ctx, nil
	}
}
