package cppn

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goESHyperNEAT/v2/eshyperneat"
	"github.com/yaricom/goNEAT/v3/neat/math"
	"testing"
)

const esHyperNeatTestConfigFile = "../data/test/test_es_hyper.neat.yml"

func TestEvolvableSubstrate_CreateNetworkSolver(t *testing.T) {
	inputCount, outputCount := 4, 2
	layout, err := NewMappedEvolvableSubstrateLayout(inputCount, outputCount)
	require.NoError(t, err, "failed to create layout")

	substr := NewEvolvableSubstrate(layout, math.SigmoidSteepenedActivation)

	cppn, err := FastSolverFromGenomeFile(cppnHyperNEATTestGenomePath)
	require.NoError(t, err, "failed to read CPPN")
	context, err := loadESHyperNeatOptions(esHyperNeatTestConfigFile)
	require.NoError(t, err, "failed to read ESHyperNEAT context")

	// test solver creation
	graph := NewSubstrateGraphMLBuilder("TestEvolvableSubstrate_CreateNetworkSolver", false)
	solver, err := substr.CreateNetworkSolver(cppn, false, graph, context)
	require.NoError(t, err, "failed to create solver")

	printGraph(graph, t)

	totalNodeCount := inputCount + outputCount + layout.HiddenCount()
	assert.Equal(t, totalNodeCount, solver.NodeCount(), "wrong total node count")
	assert.Equal(t, 8, solver.LinkCount(), "wrong link number")

	// check outputs
	outExpected := []float64{0.5, 0.5}
	checkNetworkSolverOutputs(solver, outExpected, 1e-8, t)
}

func TestEvolvableSubstrate_CreateNetworkSolver_LEO(t *testing.T) {
	inputCount, outputCount := 4, 2
	layout, err := NewMappedEvolvableSubstrateLayout(inputCount, outputCount)
	require.NoError(t, err, "failed to create layout")

	substr := NewEvolvableSubstrate(layout, math.SigmoidSteepenedActivation)

	cppn, err := FastSolverFromGenomeFile(cppnLeoHyperNEATTestGenomePath)
	require.NoError(t, err, "failed to read CPPN")
	context, err := loadESHyperNeatOptions(esHyperNeatTestConfigFile)
	require.NoError(t, err, "failed to read ESHyperNEAT context")

	// test solver creation
	graph := NewSubstrateGraphMLBuilder("TestEvolvableSubstrate_CreateNetworkSolver", false)
	solver, err := substr.CreateNetworkSolver(cppn, true, graph, context)
	require.NoError(t, err, "failed to create solver")

	printGraph(graph, t)

	totalNodeCount := inputCount + outputCount + layout.HiddenCount()
	assert.Equal(t, totalNodeCount, solver.NodeCount(), "wrong total node count")
	assert.Equal(t, 19, solver.LinkCount(), "wrong link number")

	// check outputs
	outExpected := []float64{0.5, 0.5}
	checkNetworkSolverOutputs(solver, outExpected, 1e-8, t)
}

// Loads ES-HyperNeat options from provided config file's path
func loadESHyperNeatOptions(configPath string) (*eshyperneat.Options, error) {
	if ctx, err := eshyperneat.LoadYAMLConfigFile(configPath); err != nil {
		return nil, err
	} else {
		return ctx, nil
	}
}
