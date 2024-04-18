package cppn

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goESHyperNEAT/v2/eshyperneat"
	"github.com/yaricom/goNEAT/v4/neat/math"
	"testing"
)

const esHyperNeatTestConfigFile = "../data/test/test_es_hyper.neat.yml"

func TestEvolvableSubstrate_CreateNetworkSolver(t *testing.T) {
	inputCount, outputCount := 4, 2
	layout, err := NewMappedEvolvableSubstrateLayout(inputCount, outputCount)
	require.NoError(t, err, "failed to create layout")

	substr := NewEvolvableSubstrate(layout, math.SigmoidSteepenedActivation, math.LinearActivation)

	cppn, err := FastSolverFromGenomeFile(cppnHyperNEATTestGenomePath)
	require.NoError(t, err, "failed to read CPPN")
	context, err := loadESHyperNeatOptions(esHyperNeatTestConfigFile)
	require.NoError(t, err, "failed to read ESHyperNEAT context")

	// test solver creation
	graph := NewSubstrateGraphMLBuilder("TestEvolvableSubstrate_CreateNetworkSolver", false)
	solver, err := substr.CreateNetworkSolver(cppn, graph, context)
	require.NoError(t, err, "failed to create solver")

	printGraph(graph, t)

	totalNodeCount := inputCount + outputCount + layout.HiddenCount()
	assert.Equal(t, totalNodeCount, solver.NodeCount(), "wrong total node count")
	assert.Equal(t, 30, solver.LinkCount(), "wrong link number")

	// check outputs
	outExpected := []float64{0, 0}
	checkNetworkSolverOutputs(solver, outExpected, 0.0, t)
}

func TestEvolvableSubstrate_CreateNetworkSolver_LEO(t *testing.T) {
	inputCount, outputCount := 4, 2
	layout, err := NewMappedEvolvableSubstrateLayout(inputCount, outputCount)
	require.NoError(t, err, "failed to create layout")

	substr := NewEvolvableSubstrate(layout, math.SigmoidSteepenedActivation, math.LinearActivation)

	cppn, err := FastSolverFromGenomeFile(cppnLeoHyperNEATTestGenomePath)
	require.NoError(t, err, "failed to read CPPN")
	context, err := loadESHyperNeatOptions(esHyperNeatTestConfigFile)
	require.NoError(t, err, "failed to read ESHyperNEAT context")
	context.LeoEnabled = true

	// test solver creation
	graph := NewSubstrateGraphMLBuilder("TestEvolvableSubstrate_CreateNetworkSolver", false)
	solver, err := substr.CreateNetworkSolver(cppn, graph, context)
	require.NoError(t, err, "failed to create solver")

	printGraph(graph, t)

	totalNodeCount := inputCount + outputCount + layout.HiddenCount()
	assert.Equal(t, totalNodeCount, solver.NodeCount(), "wrong total node count")
	assert.Equal(t, 12, solver.LinkCount(), "wrong link number")

	// check outputs
	outExpected := []float64{0, 0}
	checkNetworkSolverOutputs(solver, outExpected, 0.0, t)
}

// Loads ES-HyperNeat options from provided config file's path
func loadESHyperNeatOptions(configPath string) (*eshyperneat.Options, error) {
	if ctx, err := eshyperneat.LoadYAMLConfigFile(configPath); err != nil {
		return nil, err
	} else {
		return ctx, nil
	}
}
