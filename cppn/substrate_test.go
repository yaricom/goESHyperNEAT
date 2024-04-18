package cppn

import (
	"bytes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goESHyperNEAT/v2/hyperneat"
	"github.com/yaricom/goNEAT/v4/neat/math"
	"testing"
)

const hyperNeatTestConfigFile = "../data/test/test_hyper.neat.yml"

func TestNewSubstrate(t *testing.T) {
	biasCount, inputCount, hiddenCount, outputCount := 1, 4, 2, 2
	layout := NewGridSubstrateLayout(biasCount, inputCount, outputCount, hiddenCount)

	// create new substrate
	substr := NewSubstrate(layout, math.SigmoidSteepenedActivation, math.LinearActivation)
	assert.Equal(t, math.SigmoidSteepenedActivation, substr.HiddenNodesActivation)
}

func TestSubstrate_CreateNetworkSolver(t *testing.T) {
	biasCount, inputCount, hiddenCount, outputCount := 1, 4, 2, 2
	layout := NewGridSubstrateLayout(biasCount, inputCount, outputCount, hiddenCount)

	// create new substrate
	substr := NewSubstrate(layout, math.SigmoidSteepenedActivation, math.LinearActivation)
	assert.Equal(t, math.SigmoidSteepenedActivation, substr.HiddenNodesActivation)

	// create solver from substrate
	cppn, err := FastSolverFromGenomeFile(cppnHyperNEATTestGenomePath)
	require.NoError(t, err, "failed to read CPPN")

	context, err := loadHyperNeatContext(hyperNeatTestConfigFile)
	require.NoError(t, err, "failed to load HyperNEAT context options")

	graph := NewSubstrateGraphMLBuilder("", false)

	solver, err := substr.CreateNetworkSolver(cppn, false, graph, context)
	require.NoError(t, err, "failed to create network solver")

	// test solver
	totalNodeCount := biasCount + inputCount + hiddenCount + outputCount
	assert.Equal(t, totalNodeCount, solver.NodeCount(), "wrong nodes number")

	totalLinkCount := 13
	assert.Equal(t, totalLinkCount, solver.LinkCount(), "wrong links number")

	// test outputs
	outExpected := []float64{0.12926155695140656, 0.15786889573150728}
	checkNetworkSolverOutputs(solver, outExpected, 0.0, t)
}

func TestSubstrate_CreateLEONetworkSolver(t *testing.T) {
	biasCount, inputCount, hiddenCount, outputCount := 1, 4, 2, 2
	layout := NewGridSubstrateLayout(biasCount, inputCount, outputCount, hiddenCount)

	// create new substrate
	substr := NewSubstrate(layout, math.SigmoidSteepenedActivation, math.LinearActivation)
	assert.Equal(t, math.SigmoidSteepenedActivation, substr.HiddenNodesActivation)

	// create solver from substrate
	cppn, err := FastSolverFromGenomeFile(cppnLeoHyperNEATTestGenomePath)
	require.NoError(t, err, "failed to read CPPN")

	context, err := loadHyperNeatContext(hyperNeatTestConfigFile)
	require.NoError(t, err, "failed to load HyperNEAT context options")

	graph := NewSubstrateGraphMLBuilder("", false)

	solver, err := substr.CreateNetworkSolver(cppn, true, graph, context)
	require.NoError(t, err, "failed to create network solver")

	printGraph(graph, t)

	// test solver
	totalNodeCount := biasCount + inputCount + hiddenCount + outputCount
	assert.Equal(t, totalNodeCount, solver.NodeCount(), "wrong nodes number")

	totalLinkCount := 15
	assert.Equal(t, totalLinkCount, solver.LinkCount(), "wrong links number")

	// test outputs
	outExpected := []float64{0.33812764749536894, -0.017572705522999554}
	checkNetworkSolverOutputs(solver, outExpected, 0.0, t)
}

func TestSubstrate_CreateNetworkSolverWithGraphBuilder(t *testing.T) {
	biasCount, inputCount, hiddenCount, outputCount := 1, 4, 2, 2
	layout := NewGridSubstrateLayout(biasCount, inputCount, outputCount, hiddenCount)

	// create graph builder
	builder := NewSubstrateGraphMLBuilder("", false).(*graphMLBuilder)

	// create new substrate
	substr := NewSubstrate(layout, math.SigmoidSteepenedActivation, math.LinearActivation)

	// create solver from substrate
	cppn, err := FastSolverFromGenomeFile(cppnHyperNEATTestGenomePath)
	require.NoError(t, err, "failed to read CPPN")

	context, err := loadHyperNeatContext(hyperNeatTestConfigFile)
	require.NoError(t, err, "failed to load HyperNEAT context options")

	solver, err := substr.CreateNetworkSolver(cppn, false, builder, context)
	require.NoError(t, err, "failed to create network solver")

	// test builder
	graph, err := builder.graph()
	require.NoError(t, err, "failed to build graph")

	totalNodes := biasCount + inputCount + hiddenCount + outputCount
	assert.Len(t, graph.Nodes, totalNodes, "wrong nodes number")

	totalEdges := 13
	assert.Len(t, graph.Edges, totalEdges, "wrong edges number")

	var buf bytes.Buffer
	err = builder.Marshal(&buf)
	require.NoError(t, err, "failed to marshal graph")

	strOut := buf.String()
	assert.Equal(t, 5789, len(strOut), "wrong length of marshalled string")

	// test outputs
	outExpected := []float64{0.12926155695140656, 0.15786889573150728}
	checkNetworkSolverOutputs(solver, outExpected, 0.0, t)
}

// Loads HyperNeat context from provided config file's path
func loadHyperNeatContext(configPath string) (*hyperneat.Options, error) {
	if context, err := hyperneat.LoadYAMLConfigFile(configPath); err != nil {
		return nil, err
	} else {
		return context, nil
	}
}
