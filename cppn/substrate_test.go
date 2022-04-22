package cppn

import (
	"bytes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goESHyperNEAT/v2/hyperneat"
	"github.com/yaricom/goNEAT/v2/neat/math"
	"github.com/yaricom/goNEAT/v2/neat/network"
	"testing"
)

const hyperNeatTestConfigFile = "../data/test/test_hyper.neat.yml"

func TestNewSubstrate(t *testing.T) {
	biasCount, inputCount, hiddenCount, outputCount := 1, 4, 2, 2
	layout := NewGridSubstrateLayout(biasCount, inputCount, outputCount, hiddenCount)

	// create new substrate
	substr := NewSubstrate(layout, math.SigmoidSteepenedActivation)
	assert.Equal(t, math.SigmoidSteepenedActivation, substr.NodesActivation)
}

func TestSubstrate_CreateNetworkSolver(t *testing.T) {
	biasCount, inputCount, hiddenCount, outputCount := 1, 4, 2, 2
	layout := NewGridSubstrateLayout(biasCount, inputCount, outputCount, hiddenCount)

	// create new substrate
	substr := NewSubstrate(layout, math.SigmoidSteepenedActivation)
	assert.Equal(t, math.SigmoidSteepenedActivation, substr.NodesActivation)

	// create solver from substrate
	cppn, err := ReadCPPFromGenomeFile(cppnHyperNEATTestGenomePath)
	require.NoError(t, err, "failed to read CPPN")

	context, err := loadHyperNeatContext(hyperNeatTestConfigFile)
	require.NoError(t, err, "failed to load HyperNEAT context options")

	graph := NewSubstrateGraphMLBuilder("", false)

	solver, err := substr.CreateNetworkSolver(cppn, false, graph, context)
	require.NoError(t, err, "failed to create network solver")

	// test solver
	totalNodeCount := biasCount + inputCount + hiddenCount + outputCount
	assert.Equal(t, totalNodeCount, solver.NodeCount(), "wrong nodes number")

	totalLinkCount := 12 //biasCount * (hiddenCount + outputCount)
	assert.Equal(t, totalLinkCount, solver.LinkCount(), "wrong links number")

	// test outputs
	outExpected := []float64{0.6427874813512032, 0.8685335941574246}
	checkNetworkSolverOutputs(solver, outExpected, t)
}

func TestSubstrate_CreateLEONetworkSolver(t *testing.T) {
	biasCount, inputCount, hiddenCount, outputCount := 1, 4, 2, 2
	layout := NewGridSubstrateLayout(biasCount, inputCount, outputCount, hiddenCount)

	// create new substrate
	substr := NewSubstrate(layout, math.SigmoidSteepenedActivation)
	assert.Equal(t, math.SigmoidSteepenedActivation, substr.NodesActivation)

	// create solver from substrate
	cppn, err := ReadCPPFromGenomeFile(cppnLeoHyperNEATTestGenomePath)
	require.NoError(t, err, "failed to read CPPN")

	context, err := loadHyperNeatContext(hyperNeatTestConfigFile)
	require.NoError(t, err, "failed to load HyperNEAT context options")

	graph := NewSubstrateGraphMLBuilder("", false)

	solver, err := substr.CreateNetworkSolver(cppn, true, graph, context)
	require.NoError(t, err, "failed to create network solver")

	// test solver
	totalNodeCount := biasCount + inputCount + hiddenCount + outputCount
	assert.Equal(t, totalNodeCount, solver.NodeCount(), "wrong nodes number")

	totalLinkCount := 14
	assert.Equal(t, totalLinkCount, solver.LinkCount(), "wrong links number")

	// test outputs
	outExpected := []float64{0.5000001657646664, 0.5000003552761682}
	checkNetworkSolverOutputs(solver, outExpected, t)
}

func TestSubstrate_CreateNetworkSolverWithGraphBuilder(t *testing.T) {
	biasCount, inputCount, hiddenCount, outputCount := 1, 4, 2, 2
	layout := NewGridSubstrateLayout(biasCount, inputCount, outputCount, hiddenCount)

	// create graph builder
	builder := NewSubstrateGraphMLBuilder("", false).(*graphMLBuilder)

	// create new substrate
	substr := NewSubstrate(layout, math.SigmoidSteepenedActivation)

	// create solver from substrate
	cppn, err := ReadCPPFromGenomeFile(cppnHyperNEATTestGenomePath)
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

	totalEdges := 12
	assert.Len(t, graph.Edges, totalEdges, "wrong edges number")

	var buf bytes.Buffer
	err = builder.Marshal(&buf)
	require.NoError(t, err, "failed to marshal graph")

	strOut := buf.String()
	assert.Equal(t, 5597, len(strOut), "wrong length of marshalled string")

	// test outputs
	outExpected := []float64{0.6427874813512032, 0.8685335941574246}
	checkNetworkSolverOutputs(solver, outExpected, t)
}

func checkNetworkSolverOutputs(solver network.Solver, outExpected []float64, t *testing.T) {
	signals := []float64{0.9, 5.2, 1.2, 0.6}
	err := solver.LoadSensors(signals)
	require.NoError(t, err, "failed to load sensors")

	res, err := solver.RecursiveSteps()
	require.NoError(t, err, "failed to perform recursive activation")
	require.True(t, res, "failed to relax network")

	outs := solver.ReadOutputs()
	for i, out := range outs {
		assert.Equal(t, outExpected[i], out, "wrong output at: %d", i)
	}
}

// Loads HyperNeat context from provided config file's path
func loadHyperNeatContext(configPath string) (*hyperneat.Options, error) {
	if context, err := hyperneat.LoadYAMLConfigFile(configPath); err != nil {
		return nil, err
	} else {
		return context, nil
	}
}
