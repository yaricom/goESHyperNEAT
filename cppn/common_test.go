package cppn

import (
	"bytes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v4/neat/network"
	"testing"
)

func checkNetworkSolverOutputs(solver network.Solver, outExpected []float64, delta float64, t *testing.T) {
	signals := []float64{0.9, 5.2, 1.2, 0.6}
	err := solver.LoadSensors(signals)
	require.NoError(t, err, "failed to load sensors")

	res, err := solver.RecursiveSteps()
	require.NoError(t, err, "failed to perform recursive activation")
	require.True(t, res, "failed to relax network")

	outs := solver.ReadOutputs()
	for i, out := range outs {
		assert.InDelta(t, outExpected[i], out, delta, "wrong output at: %d", i)
	}
}

func printGraph(graph SubstrateGraphBuilder, t *testing.T) {
	var buf bytes.Buffer
	err := graph.Marshal(&buf)
	assert.NoError(t, err)
	t.Log(buf.String())
}
