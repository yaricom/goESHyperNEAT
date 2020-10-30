package cppn

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/neat/network"
	"testing"
)

func TestMappedEvolvableSubstrateLayout_NodePosition(t *testing.T) {
	inputCount, outputCount := 4, 2
	layout, err := NewMappedEvolvableSubstrateLayout(inputCount, outputCount)
	require.NoError(t, err, "failed to create layout")

	// check INPUT
	checkNeuronLayoutPositions([]float64{-0.75, -1.0, -0.25, -1.0, 0.25, -1.0, 0.75, -1.0}, network.InputNeuron, layout, t)
	inputPos, err := layout.NodePosition(inputCount, network.InputNeuron)
	assert.EqualError(t, err, "neuron index is out of range")
	assert.Nil(t, inputPos, "no input position should be returned")

	// check OUTPUT
	checkNeuronLayoutPositions([]float64{-0.5, 1.0, 0.5, 1.0}, network.OutputNeuron, layout, t)
	outputPos, err := layout.NodePosition(outputCount, network.OutputNeuron)
	assert.EqualError(t, err, "neuron index is out of range")
	assert.Nil(t, outputPos, "no output position should be returned")
}

func TestMappedEvolvableSubstrateLayout_AddHiddenNode(t *testing.T) {
	inputCount, outputCount := 4, 2
	layout, err := NewMappedEvolvableSubstrateLayout(inputCount, outputCount)
	require.NoError(t, err, "failed to create layout")

	index := 0
	for x := -0.7; x < 0.8; x += 0.1 {
		point := PointF{X: x, Y: 0.0}
		hIndex, err := layout.AddHiddenNode(&point)
		require.NoError(t, err, "failed to add hidden node at: %v", point)
		assert.Equal(t, index, hIndex, "wrong hidden node index")
		index++
	}
	assert.Equal(t, index, layout.HiddenCount(), "wrong number of hidden nodes")

	// test get hidden
	delta := 0.0000000001
	for i := 0; i < index; i++ {
		x := -0.7 + float64(i)*0.1
		hPoint, err := layout.NodePosition(i, network.HiddenNeuron)
		require.NoError(t, err, "failed to get node position")
		assert.InDelta(t, x, hPoint.X, delta, "wrong X coordinate: %v", hPoint.X)
		assert.InDelta(t, 0.0, hPoint.Y, delta, "wrong Y coordinate: %v", hPoint.Y)
	}

	// test index of
	index = 0
	for x := -0.7; x < 0.8; x += 0.1 {
		point := PointF{X: x, Y: 0.0}
		hIndex := layout.IndexOfHidden(&point)
		assert.Equal(t, index, hIndex, "wrong index of hidden point: %v", point)
		index++
	}
}
