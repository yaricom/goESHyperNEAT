package cppn

import (
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v4/neat/network"
	"testing"
)

func TestGridSubstrateLayout_NodePosition(t *testing.T) {
	biasCount, inputCount, hiddenCount, outputCount := 1, 4, 2, 2

	layout := NewGridSubstrateLayout(biasCount, inputCount, outputCount, hiddenCount)

	// check BIAS
	checkNeuronLayoutPositions([]float64{0.0, 0.0}, network.BiasNeuron, layout, t)
	biasPos, err := layout.NodePosition(1, network.BiasNeuron)
	require.EqualError(t, err, "the BIAS index is out of range")
	require.Nil(t, biasPos, "nil expected")

	// check INPUT
	checkNeuronLayoutPositions([]float64{-0.75, -1.0, -0.25, -1.0, 0.25, -1.0, 0.75, -1.0}, network.InputNeuron, layout, t)
	inputPos, err := layout.NodePosition(inputCount, network.InputNeuron)
	require.EqualError(t, err, "neuron index is out of range")
	require.Nil(t, inputPos, "nil expected")

	// check HIDDEN
	checkNeuronLayoutPositions([]float64{-0.5, 0.0, 0.5, 0.0}, network.HiddenNeuron, layout, t)
	hiddenPos, err := layout.NodePosition(hiddenCount, network.HiddenNeuron)
	require.EqualError(t, err, "neuron index is out of range")
	require.Nil(t, hiddenPos, "nil expected")

	// check OUTPUT
	checkNeuronLayoutPositions([]float64{-0.5, 1.0, 0.5, 1.0}, network.OutputNeuron, layout, t)
	outputPos, err := layout.NodePosition(outputCount, network.OutputNeuron)
	require.EqualError(t, err, "neuron index is out of range")
	require.Nil(t, outputPos, "nil expected")
}

func checkNeuronLayoutPositions(positions []float64, nType network.NodeNeuronType, layout SubstrateLayout, t *testing.T) {
	count := len(positions) / 2
	for i := 0; i < count; i++ {
		pos, err := layout.NodePosition(i, nType)
		if err != nil {
			t.Error(err)
			return
		}
		if pos.X != positions[i*2] || pos.Y != positions[i*2+1] {
			t.Errorf("wrong neuron position, expected: (%f, %f), but found: %s",
				positions[i*2], positions[i*2+1], pos)
			return
		}

	}
}
