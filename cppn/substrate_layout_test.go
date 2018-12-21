package cppn

import (
	"testing"
	"github.com/yaricom/goNEAT/neat/network"
)

func TestGridSubstrateLayout_NodePosition(t *testing.T) {
	biasCount, inputCount, hiddenCount, outputCount := 1, 4, 2, 2

	layout := NewGridSubstrateLayout(biasCount, inputCount, outputCount, hiddenCount)

	// check BIAS
	checkNeuronLayoutPositions([]float64{0.0, 0.0}, network.BiasNeuron, layout, t)
	biasPos, err := layout.NodePosition(1, network.BiasNeuron)
	if err == nil || biasPos != nil {
		t.Error("Error should be returned for out of bounds BIAS index")
	}

	// check INPUT
	checkNeuronLayoutPositions([]float64{-0.75, -1.0, -0.25, -1.0, 0.25, -1.0, 0.75, -1.0}, network.InputNeuron, layout, t)
	inputPos, err := layout.NodePosition(inputCount, network.InputNeuron)
	if err == nil || inputPos != nil {
		t.Error("Error should be returned for out of bounds INPUT index")
	}

	// check HIDDEN
	checkNeuronLayoutPositions([]float64{-0.5, 0.0, 0.5, 0.0}, network.HiddenNeuron, layout, t)
	hiddenPos, err := layout.NodePosition(hiddenCount, network.HiddenNeuron)
	if err == nil || hiddenPos != nil {
		t.Error("Error should be returned for out of bounds HIDDEN index")
	}

	// check OUTPUT
	checkNeuronLayoutPositions([]float64{-0.5, 1.0, 0.5, 1.0}, network.OutputNeuron, layout, t)
	outputPos, err := layout.NodePosition(outputCount, network.OutputNeuron)
	if err == nil || outputPos != nil {
		t.Error("Error should be returned for out of bounds OUTPUT index")
	}
}

func checkNeuronLayoutPositions(positions[]float64, n_type network.NodeNeuronType, layout SubstrateLayout, t *testing.T) {
	count := len(positions) / 2
	for i := 0; i < count; i++ {
		pos, err := layout.NodePosition(i, n_type)
		if err != nil {
			t.Error(err)
			return
		}
		if pos.X != positions[i * 2] || pos.Y != positions[i * 2 + 1] {
			t.Errorf("Wrong neuron position, expected: (%f, %f), but found: %s",
				positions[i * 2], positions[i * 2 + 1], pos)
			return
		}

	}
}
