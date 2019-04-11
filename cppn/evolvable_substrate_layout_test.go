package cppn

import (
	"testing"
	"github.com/yaricom/goNEAT/neat/network"
	"math"
)

func TestMappedEvolvableSubstrateLayout_NodePosition(t *testing.T) {
	inputCount, outputCount := 4, 2
	layout, err := NewMappedEvolvableSubstrateLayout(inputCount, outputCount)
	if err != nil {
		t.Error(err)
		return
	}
	// check INPUT
	checkNeuronLayoutPositions([]float64{-0.75, -1.0, -0.25, -1.0, 0.25, -1.0, 0.75, -1.0}, network.InputNeuron, layout, t)
	inputPos, err := layout.NodePosition(inputCount, network.InputNeuron)
	if err == nil || inputPos != nil {
		t.Error("Error should be returned for out of bounds INPUT index")
	}

	// check OUTPUT
	checkNeuronLayoutPositions([]float64{-0.5, 1.0, 0.5, 1.0}, network.OutputNeuron, layout, t)
	outputPos, err := layout.NodePosition(outputCount, network.OutputNeuron)
	if err == nil || outputPos != nil {
		t.Error("Error should be returned for out of bounds OUTPUT index")
	}
}

func TestMappedEvolvableSubstrateLayout_AddHiddenNode(t *testing.T) {
	inputCount, outputCount := 4, 2
	layout, err := NewMappedEvolvableSubstrateLayout(inputCount, outputCount)
	if err != nil {
		t.Error(err)
		return
	}

	index := 0
	for x := -0.7; x < 0.8; x += 0.1 {
		point := PointF{X:x, Y:0.0}
		h_index, err := layout.AddHiddenNode(&point)
		if err != nil {
			t.Error(err)
			return
		}
		if h_index != index {
			t.Error("AddHiddenNode: h_index != index", h_index, index)
		}
		index++
	}

	if layout.HiddenCount() != index {
		t.Error("layout.HiddenCount() != index - 1", layout.HiddenCount(), index )
	}

	// test get hidden
	for i := 0; i < index; i++ {
		x := -0.7 + float64(i) * 0.1
		h_point, err := layout.NodePosition(i, network.HiddenNeuron)
		if err != nil {
			t.Error(err)
			return
		}
		if math.Abs(h_point.X - x) > 0.0000000001 {
			t.Error("h_point.X != x", h_point.X, x, i)
		}
		if math.Abs(h_point.Y - 0.0) > 0.0000000001 {
			t.Error("h_point.Y != 0.0", h_point.Y, i)
		}
	}

	// test index of
	index = 0
	for x := -0.7; x < 0.8; x += 0.1 {
		point := PointF{X:x, Y:0.0}
		h_index := layout.IndexOfHidden(&point)
		if h_index != index {
			t.Error("IndexOfHidden: h_index != index", h_index, index)
		}
		index++
	}
}
