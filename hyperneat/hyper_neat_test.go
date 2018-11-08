package hyperneat

import (
	"testing"
	"bytes"
	"github.com/yaricom/goNEAT/neat/network"
)

func TestHyperNEATContext_LoadContext(t *testing.T) {
	r := bytes.NewBufferString(test_hyper_neat_yml)

	context := HyperNEATContext{}
	err := context.LoadContext(r)
	if err != nil {
		t.Error(err)
	}

	if len(context.CPPNNodeActivators) != 4 {
		t.Error("len(context.CPPNNodeActivators) != 4")
	}
	activators := []network.NodeActivationType{network.SigmoidBipolarActivation,
		network.GaussianBipolarActivation, network.LinearAbsActivation, network.SineActivation}
	probs := []float64{0.25, 0.35, 0.15, 0.25}
	for i, a := range activators {
		if context.CPPNNodeActivators[i] != a {
			t.Error("Wrong CPPN activator type at: ", i)
		}
		if context.CPPNNodeActivatorsProb[i] != probs[i] {
			t.Error("Wrong CPPN activator probability at: ", i)
		}
	}

	if context.SubstrateActivator != network.SigmoidSteepenedActivation {
		t.Error("context.SubstrateActivator != network.SigmoidSteepenedActivation")
	}
}

const test_hyper_neat_yml = "# The threshold value to indicate which links should be included\n" +
	"link_threshold: 0.2\n" +
	"# The weight range defines the minimum and maximum values for weights on substrate connections\n" +
	"weight_range: 3\n" +
	"# The substrate activation function\n" +
	"substrate_activator: SigmoidSteepenedActivation\n" +
	"# The activation functions list to choose from (activation function -> it's selection probability)\n" +
	"cppn_activators:\n" +
	"- SigmoidBipolarActivation 0.25\n" +
	"- GaussianBipolarActivation 0.35\n" +
	"- LinearAbsActivation 0.15\n" +
	"- SineActivation 0.25\n"
