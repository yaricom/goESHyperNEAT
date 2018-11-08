package hyperneat

import (
	"testing"
	"bytes"
	"github.com/yaricom/goNEAT/neat/network"
	"errors"
	"fmt"
)

func TestHyperNEATContext_LoadContext(t *testing.T) {
	r := bytes.NewBufferString(test_hyper_neat_yml)

	context := &HyperNEATContext{}
	err := context.LoadContext(r)
	if err != nil {
		t.Error(err)
	}
	err = checkHyperNEATContext(context)
	if err != nil {
		t.Error(err)
	}
}

func checkHyperNEATContext(context *HyperNEATContext) error {
	if len(context.CPPNNodeActivators) != 4 {
		return errors.New(fmt.Sprintf("len(context.CPPNNodeActivators) != 4, but: %d", len(context.CPPNNodeActivators)))
	}
	activators := []network.NodeActivationType{network.SigmoidBipolarActivation,
		network.GaussianBipolarActivation, network.LinearAbsActivation, network.SineActivation}
	probs := []float64{0.25, 0.35, 0.15, 0.25}
	for i, a := range activators {
		if context.CPPNNodeActivators[i] != a {
			return errors.New(fmt.Sprintf("Wrong CPPN activator type at: %d", i))
		}
		if context.CPPNNodeActivatorsProb[i] != probs[i] {
			return errors.New(fmt.Sprintf("Wrong CPPN activator probability at: %d", i))
		}
	}

	if context.SubstrateActivator != network.SigmoidSteepenedActivation {
		return errors.New("context.SubstrateActivator != network.SigmoidSteepenedActivation")
	}
	return nil
}

const test_hyper_neat_yml =
	"hyperneat:\n" +
	"  link_threshold: 0.2\n" +
	"  weight_range: 3\n" +
	"  substrate_activator: SigmoidSteepenedActivation\n" +
	"  cppn_activators:\n" +
	"    - SigmoidBipolarActivation 0.25\n" +
	"    - GaussianBipolarActivation 0.35\n" +
	"    - LinearAbsActivation 0.15\n" +
	"    - SineActivation 0.25\n"
