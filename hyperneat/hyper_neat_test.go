package hyperneat

import (
	"testing"
	"github.com/yaricom/goNEAT/neat/network"
	"errors"
	"fmt"
	"os"
)

func TestHyperNEATContext_LoadContext(t *testing.T) {
	r, err := os.Open("../data/test_es_hyper.neat.yml")
	if err != nil {
		t.Error("Failed to open config file", err)
	}

	context := &HyperNEATContext{}
	err = context.LoadContext(r)
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
