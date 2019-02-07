package hyperneat

import (
	"testing"
	"errors"
	"os"
	"github.com/yaricom/goNEAT/neat/utils"
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
	if context.SubstrateActivator != utils.SigmoidSteepenedActivation {
		return errors.New("context.SubstrateActivator != network.SigmoidSteepenedActivation")
	}
	return nil
}
