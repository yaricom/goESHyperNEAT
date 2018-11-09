package hyperneat

import (
	"testing"
	"errors"
	"os"
)

func TestESHyperNEATContext_LoadContext(t *testing.T) {
	r, err := os.Open("../data/test_es_hyper.neat.yml")
	if err != nil {
		t.Error("Failed to open config file", err)
	}
	context := &ESHyperNEATContext{}
	err = context.LoadContext(r)
	if err != nil {
		t.Error(err)
	}

	err = checkESHyperNEATContext(context)
	if err != nil {
		t.Error(err)
	}
}

func TestESHyperNEATContext_LoadFullContext(t *testing.T) {
	r, err := os.Open("../data/test_es_hyper.neat.yml")
	if err != nil {
		t.Error("Failed to open config file", err)
	}
	context := &ESHyperNEATContext{}
	err = context.LoadFullContext(r)
	if err != nil {
		t.Error(err)
	}

	err = checkESHyperNEATContext(context)
	if err != nil {
		t.Error(err)
	}

	err = checkHyperNEATContext(context.HyperNEAT)
	if err != nil {
		t.Error(err)
	}
}

func checkESHyperNEATContext(context *ESHyperNEATContext) error {
	if context.InitialDepth != 3 {
		return errors.New("context.InitialDepth != 3")
	}
	if context.MaximalDepth != 5 {
		return errors.New("context.MaximalDepth != 5")
	}
	if context.DivisionThreshold != 0.01 {
		return errors.New("context.DivisionThreshold != 0.01")
	}
	if context.VarianceThreshold != 0.03 {
		return errors.New("context.VarianceThreshold != 0.03")
	}
	if context.BandingThreshold != 0.3 {
		return errors.New("context.BandingThreshold != 0.3")
	}
	if context.ESIterations != 1 {
		return errors.New("context.ESIterations != 1")
	}
	return nil
}