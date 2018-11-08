package hyperneat

import (
	"testing"
	"bytes"
	"errors"
)

func TestESHyperNEATContext_LoadContext(t *testing.T) {
	r := bytes.NewBufferString(test_es_hyper_neat_yml)
	context := &ESHyperNEATContext{}
	err := context.LoadContext(r)
	if err != nil {
		t.Error(err)
	}

	err = checkESHyperNEATContext(context)
	if err != nil {
		t.Error(err)
	}
}

func TestESHyperNEATContext_LoadFullContext(t *testing.T) {
	r := bytes.NewBufferString(test_hyper_neat_yml + "\n" + test_es_hyper_neat_yml)
	context := &ESHyperNEATContext{}
	err := context.LoadFullContext(r)
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

const test_es_hyper_neat_yml =
	"es-hyperneat:\n" +
	"  initial_depth: 3\n" +
	"  maximal_depth: 5\n" +
	"  division_threshold: 0.01\n" +
	"  variance_threshold: 0.03\n" +
	"  banding_threshold: 0.3\n" +
	"  es_iterations: 1\n"
