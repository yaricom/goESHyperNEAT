package eshyperneat

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v3/neat/math"
	"os"
	"testing"
)

const EsHyperNeatCfgPath = "../data/test/test_es_hyper.neat.yml"

func TestLoadYAMLConfigFile(t *testing.T) {
	opts, err := LoadYAMLConfigFile(EsHyperNeatCfgPath)
	require.NoError(t, err, "failed to load options from config file")

	// check loaded values
	checkEsHyperNeatOptions(opts, t)
}

func TestLoadYAMLOptions(t *testing.T) {
	configFile, err := os.Open(EsHyperNeatCfgPath)
	assert.NoError(t, err, "Failed to open ES-HyperNEAT configuration file.")

	opts, err := LoadYAMLOptions(configFile)
	assert.NoError(t, err, "failed to load options from config file")

	// check loaded values
	checkEsHyperNeatOptions(opts, t)
}

func checkEsHyperNeatOptions(opts *Options, t *testing.T) {
	assert.Equal(t, 3, opts.InitialDepth)
	assert.Equal(t, 5, opts.MaximalDepth)
	assert.Equal(t, 0.01, opts.DivisionThreshold)
	assert.Equal(t, 0.03, opts.VarianceThreshold)
	assert.Equal(t, 0.3, opts.BandingThreshold)
	assert.Equal(t, 1, opts.ESIterations)

	assert.Equal(t, math.SigmoidSteepenedActivation, opts.SubstrateActivator.SubstrateActivationType)
}
