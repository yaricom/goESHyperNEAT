package hyperneat

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v4/neat/math"
	"os"
	"testing"
)

const hyperNeatCfgPath = "../data/test/test_hyper.neat.yml"

func TestLoadYAMLOptions(t *testing.T) {
	r, err := os.Open(hyperNeatCfgPath)
	require.NoError(t, err, "failed to open config file")

	opts, err := LoadYAMLOptions(r)
	require.NoError(t, err, "failed to load HyperNEAT options")

	// check values
	checkHyperNeatOptions(opts, t)
}

func TestLoadYAMLConfigFile(t *testing.T) {
	opts, err := LoadYAMLConfigFile(hyperNeatCfgPath)
	require.NoError(t, err, "failed to load HyperNEAT options")

	// check values
	checkHyperNeatOptions(opts, t)
}

func checkHyperNeatOptions(opts *Options, t *testing.T) {
	assert.Equal(t, math.SigmoidSteepenedActivation, opts.SubstrateActivator.SubstrateActivationType)
	assert.Equal(t, 0.2, opts.LinkThreshold)
	assert.Equal(t, 3.0, opts.WeightRange)
}
