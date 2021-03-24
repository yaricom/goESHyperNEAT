package eshyperneat

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/neat/utils"
	"os"
	"testing"
)

func TestESHyperNEATContext_LoadFullContext(t *testing.T) {
	r, err := os.Open("../data/test/test_es_hyper.neat.yml")
	require.NoError(t, err, "failed to open config file")

	esCtx, err := Load(r)
	require.NoError(t, err, "failed to load context")

	// check loaded values
	assert.Equal(t, 3, esCtx.InitialDepth)
	assert.Equal(t, 5, esCtx.MaximalDepth)
	assert.Equal(t, 0.01, esCtx.DivisionThreshold)
	assert.Equal(t, 0.03, esCtx.VarianceThreshold)
	assert.Equal(t, 0.3, esCtx.BandingThreshold)
	assert.Equal(t, 1, esCtx.ESIterations)

	assert.Equal(t, utils.SigmoidSteepenedActivation, esCtx.HyperNEATContext.SubstrateActivator)
}
