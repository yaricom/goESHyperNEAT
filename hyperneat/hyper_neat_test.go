package hyperneat

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/neat/utils"
	"os"
	"testing"
)

func TestHyperNEATContext_LoadContext(t *testing.T) {
	r, err := os.Open("../data/test_es_hyper.neat.yml")
	require.NoError(t, err, "failed to open config file")

	ctx, err := Load(r)
	require.NoError(t, err, "failed to load context")

	// check values
	assert.Equal(t, utils.SigmoidSteepenedActivation, ctx.SubstrateActivator)
	assert.Equal(t, 0.2, ctx.LinkThreshold)
	assert.Equal(t, 3.0, ctx.WeightRange)
}
