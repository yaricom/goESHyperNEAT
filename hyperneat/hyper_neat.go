// The package hyperneat holds implementation of HyperNEAT family of algorithms, including Evolvable-Substrate HyperNEAT
package hyperneat

import (
	"bytes"
	"errors"
	"io"

	"github.com/spf13/viper"
	"github.com/yaricom/goNEAT/neat"
	"github.com/yaricom/goNEAT/neat/utils"
)

// The HyperNEAT execution context
type HyperNEATContext struct {
	// The NEAT context included
	*neat.NeatContext

	// The threshold value to indicate which links should be included
	LinkThreshold float64
	// The weight range defines the minimum and maximum values for weights on substrate connections, they go
	// from -WeightRange to +WeightRange, and can be any integer
	WeightRange float64

	// The substrate activation function
	SubstrateActivator utils.NodeActivationType
}

// Load is to read HyperNEAT context options from the provided reader
func Load(r io.Reader) (*HyperNEATContext, error) {
	var buff bytes.Buffer
	tee := io.TeeReader(r, &buff)

	// NEAT context loading
	nCtx := &neat.NeatContext{}
	if err := nCtx.LoadContext(tee); err != nil {
		return nil, err
	}

	// Load HyperNEAT options
	ctx := &HyperNEATContext{NeatContext: nCtx}
	if err := ctx.load(&buff); err != nil {
		return nil, err
	}
	return ctx, nil
}

// Loads only HyperNEAT context from provided configuration data
func (h *HyperNEATContext) load(r io.Reader) error {
	viper.SetConfigType("YAML")
	err := viper.ReadConfig(r)
	if err != nil {
		return err
	}
	v := viper.Sub("hyperneat")
	if v == nil {
		return errors.New("hyperneat subsection not found in configuration")
	}

	h.LinkThreshold = v.GetFloat64("link_threshold")
	h.WeightRange = v.GetFloat64("weight_range")

	// read substrate activator
	subAct := v.GetString("substrate_activator")
	if h.SubstrateActivator, err = utils.NodeActivators.ActivationTypeFromName(subAct); err != nil {
		return err
	}

	return nil
}
