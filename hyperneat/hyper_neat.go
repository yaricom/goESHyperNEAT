// The package hyperneat holds implementation of HyperNEAT family of algorithms, including Evolvable-Substrate HyperNEAT
package hyperneat

import (
	"io"
	"errors"
	"bytes"

	"github.com/spf13/viper"
	"github.com/yaricom/goNEAT/neat"
	"github.com/yaricom/goNEAT/neat/utils"
)

// The HyperNEAT execution context
type HyperNEATContext struct {
	// The NEAT context included
	NeatContext        *neat.NeatContext

	// The threshold value to indicate which links should be included
	LinkThershold      float64
	// The weight range defines the minimum and maximum values for weights on substrate connections, they go from -WeightRange to +WeightRange, and can be any integer
	WeightRange        float64

	// The substrate activation function
	SubstrateActivator utils.NodeActivationType
}

// Loads context from provided configuration data
func (h *HyperNEATContext) LoadContext(r io.Reader) error {
	viper.SetConfigType("YAML")
	err := viper.ReadConfig(r)
	if err != nil {
		return err
	}
	v := viper.Sub("hyperneat")
	if v == nil {
		return errors.New("hyperneat subsection not found in configuration")
	}

	h.LinkThershold = v.GetFloat64("link_threshold")
	h.WeightRange = v.GetFloat64("weight_range")

	// read substrate activator
	subAct := v.GetString("substrate_activator")
	if h.SubstrateActivator, err = utils.NodeActivators.ActivationTypeFromName(subAct); err != nil {
		return err
	}

	return nil
}

func (e *HyperNEATContext) LoadFullContext(r io.Reader) error {
	var buff bytes.Buffer
	tee := io.TeeReader(r, &buff)

	// NEAT context loading
	e.NeatContext = &neat.NeatContext{}
	err := e.NeatContext.LoadContext(tee)
	if err != nil {
		return err
	}

	err = e.LoadContext(&buff)
	return err
}