// The package hyperneat holds implementation of HyperNEAT family of algorithms, including Evolvable-Substrate HyperNEAT
package hyperneat

import (
	"github.com/yaricom/goNEAT/neat"
	"github.com/yaricom/goNEAT/neat/network"
	"io"
	"github.com/spf13/viper"
	"strings"
	"strconv"
)

// The HyperNEAT execution context
type HyperNEATContext struct {
	// The NEAT context included
	NeatContext            *neat.NeatContext

	// The threshold value to indicate which links should be included
	LinkThershold          float64
	// The weight range defines the minimum and maximum values for weights on substrate connections, they go from -WeightRange to +WeightRange, and can be any integer
	WeightRange            int

	// The CPPN neuron nodes activation functions list to choose from
	CPPNNodeActivators     []network.NodeActivationType
	// The probabilities of CPPN node activator function selection
	CPPNNodeActivatorsProb []float64

	// The substrate activation function
	SubstrateActivator     network.NodeActivationType
}

func (h *HyperNEATContext) LoadContext(r io.Reader) error {
	viper.SetConfigType("YAML")
	err := viper.ReadConfig(r)
	if err != nil {
		return err
	}
	h.LinkThershold = viper.GetFloat64("link_threshold")
	h.WeightRange = viper.GetInt("weight_range")

	// read substrate activator
	subAct := viper.GetString("substrate_activator")
	h.SubstrateActivator = network.ActivationTypeFromName(subAct)

	// read activation functions list
	actFns := viper.GetStringSlice("cppn_activators")
	h.CPPNNodeActivators = make([]network.NodeActivationType, len(actFns))
	h.CPPNNodeActivatorsProb = make([]float64, len(actFns))
	for i, line := range actFns {
		fields := strings.Fields(line)
		h.CPPNNodeActivators[i] = network.ActivationTypeFromName(fields[0])
		prob, err := strconv.ParseFloat(fields[1], 64)
		if err != nil {
			return err
		}
		h.CPPNNodeActivatorsProb[i] = prob
	}

	return nil
}