// Package hyperneat holds implementation of HyperNEAT family of algorithms
package hyperneat

import (
	"github.com/pkg/errors"
	"github.com/yaricom/goNEAT/v3/neat/math"
	"gopkg.in/yaml.v3"
	"io"
	"io/ioutil"
	"os"
)

type SubstrateActivatorType struct {
	SubstrateActivationType math.NodeActivationType
}

// Options The HyperNEAT execution options
type Options struct {
	// The threshold value to indicate which links should be included
	LinkThreshold float64 `yaml:"link_threshold"`
	// The weight range defines the minimum and maximum values for weights on substrate connections, they go
	// from -WeightRange to +WeightRange, and can be any integer
	WeightRange float64 `yaml:"weight_range"`

	// The substrate activation function
	SubstrateActivator SubstrateActivatorType `yaml:"substrate_activator"`
	// The BIAS value for CPPN network
	CppnBias float64 `yaml:"cppn_bias,omitempty"`
}

// LoadYAMLOptions is to read HyperNEAT options from the provided reader
func LoadYAMLOptions(r io.Reader) (*Options, error) {
	content, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}
	// read options
	var opts Options
	if err = yaml.Unmarshal(content, &opts); err != nil {
		return nil, errors.Wrap(err, "failed to decode HyperNEAT options from YAML")
	}
	return &opts, nil
}

// LoadYAMLConfigFile is to load ES-HyperNEAT options from provided configuration file
func LoadYAMLConfigFile(path string) (*Options, error) {
	configFile, err := os.Open(path)
	if err != nil {
		return nil, errors.Wrap(err, "failed to open HyperNEAT configuration file")
	}
	return LoadYAMLOptions(configFile)
}

func (s *SubstrateActivatorType) UnmarshalYAML(value *yaml.Node) error {
	if activationType, err := math.NodeActivators.ActivationTypeFromName(value.Value); err != nil {
		return errors.Wrap(err, "failed to decode substrate activator function from HyperNEAT options")
	} else {
		s.SubstrateActivationType = activationType
	}
	return nil
}
