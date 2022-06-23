// Package eshyperneat holds implementation of Evolvable-Substrate HyperNEAT context
package eshyperneat

import (
	"github.com/pkg/errors"
	"github.com/yaricom/goESHyperNEAT/v2/hyperneat"
	"gopkg.in/yaml.v3"
	"io"
	"io/ioutil"
	"os"
)

// Options ES-HyperNEAT execution options
type Options struct {
	// The included HyperNEAT options
	*hyperneat.Options `yaml:",inline"`

	// InitialDepth defines the initial ES-HyperNEAT sample resolution.
	InitialDepth int `yaml:"initial_depth"`
	// Maximal ES-HyperNEAT sample resolution if the variance is still higher than the given division threshold
	MaximalDepth int `yaml:"maximal_depth"`

	// DivisionThreshold defines the division threshold. If the variance in a region is greater than this value, after
	// the initial resolution is reached, ES-HyperNEAT will sample down further (values greater than 1.0 will disable
	// this feature). Note that sampling at really high resolutions can become computationally expensive.
	DivisionThreshold float64 `yaml:"division_threshold"`
	// VarianceThreshold defines the variance threshold for the initial sampling. The bigger this value the less new
	// connections will be added directly and the more chances that the new collection will be included in bands
	// (see BandingThreshold)
	VarianceThreshold float64 `yaml:"variance_threshold"`
	// BandingThreshold defines the threshold that determines when points are regarded to be in a band. If the point
	// is in the band then no new connection will be added and as result no new hidden node will be introduced.
	// The bigger this value the fewer connections/hidden nodes will be added, i.e. wide bands approximation.
	BandingThreshold float64 `yaml:"banding_threshold"`

	// ESIterations defines how many times ES-HyperNEAT should iteratively discover new hidden nodes.
	ESIterations int `yaml:"es_iterations"`
}

// LoadYAMLOptions is to load ES-HyperNEAT options from provided reader
func LoadYAMLOptions(r io.Reader) (*Options, error) {
	content, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}
	// read options
	var opts Options
	if err = yaml.Unmarshal(content, &opts); err != nil {
		return nil, errors.Wrap(err, "failed to decode ES-HyperNEAT options from YAML")
	}
	return &opts, nil
}

// LoadYAMLConfigFile is to load ES-HyperNEAT options from provided configuration file
func LoadYAMLConfigFile(path string) (*Options, error) {
	configFile, err := os.Open(path)
	if err != nil {
		return nil, errors.Wrap(err, "failed to open ES-HyperNEAT configuration file")
	}
	return LoadYAMLOptions(configFile)
}
