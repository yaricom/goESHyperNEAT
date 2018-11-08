package hyperneat

import (
	"io"
	"bytes"
	"github.com/spf13/viper"
	"errors"
)

// ES-HyperNEAT execution context
type ESHyperNEATContext struct {
	// The included HyperNEAT context
	HyperNEAT         *HyperNEATContext

	// InitialDepth defines the initial ES-HyperNEAT sample resolution.
	InitialDepth      int
	// Maximal ES-HyperNEAT sample resolution if the variance is still higher than the given division threshold
	MaximalDepth      int

	// DivisionThreshold defines the division threshold. If the variance in a region is greater than this value, after
	// the initial resolution is reached, ES-HyperNEAT will sample down further (values greater than 1.0 will disable
	// this feature). Note that sampling at really high resolutions can become computationally expensive.
	DivisionThreshold float64
	// VarianceThreshold defines the variance threshold for the initial sampling.
	VarianceThreshold float64
	// BandingThreshold defines the threshold that determines when points are regarded to be in a band.
	BandingThreshold  float64

	// ESIterations defines how many times ES-HyperNEAT should iterativelty discover new hidden nodes.
	ESIterations      int
}

// Loads context from provided reader
func (e *ESHyperNEATContext) LoadContext(r io.Reader) error {
	viper.SetConfigType("YAML")
	err := viper.ReadConfig(r)
	if err != nil {
		return err
	}
	v := viper.Sub("es-hyperneat")
	if v == nil {
		return errors.New("es-hyperneat subsection not found in configuration")
	}

	e.InitialDepth = v.GetInt("initial_depth")
	e.MaximalDepth = v.GetInt("maximal_depth")

	e.DivisionThreshold = v.GetFloat64("division_threshold")
	e.VarianceThreshold = v.GetFloat64("variance_threshold")
	e.BandingThreshold = v.GetFloat64("banding_threshold")

	e.ESIterations = v.GetInt("es_iterations")

	return nil
}

// Loads this context and all included contexts data
func (e *ESHyperNEATContext) LoadFullContext(r io.Reader) error {
	var buff bytes.Buffer
	tee := io.TeeReader(r, &buff)

	// load included HyperNEAT context
	e.HyperNEAT = &HyperNEATContext{}
	err := e.HyperNEAT.LoadFullContext(tee)
	if err != nil {
		return err
	}

	// load this ES-HyperNEAT context
	err = e.LoadContext(&buff)
	return err
}
