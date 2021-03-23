// The package CPPN provides implementation of Compositional Pattern Producing Network
// which is a part of Hypercube-based NEAT algorithm implementation
package cppn

import (
	"errors"
	"github.com/yaricom/goNEAT/neat/genetics"
	"github.com/yaricom/goNEAT/neat/network"
	"math"
	"os"
)

// Reads CPPN from specified genome and creates network solver
func ReadCPPFromGenomeFile(genomePath string) (network.NetworkSolver, error) {
	if genomeFile, err := os.Open(genomePath); err != nil {
		return nil, err
	} else if r, err := genetics.NewGenomeReader(genomeFile, genetics.YAMLGenomeEncoding); err != nil {
		return nil, err
	} else if genome, err := r.Read(); err != nil {
		return nil, err
	} else if netw, err := genome.Genesis(genome.Id); err != nil {
		return nil, err
	} else {
		return netw.FastNetworkSolver()
	}
}

// Creates normalized by threshold value link between source and target nodes, given calculated CPPN output for their coordinates
func createThresholdNormalizedLink(cppnOutput float64, srcIndex, dstIndex int, linkThreshold, weightRange float64) *network.FastNetworkLink {
	weight := (math.Abs(cppnOutput) - linkThreshold) / (1 - linkThreshold) // normalize [0, 1]
	weight *= weightRange                                                  // scale to fit given weight range
	if math.Signbit(cppnOutput) {
		weight *= -1 // restore sign
	}
	link := network.FastNetworkLink{
		Weight:     weight,
		SourceIndx: srcIndex,
		TargetIndx: dstIndex,
	}
	return &link
}

// Creates link between source and target nodes, given calculated CPPN output for their coordinates
func createLink(cppnOutput float64, srcIndex, dstIndex int, weightRange float64) *network.FastNetworkLink {
	weight := cppnOutput
	weight *= weightRange // scale to fit given weight range
	link := network.FastNetworkLink{
		Weight:     weight,
		SourceIndx: srcIndex,
		TargetIndx: dstIndex,
	}
	return &link
}

// Calculates outputs of provided CPPN network solver with given hypercube coordinates.
func queryCPPN(coordinates []float64, cppn network.NetworkSolver) ([]float64, error) {
	// flush networks activation from previous run
	if res, err := cppn.Flush(); err != nil {
		return nil, err
	} else if !res {
		return nil, errors.New("failed to flush CPPN network")
	}
	// load inputs
	if err := cppn.LoadSensors(coordinates); err != nil {
		return nil, err
	}
	// do activations
	if res, err := cppn.RecursiveSteps(); err != nil {
		return nil, err
	} else if !res {
		return nil, errors.New("failed to relax CPPN network recursively")
	}

	return cppn.ReadOutputs(), nil
}

// Determines variance among CPPN values for certain hypercube region around specified node.
// This variance is a heuristic indicator of the heterogeneity (i.e. presence of information) of a region.
func nodeVariance(node *QuadNode) float64 {
	// quick check
	if len(node.Nodes) == 0 {
		return 0.0
	}

	cppnVals := nodeCPPNValues(node)
	// calculate median and variance
	m, v := 0.0, 0.0
	for _, f := range cppnVals {
		m += f
	}
	m /= float64(len(cppnVals))

	for _, f := range cppnVals {
		v += math.Pow(f-m, 2)
	}
	v /= float64(len(cppnVals))

	return v
}

// Collects the CPPN values stored in a given quadtree node
// Used to estimate the variance in a certain region of space around node
func nodeCPPNValues(n *QuadNode) []float64 {
	if len(n.Nodes) > 0 {
		accumulator := make([]float64, 0)
		for _, p := range n.Nodes {
			// go into child nodes
			pVals := nodeCPPNValues(p)
			accumulator = append(accumulator, pVals...)
		}
		return accumulator
	} else {
		return []float64{n.W}
	}
}
