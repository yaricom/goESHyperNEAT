// Package cppn provides implementation of Compositional Pattern Producing Network
// which is a part of Hypercube-based NEAT algorithm implementation
package cppn

import (
	"errors"
	"github.com/yaricom/goNEAT/v4/neat/genetics"
	"github.com/yaricom/goNEAT/v4/neat/network"
	"math"
)

// FastSolverFromGenomeFile Reads CPPN from specified genome and creates network solver
func FastSolverFromGenomeFile(genomePath string) (network.Solver, error) {
	if reader, err := genetics.NewGenomeReaderFromFile(genomePath); err != nil {
		return nil, err
	} else if genome, err := reader.Read(); err != nil {
		return nil, err
	} else if net, err := genome.Genesis(genome.Id); err != nil {
		return nil, err
	} else {
		return net.FastNetworkSolver()
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
		Weight:      weight,
		SourceIndex: srcIndex,
		TargetIndex: dstIndex,
	}
	return &link
}

// Creates link between source and target nodes, given calculated CPPN output for their coordinates
func createLink(cppnOutput float64, srcIndex, dstIndex int, weightRange float64) *network.FastNetworkLink {
	weight := cppnOutput
	weight *= weightRange // scale to fit given weight range
	link := network.FastNetworkLink{
		Weight:      weight,
		SourceIndex: srcIndex,
		TargetIndex: dstIndex,
	}
	return &link
}

// Calculates outputs of provided CPPN network solver with given hypercube coordinates.
func queryCPPN(coordinates []float64, cppn network.Solver) ([]float64, error) {
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

	cppnValues := nodeCPPNValues(node)
	// calculate median and variance
	meanW, variance := 0.0, 0.0
	for _, w := range cppnValues {
		meanW += w
	}
	meanW /= float64(len(cppnValues))

	for _, w := range cppnValues {
		variance += math.Pow(w-meanW, 2)
	}
	variance /= float64(len(cppnValues))

	return variance
}

// Collects the CPPN values stored in a given quadtree node
// Used to estimate the variance in a certain region of space around node
func nodeCPPNValues(n *QuadNode) []float64 {
	if len(n.Nodes) > 0 {
		accumulator := make([]float64, 0)
		for _, p := range n.Nodes {
			// go into child nodes
			cppnValues := nodeCPPNValues(p)
			accumulator = append(accumulator, cppnValues...)
		}
		return accumulator
	} else {
		return []float64{n.Weight()}
	}
}
