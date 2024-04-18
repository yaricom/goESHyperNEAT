// Package retina provides implementation of the retina experiment
package retina

import (
	"context"
	"fmt"
	"github.com/pkg/errors"
	"github.com/yaricom/goESHyperNEAT/v2/cppn"
	"github.com/yaricom/goESHyperNEAT/v2/eshyperneat"
	"github.com/yaricom/goESHyperNEAT/v2/examples"
	"github.com/yaricom/goNEAT/v4/experiment"
	"github.com/yaricom/goNEAT/v4/experiment/utils"
	"github.com/yaricom/goNEAT/v4/neat"
	"github.com/yaricom/goNEAT/v4/neat/genetics"
	"github.com/yaricom/goNEAT/v4/neat/network"
	"math"
	"os"
	"time"
)

const (
	// maxFitness Used as max value which we add error too to get an organism's fitness
	maxFitness = 1.0
	// fitnessThreshold is the fitness value for which an organism is considered to have won the experiment
	fitnessThreshold = maxFitness

	debug = true
)

type generationEvaluator struct {
	outDir string
	env    *Environment

	// The target number of species to be maintained
	numSpeciesTarget int
	// The species compatibility threshold adjustment frequency
	compatAdjustFreq int
}

// NewGenerationEvaluator is to create new generation's evaluator for retina experiment.  The numSpeciesTarget specifies the
// target number of species to maintain in the population. If the number of species differ from the numSpeciesTarget it
// will be automatically adjusted with compatAdjustFreq frequency, i.e., at each epoch % compatAdjustFreq == 0
func NewGenerationEvaluator(outDir string, env *Environment, numSpeciesTarget, compatAdjustFreq int) (experiment.GenerationEvaluator, experiment.TrialRunObserver) {
	evaluator := &generationEvaluator{
		outDir:           outDir,
		env:              env,
		numSpeciesTarget: numSpeciesTarget,
		compatAdjustFreq: compatAdjustFreq,
	}
	return evaluator, evaluator
}

// TrialRunStarted invoked to notify that new trial run just started. Invoked before any epoch evaluation in that trial run
func (e *generationEvaluator) TrialRunStarted(_ *experiment.Trial) {
	// just stub
}

// TrialRunFinished invoked to notify that the trial run just finished. Invoked after all epochs evaluated or successful solver found.
func (e *generationEvaluator) TrialRunFinished(_ *experiment.Trial) {
	// just stub
}

// EpochEvaluated invoked to notify that evaluation of specific epoch completed.
func (e *generationEvaluator) EpochEvaluated(_ *experiment.Trial, _ *experiment.Generation) {
	// just stub
}

// GenerationEvaluate evaluates a population of organisms and prints their performance on the retina experiment
func (e *generationEvaluator) GenerationEvaluate(ctx context.Context, population *genetics.Population, epoch *experiment.Generation) error {
	options, ok := neat.FromContext(ctx)
	if !ok {
		return neat.ErrNEATOptionsNotFound
	}
	// Evaluate each organism on a test
	var (
		maxPopulationFitness = 0.0
		bestLinkCount        = 0
		bestNodeCount        = 0
	)
	var bestSubstrateSolver network.Solver

	startTime := time.Now()
	for _, organism := range population.Organisms {
		isWinner, solver, err := e.organismEvaluate(ctx, organism)
		if err != nil {
			return err
		}

		if organism.Fitness > maxPopulationFitness {
			maxPopulationFitness = organism.Fitness
			if phenotype, err := organism.Phenotype(); err == nil {
				bestLinkCount = phenotype.LinkCount()
				bestNodeCount = phenotype.NodeCount()
			} else {
				neat.ErrorLog(fmt.Sprintf("Failed to get organism Phenotype, reason: %s", err))
				return err
			}
			bestSubstrateSolver = solver
		}

		if isWinner && (epoch.Champion == nil || organism.Fitness > epoch.Champion.Fitness) {
			epoch.Solved = true
			epoch.WinnerNodes = len(organism.Genotype.Nodes)
			epoch.WinnerGenes = organism.Genotype.Extrons()
			epoch.WinnerEvals = options.PopSize*epoch.Id + organism.Genotype.Id
			epoch.Champion = organism
			if epoch.WinnerNodes == 9 {
				// You could dump out optimal genomes here if desired
				if optPath, err := utils.WriteGenomePlain("retina_optimal", e.outDir, organism, epoch); err != nil {
					neat.ErrorLog(fmt.Sprintf("Failed to dump optimal genome, reason: %s\n", err))
				} else {
					neat.InfoLog(fmt.Sprintf("Dumped optimal genome to: %s\n", optPath))
				}
			}
		}
	}
	elapsedTime := time.Now().Sub(startTime)

	// Fill statistics about current epoch
	epoch.FillPopulationStatistics(population)

	// Only print to file every print_every generation
	if epoch.Solved || epoch.Id%options.PrintEvery == 0 {
		if _, err := utils.WritePopulationPlain(e.outDir, population, epoch); err != nil {
			neat.ErrorLog(fmt.Sprintf("Failed to dump population, reason: %s\n", err))
			return err
		}
	}

	if epoch.Solved {
		// print winner organism
		org := epoch.Champion
		utils.PrintActivationDepth(org, true)

		genomeFile := "retina_cppn_winner"
		// Prints the winner organism's Genome to the file!
		if orgPath, err := utils.WriteGenomePlain(genomeFile, e.outDir, org, epoch); err != nil {
			neat.ErrorLog(fmt.Sprintf("Failed to dump winner organism's genome, reason: %s\n", err))
		} else {
			neat.InfoLog(fmt.Sprintf("Generation #%d winner's genome dumped to: %s\n", epoch.Id, orgPath))
		}

		// Dump the winner substrate graph
		//
		graph := org.Data.Value.(cppn.SubstrateGraphBuilder)
		nodes, _ := graph.NodesCount()
		edges, _ := graph.EdgesCount()
		substrPath := fmt.Sprintf("%s/%s_%d-%d.xml", utils.CreateOutDirForTrial(e.outDir, epoch.TrialId),
			"retina_substrate_graph_winner", nodes, edges)
		if file, err := os.Create(substrPath); err != nil {
			neat.ErrorLog(err.Error())
		} else if err = graph.Marshal(file); err != nil {
			neat.ErrorLog(fmt.Sprintf("Failed to dump winner substrate, reason: %s\n", err))
		} else {
			neat.InfoLog(fmt.Sprintf("Generation #%d winner's substrate dumped to: %s\n", epoch.Id, substrPath))
		}
	} else if epoch.Id < options.NumGenerations-1 {
		speciesCount := len(population.Species)

		// adjust species count by keeping it constant
		examples.AdjustSpeciesNumber(speciesCount, epoch.Id, e.compatAdjustFreq, e.numSpeciesTarget, options)

		bestSolverLinks, bestSolverNodes := -1, -1
		if bestSubstrateSolver != nil {
			bestSolverLinks, bestSolverNodes = bestSubstrateSolver.LinkCount(), bestSubstrateSolver.NodeCount()
		}

		neat.InfoLog(
			fmt.Sprintf("%d species -> %d organisms [compatibility threshold: %.1f, target: %d]\nbest CPNN organism [fitness: %.2f, links: %d, nodes: %d], best solver [links: %d, nodes: %d], population evaluation time: %v",
				speciesCount, len(population.Organisms), options.CompatThreshold, e.numSpeciesTarget,
				maxPopulationFitness, bestLinkCount, bestNodeCount, bestSolverLinks, bestSolverNodes, elapsedTime))
	}
	return nil
}

// organismEvaluate evaluates an individual phenotype network with retina experiment and returns true if it's a winner
func (e *generationEvaluator) organismEvaluate(ctx context.Context, organism *genetics.Organism) (bool, network.Solver, error) {
	options, ok := eshyperneat.FromContext(ctx)
	if !ok {
		return false, nil, eshyperneat.ErrESHyperNEATOptionsNotFound
	}
	// get CPPN network solver
	cppnSolver, err := organism.Phenotype()
	if err != nil {
		return false, nil, errors.Wrap(err, "failed to create CPPN solver")
	}

	// create substrate layout
	inputCount := e.env.inputSize * 2 // left + right pixels of visual object
	layout, err := cppn.NewMappedEvolvableSubstrateLayout(inputCount, 2)
	if err != nil {
		return false, nil, err
	}
	// create ES-HyperNEAT solver
	substr := cppn.NewEvolvableSubstrateWithBias(layout, options.SubstrateActivator.SubstrateActivationType, options.CppnBias)
	graph := cppn.NewSubstrateGraphMLBuilder("retina ES-HyperNEAT", false)
	createSolverTime := time.Now()
	solver, err := substr.CreateNetworkSolver(cppnSolver, graph, options)
	if err != nil {
		return false, nil, errors.Wrap(err, fmt.Sprintf("failed to evaluate organism: %s", organism))
	}
	createSolverElapsedTime := time.Now().Sub(createSolverTime)

	// Evaluate the detector ANN against 256 combinations of the left and the right visual objects
	// at correct and incorrect sides of retina
	startTime := time.Now()
	errorSum, count, detectionErrorCount := 0.0, 0.0, 0.0
	for _, leftObj := range e.env.visualObjects {
		for _, rightObj := range e.env.visualObjects {
			// Evaluate outputted predictions
			loss, err := evaluateNetwork(solver, leftObj, rightObj)
			if err != nil {
				return false, nil, err
			}
			errorSum += loss
			count += 1.0
			if loss > 0 {
				detectionErrorCount += 1.0
			}
			// flush solver
			if flushed, err := solver.Flush(); err != nil {
				return false, nil, err
			} else if !flushed {
				return false, nil, errors.New("failed to flush solver after evaluation")
			}
		}
	}
	elapsed := time.Since(startTime)

	// Calculate the fitness score
	fitness := maxFitness / (1.0 + errorSum)
	avgError := errorSum / count

	isWinner := false
	if fitness >= fitnessThreshold {
		isWinner = true
		fmt.Printf("Found a Winner! \n")
		// save solver graph to the winner organism
		organism.Data = &genetics.OrganismData{Value: graph}
	}
	// Save properties to organism struct
	organism.IsWinner = isWinner
	organism.Error = avgError
	organism.Fitness = fitness

	if debug {
		neat.InfoLog(fmt.Sprintf("Average error: %f, errors sum: %f, false detections: %f from: %f",
			avgError, errorSum, detectionErrorCount, count))
		neat.InfoLog(fmt.Sprintf("Substrate: nodes = %d, edges = %d | CPPN phenotype: nodes = %d, edges = %d",
			solver.NodeCount(), solver.LinkCount(), cppnSolver.NodeCount(), cppnSolver.LinkCount()))
		neat.InfoLog(fmt.Sprintf("Substrate: evaluation time = %v, create solver time = %v", elapsed, createSolverElapsedTime))
	}

	return isWinner, solver, nil
}

// evaluateNetwork is to evaluate provided network solver using provided visual objects to test prediction performance.
// Returns the prediction loss value or error if failed to evaluate.
func evaluateNetwork(solver network.Solver, leftObj VisualObject, rightObj VisualObject) (float64, error) {
	// flush current network state
	if _, err := solver.Flush(); err != nil {
		return -1, err
	}

	// Create input by joining data from left and right visual objects
	inputs := append(leftObj.data, rightObj.data...)

	// run evaluation
	loss := math.MaxFloat64
	if err := solver.LoadSensors(inputs); err != nil {
		return loss, err
	}

	// Propagate activation
	if relaxed, err := solver.RecursiveSteps(); err != nil {
		return loss, err
	} else if !relaxed {
		return loss, errors.New("failed to relax network solver of the ES substrate")
	}

	// get outputs and evaluate against ground truth
	outs := solver.ReadOutputs()
	loss = evaluatePredictions(outs, leftObj, rightObj)
	return loss, nil
}

// evaluatePredictions returns the loss between predictions and ground truth of leftObj and rightObj
func evaluatePredictions(predictions []float64, leftObj VisualObject, rightObj VisualObject) float64 {
	// Convert predictions[i] to 1.0 or 0.0 about 0.5 threshold
	normPredictions := make([]float64, len(predictions))
	for i := 0; i < len(normPredictions); i++ {
		if predictions[i] >= 0.5 {
			normPredictions[i] = 1.0
		} else {
			predictions[i] = 0.0
		}
	}

	// Get ground truth values
	targets := make([]float64, 2)

	// Set target[0] to 1.0 if LeftObj is suitable for Left side, otherwise set to 0.0
	if leftObj.Side == LeftSide || leftObj.Side == BothSide {
		targets[0] = 1.0
	} else {
		targets[0] = 0.0
	}

	// Repeat for target[1], the right side truth value
	if rightObj.Side == RightSide || rightObj.Side == BothSide {
		targets[1] = 1.0
	} else {
		targets[1] = 0.0
	}

	// Find normalized loss
	normLoss := (normPredictions[0]-targets[0])*(normPredictions[0]-targets[0]) + (normPredictions[1]-targets[1])*(normPredictions[1]-targets[1])
	if normLoss == 0 {
		return 0.0
	}

	// find loss
	loss := histDiff(predictions, targets)

	neat.DebugLog(fmt.Sprintf("[%.2f, %.2f] -> [%.2f, %.2f] loss: %.2f",
		targets[0], targets[1], normPredictions[0], normPredictions[1], normLoss))

	return loss

}

// calculates item-wise difference between two vectors
func histDiff(left, right []float64) float64 {
	size := len(left)
	diffAccum := 0.0
	for i := 0; i < size; i++ {
		diff := left[i] - right[i]
		diffAccum += math.Abs(diff)
	}
	return diffAccum / float64(size)
}
