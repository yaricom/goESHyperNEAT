// Package retina provides implementation of the retina experiment
package retina

import (
	"errors"
	"fmt"
	"github.com/yaricom/goESHyperNEAT/v2/cppn"
	"github.com/yaricom/goNEAT/experiments"
	"github.com/yaricom/goNEAT/neat"
	"github.com/yaricom/goNEAT/neat/genetics"
	"github.com/yaricom/goNEAT/neat/network"
	"math"
	"os"
)

const (
	// maxFitness Used as max value which we add error too to get an organism's fitness
	maxFitness = 1000.0
	// fitnessThreshold is the fitness value for which an organism is considered to have won the experiment
	fitnessThreshold = maxFitness
)

type generationEvaluator struct {
	outDir string
	env    *Environment
}

// NewGenerationEvaluator is to create new generations evaluator for retina experiment.
func NewGenerationEvaluator(outDir string, env *Environment) experiments.GenerationEvaluator {
	return &generationEvaluator{
		outDir: outDir,
		env:    env,
	}
}

// GenerationEvaluate evaluates a population of organisms and prints their performance on the retina experiment
func (e *generationEvaluator) GenerationEvaluate(population *genetics.Population, epoch *experiments.Generation, context *neat.NeatContext) (err error) {
	// Evaluate each organism on a test
	for idx, organism := range population.Organisms {
		isWinner, err := e.organismEvaluate(organism)
		if err != nil {
			return err
		}
		neat.InfoLog(fmt.Sprintf("organism #%d fitness %f \n", idx, organism.Fitness))

		if isWinner && (epoch.Best == nil || organism.Fitness > epoch.Best.Fitness) {
			epoch.Solved = true
			epoch.WinnerNodes = len(organism.Genotype.Nodes)
			epoch.WinnerGenes = organism.Genotype.Extrons()
			epoch.WinnerEvals = context.PopSize*epoch.Id + organism.Genotype.Id
			epoch.Best = organism
			if epoch.WinnerNodes == 9 {
				// You could dump out optimal genomes here if desired
				optPath := fmt.Sprintf("%s/%s_%d-%d", experiments.OutDirForTrial(e.outDir, epoch.TrialId),
					"retina_optimal", organism.Phenotype.NodeCount(), organism.Phenotype.LinkCount())
				file, err := os.Create(optPath)
				if err != nil {
					neat.ErrorLog(fmt.Sprintf("Failed to dump optimal genome, reason: %s\n", err))
				} else if err = organism.Genotype.Write(file); err != nil {
					neat.ErrorLog("Failed to save optimal genotype")
				} else {
					neat.InfoLog(fmt.Sprintf("Dumped optimal genome to: %s\n", optPath))
				}
			}
		}
	}

	// Fill statistics about current epoch
	epoch.FillPopulationStatistics(population)

	// Only print to file every print_every generations
	if epoch.Solved || epoch.Id%context.PrintEvery == 0 {
		popPath := fmt.Sprintf("%s/gen_%d", experiments.OutDirForTrial(e.outDir, epoch.TrialId), epoch.Id)
		file, err := os.Create(popPath)
		if err != nil {
			neat.ErrorLog(fmt.Sprintf("Failed to dump population, reason: %s\n", err))
		} else {
			population.WriteBySpecies(file)
		}
	}

	if epoch.Solved {
		// print winner organism
		for _, org := range population.Organisms {
			if org.IsWinner {
				// Prints the winner CPPN organism to file!
				//
				orgPath := fmt.Sprintf("%s/%s_%d-%d", experiments.OutDirForTrial(e.outDir, epoch.TrialId),
					"retina_cppn_winner", org.Phenotype.NodeCount(), org.Phenotype.LinkCount())
				if file, err := os.Create(orgPath); err != nil {
					neat.ErrorLog(err.Error())
				} else if err = org.Genotype.Write(file); err != nil {
					neat.ErrorLog(fmt.Sprintf("Failed to dump winner CPPN genotype, reason: %s\n", err))
				} else {
					neat.InfoLog(fmt.Sprintf("Generation #%d winner CPPN dumped to: %s\n", epoch.Id, orgPath))
				}
				// Dump the winner substrate graph
				//
				graph := org.Data.Value.(cppn.SubstrateGraphBuilder)
				nodes, _ := graph.NodesCount()
				edges, _ := graph.EdgesCount()
				substrPath := fmt.Sprintf("%s/%s_%d-%d.xml", experiments.OutDirForTrial(e.outDir, epoch.TrialId),
					"retina_substrate_graph_winner", nodes, edges)
				if file, err := os.Create(substrPath); err != nil {
					neat.ErrorLog(err.Error())
				} else if err = graph.Marshal(file); err != nil {
					neat.ErrorLog(fmt.Sprintf("Failed to dump winner substrate, reason: %s\n", err))
				} else {
					neat.InfoLog(fmt.Sprintf("Generation #%d winner's substrate dumped to: %s\n", epoch.Id, substrPath))
				}
				break
			}
		}
	}
	return err
}

// organismEvaluate evaluates an individual phenotype network with retina experiment and returns true if its won
func (e generationEvaluator) organismEvaluate(organism *genetics.Organism) (bool, error) {
	// get CPPN network solver
	cppnSolver, err := organism.Phenotype.FastNetworkSolver()
	if err != nil {
		return false, err
	}

	// create substrate layout
	inputCount := e.env.inputSize * 2 // left + right pixels of visual object
	layout, err := cppn.NewMappedEvolvableSubstrateLayout(inputCount, 2)
	if err != nil {
		return false, err
	}
	// create ES-HyperNEAT solver
	substr := cppn.NewEvolvableSubstrate(layout, e.env.context.SubstrateActivator)
	graph := cppn.NewSubstrateGraphMLBuilder("retina ES-HyperNEAT", false)
	solver, err := substr.CreateNetworkSolver(cppnSolver, graph, e.env.context)
	if err != nil {
		return false, err
	}

	// Evaluate the detector ANN against 256 combinations of the left and the right visual objects
	// at correct and incorrect sides of retina
	errorSum, count, detectionErrorCount := 0.0, 0.0, 0.0
	for _, leftObj := range e.env.visualObjects {
		for _, rightObj := range e.env.visualObjects {
			// Evaluate outputted predictions
			loss, err := evaluateNetwork(solver, leftObj, rightObj)
			if err != nil {
				return false, err
			}
			errorSum += loss
			count += 1.0
			if loss > 0 {
				detectionErrorCount += 1.0
			}
			// flush solver
			if flushed, err := solver.Flush(); err != nil {
				return false, err
			} else if !flushed {
				return false, errors.New("failed to flush solver after evaluation")
			}
		}
	}

	// Calculate the fitness score
	fitness := maxFitness / (1.0 + errorSum)
	avgError := errorSum / count

	neat.InfoLog(fmt.Sprintf("Average error: %f, errors sum: %f, false detections: %f from: %f",
		avgError, errorSum, detectionErrorCount, count))
	nodes, _ := graph.NodesCount()
	edges, _ := graph.EdgesCount()
	neat.InfoLog(fmt.Sprintf("Substrate: #nodes = %d, #edges = %d | CPPN phenotype: #nodes = %d, #edges = %d",
		nodes, edges, cppnSolver.NodeCount(), cppnSolver.LinkCount()))

	isWinner := false
	if organism.Fitness > fitnessThreshold {
		isWinner = true
		fmt.Printf("Found a Winner! \n")
		// save solver graph to the winner organism
		organism.Data = &genetics.OrganismData{Value: graph}
	}
	// Save properties to organism struct
	organism.IsWinner = isWinner
	organism.Error = avgError
	organism.Fitness = fitness

	return organism.IsWinner, nil
}

// evaluateNetwork is to evaluate provided network solver using provided visual objects to test prediction performance.
// Returns the prediction loss value or error if failed to evaluate.
func evaluateNetwork(solver network.NetworkSolver, leftObj VisualObject, rightObj VisualObject) (float64, error) {
	// Create input by joining data from left and right visual objects
	inputs := append(leftObj.data, rightObj.data...)

	// run evaluation
	loss := math.MaxFloat64
	if err := solver.LoadSensors(inputs); err != nil {
		return loss, err
	}
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
	normPreds := make([]float64, len(predictions))
	for i := 0; i < len(normPreds); i++ {
		if normPreds[i] >= 0.5 {
			normPreds[i] = 1.0
		} else {
			normPreds[i] = 0.0
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

	// Find loss as an Euclidean distance between outputs and ground truth
	loss := (normPreds[0]-targets[0])*(normPreds[0]-targets[0]) + (normPreds[1]-targets[1])*(normPreds[1]-targets[1])

	flag := "match"
	if loss != 0 {
		flag = "-"
	}

	neat.DebugLog(fmt.Sprintf("[%.2f, %.2f] -> [%.2f, %.2f] '%s'",
		targets[0], targets[1], normPreds[0], normPreds[1], flag))

	return loss

}
