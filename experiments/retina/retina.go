// Package retina provides implementation of the retina experiment
package retina

import (
	"fmt"
	"github.com/yaricom/goNEAT/experiments"
	"github.com/yaricom/goNEAT/neat"
	"github.com/yaricom/goNEAT/neat/genetics"
	"github.com/yaricom/goNEAT/neat/network"
	"log"
	"os"
)

const (
	// MaxFitness Used as max value which we add error too to get an organism's fitness
	MaxFitness = 1000.0

	// FitnessThreshold is the fitness value for which an organism is considered to have won the experiment
	// TODO update to value which beats the environment (not yet determined)
	FitnessThreshold = 55
)

type generationEvaluator struct {
	OutputPath         string
	Environment        *Environment
	CPPNNetworkBuilder *CPPNNetworkBuilder
}

// NewGenerationEvaluator is to create new generations evaluator for retina experiment.
func NewGenerationEvaluator(outDir string, env *Environment) experiments.GenerationEvaluator {
	return &generationEvaluator{
		OutputPath:  outDir,
		Environment: env,
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
		fmt.Printf("organism #%d fitness %f \n", idx, organism.Fitness)

		if isWinner && (epoch.Best == nil || organism.Fitness > epoch.Best.Fitness) {
			epoch.Solved = true
			epoch.WinnerNodes = len(organism.Genotype.Nodes)
			epoch.WinnerGenes = organism.Genotype.Extrons()
			epoch.WinnerEvals = context.PopSize*epoch.Id + organism.Genotype.Id
			epoch.Best = organism
			if epoch.WinnerNodes == 21 {
				// You could dump out optimal genomes here if desired
				optPath := fmt.Sprintf("%s/%s_%d-%d", experiments.OutDirForTrial(e.OutputPath, epoch.TrialId),
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
		popPath := fmt.Sprintf("%s/gen_%d", experiments.OutDirForTrial(e.OutputPath, epoch.TrialId), epoch.Id)
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
				// Prints the winner organism to file!
				orgPath := fmt.Sprintf("%s/%s_%d-%d", experiments.OutDirForTrial(e.OutputPath, epoch.TrialId),
					"retina_optimal", org.Phenotype.NodeCount(), org.Phenotype.LinkCount())
				file, err := os.Create(orgPath)
				if err != nil {
					neat.ErrorLog(fmt.Sprintf("Failed to dump winner organism genome, reason: %s\n", err))
				} else if err = org.Genotype.Write(file); err != nil {
					neat.ErrorLog("Failed to save winner genotype")
				} else {
					neat.InfoLog(fmt.Sprintf("Generation #%d winner dumped to: %s\n", epoch.Id, orgPath))
				}
				break
			}
		}
	}
	return err
}

// organismEvaluate evaluates an individual phenotype network with retina experiment and returns true if its won
func (e generationEvaluator) organismEvaluate(organism *genetics.Organism) (bool, error) {
	// Prepare underlying network properties
	predictorNetwork, err := e.CPPNNetworkBuilder.CreateANNFromCPPNOrganism(organism)
	if err != nil {
		return false, err
	}

	// Predict on dataset
	errorSum := 0.0
	count := 0.0
	detectionErrorCount := 0.0

	// TODO fix this function, how to activate a NetworkSolver? only can a Network
	// use the network by passing in inputs to get outputs
	networkPredict := func(network network.NetworkSolver, inputs []float64) ([]float64, error) {
		// Put network into initial state
		if _, err := network.Flush(); err != nil {
			return nil, err
		}

		// Load input array into sensors (inputs) of network
		err := network.LoadSensors(inputs)
		if err != nil {
			log.Fatal(err)
			return nil, err
		}

		// TODO Activate the network! Pass the inputs through to the outputs!

		// Get outputs from network
		outputs := network.ReadOutputs()

		return outputs, nil
	}

	// Evaluate the detector ANN against 256 combinations of the left and the right visual objects
	// at correct and incorrect sides of retina
	for _, leftObj := range e.Environment.VisualObjects {
		for _, rightObj := range e.Environment.VisualObjects {
			// Create input by joining data from left and right visual objects
			inputs := append(leftObj.Data, rightObj.Data...)
			// Get outputs by feeding it through our model
			outputs, err := networkPredict(predictorNetwork, inputs)
			if err != nil {
				return false, err
			}
			// Evaluate outputted predictions
			loss := evaluatePredictions(outputs, leftObj, rightObj)
			errorSum += loss
			count += 1.0
			if loss > 0 {
				detectionErrorCount += 1.0
			}
		}
	}

	// Calculate the fitness score
	fitness := MaxFitness / (1.0 + errorSum)
	avgError := errorSum / count

	neat.DebugLog(fmt.Sprintf("Average error: %f, errors sum: %f, false detections: %f", avgError, errorSum, detectionErrorCount))

	isWinner := false
	if organism.Fitness > FitnessThreshold {
		isWinner = true
		fmt.Printf("Found a Winner! \n")
	}
	// Save properties to organism struct
	organism.IsWinner = isWinner
	organism.Error = avgError
	organism.Fitness = fitness

	return organism.IsWinner, nil
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

	// Set target[0] to 1.0 if it contains a valid LeftObj, 0.0 else
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

	// Find loss as a distance between outputs and ground truth
	loss := (normPreds[0]-targets[0])*(normPreds[0]-targets[0]) +
		(normPreds[1]-targets[1])*(normPreds[1]-targets[1])

	flag := "match"
	if loss != 0 {
		flag = "-"
	}

	neat.DebugLog(fmt.Sprintf("[%.2f, %.2f] -> [%.2f, %.2f] '%s'",
		targets[0], targets[1], normPreds[0], normPreds[1], flag))

	return loss

}
