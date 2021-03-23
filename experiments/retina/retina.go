package retina

import (
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/yaricom/goNEAT/experiments"
	"github.com/yaricom/goNEAT/neat"
	"github.com/yaricom/goNEAT/neat/genetics"
	"github.com/yaricom/goNEAT/neat/network"
)

const (
	// MaxFitness Used as max value which we add error too to get an organism's fitness
	MaxFitness = 1000.0

	// FitnessThreshold is the fitness value for which an organism is considered to have won the experiment
	// TODO update to value which beats the environment (not yet determined)
	FitnessThreshold = 55
)

// GenerationEvaluator holds the Retina environment and setup instructions.
// it implemens the GenerationEvaluator interface by implementing GenerationEvaluate
type GenerationEvaluator struct {
	OutputPath         string
	Environment        *Environment
	CPPNNetworkBuilder *CPPNNetworkBuilder
}

// Environment holds the dataset and evaluation methods for the modular retina experiment
type Environment struct {
	VisualObjects []VisualObject
}

// NewRetinaEnvironment creates a new Retina Environment with a dataset of all possible Visual Object
func NewRetinaEnvironment() Environment {
	visualObjs := CreateRetinaDataset()
	return Environment{VisualObjects: visualObjs}
}

// GenerationEvaluate evaluates a population of organisms and prints their performance on the retina experiment
func (eval GenerationEvaluator) GenerationEvaluate(population *genetics.Population, epoch *experiments.Generation, context *neat.NeatContext) (err error) {
	// Evaluate each organism on a test
	for idx, organism := range population.Organisms {
		isWinner, err := eval.OrganismEvaluate(organism, context)
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
			if epoch.WinnerNodes == 7 {
				// You could dump out optimal genomes here if desired
				optPath := fmt.Sprintf("%s/%s_%d-%d", experiments.OutDirForTrial(eval.OutputPath, epoch.TrialId),
					"retina_optimal", organism.Phenotype.NodeCount(), organism.Phenotype.LinkCount())
				file, err := os.Create(optPath)
				if err != nil {
					neat.ErrorLog(fmt.Sprintf("Failed to dump optimal genome, reason: %s\n", err))
				} else {
					organism.Genotype.Write(file)
					neat.InfoLog(fmt.Sprintf("Dumped optimal genome to: %s\n", optPath))
				}
			}
		}
	}

	// Fill statistics about current epoch
	epoch.FillPopulationStatistics(population)

	// Only print to file every print_every generations
	if epoch.Solved || epoch.Id%context.PrintEvery == 0 {
		popPath := fmt.Sprintf("%s/gen_%d", experiments.OutDirForTrial(eval.OutputPath, epoch.TrialId), epoch.Id)
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
				orgPath := fmt.Sprintf("%s/%s_%d-%d", experiments.OutDirForTrial(eval.OutputPath, epoch.TrialId),
					"retina_optimal", org.Phenotype.NodeCount(), org.Phenotype.LinkCount())
				file, err := os.Create(orgPath)
				if err != nil {
					neat.ErrorLog(fmt.Sprintf("Failed to dump winner organism genome, reason: %s\n", err))
				} else {
					org.Genotype.Write(file)
					neat.InfoLog(fmt.Sprintf("Generation #%d winner dumped to: %s\n", epoch.Id, orgPath))
				}
				break
			}
		}
	}
	return err
}

// OrganismEvaluate evalutes an individual phenotype network with retina experiment and returns true if its won
func (eval GenerationEvaluator) OrganismEvaluate(organism *genetics.Organism, context *neat.NeatContext) (bool, error) {
	// Prepare underlying network properties
	predictorNetwork, err := eval.CPPNNetworkBuilder.CreateANNFromCPPNOrganism(organism)
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
		network.Flush()

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

	// Evaluate the detector ANN against 256 combintaions of the left and the right visual objects
	// at correct and incorrect sides of retina
	for _, leftObj := range eval.Environment.VisualObjects {
		for _, rightObj := range eval.Environment.VisualObjects {
			// Create input by joining data from left and right visual objects
			inputs := append(leftObj.Flatten(), rightObj.Flatten()...)
			// Get outputs by feeding it through our model
			outputs, err := networkPredict(predictorNetwork, inputs)
			if err != nil {
				return false, err
			}
			// Evaluate outputted predictions
			loss := EvaluatePredictions(outputs, leftObj, rightObj)

			if err != nil {
				return false, err
			}
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

// EvaluatePredictions returns the loss between predictions and ground truth of leftObj and rightObj
func EvaluatePredictions(predictions []float64, leftObj VisualObject, rightObj VisualObject) float64 {
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

	// Find loss as a distance between outputs and groud truth
	loss := (normPreds[0]-targets[0])*(normPreds[0]-targets[0]) +
		(normPreds[1]-targets[1])*(normPreds[1]-targets[1])

	return loss

}

// CreateRetinaDataset returns the 16 possible permuations of visual objects
func CreateRetinaDataset() []VisualObject {
	objs := make([]VisualObject, 0)
	size := 2
	// set left side objects
	objs = append(objs, NewVisualObject(BothSide, size, ". .\n. ."))
	objs = append(objs, NewVisualObject(BothSide, size, ". .\n. o"))
	objs = append(objs, NewVisualObject(LeftSide, size, ". o\n. o"))
	objs = append(objs, NewVisualObject(BothSide, size, ". o\n. ."))
	objs = append(objs, NewVisualObject(LeftSide, size, ". o\no o"))
	objs = append(objs, NewVisualObject(BothSide, size, ". .\no ."))
	objs = append(objs, NewVisualObject(LeftSide, size, "o o\n. o"))
	objs = append(objs, NewVisualObject(BothSide, size, "o .\n. ."))

	// set right side objects
	objs = append(objs, NewVisualObject(BothSide, size, ". .\n. ."))
	objs = append(objs, NewVisualObject(BothSide, size, "o .\n. ."))
	objs = append(objs, NewVisualObject(RightSide, size, "o .\no ."))
	objs = append(objs, NewVisualObject(BothSide, size, ". .\no ."))
	objs = append(objs, NewVisualObject(RightSide, size, "o o\no ."))
	objs = append(objs, NewVisualObject(BothSide, size, ". o\n. ."))
	objs = append(objs, NewVisualObject(RightSide, size, "o .\no o"))
	objs = append(objs, NewVisualObject(BothSide, size, ". .\n. o"))

	return objs
}

// VisualObject represents a left, right, or both, object classified by retina
type VisualObject struct {
	// Setup visual object
	Side   Side
	Size   int
	Config string

	// Inner computed values from Config parsing
	Data [][]float64
}

// NewVisualObject creates a new VisuaObject by first parsing the config string into a VisualObject
func NewVisualObject(side Side, size int, config string) VisualObject {
	// Setup visual object data multi-array from config string
	data := ParseVisualObjectConfig(config)
	return VisualObject{
		Side: side, Size: size, Config: config,
		Data: data,
	}
}

// ParseVisualObjectConfig parses config semantically in the format
// (config = "x1 x2 \n x3 x4") to [ [x1, x2], [x3, x4] ]  where if xi == "o" => xi = 1
func ParseVisualObjectConfig(config string) [][]float64 {
	data := make([][]float64, 0)
	lines := strings.Split(config, "\n")
	for r := 0; r < len(lines); r++ {
		dataRow := make([]float64, 0)
		line := lines[r]
		chars := strings.Split(line, " ")
		for c := 0; c < len(chars); c++ {
			char := chars[c]
			if char == "o" {
				// pixel is ON
				dataRow = append(dataRow, 1.0)
			} else {
				// pixel is OFF
				dataRow = append(dataRow, 0.0)
			}
		}
		data = append(data, dataRow)
	}
	return data
}

// Flatten returns a 1D array from the values in the 2D obj
func (obj *VisualObject) Flatten() []float64 {
	l := make([]float64, 0)
	for i := 0; i < len(obj.Data); i++ {
		for j := 0; j < len(obj.Data[i]); j++ {
			l = append(l, obj.Data[i][j])
		}
	}
	return l
}

func (obj *VisualObject) String() string {
	return fmt.Sprintf("%s\n%s", obj.Side, obj.Config)
}

// Side that a Visual Object is valid in
type Side string

const (
	RightSide = Side("right")
	LeftSide  = Side("left")
	BothSide  = Side("both")
)

func (s Side) String() string {
	return string(s)
}
