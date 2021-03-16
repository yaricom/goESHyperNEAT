package retina

import (
	"fmt"
	"os"
	"strings"

	"github.com/yaricom/goNEAT/experiments"
	"github.com/yaricom/goNEAT/neat"
	"github.com/yaricom/goNEAT/neat/genetics"
	"github.com/yaricom/goNEAT/neat/network"
)

const (
	MaxFitness       = 1000.0
	FitnessThreshold = 77 // TODO
)

// RetinaGenerationEvaluator holds the Retina environment and setup instructions.
// it implemens the GenerationEvaluator interface by implementing GenerationEvaluate
type RetinaGenerationEvaluator struct {
	OutputPath  string
	Environment *RetinaEnvironment
}

// RetinaEnvironment holds the dataset and evaluation methods for the modular retina experiment
type RetinaEnvironment struct {
	VisualObjects []VisualObject
}

func NewRetinaEnvironment() RetinaEnvironment {
	visualObjs := CreateRetinaDataset()
	return RetinaEnvironment{VisualObjects: visualObjs}
}

// EvaluateGeneration evaluates a population of organisms and prints their performance on the retina experiment
func (eval RetinaGenerationEvaluator) GenerationEvaluate(population *genetics.Population, epoch *experiments.Generation, context *neat.NeatContext) (err error) {
	// Evaluate each organism on a test
	for _, organism := range population.Organisms {
		isWinner, err := eval.OrganismEvaluate(organism, context)
		if err != nil {
			return err
		}

		if isWinner && (epoch.Best == nil || organism.Fitness > epoch.Best.Fitness) {
			epoch.Solved = true
			epoch.WinnerNodes = len(organism.Genotype.Nodes)
			epoch.WinnerGenes = organism.Genotype.Extrons()
			epoch.WinnerEvals = context.PopSize*epoch.Id + organism.Genotype.Id
			epoch.Best = organism
			if epoch.WinnerNodes == 7 {
				// You could dump out optimal genomes here if desired
				opt_path := fmt.Sprintf("%s/%s_%d-%d", experiments.OutDirForTrial(eval.OutputPath, epoch.TrialId),
					"retina_optimal", organism.Phenotype.NodeCount(), organism.Phenotype.LinkCount())
				file, err := os.Create(opt_path)
				if err != nil {
					neat.ErrorLog(fmt.Sprintf("Failed to dump optimal genome, reason: %s\n", err))
				} else {
					organism.Genotype.Write(file)
					neat.InfoLog(fmt.Sprintf("Dumped optimal genome to: %s\n", opt_path))
				}
			}
		}
	}

	// Fill statistics about current epoch
	epoch.FillPopulationStatistics(population)

	// Only print to file every print_every generations
	if epoch.Solved || epoch.Id%context.PrintEvery == 0 {
		pop_path := fmt.Sprintf("%s/gen_%d", experiments.OutDirForTrial(eval.OutputPath, epoch.TrialId), epoch.Id)
		file, err := os.Create(pop_path)
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
				org_path := fmt.Sprintf("%s/%s_%d-%d", experiments.OutDirForTrial(eval.OutputPath, epoch.TrialId),
					"retina_optimal", org.Phenotype.NodeCount(), org.Phenotype.LinkCount())
				file, err := os.Create(org_path)
				if err != nil {
					neat.ErrorLog(fmt.Sprintf("Failed to dump winner organism genome, reason: %s\n", err))
				} else {
					org.Genotype.Write(file)
					neat.InfoLog(fmt.Sprintf("Generation #%d winner dumped to: %s\n", epoch.Id, org_path))
				}
				break
			}
		}
	}
	return err
}

// EvaluateOrganism evalutes an individual phenotype network with retina experiment and returns true if its won
func (eval RetinaGenerationEvaluator) OrganismEvaluate(organism *genetics.Organism, context *neat.NeatContext) (bool, error) {
	// Prepare underlying network properties

	// The max depth of the network to be activated
	net_depth, err := organism.Phenotype.MaxDepth()
	if err != nil {
		neat.ErrorLog(fmt.Sprintf("Failed to estimate maximal depth of the network with loop:\n%s\nUsing default dpeth: %d", organism.Genotype, net_depth))
	}
	neat.InfoLog(fmt.Sprintf("Network depth: %d for organism: %d\n", net_depth, organism.Genotype.Id))
	if net_depth == 0 {
		neat.WarnLog(fmt.Sprintf("ALERT: Network depth is ZERO for Genome: %s", organism.Genotype))
	}

	// Predict on dataset
	error_sum := 0.0
	count := 0.0
	detection_error_count := 0.0

	healthy_relaxing_organism := true // true if network can relax to the specified depth

	// Evaluate the detector ANN against 256 combintaions of the left and the right visual objects
	// at correct and incorrect sides of retina
	for i := 0; i < len(eval.Environment.VisualObjects); i++ {
		for j := 0; j < len(eval.Environment.VisualObjects); j++ {
			leftObj, rightObj := eval.Environment.VisualObjects[i], eval.Environment.VisualObjects[j]
			loss, _, good_relax, err := evaluateNetwork(organism.Phenotype, leftObj, rightObj, net_depth, true)
			if err != nil {
				return false, err
			}
			if !good_relax {
				healthy_relaxing_organism = false
			}
			error_sum += loss
			count += 1.0
			if loss > 0 {
				detection_error_count += 1.0
			}
		}
	}

	// Calculate the fitness score
	var fitness float64
	var avg_error float64
	if healthy_relaxing_organism {
		fitness = MaxFitness / (1.0 + error_sum)
		avg_error = error_sum / count
	} else {
		// something wrong with relaxing (propogating) inputs thru the network, kill it
		fitness = 0.0
		avg_error = 1.0
	}

	neat.DebugLog(fmt.Sprintf("Average error: %f, errors sum: %f, false detections: %f", avg_error, error_sum, detection_error_count))

	isWinner := false
	if organism.Fitness > FitnessThreshold {
		isWinner = true
		fmt.Printf("Found a Winner! \n")
	}
	// Save properties to organism struct
	organism.IsWinner = isWinner
	organism.Error = avg_error
	organism.Fitness = fitness

	return organism.IsWinner, nil
}

// evaluateNetwork will evaluate ANN against specific visual objects at left and right side
func evaluateNetwork(network *network.Network, leftObj VisualObject, rightObj VisualObject, depth int, debug bool) (float64, []float64, bool, error) {
	// Put network into initial state
	network.Flush()

	// Create input by joining data from left and right visual objects
	inputs := append(leftObj.Flatten(), rightObj.Flatten()...)
	inputs = append(inputs, 0.5) // the bias

	// Load input array into sensors (inputs) of network
	err := network.LoadSensors(inputs)
	if err != nil {
		return 0.0, nil, false, err
	}

	// Relax network and get outputs
	good_relax, err := network.Activate()
	if err != nil {
		return 0.0, nil, false, err
	}

	// Use depth to ensure relaxation
	for relax := 0; relax <= depth; relax++ {
		good_relax, err = network.Activate()
		if err != nil {
			return 0.0, nil, false, err
		}
	}

	// Get outputs from network. Use 0.5 as threshold for classification
	n_outputs := len(network.Outputs)
	outputs := make([]float64, n_outputs)
	for i := 0; i < n_outputs; i++ {
		if network.Outputs[i].Activation >= 0.5 {
			outputs[i] = 1.0
		} else {
			outputs[i] = 0.0
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
	loss := (outputs[0]-targets[0])*(outputs[0]-targets[0]) + (outputs[1]-targets[1])*(outputs[1]-targets[1])

	var flag string
	if loss > 0 {
		flag = "+"
	} else {
		flag = "-"
	}

	if debug {
		fmt.Printf("[%.2f, %.2f] -> [%.2f, %.2f] %s", targets[0], targets[1], outputs[0], outputs[1], flag)
	}

	// Return network to initial state
	network.Flush()

	return loss, outputs, good_relax, nil
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

func NewVisualObject(side Side, size int, config string) VisualObject {
	// Setup visual object data multi-array from config string
	data := ParseVisualObjectConfig(config)
	return VisualObject{
		Side: side, Size: size, Config: config,
		Data: data,
	}
}

func ParseVisualObjectConfig(config string) [][]float64 {
	data := make([][]float64, 0)
	lines := strings.Split(config, "\n")
	for r := 0; r < len(lines); r++ {
		line := lines[r]
		chars := strings.Split(line, " ")
		for c := 0; c < len(chars); r++ {
			char := chars[c]
			if char == "o" {
				// pixel is ON
				data[r][c] = 1.0
			} else {
				// pixel is OFF
				data[r][c] = 0.0
			}
		}
	}
	return data
}

func (obj *VisualObject) Flatten() []float64 {
	l := make([]float64, 0)
	for i := 0; i <= len(obj.Data); i++ {
		for j := 0; j <= len(obj.Data[i]); j++ {
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
