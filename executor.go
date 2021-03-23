package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/vdyagilev/goESHyperNEAT/experiments/retina"
	"github.com/yaricom/goESHyperNEAT/cppn"
	"github.com/yaricom/goESHyperNEAT/eshyperneat"
	"github.com/yaricom/goNEAT/experiments"
	"github.com/yaricom/goNEAT/neat"
	"github.com/yaricom/goNEAT/neat/genetics"
	"github.com/yaricom/goNEAT/neat/utils"
)

// Current bugs:
// - NetworkSolver doesn't have an Activate function, how do we pass through inputs to get outputs? (If its with Relax(), how to determine maxDepth?)
// 		-read more at retina/retina.go:122
//
// - !!! Mysterious Bug in CreateNetworkSolver. With GraphBuilder != nil, crashes when graph.AddEdgeToBuilder since "source node does not exist". (also target node too)
// - With GraphBuilder disabled (set to nil), there are no runtime crashes but an infinite loop somewhere in the first generation.
// I think something is wrong with adding in edges in the eshyperneat algorithm, but its very hard to track down since this bug changes run-to-run.
//		- evolvable_substrate.go:44

// The experiment runner boilerplate code
func main() {
	var outDirPath = flag.String("out", "./out", "The output directory to store results.")
	var contextPath = flag.String("context", "./data/test_es_hyper.neat.yml", "The execution context configuration file.")
	var genomePath = flag.String("genome", "./data/test_cppn_hyperneat_genome.yml", "The seed genome to start with.")
	//var retina_config_path = flag.String("env", "./data/retina.conf", "The retina environment config file")
	var experimentName = flag.String("experiment", "retina", "The name of experiment to run. [retina]")

	var logLevel = flag.Int("log_level", -1, "The logger level to be used. Overrides the one set in configuration.")
	var trialsCount = flag.Int("trials", 0, "The number of trials for experiment. Overrides the one set in configuration.")
	// var species_target = flag.Int("species_target", 20, "The target number of species to maintain.")
	// var species_compat_adjust_freq = flag.Int("species_adjust_freq", 10, "The frequency of species compatibility theshold adjustments when trying to maintain their number.")
	var useLEO = flag.Bool("use_leo", false, "Set whether or not to use LEO to trim connections in the ESHyperNEAT algorithm.")

	flag.Parse()

	// Seed the random-number generator with current time so that
	// the numbers will be different every time we run.
	rand.Seed(time.Now().Unix())

	// Load context configuration
	configFile, err := os.Open(*contextPath)
	if err != nil {
		log.Fatal("Failed to open context configuration file: ", err)
	}
	context, err := eshyperneat.Load(configFile)
	if err != nil {
		log.Fatal("Failed to load context from config file: ", err)
	}

	// Load Genome
	log.Printf("Loading start genome for %s experiment\n", *experimentName)
	genomeFile, err := os.Open(*genomePath)
	if err != nil {
		log.Fatal("Failed to open genome file: ", err)
	}
	encoding := genetics.PlainGenomeEncoding
	if strings.HasSuffix(*genomePath, ".yml") {
		encoding = genetics.YAMLGenomeEncoding
	}
	decoder, err := genetics.NewGenomeReader(genomeFile, encoding)
	if err != nil {
		log.Fatal("Failed to create genome decoder: ", err)
	}
	startGenome, err := decoder.Read()
	if err != nil {
		log.Fatal("Failed to read start genome: ", err)
	}
	fmt.Println(startGenome)

	// Load Modular Retina environment
	var environment retina.Environment
	environment = retina.NewRetinaEnvironment()

	// Check if output dir exists
	outDir := *outDirPath
	if _, err := os.Stat(outDir); err == nil {
		// backup it
		backUpDir := fmt.Sprintf("%s-%s", outDir, time.Now().Format("2006-01-02T15_04_05"))
		// clear it
		err = os.Rename(outDir, backUpDir)
		if err != nil {
			log.Fatal("Failed to do previous results backup: ", err)
		}
	}
	// create output dir
	err = os.MkdirAll(outDir, os.ModePerm)
	if err != nil {
		log.Fatal("Failed to create output directory: ", err)
	}

	// Override context configuration parameters with ones set from command line
	if *trialsCount > 0 {
		context.NumRuns = *trialsCount
	}
	if *logLevel >= 0 {
		neat.LogLevel = neat.LoggerLevel(*logLevel)
	}

	// Create CPPNNetworkBuilder struct that builds prediction anns from organisms cppn phenotype
	substrateLayout, err := cppn.NewMappedEvolvableSubstrateLayout(8, 2) // 8 inputs, 2 outputs in the retina experiment
	if err != nil {
		log.Fatal(err)
	}
	cppnNetworkBuilder := retina.CPPNNetworkBuilder{
		UseLEO:          *useLEO,
		SubstrateLayout: substrateLayout,
		// TODO I chose a random activation function, not sure what to choose!
		NodesActivation: utils.GaussianBipolarActivation,
		Context:         context,
		NewGraphBuilder: func() *cppn.GraphBuilder {
			// TODO removed GraphBuilder from algorithm since
			// there is a BUG in CreateNetworkSolver (I think) which
			// PREVENTS graphbuilder from executing correctly!
			return nil
		}, // cppn.NewGraphMLBuilder("eshyperneat-ann", false) },
	}

	// Create Retina Experiment
	experiment := experiments.Experiment{
		Id:     0,
		Trials: make(experiments.Trials, context.NumRuns),
	}
	var generationEvaluator experiments.GenerationEvaluator
	if *experimentName == "retina" {
		generationEvaluator = retina.GenerationEvaluator{
			OutputPath:         *outDirPath,
			Environment:        &environment,
			CPPNNetworkBuilder: &cppnNetworkBuilder,
		}
	} else {
		log.Fatalf("Unsupported experiment name requested: %s\n", *experimentName)
	}

	fmt.Println("Done Setup, Starting Experiment!")

	// Perform Retina Experiment with goNEAT training a population of CPPNs each epoch
	err = experiment.Execute(context.NeatContext, startGenome, generationEvaluator)
	if err != nil {
		log.Fatalf("Failed to perform %s experiment! Reason: %s\n", *experimentName, err)
	}

}
