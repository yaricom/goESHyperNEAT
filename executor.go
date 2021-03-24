package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/yaricom/goESHyperNEAT/eshyperneat"
	"github.com/yaricom/goESHyperNEAT/experiments/retina"
	"github.com/yaricom/goNEAT/experiments"
	"github.com/yaricom/goNEAT/neat"
	"github.com/yaricom/goNEAT/neat/genetics"
)

// The experiment runner boilerplate code
func main() {
	var outDirPath = flag.String("out", "./out", "The output directory to store results.")
	var contextPath = flag.String("context", "./data/retina/es_hyper.neat.yml", "The execution context configuration file.")
	var genomePath = flag.String("genome", "./data/retina/cppn_genome.yml", "The seed genome to start with.")
	var experimentName = flag.String("experiment", "retina", "The name of experiment to run. [retina]")

	var logLevel = flag.Int("log-level", -1, "The logger level to be used. Overrides the one set in configuration.")
	var trialsCount = flag.Int("trials", 0, "The number of trials for experiment. Overrides the one set in configuration.")

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
		log.Fatal("Failed to read start genome of CPPN: ", err)
	}
	fmt.Println(startGenome)

	// Load Modular Retina environment

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

	// Create Retina Experiment
	experiment := experiments.Experiment{
		Id:     0,
		Trials: make(experiments.Trials, context.NumRuns),
	}
	var generationEvaluator experiments.GenerationEvaluator
	switch *experimentName {
	case "retina":
		if env, err := retina.NewRetinaEnvironment(retina.CreateRetinaDataset(), 4, context); err != nil {
			log.Fatalf("Failed to create retina environment, reason: %s", err)
		} else {
			generationEvaluator = retina.NewGenerationEvaluator(*outDirPath, env)
		}
	default:
		log.Fatalf("Unsupported experiment name requested: %s\n", *experimentName)
	}

	fmt.Println("Done Setup, Starting Experiment!")

	err = experiment.Execute(context.NeatContext, startGenome, generationEvaluator)
	if err != nil {
		neat.ErrorLog(fmt.Sprintf("Failed to perform %s experiment! Reason: %s\n", *experimentName, err))
		os.Exit(1)
	}

	// Output winner statistics
	experiment.PrintStatistics()
}
