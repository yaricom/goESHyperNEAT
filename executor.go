package main

import (
	"context"
	"flag"
	"fmt"
	"github.com/yaricom/goESHyperNEAT/v2/eshyperneat"
	"github.com/yaricom/goESHyperNEAT/v2/examples/retina"
	"github.com/yaricom/goNEAT/v3/experiment"
	"github.com/yaricom/goNEAT/v3/neat"
	"github.com/yaricom/goNEAT/v3/neat/genetics"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"syscall"
	"time"
)

// The experiment runner code
func main() {
	var outDirPath = flag.String("out", "./out", "The output directory to store results.")
	var contextPath = flag.String("context", "./data/retina/es_hyper.neat.yml", "The execution context configuration file.")
	var genomePath = flag.String("genome", "./data/retina/cppn_genome.yml", "The seed genome to start with.")
	var experimentName = flag.String("experiment", "retina", "The name of experiment to run. [retina]")
	var speciesTarget = flag.Int("species_target", 15, "The target number of species to maintain.")
	var speciesCompatAdjustFreq = flag.Int("species_adjust_freq", 10, "The frequency of species compatibility threshold adjustments when trying to maintain their number.")

	var logLevel = flag.String("log-level", "", "The logger level to be used. Overrides the one set in configuration.")
	var trialsCount = flag.Int("trials", 0, "The number of trials for experiment. Overrides the one set in configuration.")

	flag.Parse()

	// Seed the random-number generator with current time so that
	// the numbers will be different every time we run.
	seed := time.Now().Unix()
	rand.Seed(seed)

	// Load context configuration
	neatOptions, err := neat.ReadNeatOptionsFromFile(*contextPath)
	if err != nil {
		log.Fatal("Failed to load NEAT options: ", err)
	}

	// Load Genome
	log.Printf("Loading start genome for %s experiment\n", *experimentName)
	reader, err := genetics.NewGenomeReaderFromFile(*genomePath)
	if err != nil {
		log.Fatalf("Failed to open genome file, reason: '%s'", err)
	}
	startGenome, err := reader.Read()
	if err != nil {
		log.Fatalf("Failed to read start genome, reason: '%s'", err)
	}
	fmt.Println(startGenome)

	// Load Modular Retina environment

	// Check if output dir exists
	outDir := *outDirPath
	if _, err = os.Stat(outDir); err == nil {
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
		neatOptions.NumRuns = *trialsCount
	}
	if len(*logLevel) > 0 {
		neat.LogLevel = neat.LoggerLevel(*logLevel)
	}

	// Create Experiment
	experimentContext := neatOptions.NeatContext()
	exp := experiment.Experiment{
		Id:       0,
		Trials:   make(experiment.Trials, neatOptions.NumRuns),
		RandSeed: seed,
	}
	var generationEvaluator experiment.GenerationEvaluator
	var trialObserver experiment.TrialRunObserver
	switch *experimentName {
	case "retina":
		opts, err := eshyperneat.LoadYAMLConfigFile(*contextPath)
		if err != nil {
			log.Fatal("Failed to load ES-HyperNEAT options from config file: ", err)
		} else {
			experimentContext = eshyperneat.NewContext(experimentContext, opts)
		}
		if env, err := retina.NewRetinaEnvironment(retina.CreateRetinaDataset(), 4); err != nil {
			log.Fatalf("Failed to create retina environment, reason: %s", err)
		} else {
			generationEvaluator, trialObserver = retina.NewGenerationEvaluator(*outDirPath, env, *speciesTarget, *speciesCompatAdjustFreq)
		}
	default:
		log.Fatalf("Unsupported experiment name requested: %s\n", *experimentName)
	}

	// prepare to execute
	errChan := make(chan error)
	ctx, cancel := context.WithCancel(experimentContext)

	// run experiment in the separate GO routine
	go func() {
		if err = exp.Execute(ctx, startGenome, generationEvaluator, trialObserver); err != nil {
			errChan <- err
		} else {
			errChan <- nil
		}
	}()

	// register handler to wait for termination signals
	//
	go func(cancel context.CancelFunc) {
		fmt.Println("\nPress Ctrl+C to stop")

		signals := make(chan os.Signal, 1)
		signal.Notify(signals, syscall.SIGHUP, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT)
		select {
		case <-signals:
			// signal to stop test fixture
			cancel()
		case err = <-errChan:
			// stop waiting
		}
	}(cancel)

	// Wait for experiment completion
	//
	err = <-errChan
	if err != nil {
		// error during execution
		log.Fatalf("Experiment execution failed: %s", err)
	}

	// Print experiment results statistics
	//
	exp.PrintStatistics()

	// Save experiment data in native format
	//
	expResPath := fmt.Sprintf("%s/%s.dat", outDir, *experimentName)
	if expResFile, err := os.Create(expResPath); err != nil {
		log.Fatal("Failed to create file for experiment results", err)
	} else if err = exp.Write(expResFile); err != nil {
		log.Fatal("Failed to save experiment results", err)
	}

	// Save experiment data in Numpy NPZ format if requested
	//
	npzResPath := fmt.Sprintf("%s/%s.npz", outDir, *experimentName)
	if npzResFile, err := os.Create(npzResPath); err != nil {
		log.Fatalf("Failed to create file for experiment results: [%s], reason: %s", npzResPath, err)
	} else if err = exp.WriteNPZ(npzResFile); err != nil {
		log.Fatal("Failed to save experiment results as NPZ file", err)
	}
}
