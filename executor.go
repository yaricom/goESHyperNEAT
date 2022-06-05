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
	"strings"
	"syscall"
	"time"
)

// The experiment runner boilerplate code
func main() {
	var outDirPath = flag.String("out", "./out", "The output directory to store results.")
	var contextPath = flag.String("context", "./data/retina/es_hyper.neat.yml", "The execution context configuration file.")
	var genomePath = flag.String("genome", "./data/retina/cppn_genome.yml", "The seed genome to start with.")
	var experimentName = flag.String("experiment", "retina", "The name of experiment to run. [retina]")

	var logLevel = flag.String("log-level", "", "The logger level to be used. Overrides the one set in configuration.")
	var trialsCount = flag.Int("trials", 0, "The number of trials for experiment. Overrides the one set in configuration.")

	flag.Parse()

	// Seed the random-number generator with current time so that
	// the numbers will be different every time we run.
	seed := time.Now().Unix()
	rand.Seed(seed)

	// Load context configuration
	configFile, err := os.Open(*contextPath)
	if err != nil {
		log.Fatal("Failed to open context configuration file: ", err)
	}
	neatOptions, err := neat.LoadYAMLOptions(configFile)
	if err != nil {
		log.Fatal("Failed to load NEAT options from config file: ", err)
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
		if env, err := retina.NewRetinaEnvironment(retina.CreateRetinaDataset(), 4, opts); err != nil {
			log.Fatalf("Failed to create retina environment, reason: %s", err)
		} else {
			generationEvaluator, trialObserver = retina.NewGenerationEvaluator(*outDirPath, env)
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
}
