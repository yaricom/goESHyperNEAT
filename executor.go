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
	"github.com/yaricom/goESHyperNEAT/eshyperneat"
	"github.com/yaricom/goNEAT/experiments"
	"github.com/yaricom/goNEAT/neat"
	"github.com/yaricom/goNEAT/neat/genetics"
)

// The experiment runner boilerplate code
func main() {
	var out_dir_path = flag.String("out", "./out", "The output directory to store results.")
	var context_path = flag.String("context", "./data/context.es", "The execution context configuration file.")
	var genome_path = flag.String("genome", "./data/babygenome.es", "The seed genome to start with.")
	//var retina_config_path = flag.String("env", "./data/retina.conf", "The retina environment config file")
	var experiment_name = flag.String("experiment", "retina", "The name of experiment to run. [retina]")

	var log_level = flag.Int("log_level", -1, "The logger level to be used. Overrides the one set in configuration.")
	var trials_count = flag.Int("trials", 0, "The number of trials for experiment. Overrides the one set in configuration.")
	var species_target = flag.Int("species_target", 20, "The target number of species to maintain.")
	var species_compat_adjust_freq = flag.Int("species_adjust_freq", 10, "The frequency of species compatibility theshold adjustments when trying to maintain their number.")

	flag.Parse()

	// Seed the random-number generator with current time so that
	// the numbers will be different every time we run.
	rand.Seed(time.Now().Unix())

	// Load context configuration
	configFile, err := os.Open(*context_path)
	if err != nil {
		log.Fatal("Failed to open context configuration file: ", err)
	}

	context, err := eshyperneat.Load(configFile)
	if err != nil {
		log.Fatal("Failed to load context from config file: ", err)
	}
	// Load Genome
	log.Printf("Loading start genome for %s experiment\n", *experiment_name)
	genomeFile, err := os.Open(*genome_path)
	if err != nil {
		log.Fatal("Failed to open genome file: ", err)
	}
	encoding := genetics.PlainGenomeEncoding
	if strings.HasSuffix(*genome_path, ".yml") {
		encoding = genetics.YAMLGenomeEncoding
	}
	decoder, err := genetics.NewGenomeReader(genomeFile, encoding)
	if err != nil {
		log.Fatal("Failed to create genome decoder: ", err)
	}
	start_genome, err := decoder.Read()
	if err != nil {
		log.Fatal("Failed to read start genome: ", err)
	}
	fmt.Println(start_genome)

	// Load Modular Retina environment
	//log.Printf("Reading retina environment: %s\n", *retina_config_path)
	//retinaFile, err := os.Open(*retina_config_path)
	var environment retina.RetinaEnvironment
	environment = retina.NewRetinaEnvironment()

	// Check if output dir exists
	out_dir := *out_dir_path
	if _, err := os.Stat(out_dir); err == nil {
		// backup it
		back_up_dir := fmt.Sprintf("%s-%s", out_dir, time.Now().Format("2006-01-02T15_04_05"))
		// clear it
		err = os.Rename(out_dir, back_up_dir)
		if err != nil {
			log.Fatal("Failed to do previous results backup: ", err)
		}
	}
	// create output dir
	err = os.MkdirAll(out_dir, os.ModePerm)
	if err != nil {
		log.Fatal("Failed to create output directory: ", err)
	}

	// Override context configuration parameters with ones set from command line
	if *trials_count > 0 {
		context.NumRuns = *trials_count
	}
	if *log_level >= 0 {
		neat.LogLevel = neat.LoggerLevel(*log_level)
	}

	experiment := experiments.Experiment{
		Id:     0,
		Trials: make(experiments.Trials, context.NumRuns),
	}
	var generationEvaluator experiments.GenerationEvaluator
	if *experiment_name == "retina" {
		generationEvaluator = retina.RetinaGenerationEvaluator{
			OutputPath:  *out_dir_path,
			Environment: &environment,
		}
	} else {
		log.Fatalf("Unsupported experiment name requested: %s\n", *experiment_name)
	}

	err = experiment.Execute(context, start_genome, generationEvaluator)
	if err != nil {
		log.Fatalf("Failed to perform %s experiment! Reason: %s\n", *experiment_name, err)
	}

}
