// Package retina provides implementation of the retina experiment
package retina

import (
	"context"
	"fmt"
	"github.com/yaricom/goESHyperNEAT/v2/cppn"
	"github.com/yaricom/goESHyperNEAT/v2/examples"
	"github.com/yaricom/goNEAT/v4/experiment"
	"github.com/yaricom/goNEAT/v4/experiment/utils"
	"github.com/yaricom/goNEAT/v4/neat"
	"github.com/yaricom/goNEAT/v4/neat/genetics"
	"os"
	"sync"
	"time"
)

type parallelGenerationEvaluator struct {
	generationEvaluator
	maxWorkers int
}

type evaluationJob struct {
	organism *genetics.Organism
}

type evaluationJobResult struct {
	genomeID    int
	fitness     float64
	errorRate   float64
	winner      bool
	solverLinks int
	solverNodes int
	err         error
}

func retinaEvaluationWorker(ctx context.Context, evaluator generationEvaluator, jobs <-chan evaluationJob,
	results chan<- evaluationJobResult, wg *sync.WaitGroup) {
	defer wg.Done()

	for job := range jobs {
		winner, solver, err := evaluator.organismEvaluate(ctx, job.organism)
		if err != nil {
			results <- evaluationJobResult{err: err}
			return
		}

		results <- evaluationJobResult{
			genomeID:    job.organism.Genotype.Id,
			fitness:     job.organism.Fitness,
			errorRate:   job.organism.Error,
			winner:      winner,
			solverLinks: solver.LinkCount(),
			solverNodes: solver.NodeCount(),
		}
	}
}

// NewParallelGenerationEvaluator creates parallel generation evaluator for retina experiment.
// The numSpeciesTarget specifies the target number of species to maintain in the population.
// If the number of species differs from the numSpeciesTarget it will be automatically adjusted
// with compatAdjustFreq frequency, i.e., at each epoch % compatAdjustFreq == 0.
func NewParallelGenerationEvaluator(outDir string, env *Environment, numSpeciesTarget, compatAdjustFreq,
	maxWorkers int) (experiment.GenerationEvaluator, experiment.TrialRunObserver) {
	if maxWorkers < 1 {
		maxWorkers = 1
	}
	evaluator := &parallelGenerationEvaluator{
		generationEvaluator: generationEvaluator{
			outDir:           outDir,
			env:              env,
			numSpeciesTarget: numSpeciesTarget,
			compatAdjustFreq: compatAdjustFreq,
		},
		maxWorkers: maxWorkers,
	}
	return evaluator, evaluator
}

// GenerationEvaluate evaluates a population of organisms in parallel on the retina experiment.
func (e *parallelGenerationEvaluator) GenerationEvaluate(ctx context.Context, population *genetics.Population,
	epoch *experiment.Generation) error {
	options, ok := neat.FromContext(ctx)
	if !ok {
		return neat.ErrNEATOptionsNotFound
	}

	popSize := len(population.Organisms)
	resultsChan := make(chan evaluationJobResult, popSize)
	jobsChan := make(chan evaluationJob, popSize)
	organismMapping := make(map[int]*genetics.Organism, popSize)

	var wg sync.WaitGroup
	for i := 0; i < e.maxWorkers; i++ {
		wg.Add(1)
		// Create private evaluator instance per worker to avoid races on mutable state.
		workerEvaluator := generationEvaluator{
			outDir:           e.outDir,
			env:              e.env,
			numSpeciesTarget: e.numSpeciesTarget,
			compatAdjustFreq: e.compatAdjustFreq,
		}
		go retinaEvaluationWorker(ctx, workerEvaluator, jobsChan, resultsChan, &wg)
	}

	startTime := time.Now()
	for _, organism := range population.Organisms {
		if _, exists := organismMapping[organism.Genotype.Id]; exists {
			return fmt.Errorf("organism with %d already exists in mapping", organism.Genotype.Id)
		}
		organismMapping[organism.Genotype.Id] = organism
		jobsChan <- evaluationJob{organism: organism}
	}
	close(jobsChan)

	wg.Wait()
	close(resultsChan)

	maxPopulationFitness := 0.0
	bestLinkCount := 0
	bestNodeCount := 0
	bestSolverLinks := -1
	bestSolverNodes := -1

	for result := range resultsChan {
		if result.err != nil {
			return result.err
		}

		organism, exists := organismMapping[result.genomeID]
		if !exists {
			return fmt.Errorf("organism not found in mapping for id: %d", result.genomeID)
		}

		organism.Fitness = result.fitness
		organism.Error = result.errorRate

		if organism.Fitness > maxPopulationFitness {
			maxPopulationFitness = organism.Fitness
			if phenotype, err := organism.Phenotype(); err == nil {
				bestLinkCount = phenotype.LinkCount()
				bestNodeCount = phenotype.NodeCount()
			} else {
				neat.ErrorLog(fmt.Sprintf("Failed to get organism Phenotype, reason: %s", err))
				return err
			}
			bestSolverLinks = result.solverLinks
			bestSolverNodes = result.solverNodes
		}

		if result.winner && (epoch.Champion == nil || organism.Fitness > epoch.Champion.Fitness) {
			epoch.Solved = true
			epoch.WinnerNodes = len(organism.Genotype.Nodes)
			epoch.WinnerGenes = organism.Genotype.Extrons()
			epoch.WinnerEvals = options.PopSize*epoch.Id + organism.Genotype.Id
			epoch.Champion = organism
			organism.IsWinner = true
		}
	}
	elapsedTime := time.Since(startTime)

	epoch.FillPopulationStatistics(population)

	if epoch.Solved || epoch.Id%options.PrintEvery == 0 {
		if _, err := utils.WritePopulationPlain(e.outDir, population, epoch); err != nil {
			neat.ErrorLog(fmt.Sprintf("Failed to dump population, reason: %s\n", err))
			return err
		}
	}

	if epoch.Solved {
		org := epoch.Champion
		utils.PrintActivationDepth(org, true)

		genomeFile := "retina_cppn_winner"
		if orgPath, err := utils.WriteGenomePlain(genomeFile, e.outDir, org, epoch); err != nil {
			neat.ErrorLog(fmt.Sprintf("Failed to dump winner organism's genome, reason: %s\n", err))
		} else {
			neat.InfoLog(fmt.Sprintf("Generation #%d winner's genome dumped to: %s\n", epoch.Id, orgPath))
		}

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

		examples.AdjustSpeciesNumber(speciesCount, epoch.Id, e.compatAdjustFreq, e.numSpeciesTarget, options)

		neat.InfoLog(
			fmt.Sprintf("%d species -> %d organisms [compatibility threshold: %.1f, target: %d]\nbest CPNN organism [fitness: %.2f, links: %d, nodes: %d], best solver [links: %d, nodes: %d], population evaluation time: %v",
				speciesCount, len(population.Organisms), options.CompatThreshold, e.numSpeciesTarget,
				maxPopulationFitness, bestLinkCount, bestNodeCount, bestSolverLinks, bestSolverNodes, elapsedTime))
	}

	return nil
}
