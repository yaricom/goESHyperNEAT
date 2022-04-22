#
# Go parameters
#
GOCMD = go
GOBUILD = $(GOCMD) build
GOCLEAN = $(GOCMD) clean
GOTEST = $(GOCMD) test -count=1
GOGET = $(GOCMD) get
GORUN = $(GOCMD) run

# The common parameters
BINARY_NAME = eshyperneat
OUT_DIR = out
LOG_LEVEL = info

#
# The retina experiment parameters
#
RETINA_CONTEXT_FILE = "./data/retina/es_hyper.neat.yml"
RETINA_GENOME_FILE = "./data/retina/cppn_genome.yml"
RETINA_TRIALS_COUNT = 0
RETINA_OUT_DIR = $(OUT_DIR)/retina

# The default targets to run
#
all: test

# Run retina experiment
#
execute-retina:
	$(GORUN) executor.go --out $(OUT_DIR) \
						 --context $(RETINA_CONTEXT_FILE) \
						 --genome $(RETINA_GENOME_FILE) \
						 --experiment retina \
						 --trials $(RETINA_TRIALS_COUNT) \
						 --log-level $(LOG_LEVEL)

# Run unit tests
#
test:
	$(GOTEST) -v --short ./...

# Builds binary
#
build: | $(OUT_DIR)
	$(GOBUILD) -o $(OUT_DIR)/$(BINARY_NAME) -v

# Creates the output directory for build artefacts
#
$(OUT_DIR):
	mkdir -p $@

#
# Clean build targets
#
clean:
	$(GOCLEAN)
	rm -f $(OUT_DIR)/$(BINARY_NAME)
	rm -f $(OUT_DIR)/$(BINARY_UNIX)