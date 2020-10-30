#
# Go parameters
#
GOCMD = go
GOBUILD = $(GOCMD) build
GOCLEAN = $(GOCMD) clean
GOTEST = $(GOCMD) test -count=1
GOGET = $(GOCMD) get
BINARY_NAME = eshyperneat
OUT_DIR = out

# The default targets to run
#
all: test

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