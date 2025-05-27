package examples

import "github.com/yaricom/goNEAT/v4/neat"

const (
	compatibilityThresholdStep     = 0.1
	compatibilityThresholdMinValue = 0.3
)

// AdjustSpeciesNumber is to adjust species count by keeping it constant
func AdjustSpeciesNumber(speciesCount, epochId, adjustFrequency, numberSpeciesTarget int, options *neat.Options) {
	if epochId%adjustFrequency == 0 {
		if speciesCount < numberSpeciesTarget {
			options.CompatThreshold -= compatibilityThresholdStep
		} else if speciesCount > numberSpeciesTarget {
			options.CompatThreshold += compatibilityThresholdStep
		}

		// to avoid dropping too low
		if options.CompatThreshold < compatibilityThresholdMinValue {
			options.CompatThreshold = compatibilityThresholdMinValue
		}
	}
}
