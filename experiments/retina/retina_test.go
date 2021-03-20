package retina

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRetinaEnvironment(t *testing.T) {
	dataset := CreateRetinaDataset()

	assert.Equal(t, 16, len(dataset))
	assert.Equal(t, 4, len(dataset[0].Flatten()))

	// "o .\n. ."
	assert.Equal(t, []float64{1, 0, 0, 0}, dataset[9].Flatten())

	sumLoss := 0.0
	for _, leftObj := range dataset {
		for _, rightObj := range dataset {
			sumLoss += EvaluatePredictions([]float64{0, 0, 0, 0}, leftObj, rightObj)
		}
	}
	assert.Equal(t, 416.0, sumLoss)
}
