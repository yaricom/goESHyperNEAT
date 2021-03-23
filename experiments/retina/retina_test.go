package retina

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_evaluatePredictions(t *testing.T) {
	dataset := CreateRetinaDataset()
	sumLoss := 0.0
	for _, leftObj := range dataset {
		for _, rightObj := range dataset {
			sumLoss += evaluatePredictions([]float64{0, 0, 0, 0}, leftObj, rightObj)
		}
	}
	assert.Equal(t, 416.0, sumLoss)
}
