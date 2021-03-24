package retina

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestCreateRetinaDataset(t *testing.T) {
	dataset := CreateRetinaDataset()

	assert.Equal(t, 16, len(dataset))
	for _, vo := range dataset {
		assert.Len(t, vo.data, 4, "wrong data length for VO: %s", vo)
	}
}
