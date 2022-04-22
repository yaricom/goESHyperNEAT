package retina

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_parseVisualObjectConfig(t *testing.T) {
	resources := []struct {
		config   string
		expected []float64
	}{
		{
			config:   ". o\n. o",
			expected: []float64{0, 1, 0, 1},
		},
		{
			config:   "o .\n. o",
			expected: []float64{1, 0, 0, 1},
		},
		{
			config:   "o .\n. .",
			expected: []float64{1, 0, 0, 0},
		},
		{
			config:   ". .\n. .",
			expected: []float64{0, 0, 0, 0},
		},
	}

	for i, res := range resources {
		data := parseVisualObjectConfig(res.config)
		assert.ElementsMatch(t, res.expected, data, "elements didn't match at: %d", i)
	}
}
