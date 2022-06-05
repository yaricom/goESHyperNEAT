package retina

import (
	"fmt"
	"github.com/pkg/errors"
	"strings"
)

// DetectionSide the side of retina where VisualObject is valid to be detected
type DetectionSide string

const (
	RightSide = DetectionSide("right")
	LeftSide  = DetectionSide("left")
	BothSide  = DetectionSide("both")
)

func (s DetectionSide) String() string {
	return string(s)
}

// Environment holds the dataset and evaluation methods for the modular retina experiment
type Environment struct {
	visualObjects []VisualObject
	inputSize     int
}

// NewRetinaEnvironment creates a new Retina Environment with a dataset of all possible Visual Object with specified
// number of inputs to be acquired from provided objects
func NewRetinaEnvironment(dataSet []VisualObject, inputSize int) (*Environment, error) {
	// check that provided visual objects has data points equal to the inputSize
	for _, o := range dataSet {
		if len(o.data) != inputSize {
			return nil, errors.Errorf(
				"all viasual objects expected to have %d data points, but found %d at %v",
				inputSize, len(o.data), o)
		}
	}
	return &Environment{visualObjects: dataSet, inputSize: inputSize}, nil
}

// VisualObject represents a left, right, or both, object classified by retina
type VisualObject struct {
	Side   DetectionSide // the side(-s) of retina where this visual object accepted as valid
	Config string        // the configuration string

	// Inner computed values from visual objects configuration parsing
	data []float64 // the visual object is rectangular, it can be encoded as 1D array
}

// NewVisualObject creates a new VisualObject by first parsing the config string into a VisualObject
func NewVisualObject(side DetectionSide, config string) VisualObject {
	// Setup visual object data multi-array from config string
	return VisualObject{
		Side:   side,
		Config: config,
		data:   parseVisualObjectConfig(config),
	}
}

func (o *VisualObject) String() string {
	return fmt.Sprintf("%s\n%s", o.Side, o.Config)
}

// parseVisualObjectConfig parses config semantically in the format
// (config = "x1 x2 \n x3 x4") to [ x1, x2, x3, x4 ]  where if xi == "o" => xi = 1
func parseVisualObjectConfig(config string) []float64 {
	data := make([]float64, 0)
	lines := strings.Split(config, "\n")
	for _, line := range lines {
		chars := strings.Split(line, " ")
		for _, char := range chars {
			switch char {
			case "o":
				// pixel is ON
				data = append(data, 1.0)
			case ".":
				// pixel is OFF
				data = append(data, 0.0)
			default:
				panic(fmt.Sprintf("unsupported configuration character [%s]", char))
			}
		}
	}
	return data
}
