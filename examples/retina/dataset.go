package retina

// CreateRetinaDataset returns the 16 possible permutations of visual objects
func CreateRetinaDataset() []VisualObject {
	objs := make([]VisualObject, 0)
	// set left side objects
	objs = append(objs, NewVisualObject(BothSide, ". .\n. ."))
	objs = append(objs, NewVisualObject(BothSide, ". .\n. o"))
	objs = append(objs, NewVisualObject(BothSide, ". .\no ."))
	objs = append(objs, NewVisualObject(BothSide, ". o\n. ."))
	objs = append(objs, NewVisualObject(BothSide, "o .\n. ."))
	objs = append(objs, NewVisualObject(LeftSide, ". .\no o"))
	objs = append(objs, NewVisualObject(LeftSide, ". o\no o"))
	objs = append(objs, NewVisualObject(LeftSide, "o .\no o"))

	// set right side objects
	objs = append(objs, NewVisualObject(BothSide, ". .\n. ."))
	objs = append(objs, NewVisualObject(BothSide, ". .\n. o"))
	objs = append(objs, NewVisualObject(BothSide, ". .\no ."))
	objs = append(objs, NewVisualObject(BothSide, ". o\n. ."))
	objs = append(objs, NewVisualObject(BothSide, "o .\n. ."))
	objs = append(objs, NewVisualObject(RightSide, "o o\n. ."))
	objs = append(objs, NewVisualObject(RightSide, "o o\no ."))
	objs = append(objs, NewVisualObject(RightSide, "o o\n. o"))

	return objs
}
