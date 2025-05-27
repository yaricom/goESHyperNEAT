package eshyperneat

import (
	"context"
	"errors"
)

var ErrESHyperNEATOptionsNotFound = errors.New("ES-HyperNEAT options not found in the context")

// key is an unexported type for keys defined in this package.
// This prevents collisions with keys defined in other packages.
type key int

// esHyperNeatOptionsKey is the key for eshyperneat.Options value in Contexts. It is
// unexported; clients use eshyperneat.NewContext and eshyperneat.FromContext
// instead of using this key directly.
var esHyperNeatOptionsKey key

// NewContext returns a new Context that carries value of ES-HyperNEAT options.
func NewContext(ctx context.Context, opts *Options) context.Context {
	return context.WithValue(ctx, esHyperNeatOptionsKey, opts)
}

// FromContext returns the ES-HyperNEAT Options value stored in ctx, if any.
func FromContext(ctx context.Context) (*Options, bool) {
	u, ok := ctx.Value(esHyperNeatOptionsKey).(*Options)
	return u, ok
}
