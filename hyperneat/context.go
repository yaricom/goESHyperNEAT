package hyperneat

import "context"

// key is an unexported type for keys defined in this package.
// This prevents collisions with keys defined in other packages.
type key int

// hyperNeatOptionsKey is the key for hyperneat.Options value in Contexts. It is
// unexported; clients use hyperneat.NewContext and hyperneat.FromContext
// instead of using this key directly.
var hyperNeatOptionsKey key

// NewContext returns a new Context that carries the value of HyperNEAT options.
func NewContext(ctx context.Context, opts *Options) context.Context {
	return context.WithValue(ctx, hyperNeatOptionsKey, opts)
}

// FromContext returns the HyperNEAT Options value stored in ctx, if any.
func FromContext(ctx context.Context) (*Options, bool) {
	u, ok := ctx.Value(hyperNeatOptionsKey).(*Options)
	return u, ok
}
