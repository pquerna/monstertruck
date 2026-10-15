# `monstertruck-modeling`

<!-- cargo-rdme start -->

Integrated geometric and topological modeling algorithms.

## Examples

```rust
use monstertruck_modeling::*;

// Build a unit cube by three successive extrusions: point -> edge -> face -> solid
let v = builder::vertex(Point3::new(-0.5, -0.5, -0.5));
let e = builder::extrude(&v, Vector3::unit_x());
let f = builder::extrude(&e, Vector3::unit_y());
let cube: Solid = builder::extrude(&f, Vector3::unit_z());

assert_eq!(cube.boundaries()[0].len(), 6); // six faces
```

<!-- cargo-rdme end -->

> Forked from [`truck-modeling`](https://crates.io/crates/truck-modeling) v0.6.0 by [ricosjp](https://github.com/ricosjp/truck).

## Native curve consistency

A parameter curve's carrier is not necessarily the face being inspected. An
original analytic circle can be parameterized on a plane while also lying on a
cylinder, or on a cap with a different plane frame. Carrier equality is a fast
path, not a requirement for geometric consistency.

Other carriers use the existing native curve-consistency predicate: evaluate
the actual bounded curve, preserving nested trims, poses and inversion, recover
parameters on the inspected carrier, and check the unchanged world-space
residual. This sampled consistency heuristic is not a whole-domain containment
or intersection-completeness certificate. Consumers still require their
independent whole-native trim and plane-coverage proofs before accepting a
section; neither sampled consistency nor mesh discovery replaces those proofs.

## License

Apache License 2.0
