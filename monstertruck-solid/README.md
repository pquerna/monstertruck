# `monstertruck-solid`

<!-- cargo-rdme start -->

Boolean operations for solids. Fillets live in `monstertruck-fillet` and shape
healing in `monstertruck-healing`; both are post-CSG and kernel-independent.

## Examples

```rust
use monstertruck_modeling::*;
use monstertruck_solid::or;

// Two unit cubes overlapping in a 0.5 cube at one corner.
let v = builder::vertex(Point3::origin());
let cube_a: Solid = builder::extrude(
    &builder::extrude(&builder::extrude(&v, Vector3::unit_x()), Vector3::unit_y()),
    Vector3::unit_z(),
);
let cube_b = builder::translated(&cube_a, Vector3::new(0.5, 0.5, 0.5));

// Boolean entry points are Result-shaped: `Ok(Solid)`, or a typed
// `ShapeOpsError` -- never a silent `None`.
// `monstertruck_modeling::*` brings its own 1-generic `Result` alias into
// scope, so leave the binding unannotated rather than shadowing it.
let union = or(&cube_a, &cube_b, 0.05);
assert!(union.is_ok());
```

<!-- cargo-rdme end -->

> Forked from [`truck-shapeops`](https://crates.io/crates/truck-shapeops) v0.4.0 by [ricosjp](https://github.com/ricosjp/truck).

## Repeated native intersections

`prepare_boolean_source(&source, tolerance, limits, &cancel)` returns an opaque
`PreparedBooleanSource<'_, C, S>` that borrows the exact native solid and owns only
its source-tolerance marching seeds. Keep the source immutable, including
mutations through aliased topology handles, until the prepared value is dropped.
`and_prepared(&prepared, &clip, &cancel)` uses the same native intersection,
classification, carrier reconstruction and release-active manifold checks as
`and`; it does not accept outside meshes or replace a failed branch.

`BooleanWorkLimits` requires explicit `usize` limits for input faces and oriented
edge occurrences, seed points and triangles, conservative triangle-pair work,
interference segments, output faces and edge occurrences, and
`max_prepared_bytes` for combined source/query seeds,
`max_source_prepared_bytes` for source-cache preparation,
`max_other_prepared_bytes` for the derived query input, and
`max_intermediate_prepared_bytes` for intermediate seeds, native component
certificates, candidate metadata and curve subdivisions (each owned object is
admitted separately).
Other-input meshing uses the checked remaining combined point/triangle/byte
allowances before retaining each completed shell. An exact-fit source-cache
admission therefore does not consume a separately reserved clipping-query
allowance. `retained_bytes()` reports actual owned buffer capacities
(including mesh topology vectors and shared curve buffers counted once), not the
borrowed BREP, allocator/Arc bookkeeping or process RSS. Deduct those bytes from
the caller's aggregate cache budget; both inputs' seed admissions apply together
to each query. `retained_mesh_vertices()`, `retained_mesh_triangles()` and
`tolerance()` expose the remaining seed accounting.

Cancellation returns `ShapeOpsError::Cancelled`; explicit admission excess
returns `ShapeOpsError::WorkLimitExceeded { resource, actual, limit }`.
Cancellation checks occur at native stages, face pairs and leader points.
Existing generic tessellation, homogeneous conversion and parameter-search calls
remain indivisible: seed limits bound retained buffers after meshing, not opaque
transient allocations or a hard wall-clock deadline. Bound source geometry and
execution separately. Collision scratch is bounded by admitted seed counts and
conservative triangle pairs; reconstruction scratch is bounded by admitted
interference segments and native input incidences. Multi-shell queries also
admit each intermediate seed mesh.
Face division shares a query-local cache of canonical native edge subdivisions
across both input shells. The unchanged curve sampler runs once per immutable
`EdgeId`, with the exact same tolerance and sample order; inverse coedges replay
the same samples in reverse. Point-buffer capacities and conservative hash-table
growth are admitted against intermediate scratch bytes, and sample counts are
bounded by the mesh-position limit. Cancellation is checked on cache hits,
before/after indivisible generic subdivision and during native UV projection.
The cache does not retain source BREP or survive the shell-pair query.

Interference stitching certifies shared vertices with the unchanged native
residual rather than equating spatial buckets with vertex identity. Reconstructed
leaders must remain on both original carriers with nonzero finite
transversality. Native curve evaluation borrows its carriers and leader rather
than copying them per sample, and AND does not reconstruct the unused OR output.
Open-branch endpoints originate from the certified native leader, not raw mesh
seeds. Boundary recovery jointly searches immutable original input-face coedges
for the closest contact certified against both carriers; subsequently generated
cuts never become new boundary authorities.
Only coedges containing that same native contact can attach an endpoint; polygon
parameters then replay the corresponding shared subdivisions without defining
the native contact or imposing a mesh-distance acceptance window. Ambiguous or
conflicting contacts and endpoints unattached to either original trim refuse
assembly.
Curve/surface root convergence uses native tangent displacement in world units,
not merely a small carrier residual at shallow crossings. Endpoint kinds require
actual native corner proximity, not closeness of scalar curve parameters.
Certified native parameters and polygon-guide parameters are carried separately;
shared native subdivision does not re-solve the root, and corner reuse is
additionally certified in native world distance.

When multiple mesh chains describe a native intersection, certified component
interval coverage is resolved before adding a redundant cut. The certificate
is either the unique transverse plane/plane line or a positive-weight,
degree-one homogeneous fibre with uniformly nonzero plane coefficients.
Coefficient signs are checked conservatively against floating cancellation.
Ordered component coordinates explicitly unwrap a certified periodic seam.
The finite fibre endpoints do not truncate this certificate: equal positive
weights allow an affine extension, and original trims determine material.
Candidate native coordinates and original contacts are prepared before splicing;
supported original intervals are processed before collapsed/uncertain chains,
independently of raw graph encounter order. Metadata scratch is admitted before
allocation, and cached root coordinates are reused rather than re-solved.
An interval may clip a terminal seed segment only at the same immutable original
trim contact with the correct material-crossing direction. Distinct intervals,
uncertain certificates, nontransverse curves and collapsed uncovered contacts
are not discarded or converted into successful empty results. Output still uses
the common native `IntersectionCurve` on the original carriers, not a sampled
section or a replacement analytic shape.

## Runtime and formatting toolchains

Explicitly select the consumer runtime compiler for native prepared-section
qualification: `cargo +1.99.0 test -p monstertruck-solid --lib` and the named
integration regressions. The unchanged Spiling facade probes use this compiler.
The workspace's floating `nightly` toolchain remains available for
`cargo +nightly fmt`; unstable formatting options require it.

Floating-nightly runtime compilation currently fails in the cached
`branches 0.4.6` dependency because `core::intrinsics::abort` was removed, before
this crate is compiled. Runtime qualification on Rust 1.99.0 is not a claim of
floating-nightly runtime compatibility. No dependency patch or failure
suppression is part of this capability.

## License

Apache License 2.0
