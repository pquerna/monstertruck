// The classic (0.3.2) boolean pipeline: the self-contained default backend.
// Compiled only when the marching SSI backend is enabled (the published
// default). Under `--no-default-features` it is absent and the boolean entry
// points in `integrate` return `ShapeOpsError::NoBackend`.
#[cfg(feature = "marching-ssi")]
mod classic;
mod edge_lineage;
mod integrate;
mod prepared;
pub use edge_lineage::{
    BooleanCoedgeOrigin, BooleanCoedgeProvenance, BooleanCutSide, BooleanEdgeDerivation,
};
mod provenance;
pub use prepared::{
    BooleanPreparationOptions, BooleanWorkBudget, BooleanWorkLimits, PreparedBooleanSource,
    and_prepared, and_prepared_counted, prepare_boolean_source, prepare_boolean_source_counted,
    prepare_boolean_source_counted_with_options,
};
pub use provenance::{
    BooleanFaceOrigin, BooleanFaceProvenance, BooleanOperand, BooleanWithProvenance,
    and_prepared_with_provenance, and_prepared_with_provenance_counted,
    and_prepared_with_provenance_counted_with_query_bytes,
};
// Marching-SSI polyline chaining support consumed by the classic backend.
#[cfg(feature = "marching-ssi")]
mod polyline_construction;
pub use integrate::{
    PlaneCut, ShapeOpsCurve, ShapeOpsError, ShapeOpsSurface, ShellOrientationHints, and,
    and_with_orientation_hints, clip_half_space_z, difference, or, plane_cut, symmetric_difference,
};
pub use monstertruck_traits::SnapCurveEndpoints;
