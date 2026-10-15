use super::*;
use derive_more::From;
use monstertruck_core::{ContentHasher, DeterministicContentHash};
use monstertruck_geometry::prelude::{
    AnalyticSurfaceKind, BoundaryCurve2D, HomogeneousSurfaceConversion, SupportsExactPatchDomains,
    SurfaceParameterRectangle, TryIntoAnalyticSurfaceKind, TryIntoBsplineSurface,
    TryIntoHomogeneousBsplineCurve, TryIntoHomogeneousBsplineSurface,
};
#[doc(hidden)]
pub use monstertruck_geometry::prelude::{algo, inv_or_zero};
pub use monstertruck_geometry::{decorators::*, nurbs::*, specifieds::*, t_spline::*};
pub use monstertruck_mesh::PolylineCurve;
use monstertruck_topology::compress::{CompressedTrimmedShell, CompressedTrimmedSolid};
use monstertruck_topology::trimmed::{TrimmedShell, TrimmedSolid};
// Only the rayon-parallel (native) `to_trimmed_with_parameter_curves` builds
// faces directly; the wasm32 arm goes through `to_trimmed_with_face_trims`.
#[cfg(not(target_arch = "wasm32"))]
use monstertruck_topology::trimmed::TrimmedFace;
use monstertruck_traits::SnapCurveEndpoints;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::hash::Hasher;
use std::{env, iter};
// `web_time::Instant` is `std::time::Instant` on native and falls back to the
// browser performance clock on wasm32, where `std::time::Instant::now()` panics.
use web_time::Instant;

mod content_hash;
mod curve;
mod domain_projection;
mod parameter_curves;
mod storage;
mod surface;
/// Immutable validated implicit-curve continuation.
pub mod validated_curve;
/// Outward-rounded native source-geometry enclosures.
pub mod validated_interval;
mod validated_phase;
mod validated_trim;

pub use validated_trim::{
    ExtrusionChartCurve, NativeSurfaceBoundary, NativeTrimCurve, NaturalBoundaryJointEvidence,
    ParameterBoxClassification, PreparedFaceParameterDomain, PreparedTrimCoedge, PreparedTrimJoint,
    SphericalPole, TrimDomainError, TrimDomainLimits, TrimJointKind,
    certified_affine_surface_parameter_curve, certified_curve2d_endpoint_coordinate,
    certified_curve2d_is_straight_regular, certified_extrusion_parameter_curve,
    certified_planar_parameter_curve, certified_surface_boundary_parameter_curve,
    certified_surface_isocurve_parameter_curve, prepare_face_parameter_domain,
    prepare_face_parameter_domain_with_natural_boundaries,
    prepare_face_parameter_domain_with_uncertainty,
};

pub use curve::{Conic2D, Curve, Curve2D};
pub use parameter_curves::{ToCompressedTrimmedParameterCurves, ToTrimmedParameterCurves};
pub use storage::{MAX_NATIVE_GEOMETRY_NESTING, curve_owned_bytes, surface_owned_bytes};
pub use surface::Surface;
pub use validated_curve::{
    CoordinatePlane, PreparedValidatedCurve, ValidatedCurveOptions, ValidatedCurvePiece,
    ValidatedCurvePoint, ValidatedFacePlaneSource, ValidatedPlaneCoverage,
    ValidatedPlaneCoverageOptions, ValidatedPlaneTrimEndpoint, ValidatedPlaneTrimEndpointBinding,
    prepared_face_plane_curve, prepared_validated_curve, prepared_validated_curve_owned,
    validated_face_plane_coverage, validated_plane_trim_root_chart,
    validated_same_surface_coefficients,
};
pub use validated_interval::{
    ValidatedCurveJet, ValidatedCurveJet2D, ValidatedGeometryBudget, ValidatedGeometryError,
    ValidatedInterval, ValidatedPlaneCandidateWitness, ValidatedPlaneCoverageRejection,
    ValidatedPlaneCoverageWitness, ValidatedPlaneJunctionCondition,
    ValidatedPlaneJunctionRejection, ValidatedPlaneJunctionWitness, ValidatedScalarCurveJet,
    ValidatedSurfaceJet, validated_boundary_jet, validated_curve_jet,
    validated_curve_projected_jet, validated_curve2d_jet, validated_sin_cos, validated_surface_jet,
};
pub use validated_phase::{validated_asin, validated_atan2};
