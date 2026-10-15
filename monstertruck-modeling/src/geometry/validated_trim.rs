//! Validated winding of authoritative stored native UV trim charts.
//!
//! No inverse-map samples are used. A curve arc is replaced for winding only
//! when its entire outward interval enclosure misses the query box: the convex
//! enclosure then supplies a homotopy avoiding every point of that box. Ray
//! crossings of the replacement chord are evaluated with interval endpoints.
//! Stored UV coefficients are authoritative; this module does not certify that
//! an arbitrary caller-supplied UV payload corresponds to its independent 3D edge.

use super::{
    Conic2D, Curve, Curve2D, ParameterCurve, Surface, ValidatedCurveJet2D, ValidatedGeometryBudget,
    ValidatedGeometryError, ValidatedInterval, validated_curve2d_jet,
};
use crate::{Face, ParametricCurve, ParametricSurface, Point3};
use monstertruck_topology::VertexId;
use monstertruck_topology::trimmed::{ChartCoedgeIdentity, TrimmedFace};
use monstertruck_traits::{BoundedCurve, ParameterRange};
use std::{mem::size_of, ops::Range, sync::atomic::AtomicBool};
use thiserror::Error;

mod affine;
mod closure;
mod extrusion;
mod isocurve;
mod natural_boundary;
mod planar;
mod poles;
mod rechart;
pub use affine::certified_affine_surface_parameter_curve;
pub use extrusion::{ExtrusionChartCurve, certified_extrusion_parameter_curve};
pub use isocurve::{
    NativeSurfaceBoundary, certified_surface_boundary_parameter_curve,
    certified_surface_isocurve_parameter_curve,
};
pub use natural_boundary::NaturalBoundaryJointEvidence;
pub use natural_boundary::certified_curve2d_endpoint_coordinate;
pub use natural_boundary::certified_curve2d_is_straight_regular;
pub use planar::certified_planar_parameter_curve;
pub use poles::{SphericalPole, TrimJointKind, certified_surface_parameter_joint};

/// Borrowed access to an authoritative native UV trim without carrier copies.
pub trait NativeTrimCurve {
    /// Returns the stored absolute-edge-direction UV curve.
    fn native_uv_curve(&self) -> &Curve2D;
}

impl NativeTrimCurve for Curve2D {
    fn native_uv_curve(&self) -> &Curve2D { self }
}

impl NativeTrimCurve for ParameterCurve<Curve2D, Box<Surface>> {
    fn native_uv_curve(&self) -> &Curve2D { self.curve() }
}

/// Retained chart metadata and query subdivision caps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrimDomainLimits {
    /// Maximum oriented coedge occurrences, including repeated seam uses.
    pub max_coedges: usize,
    /// Maximum retained knot spans and adaptive nodes per classification.
    pub max_subdivisions: usize,
    /// Maximum combined owned chart metadata and classification scratch bytes.
    pub max_retained_bytes: usize,
}

/// A sound classification of the complete closed query box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterBoxClassification {
    /// Every point of the box has nonzero material winding.
    Inside,
    /// Every point of the box has zero material winding.
    Outside,
    /// Boundary contact or bounded interval uncertainty prevents strict classification.
    Boundary,
}

/// Failure to admit or query an authoritative closed covering chart.
#[derive(Clone, Copy, Debug, PartialEq, Error)]
pub enum TrimDomainError {
    /// Native interval arithmetic or the cumulative work allowance refused the query.
    #[error(transparent)]
    Geometry(#[from] ValidatedGeometryError),
    /// A stored exact pcurve is missing from one oriented boundary occurrence.
    #[error("missing exact trim at wire {wire_index}, coedge {coedge_index}")]
    MissingExactTrim {
        /// Absolute boundary wire ordinal.
        wire_index: usize,
        /// Absolute coedge ordinal.
        coedge_index: usize,
    },
    /// A retained metadata or subdivision cap was exceeded.
    #[error("native trim resource cap exceeded")]
    ResourceLimit,
    /// A trim domain or closed query box is invalid or nonfinite.
    #[error("invalid native trim chart")]
    InvalidChart,
    /// Successive UV endpoints do not have a structural equality certificate.
    #[error("native UV chart closure is not structurally certified")]
    UnclosedChart,
    /// A nonclosed covering chart requires unstored seam or pole correspondence.
    #[error("native trim seam or pole correspondence unavailable")]
    UnsupportedSeam,
}

/// One borrowed, completely identified oriented trim occurrence.
#[derive(Debug)]
pub struct PreparedTrimCoedge<'a> {
    /// Native face, shared edge, wire and occurrence identity with material orientation.
    pub identity: ChartCoedgeIdentity<Curve, Surface>,
    /// Exact authoritative stored UV curve, in absolute edge direction.
    pub curve: &'a Curve2D,
    /// Exact native bounds, retaining endpoint inclusion.
    pub domain: ParameterRange,
    /// Closed numeric bounds ordered in material traversal direction.
    pub oriented_range: [f64; 2],
    span_range: Range<usize>,
}

/// Bounded endpoint closure uncertainty tied to an original native vertex use.
///
/// This is not an assertion that independently rounded endpoint expressions are
/// equal. It describes all local chart completions within the convex endpoint
/// envelope. Strict winding results hold for every such completion; boxes meeting
/// this envelope remain `Boundary`. Exact source-chart association is the caller's
/// responsibility, as for the stored coedge curves themselves.
#[derive(Debug)]
pub struct PreparedTrimJoint {
    /// The common native boundary vertex, not a nearest-point association.
    pub vertex_id: VertexId<Point3>,
    /// Original shared native vertex position captured without copying its geometry carrier.
    pub vertex_point: Point3,
    /// Original absolute occurrence arriving at this joint in material direction.
    pub incoming: ChartCoedgeIdentity<Curve, Surface>,
    /// Original absolute occurrence leaving this joint in material direction.
    pub outgoing: ChartCoedgeIdentity<Curve, Surface>,
    /// Complete closed UV envelope of local endpoint/closure uncertainty.
    pub enclosure: [ValidatedInterval; 2],
    /// Declared singular-stratum completion or ordinary endpoint uncertainty.
    pub kind: TrimJointKind,
    front: [ValidatedInterval; 2],
    back: [ValidatedInterval; 2],
}

#[derive(Clone, Copy, Debug)]
struct TrimSpan {
    coedge: usize,
    range: [f64; 2],
}

/// Borrowed prepared UV chart, retaining every oriented use and exact knot span.
///
/// Opposite seam uses are not merged. A stored closed covering-chart boundary
/// works directly, including explicit periodic seam connectors. Local rounded
/// endpoint disagreements retain bounded joint uncertainty with shared native
/// vertex identity. Unstored period lifts and pole maps are refused rather than
/// reconstructed. Face inversion changes winding sign, not material membership.
/// Metadata is bounded by [`TrimDomainLimits`]; every query shares the caller's
/// cumulative [`ValidatedGeometryBudget`] and cancellation flag.
#[derive(Debug)]
pub struct PreparedFaceParameterDomain<'a> {
    face: &'a Face,
    coedges: Vec<PreparedTrimCoedge<'a>>,
    joints: Vec<PreparedTrimJoint>,
    spans: Vec<TrimSpan>,
    bounds: [[f64; 2]; 2],
    uncertainty_uv: [f64; 2],
    limits: TrimDomainLimits,
}

fn validate_box(bounds: [[f64; 2]; 2]) -> Result<[ValidatedInterval; 2], TrimDomainError> {
    Ok([
        ValidatedInterval::new(bounds[0][0], bounds[0][1])?,
        ValidatedInterval::new(bounds[1][0], bounds[1][1])?,
    ])
}

fn disjoint(a: [ValidatedInterval; 2], b: [ValidatedInterval; 2]) -> bool {
    (0..2).any(|axis| a[axis].upper < b[axis].lower || b[axis].upper < a[axis].lower)
}

fn metadata_bytes(coedges: usize, spans: usize, joints: usize) -> usize {
    size_of::<PreparedFaceParameterDomain<'_>>()
        .saturating_add(coedges.saturating_mul(size_of::<PreparedTrimCoedge<'_>>()))
        .saturating_add(spans.saturating_mul(size_of::<TrimSpan>()))
        .saturating_add(joints.saturating_mul(size_of::<PreparedTrimJoint>()))
}

fn growing_capacity(length: usize, capacity: usize) -> usize {
    if length == capacity {
        capacity.saturating_mul(2).max(4)
    } else {
        capacity
    }
}

fn admit_bytes(bytes: usize, limit: usize) -> Result<(), TrimDomainError> {
    if bytes > limit {
        Err(TrimDomainError::ResourceLimit)
    } else {
        Ok(())
    }
}

fn curve_storage_units(
    mut curve: &Curve2D,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<usize, TrimDomainError> {
    let mut units = 0_usize;
    for _ in 0..32 {
        budget.charge(1, cancel)?;
        units = units
            .checked_add(12)
            .ok_or(TrimDomainError::ResourceLimit)?;
        match curve {
            Curve2D::TrimmedCurve(selected) => curve = selected.entity().curve().as_ref(),
            Curve2D::BsplineCurve(curve) => {
                return units
                    .checked_add(curve.knot_vector().len())
                    .and_then(|units| {
                        units.checked_add(curve.control_points().len().checked_mul(2)?)
                    })
                    .ok_or(TrimDomainError::ResourceLimit);
            }
            Curve2D::NurbsCurve(curve) => {
                return units
                    .checked_add(curve.knot_vector().len())
                    .and_then(|units| {
                        units.checked_add(curve.control_points().len().checked_mul(3)?)
                    })
                    .ok_or(TrimDomainError::ResourceLimit);
            }
            Curve2D::Polyline(curve) => {
                return units
                    .checked_add(
                        curve
                            .as_slice()
                            .len()
                            .checked_mul(2)
                            .ok_or(TrimDomainError::ResourceLimit)?,
                    )
                    .ok_or(TrimDomainError::ResourceLimit);
            }
            _ => return Ok(units),
        }
    }
    Err(TrimDomainError::ResourceLimit)
}

fn breaks(
    mut curve: &Curve2D,
    range: [f64; 2],
    limit: usize,
    byte_limit: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Vec<f64>, TrimDomainError> {
    let zero = planar::Dyadic::scalar(0.0).ok_or(TrimDomainError::InvalidChart)?;
    let mut offset = zero;
    let mut reversed = false;
    let mut depth = 0_usize;
    while let Curve2D::TrimmedCurve(selected) = curve {
        budget.charge(1, cancel)?;
        depth += 1;
        if depth > 32 {
            return Err(TrimDomainError::ResourceLimit);
        }
        if !selected.orientation() {
            let (a, b) = selected.entity().range();
            offset = planar::Dyadic::scalar(a)
                .and_then(|a| a.add(planar::Dyadic::scalar(b)?))
                .and_then(|sum| sum.add(offset.negated()?))
                .ok_or(TrimDomainError::Geometry(
                    ValidatedGeometryError::Unresolved,
                ))?;
            reversed = !reversed;
        }
        curve = selected.entity().curve().as_ref();
    }
    let count = match curve {
        Curve2D::BsplineCurve(curve) => curve.knot_vector().len(),
        Curve2D::NurbsCurve(curve) => curve.knot_vector().len(),
        Curve2D::Polyline(curve) => curve.as_slice().len(),
        _ => 0,
    };
    if count.saturating_add(2) > limit {
        return Err(TrimDomainError::ResourceLimit);
    }
    admit_bytes(
        count.saturating_add(2).saturating_mul(size_of::<f64>()),
        byte_limit,
    )?;
    budget.charge(count.saturating_add(2), cancel)?;
    let mut result = Vec::with_capacity(count + 2);
    result.push(range[0]);
    let mut insert = |parameter: f64| -> Result<(), TrimDomainError> {
        budget.charge(1, cancel)?;
        let parameter = planar::Dyadic::scalar(parameter).ok_or(TrimDomainError::InvalidChart)?;
        let mapped = if reversed {
            offset.add(parameter.negated().ok_or(TrimDomainError::InvalidChart)?)
        } else {
            parameter.add(offset.negated().ok_or(TrimDomainError::InvalidChart)?)
        }
        .ok_or(TrimDomainError::Geometry(
            ValidatedGeometryError::Unresolved,
        ))?;
        let lower = planar::Dyadic::scalar(range[0]).ok_or(TrimDomainError::InvalidChart)?;
        let upper = planar::Dyadic::scalar(range[1]).ok_or(TrimDomainError::InvalidChart)?;
        if mapped.compare(lower).is_lt() || mapped.compare(upper).is_gt() {
            return Ok(());
        }
        if mapped == lower || mapped == upper {
            return Ok(());
        }
        let parameter = mapped.exact_float().ok_or(TrimDomainError::Geometry(
            ValidatedGeometryError::Unresolved,
        ))?;
        if range[0] < parameter && parameter < range[1] {
            result.push(parameter);
        }
        Ok(())
    };
    match curve {
        Curve2D::BsplineCurve(curve) => curve
            .knot_vector()
            .iter()
            .try_for_each(|parameter| insert(*parameter))?,
        Curve2D::NurbsCurve(curve) => curve
            .knot_vector()
            .iter()
            .try_for_each(|parameter| insert(*parameter))?,
        Curve2D::Polyline(_) => (0..count).try_for_each(|parameter| insert(parameter as f64))?,
        _ => {}
    }
    result.push(range[1]);
    budget.charge(
        result
            .len()
            .saturating_mul(usize::BITS as usize - result.len().leading_zeros() as usize),
        cancel,
    )?;
    result.sort_by(f64::total_cmp);
    result.dedup();
    Ok(result)
}

struct SpanAdmission<'a> {
    spans: &'a mut Vec<TrimSpan>,
    bounds: &'a mut Option<[ValidatedInterval; 2]>,
    nodes: usize,
    limit: usize,
    base_bytes: usize,
    byte_limit: usize,
}

impl SpanAdmission<'_> {
    fn admit(
        &mut self,
        curve: &Curve2D,
        span: TrimSpan,
        depth: usize,
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> Result<(), TrimDomainError> {
        budget.charge(1, cancel)?;
        self.nodes = self
            .nodes
            .checked_add(1)
            .ok_or(TrimDomainError::ResourceLimit)?;
        if self.nodes > self.limit {
            return Err(TrimDomainError::ResourceLimit);
        }
        match validated_curve2d_jet(curve, span.range, budget, cancel) {
            Ok(jet) => {
                if !jet.position.into_iter().all(ValidatedInterval::is_finite) {
                    return Err(TrimDomainError::InvalidChart);
                }
                *self.bounds = Some(match *self.bounds {
                    Some(old) => [old[0].hull(jet.position[0]), old[1].hull(jet.position[1])],
                    None => jet.position,
                });
                let bytes = self.base_bytes.saturating_add(
                    growing_capacity(self.spans.len(), self.spans.capacity())
                        .saturating_mul(size_of::<TrimSpan>()),
                );
                admit_bytes(bytes, self.byte_limit)?;
                self.spans.push(span);
                admit_bytes(
                    self.base_bytes.saturating_add(
                        self.spans.capacity().saturating_mul(size_of::<TrimSpan>()),
                    ),
                    self.byte_limit,
                )?;
                Ok(())
            }
            Err(
                error @ (ValidatedGeometryError::DenominatorContainsZero
                | ValidatedGeometryError::ConditioningFailure),
            ) => {
                let middle = span.range[0] * 0.5 + span.range[1] * 0.5;
                if depth >= 64 || middle <= span.range[0] || middle >= span.range[1] {
                    return Err(error.into());
                }
                self.admit(
                    curve,
                    TrimSpan {
                        range: [span.range[0], middle],
                        ..span
                    },
                    depth + 1,
                    budget,
                    cancel,
                )?;
                self.admit(
                    curve,
                    TrimSpan {
                        range: [middle, span.range[1]],
                        ..span
                    },
                    depth + 1,
                    budget,
                    cancel,
                )
            }
            Err(error) => Err(error.into()),
        }
    }
}

/// Prepares authoritative stored native trim charts without reconstructing pcurves.
///
/// The caller must retain its exact source-to-native pcurve correspondence; plain
/// 3D topology and sampled payloads do not establish that correspondence. Exact
/// clamped rational and polynomial spline endpoints use dyadic cross products.
/// Independently rounded endpoint expressions retain a bounded convex joint
/// envelope tied to the shared original native vertex; they are never declared
/// equal. Strict results cover every local chart completion in that envelope,
/// while its neighborhood remains `Boundary`. Source-map uncertainty beyond
/// these native endpoint envelopes must be separately retained by the importer.
/// Unstored periodic lifts and pole identifications return a typed refusal.
/// Preparation encloses every complete knot span, joint uncertainty and bounded
/// nonzero-winding region in the finite returned box.
pub fn prepare_face_parameter_domain<'a, T: NativeTrimCurve>(
    trimmed: &'a TrimmedFace<Point3, Curve, Surface, T>,
    limits: TrimDomainLimits,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<PreparedFaceParameterDomain<'a>, TrimDomainError> {
    prepare_face_parameter_domain_with_uncertainty(trimmed, [0.0; 2], limits, budget, cancel)
}

/// Prepares a complete source-chart tube with a certified whole-boundary UV margin.
///
/// `margin_uv` must enclose coordinate error of every true oriented source coedge
/// and its source-chart joints, with an orientation-preserving continuous
/// correspondence to the supplied stored curves. It is a numerical enclosure,
/// not an unchecked exact-coverage flag. The importer owns that source-map proof.
/// Native strict classifications avoid the whole tube and therefore hold for all
/// source boundaries homotopic inside it. Positional margins do not certify
/// derivative errors; nominal boundary jets remain explicitly nominal.
pub fn prepare_face_parameter_domain_with_uncertainty<'a, T: NativeTrimCurve>(
    trimmed: &'a TrimmedFace<Point3, Curve, Surface, T>,
    margin_uv: [f64; 2],
    limits: TrimDomainLimits,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<PreparedFaceParameterDomain<'a>, TrimDomainError> {
    prepare_face_parameter_domain_with_natural_boundaries(
        trimmed,
        margin_uv,
        &[],
        limits,
        budget,
        cancel,
    )
}

/// Prepares stored charts with explicit original-source natural-boundary authority.
///
/// Every evidence item must identify one actual unresolved material-oriented
/// joint of this face. Its original-source natural-boundary correspondence is
/// caller-established; proximity does not supply it. Native preparation verifies
/// the shared vertex/use identities and actual clamped carrier boundary, then
/// bounds the complete supported UV joint tube relative to that vertex using
/// coefficient-first rational residuals and transverse derivative inflation.
/// Only numeric residuals within the supplied per-coordinate allowance are
/// accepted. This does not establish normal, triangle or surface coverage.
pub fn prepare_face_parameter_domain_with_natural_boundaries<'a, T: NativeTrimCurve>(
    trimmed: &'a TrimmedFace<Point3, Curve, Surface, T>,
    margin_uv: [f64; 2],
    natural_boundaries: &[NaturalBoundaryJointEvidence],
    limits: TrimDomainLimits,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<PreparedFaceParameterDomain<'a>, TrimDomainError> {
    budget.charge(1, cancel)?;
    if !margin_uv
        .into_iter()
        .all(|value| value.is_finite() && value >= 0.0)
    {
        return Err(TrimDomainError::InvalidChart);
    }
    if natural_boundaries.len() > limits.max_coedges {
        return Err(TrimDomainError::ResourceLimit);
    }
    budget.charge(natural_boundaries.len(), cancel)?;
    let mut used_evidence = 0_usize;
    let face = trimmed.face();
    if face.absolute_boundaries().is_empty() {
        return Err(TrimDomainError::InvalidChart);
    }
    let count = face
        .absolute_boundaries()
        .iter()
        .try_fold(0_usize, |count, wire| {
            budget.charge(1, cancel)?;
            let next = count
                .checked_add(wire.len())
                .ok_or(TrimDomainError::ResourceLimit)?;
            if next > limits.max_coedges {
                Err(TrimDomainError::ResourceLimit)
            } else {
                Ok(next)
            }
        })?;
    admit_bytes(metadata_bytes(count, 0, 0), limits.max_retained_bytes)?;
    budget.charge(count, cancel)?;
    let mut coedges = Vec::with_capacity(count);
    admit_bytes(
        metadata_bytes(coedges.capacity(), 0, 0),
        limits.max_retained_bytes,
    )?;
    let mut joints = Vec::new();
    let periods = face.with_surface(|surface| [surface.period_u(), surface.period_v()]);
    let mut spans = Vec::new();
    let mut bounds: Option<[ValidatedInterval; 2]> = None;
    let mut preparation_nodes = 0_usize;
    for (wire_index, wire) in face.absolute_boundaries().iter().enumerate() {
        if wire.is_empty() {
            return Err(TrimDomainError::InvalidChart);
        }
        let first = coedges.len();
        for (coedge_index, edge) in wire.iter().enumerate() {
            budget.charge(1, cancel)?;
            let curve = trimmed.trims()[wire_index][coedge_index]
                .as_ref()
                .ok_or(TrimDomainError::MissingExactTrim {
                    wire_index,
                    coedge_index,
                })?
                .native_uv_curve();
            let (a, b) = curve
                .try_range_tuple()
                .ok_or(TrimDomainError::InvalidChart)?;
            if !a.is_finite() || !b.is_finite() || a >= b {
                return Err(TrimDomainError::InvalidChart);
            }
            let orientation = edge.orientation() == face.orientation();
            let index = coedges.len();
            coedges.push(PreparedTrimCoedge {
                identity: ChartCoedgeIdentity {
                    face_id: face.id(),
                    wire_index,
                    coedge_index,
                    edge_id: edge.id(),
                    orientation,
                },
                curve,
                domain: curve.parameter_range(),
                oriented_range: if orientation { [a, b] } else { [b, a] },
                span_range: spans.len()..spans.len(),
            });
            let retained = metadata_bytes(coedges.capacity(), spans.capacity(), joints.capacity());
            let knots = breaks(
                curve,
                [a, b],
                limits.max_subdivisions,
                limits.max_retained_bytes.saturating_sub(retained),
                budget,
                cancel,
            )?;
            for pair in knots.windows(2) {
                let mut admission = SpanAdmission {
                    spans: &mut spans,
                    bounds: &mut bounds,
                    nodes: preparation_nodes,
                    limit: limits.max_subdivisions,
                    base_bytes: metadata_bytes(coedges.capacity(), 0, joints.capacity())
                        .saturating_add(knots.capacity().saturating_mul(size_of::<f64>())),
                    byte_limit: limits.max_retained_bytes,
                };
                admission.admit(
                    curve,
                    TrimSpan {
                        coedge: index,
                        range: [pair[0], pair[1]],
                    },
                    0,
                    budget,
                    cancel,
                )?;
                preparation_nodes = admission.nodes;
            }
            coedges[index].span_range.end = spans.len();
        }
        for index in first..coedges.len() {
            budget.charge(1, cancel)?;
            let next = if index + 1 == coedges.len() {
                first
            } else {
                index + 1
            };
            let current = &coedges[index];
            let next = &coedges[next];
            // Closure follows absolute wire order, independently of face inversion.
            let current_back = wire[current.identity.coedge_index].orientation();
            let next_back = !wire[next.identity.coedge_index].orientation();
            if !closure::endpoints_equal(current.curve, current_back, next.curve, next_back) {
                let incoming_edge = &wire[current.identity.coedge_index];
                let outgoing_edge = &wire[next.identity.coedge_index];
                if incoming_edge.back().id() != outgoing_edge.front().id() {
                    return Err(TrimDomainError::UnclosedChart);
                }
                let current_range = current.curve.range_tuple();
                let next_range = next.curve.range_tuple();
                let ta = if current_back {
                    current_range.1
                } else {
                    current_range.0
                };
                let tb = if next_back {
                    next_range.1
                } else {
                    next_range.0
                };
                let mut front =
                    validated_curve2d_jet(current.curve, [ta; 2], budget, cancel)?.position;
                let mut back = validated_curve2d_jet(next.curve, [tb; 2], budget, cancel)?.position;
                let mut enclosure = [front[0].hull(back[0]), front[1].hull(back[1])];
                if !face.orientation() {
                    std::mem::swap(&mut front, &mut back);
                }
                let (incoming, outgoing) = if face.orientation() {
                    (current.identity, next.identity)
                } else {
                    (next.identity, current.identity)
                };
                let mut selected = None;
                for evidence in natural_boundaries {
                    budget.charge(1, cancel)?;
                    if evidence.vertex_id == incoming_edge.back().id()
                        && natural_boundary::same_use(evidence.incoming, incoming)
                        && natural_boundary::same_use(evidence.outgoing, outgoing)
                    {
                        if selected.is_some() {
                            return Err(TrimDomainError::InvalidChart);
                        }
                        selected = Some(evidence);
                    }
                }
                let kind = if let Some(evidence) = selected {
                    used_evidence += 1;
                    natural_boundary::completion(
                        face,
                        incoming_edge.back().point(),
                        evidence,
                        [(current.curve, ta), (next.curve, tb)],
                        poles::JointNeighborhood {
                            enclosure: &mut enclosure,
                            front,
                            back,
                            margin: margin_uv,
                        },
                        (
                            limits,
                            limits.max_retained_bytes.saturating_sub(metadata_bytes(
                                coedges.capacity(),
                                spans.capacity(),
                                joints.capacity(),
                            )),
                        ),
                        (budget, cancel),
                    )?
                } else {
                    poles::pole_completion(
                        face,
                        incoming_edge.back().point(),
                        poles::JointNeighborhood {
                            enclosure: &mut enclosure,
                            front,
                            back,
                            margin: margin_uv,
                        },
                        budget,
                        cancel,
                    )?
                    .unwrap_or(TrimJointKind::EndpointUncertainty)
                };
                if matches!(kind, TrimJointKind::EndpointUncertainty)
                    && (0..2).any(|axis| {
                        periods[axis].is_some_and(|period| {
                            !period.is_finite()
                                || period <= 0.0
                                || enclosure[axis].width() >= period * 0.5
                        })
                    })
                {
                    return Err(TrimDomainError::UnsupportedSeam);
                }
                if joints.len() >= limits.max_coedges {
                    return Err(TrimDomainError::ResourceLimit);
                }
                budget.charge(1, cancel)?;
                admit_bytes(
                    metadata_bytes(
                        coedges.capacity(),
                        spans.capacity(),
                        growing_capacity(joints.len(), joints.capacity()),
                    ),
                    limits.max_retained_bytes,
                )?;
                joints.push(PreparedTrimJoint {
                    vertex_id: incoming_edge.back().id(),
                    vertex_point: incoming_edge.back().point(),
                    incoming,
                    outgoing,
                    enclosure,
                    front,
                    back,
                    kind,
                });
                admit_bytes(
                    metadata_bytes(coedges.capacity(), spans.capacity(), joints.capacity()),
                    limits.max_retained_bytes,
                )?;
                bounds = Some(match bounds {
                    Some(old) => [old[0].hull(enclosure[0]), old[1].hull(enclosure[1])],
                    None => enclosure,
                });
            }
        }
    }
    if used_evidence != natural_boundaries.len() {
        return Err(TrimDomainError::InvalidChart);
    }
    let bounds = bounds.ok_or(TrimDomainError::InvalidChart)?;
    let bounds = [
        bounds[0] + ValidatedInterval::new(-margin_uv[0], margin_uv[0])?,
        bounds[1] + ValidatedInterval::new(-margin_uv[1], margin_uv[1])?,
    ];
    if !bounds.into_iter().all(ValidatedInterval::is_finite) {
        return Err(TrimDomainError::InvalidChart);
    }
    Ok(PreparedFaceParameterDomain {
        face,
        coedges,
        joints,
        spans,
        bounds: bounds.map(|interval| [interval.lower, interval.upper]),
        limits,
        uncertainty_uv: margin_uv,
    })
}

#[derive(Clone, Copy)]
struct WindingArc {
    front: [ValidatedInterval; 2],
    back: [ValidatedInterval; 2],
}

fn ray_winding(arcs: &[WindingArc], query: [ValidatedInterval; 2], slope: f64) -> Option<isize> {
    let slope = ValidatedInterval::point(slope);
    arcs.iter().try_fold(0_isize, |winding, arc| {
        let a = [arc.front[0] - query[0], arc.front[1] - query[1]];
        let b = [arc.back[0] - query[0], arc.back[1] - query[1]];
        let ay = a[1] - slope * a[0];
        let by = b[1] - slope * b[0];
        if !ay.is_finite() || !by.is_finite() || ay.contains_zero() || by.contains_zero() {
            return None;
        }
        if (ay.lower > 0.0) == (by.lower > 0.0) {
            return Some(winding);
        }
        let determinant = a[0] * b[1] - a[1] * b[0];
        if !determinant.is_finite() || determinant.contains_zero() {
            return None;
        }
        let upward = ay.upper < 0.0;
        if (determinant.lower > 0.0) == upward {
            if upward {
                winding.checked_add(1)
            } else {
                winding.checked_sub(1)
            }
        } else {
            Some(winding)
        }
    })
}

impl<'a> PreparedFaceParameterDomain<'a> {
    /// Returns the borrowed original face for carrier interval evaluation.
    pub const fn face(&self) -> &'a Face { self.face }

    /// Returns complete ordered native coedge metadata without cloning geometry.
    ///
    /// Entries are grouped by absolute wire ordinal. Face inversion reverses each
    /// occurrence's material direction, explicitly recorded in `oriented_range`.
    pub fn coedges(&self) -> &[PreparedTrimCoedge<'a>] { &self.coedges }

    /// Returns every bounded shared-vertex endpoint uncertainty envelope.
    ///
    /// These are not synthetic exact pcurves. Coverage consumers must include
    /// their complete envelopes when bounding display error or trim contact.
    pub fn joints(&self) -> &[PreparedTrimJoint] { &self.joints }

    /// Returns a finite closed box covering every bounded material chart region.
    pub const fn closed_parameter_box(&self) -> [[f64; 2]; 2] { self.bounds }

    /// Returns the certified source-map positional UV error retained by this domain.
    pub const fn uncertainty_uv(&self) -> [f64; 2] { self.uncertainty_uv }

    /// Borrows every ascending closed native parameter interval of an original coedge.
    ///
    /// Includes all exact transported knots and retained adaptive denominator spans.
    /// The borrow is constant-time and does not scan other coedges. Material direction
    /// remains in the coedge's `oriented_range`; no interval or circuit is omitted.
    /// Invalid original occurrence indices return a typed chart error.
    pub fn boundary_parameter_intervals(
        &self,
        coedge_index: usize,
    ) -> Result<impl Iterator<Item = [f64; 2]> + '_, TrimDomainError> {
        let coedge = self
            .coedges
            .get(coedge_index)
            .ok_or(TrimDomainError::InvalidChart)?;
        let spans = self
            .spans
            .get(coedge.span_range.clone())
            .ok_or(TrimDomainError::InvalidChart)?;
        Ok(spans.iter().map(|span| span.range))
    }

    /// Returns retained object and owned buffer capacity bytes, excluding borrowed geometry.
    pub fn retained_bytes(&self) -> usize {
        metadata_bytes(
            self.coedges.capacity(),
            self.spans.capacity(),
            self.joints.capacity(),
        )
    }

    /// Intersects the cover with an independently certified complete source support domain.
    ///
    /// The caller must prove the original material face lies in `bounds`; a
    /// finite native parameter range alone is not that proof. Stored trim curves,
    /// source-map uncertainty and joint lineage are not discarded or rewritten.
    pub fn restrict_parameter_domain(
        &mut self,
        bounds: [[f64; 2]; 2],
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> Result<(), TrimDomainError> {
        budget.charge(1, cancel)?;
        let restriction = validate_box(bounds)?;
        let current = validate_box(self.bounds)?;
        let clipped = [
            current[0]
                .intersection(restriction[0])
                .ok_or(TrimDomainError::InvalidChart)?,
            current[1]
                .intersection(restriction[1])
                .ok_or(TrimDomainError::InvalidChart)?,
        ];
        self.bounds = clipped.map(|axis| [axis.lower, axis.upper]);
        Ok(())
    }

    /// Encloses a bounded native boundary subarc and its first three UV derivatives.
    ///
    /// Uses the same genuine stored coefficients as winding. A range crossing a
    /// nonsmooth knot is refused; callers may split at the native curve knots.
    pub fn boundary_span_enclosure(
        &self,
        coedge_index: usize,
        range: [f64; 2],
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> Result<ValidatedCurveJet2D, TrimDomainError> {
        budget.charge(1, cancel)?;
        let coedge = self
            .coedges
            .get(coedge_index)
            .ok_or(TrimDomainError::InvalidChart)?;
        let (a, b) = coedge.curve.range_tuple();
        if range[0] < a || range[1] > b {
            return Err(TrimDomainError::InvalidChart);
        }
        Ok(validated_curve2d_jet(coedge.curve, range, budget, cancel)?)
    }

    /// Encloses the entire true source-boundary position tube on a nominal subarc.
    ///
    /// Unlike nominal boundary jets, this includes the supplied whole-source UV
    /// mapping error. It does not assert any bound on true-source derivatives.
    /// The tube is intersected with independently certified restricted source
    /// support, if supplied; an empty intersection is a typed unresolved chart.
    pub fn boundary_position_enclosure(
        &self,
        coedge_index: usize,
        range: [f64; 2],
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> Result<[ValidatedInterval; 2], TrimDomainError> {
        let nominal = self
            .boundary_span_enclosure(coedge_index, range, budget, cancel)?
            .position;
        let position = [
            nominal[0] + ValidatedInterval::new(-self.uncertainty_uv[0], self.uncertainty_uv[0])?,
            nominal[1] + ValidatedInterval::new(-self.uncertainty_uv[1], self.uncertainty_uv[1])?,
        ];
        if position.into_iter().all(ValidatedInterval::is_finite) {
            let domain = validate_box(self.bounds)?;
            Ok([
                position[0]
                    .intersection(domain[0])
                    .ok_or(TrimDomainError::Geometry(
                        ValidatedGeometryError::Unresolved,
                    ))?,
                position[1]
                    .intersection(domain[1])
                    .ok_or(TrimDomainError::Geometry(
                        ValidatedGeometryError::Unresolved,
                    ))?,
            ])
        } else {
            Err(TrimDomainError::InvalidChart)
        }
    }

    /// Classifies an entire UV point through the same interval winding algorithm.
    pub fn classify_parameter_point(
        &self,
        point: [f64; 2],
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> Result<ParameterBoxClassification, TrimDomainError> {
        self.classify_parameter_box([[point[0]; 2], [point[1]; 2]], budget, cancel)
    }

    fn admit_winding_scratch(&self, arcs: usize, pending: usize) -> Result<(), TrimDomainError> {
        let bytes = self
            .retained_bytes()
            .checked_add(size_of::<Vec<WindingArc>>())
            .and_then(|bytes| bytes.checked_add(size_of::<Vec<TrimSpan>>()))
            .and_then(|bytes| bytes.checked_add(arcs.checked_mul(size_of::<WindingArc>())?))
            .and_then(|bytes| bytes.checked_add(pending.checked_mul(size_of::<TrimSpan>())?))
            .ok_or(TrimDomainError::ResourceLimit)?;
        admit_bytes(bytes, self.limits.max_retained_bytes)
    }

    /// Classifies the complete connected query box, not a finite set of probes.
    ///
    /// Strict results require every full trim-arc enclosure to miss the box and
    /// an interval-certified chord winding. Contact or finite subdivision/ray
    /// ambiguity returns `Boundary`; exhausted cumulative work returns an error.
    /// No missed, unbounded or unavailable boundary occurrence is discarded.
    pub fn classify_parameter_box(
        &self,
        bounds: [[f64; 2]; 2],
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> Result<ParameterBoxClassification, TrimDomainError> {
        budget.charge(1, cancel)?;
        let query = validate_box(bounds)?;
        let query = [
            query[0] + ValidatedInterval::new(-self.uncertainty_uv[0], self.uncertainty_uv[0])?,
            query[1] + ValidatedInterval::new(-self.uncertainty_uv[1], self.uncertainty_uv[1])?,
        ];
        if !query.into_iter().all(ValidatedInterval::is_finite) {
            return Err(TrimDomainError::InvalidChart);
        }
        let domain = validate_box(self.bounds)?;
        if disjoint(query, domain) {
            return Ok(ParameterBoxClassification::Outside);
        }
        if (0..2).any(|axis| {
            query[axis].lower < domain[axis].lower || query[axis].upper > domain[axis].upper
        }) {
            return Ok(ParameterBoxClassification::Boundary);
        }
        let initial_arcs = self
            .spans
            .len()
            .checked_add(self.joints.len())
            .ok_or(TrimDomainError::ResourceLimit)?;
        self.admit_winding_scratch(initial_arcs, 0)?;
        budget.charge(initial_arcs, cancel)?;
        let mut arcs = Vec::with_capacity(initial_arcs);
        self.admit_winding_scratch(arcs.capacity(), 0)?;
        for joint in &self.joints {
            budget.charge(1, cancel)?;
            if !disjoint(joint.enclosure, query) {
                return Ok(ParameterBoxClassification::Boundary);
            }
            arcs.push(WindingArc {
                front: joint.front,
                back: joint.back,
            });
        }
        let mut pending = Vec::new();
        let mut nodes = 0_usize;
        for span in &self.spans {
            self.admit_winding_scratch(
                arcs.capacity(),
                growing_capacity(pending.len(), pending.capacity()),
            )?;
            pending.push(*span);
            self.admit_winding_scratch(arcs.capacity(), pending.capacity())?;
            while let Some(span) = pending.pop() {
                budget.charge(1, cancel)?;
                nodes = nodes.checked_add(1).ok_or(TrimDomainError::ResourceLimit)?;
                if nodes > self.limits.max_subdivisions {
                    return Ok(ParameterBoxClassification::Boundary);
                }
                let coedge = &self.coedges[span.coedge];
                let position =
                    validated_curve2d_jet(coedge.curve, span.range, budget, cancel)?.position;
                if disjoint(position, query) {
                    let mut front =
                        validated_curve2d_jet(coedge.curve, [span.range[0]; 2], budget, cancel)?
                            .position;
                    let mut back =
                        validated_curve2d_jet(coedge.curve, [span.range[1]; 2], budget, cancel)?
                            .position;
                    if !coedge.identity.orientation {
                        std::mem::swap(&mut front, &mut back);
                    }
                    budget.charge(1, cancel)?;
                    self.admit_winding_scratch(
                        growing_capacity(arcs.len(), arcs.capacity()),
                        pending.capacity(),
                    )?;
                    arcs.push(WindingArc { front, back });
                    self.admit_winding_scratch(arcs.capacity(), pending.capacity())?;
                } else {
                    let middle = span.range[0] * 0.5 + span.range[1] * 0.5;
                    if middle <= span.range[0] || middle >= span.range[1] {
                        return Ok(ParameterBoxClassification::Boundary);
                    }
                    budget.charge(2, cancel)?;
                    self.admit_winding_scratch(
                        arcs.capacity(),
                        growing_capacity(pending.len(), pending.capacity()),
                    )?;
                    pending.push(TrimSpan {
                        range: [middle, span.range[1]],
                        ..span
                    });
                    self.admit_winding_scratch(arcs.capacity(), pending.capacity())?;
                    self.admit_winding_scratch(
                        arcs.capacity(),
                        growing_capacity(pending.len(), pending.capacity()),
                    )?;
                    pending.push(TrimSpan {
                        range: [span.range[0], middle],
                        ..span
                    });
                    self.admit_winding_scratch(arcs.capacity(), pending.capacity())?;
                }
            }
        }
        let representative = query.map(|axis| ValidatedInterval::point(axis.lower));
        for ray in 0..arcs.len().saturating_mul(2).saturating_add(1) {
            budget.charge(arcs.len().saturating_add(1), cancel)?;
            let slope = if ray == 0 {
                0.0
            } else if ray % 2 == 0 {
                -((ray / 2) as f64)
            } else {
                (ray / 2 + 1) as f64
            };
            if let Some(winding) = ray_winding(&arcs, representative, slope) {
                return Ok(if winding == 0 {
                    ParameterBoxClassification::Outside
                } else {
                    ParameterBoxClassification::Inside
                });
            }
        }
        Ok(ParameterBoxClassification::Boundary)
    }
}

#[cfg(test)]
mod native_trim_regressions {
    use super::*;
    use crate::{
        BsplineCurve, BsplineSurface, EuclideanSpace, Invertible, KnotVector, Line, Matrix3, Plane,
        Point2, Processor, Sphere, TrimmedCurve, UnitCircle, Vector3,
    };
    use monstertruck_topology::{Edge, Vertex, Wire};

    fn limits() -> TrimDomainLimits {
        TrimDomainLimits {
            max_coedges: 16,
            max_subdivisions: 4096,
            max_retained_bytes: 1_048_576,
        }
    }

    fn square_with_hole() -> TrimmedFace<Point3, Curve, Surface, Curve2D> {
        let loops = [
            [
                Point2::new(0.0, 0.0),
                Point2::new(4.0, 0.0),
                Point2::new(4.0, 4.0),
                Point2::new(0.0, 4.0),
            ],
            [
                Point2::new(1.0, 1.0),
                Point2::new(1.0, 3.0),
                Point2::new(3.0, 3.0),
                Point2::new(3.0, 1.0),
            ],
        ];
        let mut wires = Vec::new();
        let mut trims = Vec::new();
        for points in loops {
            let vertices = points.map(|point| Vertex::new(Point3::new(point.x, point.y, 0.0)));
            let wire: Wire<Point3, Curve> = (0..4)
                .map(|i| {
                    Edge::new(
                        &vertices[i],
                        &vertices[(i + 1) % 4],
                        Curve::Line(Line(vertices[i].point(), vertices[(i + 1) % 4].point())),
                    )
                })
                .collect();
            wires.push(wire);
            trims.push(
                (0..4)
                    .map(|i| Some(Curve2D::Line(Line(points[i], points[(i + 1) % 4]))))
                    .collect(),
            );
        }
        TrimmedFace::new(
            Face::new(
                wires,
                Surface::Plane(Plane::from_origin_vectors(
                    Point3::origin(),
                    Vector3::unit_x(),
                    Vector3::unit_y(),
                )),
            ),
            trims,
        )
    }

    #[test]
    fn winding_respects_holes_complete_boxes_and_oriented_use_spans() {
        let face = square_with_hole();
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(1_000_000);
        let domain = prepare_face_parameter_domain(&face, limits(), &mut budget, &cancel).unwrap();
        assert_eq!(domain.coedges().len(), 8);
        for (index, coedge) in domain.coedges().iter().enumerate() {
            assert_eq!(coedge.identity.coedge_index, index % 4);
            assert_eq!(coedge.identity.wire_index, index / 4);
            assert_eq!(
                domain
                    .boundary_parameter_intervals(index)
                    .unwrap()
                    .collect::<Vec<_>>(),
                vec![[0.0, 1.0]]
            );
        }
        let query = |x: (f64, f64), y: (f64, f64)| [[x.0, x.1], [y.0, y.1]];
        assert_eq!(
            domain
                .classify_parameter_box(query((0.25, 0.75), (1.5, 2.5)), &mut budget, &cancel)
                .unwrap(),
            ParameterBoxClassification::Inside
        );
        assert_eq!(
            domain
                .classify_parameter_box(query((1.5, 2.5), (1.5, 2.5)), &mut budget, &cancel)
                .unwrap(),
            ParameterBoxClassification::Outside
        );
        assert_eq!(
            domain
                .classify_parameter_point([1.0, 2.0], &mut budget, &cancel)
                .unwrap(),
            ParameterBoxClassification::Boundary
        );
        assert_eq!(
            domain
                .classify_parameter_box(query((6.0, 7.0), (0.0, 1.0)), &mut budget, &cancel)
                .unwrap(),
            ParameterBoxClassification::Outside
        );
    }

    #[test]
    fn nested_inverse_uv_knots_keep_independent_trim_anchors() {
        let basis = Curve2D::BsplineCurve(BsplineCurve::new(
            KnotVector::from(vec![10.0, 10.0, 12.0, 14.0, 16.0, 18.0, 18.0]),
            (0..5).map(|i| Point2::new(i as f64, 0.0)).collect(),
        ));
        let mut inner = Processor::new(TrimmedCurve::new(Box::new(basis), (11.0, 17.0)));
        inner.invert();
        let mut outer = Processor::new(TrimmedCurve::new(
            Box::new(Curve2D::TrimmedCurve(inner)),
            (11.0, 14.0),
        ));
        outer.invert();
        assert_eq!(
            breaks(
                &Curve2D::TrimmedCurve(outer),
                [11.0, 14.0],
                limits().max_subdivisions,
                limits().max_retained_bytes,
                &mut ValidatedGeometryBudget::new(100_000),
                &AtomicBool::new(false)
            )
            .unwrap(),
            vec![11.0, 13.0, 14.0]
        );
    }

    #[test]
    fn planar_conic_rechart_retains_original_basis_and_homogeneous_factors() {
        let uv = Curve2D::Conic(Conic2D::Ellipse(Processor::new(TrimmedCurve::new(
            UnitCircle::<Point2>::new(),
            (0.0, std::f64::consts::TAU),
        ))));
        let source = Surface::Plane(Plane::from_origin_vectors(
            Point3::new(4.0, 5.0, 0.0),
            Vector3::new(1.1, 0.0, 0.0),
            Vector3::new(0.0, 1.1, 0.0),
        ));
        let curve = Curve::ParameterCurve(ParameterCurve::new(uv, Box::new(source)));
        let target = Plane::from_origin_vectors(
            Point3::origin(),
            Vector3::new(37.0, 0.0, 0.0),
            Vector3::new(0.0, 31.0, 0.0),
        );
        let projected = certified_planar_parameter_curve(
            &curve,
            &target,
            &mut ValidatedGeometryBudget::new(100_000),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(projected.range_tuple(), (0.0, std::f64::consts::TAU));
        let mut basis = &projected;
        let mut factors = 0;
        while let Curve2D::TrimmedCurve(wrapper) = basis {
            factors += 1;
            assert!(wrapper.orientation());
            basis = wrapper.entity().curve();
        }
        assert!(factors >= 2);
        assert!(matches!(basis, Curve2D::Conic(Conic2D::Ellipse(_))));
    }

    #[test]
    fn restricted_tensor_row_preserves_native_parameter_and_inverse_use() {
        let surface = Surface::BsplineSurface(BsplineSurface::new(
            (
                KnotVector::from(vec![0.0, 0.0, 1.0, 1.0]),
                KnotVector::from(vec![0.0, 0.0, 1.0, 2.0, 2.0]),
            ),
            vec![
                vec![
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(0.0, 1.0, 0.0),
                    Point3::new(0.0, 2.0, 0.0),
                ],
                vec![
                    Point3::new(1.0, 0.0, 0.0),
                    Point3::new(1.0, 1.0, 0.0),
                    Point3::new(1.0, 2.0, 0.0),
                ],
            ],
        ));
        let basis = Curve::BsplineCurve(BsplineCurve::new(
            KnotVector::from(vec![10.0, 10.0, 14.0, 14.0]),
            vec![Point3::origin(), Point3::new(0.0, 1.0, 0.0)],
        ));
        let mut wrapper = Processor::new(TrimmedCurve::new(Box::new(basis), (10.0, 14.0)));
        wrapper.invert();
        let uv = certified_surface_boundary_parameter_curve(
            &Curve::TrimmedCurve(wrapper),
            &surface,
            NativeSurfaceBoundary::UFront,
            [0.0, 1.0],
            &mut ValidatedGeometryBudget::new(100_000),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(uv.range_tuple(), (10.0, 14.0));
        let Curve2D::TrimmedCurve(wrapper) = uv else {
            panic!("lost original trim use")
        };
        assert!(!wrapper.orientation());
        assert_eq!(*wrapper.transform(), Matrix3::from_scale(1.0));
        assert_eq!(wrapper.entity().range(), (10.0, 14.0));
    }

    #[test]
    fn spherical_quarter_lune_has_bounded_native_pole_completions() {
        let sphere = Surface::SphericalSurface(Processor::new(Sphere::new(Point3::origin(), 1.0)));
        let north = Vertex::new(Point3::new(0.0, 0.0, 1.0));
        let south = Vertex::new(Point3::new(0.0, 0.0, -1.0));
        let uv = [0.0, std::f64::consts::FRAC_PI_2].map(|phi| {
            Curve2D::Line(Line(
                Point2::new(0.0, phi),
                Point2::new(std::f64::consts::PI, phi),
            ))
        });
        let first = Edge::new(
            &north,
            &south,
            Curve::ParameterCurve(ParameterCurve::new(uv[0].clone(), Box::new(sphere.clone()))),
        );
        let second = Edge::new(
            &north,
            &south,
            Curve::ParameterCurve(ParameterCurve::new(uv[1].clone(), Box::new(sphere.clone()))),
        )
        .inverse();
        let wire: Wire<Point3, Curve> = [first, second].into_iter().collect();
        let face = TrimmedFace::new(
            Face::new(vec![wire], sphere),
            vec![uv.into_iter().map(Some).collect()],
        );
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(1_000_000);
        let domain = prepare_face_parameter_domain(&face, limits(), &mut budget, &cancel).unwrap();
        assert_eq!(domain.joints().len(), 2);
        for joint in domain.joints() {
            let TrimJointKind::SphericalPole {
                vertex_residual, ..
            } = joint.kind
            else {
                panic!("missing analytic pole completion")
            };
            assert!(
                vertex_residual
                    .iter()
                    .all(|coordinate| coordinate.contains_zero())
            );
        }
        assert_eq!(
            domain
                .classify_parameter_point(
                    [std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_4],
                    &mut budget,
                    &cancel
                )
                .unwrap(),
            ParameterBoxClassification::Inside
        );
        assert_eq!(
            domain
                .classify_parameter_point([0.0, std::f64::consts::FRAC_PI_4], &mut budget, &cancel)
                .unwrap(),
            ParameterBoxClassification::Boundary
        );
    }
}
