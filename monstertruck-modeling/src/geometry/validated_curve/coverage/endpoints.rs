use super::super::{CoordinatePlane, Projection, certify, implicit_jet};
use super::*;
use crate::ValidatedScalarCurveJet;
use crate::geometry::validated_interval::{
    ProjectionExecution, ScalarProjection, Taylor, projected_surface_curve_jet,
    projected_surface_taylor, validated_surface_position_bounds,
};
use monstertruck_topology::EdgeId;
use monstertruck_traits::BoundedCurve;
use std::fmt;
/// Original source geometry for one oriented trim occurrence, without a candidate.
///
/// This binding retains the original native gauge and the independently proved
/// chart correspondence. Neither an edge identity nor a numerical cut is source
/// geometry authority without the complete retained coefficient association.
#[derive(Clone, Copy, Debug)]
pub struct ValidatedPlaneTrimBoundaryBinding<'a> {
    /// Original occurrence in the prepared face domain.
    pub coedge_index: usize,
    /// Original shared edge identity, checked against the occurrence.
    pub original_edge_id: EdgeId<Curve>,
    /// Immutable original native boundary continuation.
    pub boundary: &'a PreparedValidatedCurve<'a>,
    /// Actual active native domain, without an assumed affine rescaling.
    pub boundary_range: [f64; 2],
    /// Certified source-to-query spatial unit map.
    pub boundary_scale: f64,
    /// Already certified physical source/chart correspondence error.
    pub source_chart_error_mm: f64,
}

/// Original coedge incidence of an actual selected Boolean curve endpoint.
///
/// The caller validates the original cut genealogy and shared output vertex.
/// The cut parameter proposes a root; it never proves root existence or identity.
/// `boundary` must be the retained original branch, not a newly selected guide.
#[derive(Clone, Copy, Debug)]
pub struct ValidatedPlaneTrimEndpointBinding<'a> {
    /// Original occurrence in the prepared face domain.
    pub coedge_index: usize,
    /// Original shared edge identity, checked against the occurrence.
    pub original_edge_id: EdgeId<Curve>,
    /// Immutable original native boundary continuation.
    pub boundary: &'a PreparedValidatedCurve<'a>,
    /// Actual active native domain, without an assumed affine rescaling.
    pub boundary_range: [f64; 2],
    /// Certified source-to-query spatial unit map.
    pub boundary_scale: f64,
    /// Candidate in the complete coverage input.
    pub candidate_index: usize,
    /// Whether this is the ascending candidate range's back endpoint.
    pub endpoint_is_back: bool,
    /// Actual original native cut parameter from construction genealogy.
    pub source_cut_parameter: f64,
    /// Already certified physical source/chart correspondence error.
    pub source_chart_error_mm: f64,
}

impl<'a> ValidatedPlaneTrimEndpointBinding<'a> {
    /// Borrows only the original source association, without candidate or cut tags.
    pub const fn boundary_binding(&self) -> ValidatedPlaneTrimBoundaryBinding<'a> {
        ValidatedPlaneTrimBoundaryBinding {
            coedge_index: self.coedge_index,
            original_edge_id: self.original_edge_id,
            boundary: self.boundary,
            boundary_range: self.boundary_range,
            boundary_scale: self.boundary_scale,
            source_chart_error_mm: self.source_chart_error_mm,
        }
    }
}

/// A proved original trim-root stratum attached to one retained candidate end.
///
/// The private fields are produced only after complete original parameter/root
/// coverage, an oriented transverse crossing and common unique continuation.
/// Consumers add the achieved extension to their existing curve error and bound
/// the additional Green integral using the retained position and arc length.
#[derive(Clone, Debug)]
pub struct ValidatedPlaneTrimEndpoint {
    candidate_index: usize,
    endpoint_is_back: bool,
    coedge_index: usize,
    original_edge_id: EdgeId<Curve>,
    source_root_parameter: [f64; 2],
    extension_mm: f64,
    arc_length_mm: f64,
    position: [ValidatedInterval; 3],
    plane_error_mm: f64,
}
impl ValidatedPlaneTrimEndpoint {
    /// Returns the retained candidate index.
    pub const fn candidate_index(&self) -> usize { self.candidate_index }
    /// Returns which ascending candidate endpoint owns this stratum.
    pub const fn endpoint_is_back(&self) -> bool { self.endpoint_is_back }
    /// Returns the original coedge occurrence.
    pub const fn coedge_index(&self) -> usize { self.coedge_index }
    /// Returns the immutable original edge identity.
    pub const fn original_edge_id(&self) -> EdgeId<Curve> { self.original_edge_id }
    /// Returns the isolated original native trim-root parameter enclosure.
    pub const fn source_root_parameter(&self) -> [f64; 2] { self.source_root_parameter }
    /// Returns additional physical distance beyond the nominal candidate end.
    pub const fn maximum_extension_mm(&self) -> f64 { self.extension_mm }
    /// Returns an upper bound on the additional native continuation arc length.
    pub const fn maximum_arc_length_mm(&self) -> f64 { self.arc_length_mm }
    /// Returns a position enclosure for that continuation and original root.
    pub const fn position_bounds(&self) -> [ValidatedInterval; 3] { self.position }
    /// Returns the achieved original root's distance to the actual plane.
    pub const fn maximum_plane_error_mm(&self) -> f64 { self.plane_error_mm }
}

#[derive(Clone, Copy)]
struct Root {
    parameter: [f64; 2],
    witness: [f64; 2],
    derivative: ValidatedInterval,
}
fn sign(value: ValidatedInterval) -> Option<bool> {
    if value.lower > 0.0 {
        Some(true)
    } else if value.upper < 0.0 {
        Some(false)
    } else {
        None
    }
}
fn contains(outer: ValidatedInterval, inner: ValidatedInterval) -> bool {
    outer.lower <= inner.lower && inner.upper <= outer.upper
}
fn disjoint2(a: [ValidatedInterval; 2], b: [ValidatedInterval; 2]) -> bool {
    (0..2).any(|i| a[i].upper < b[i].lower || b[i].upper < a[i].lower)
}
fn norm_upper(vector: [ValidatedInterval; 3]) -> Result<f64, ValidatedGeometryError> {
    let sum = vector
        .into_iter()
        .fold(super::super::ZERO, |sum, x| sum + x.square());
    if !sum.is_finite() || sum.upper < 0.0 {
        return Err(ValidatedGeometryError::ConditioningFailure);
    }
    Ok(sum.upper.max(0.0).sqrt().next_up())
}

// Every span is excluded or has a proved unique transverse root. In particular,
// a cut seed does not excuse an unexamined original parameter neighborhood.
fn isolate_roots(
    range: [f64; 2],
    maximum_bytes: usize,
    mut jet: impl FnMut(
        [f64; 2],
        &mut ValidatedGeometryBudget,
    ) -> Result<ValidatedScalarCurveJet, ValidatedGeometryError>,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Vec<Root>, ValidatedGeometryError> {
    interval_range(range)?;
    if range[0] >= range[1] {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    let mut pending = [(range, 0usize); 49];
    let mut length = 1;
    let mut roots: Vec<Root> = Vec::new();
    while length != 0 {
        budget.charge(1, cancel)?;
        length -= 1;
        let (span, depth) = pending[length];
        let whole = jet(span, budget)?;
        if !whole.position.contains_zero() {
            continue;
        }
        let mut witness = span;
        let mut derivative = whole.first;
        let mut left = jet([span[0]; 2], budget)?.position;
        let mut right = jet([span[1]; 2], budget)?.position;
        if sign(derivative).is_some() && (sign(left).is_none() || sign(right).is_none()) {
            // A root on an artificial binary joint needs a two-sided witness
            // in the SAME original owner, not two independent branch captures.
            let width = span[1] - span[0];
            witness = [
                (span[0] - width).max(range[0]),
                (span[1] + width).min(range[1]),
            ];
            derivative = jet(witness, budget)?.first;
            left = jet([witness[0]; 2], budget)?.position;
            right = jet([witness[1]; 2], budget)?.position;
        }
        let crossing = matches!((sign(left), sign(right)), (Some(a), Some(b)) if a != b);
        if sign(derivative).is_some() && crossing {
            let mut root = interval_range(witness)?;
            for _ in 0..48 {
                budget.charge(1, cancel)?;
                let midpoint = root.midpoint();
                let value = jet([midpoint; 2], budget)?.position;
                let slope = jet([root.lower, root.upper], budget)?.first;
                if slope.contains_zero() {
                    return Err(ValidatedGeometryError::NonTransverse);
                }
                let newton = ValidatedInterval::point(midpoint) - value * slope.reciprocal()?;
                let next = root
                    .intersection(newton)
                    .ok_or(ValidatedGeometryError::ConditioningFailure)?;
                if next == root {
                    break;
                }
                root = next;
            }
            let found = Root {
                parameter: [root.lower, root.upper],
                witness,
                derivative,
            };
            let mut duplicate = false;
            for previous in &roots {
                budget.charge(1, cancel)?;
                if previous.parameter[1] < found.parameter[0]
                    || found.parameter[1] < previous.parameter[0]
                {
                    continue;
                }
                // Each root enclosure lies inside its independently certified
                // monotone witness. Their overlap makes the witness union
                // connected; matching strict derivative signs prove one root
                // on that complete cover without a looser union reevaluation.
                // Keep the retained witness and its own derivative together.
                if sign(previous.derivative.hull(found.derivative)).is_none() {
                    return Err(ValidatedGeometryError::AmbiguousBranch);
                }
                duplicate = true;
                break;
            }
            if !duplicate {
                super::super::reserve_native_vec(&mut roots, maximum_bytes, 0, budget, cancel)?;
                roots.push(found);
            }
            continue;
        }
        if sign(derivative).is_some()
            && matches!((sign(left), sign(right)), (Some(a), Some(b)) if a == b)
        {
            continue;
        }
        let midpoint = span[0] * 0.5 + span[1] * 0.5;
        if depth == 48 || midpoint <= span[0] || midpoint >= span[1] {
            return Err(ValidatedGeometryError::Unresolved);
        }
        budget.charge(2, cancel)?;
        pending[length] = ([midpoint, span[1]], depth + 1);
        pending[length + 1] = ([span[0], midpoint], depth + 1);
        length += 2;
    }
    Ok(roots)
}

pub(super) struct PreparedEndpoint<'a> {
    pub(super) binding: ValidatedPlaneTrimEndpointBinding<'a>,
    original: Root,
    root: Root,
    original_position: [ValidatedInterval; 3],
    nominal_position: [ValidatedInterval; 3],
    uv: [ValidatedInterval; 2],
    nominal_end: [ValidatedInterval; 3],
}

impl PreparedEndpoint<'_> {
    pub(super) fn root_cell(&self) -> [[f64; 2]; 2] {
        self.uv.map(|value| [value.lower, value.upper])
    }
}

pub(super) struct FacePlaneContext<'a, 'd> {
    pub(super) domain: &'a PreparedFaceParameterDomain<'d>,
    pub(super) surface: &'a Surface,
    pub(super) plane: &'a Plane,
    pub(super) normal: [ValidatedInterval; 3],
    pub(super) origin: [ValidatedInterval; 3],
    pub(super) maximum_extension: f64,
}

pub(super) struct CoverageCell {
    pub(super) source_index: usize,
    pub(super) uv: [[f64; 2]; 2],
    pub(super) position: [ValidatedInterval; 3],
}
impl<'a, 'd> FacePlaneContext<'a, 'd> {
    pub(super) fn new(
        domain: &'a PreparedFaceParameterDomain<'d>,
        surface: &'a Surface,
        plane: &'a Plane,
        maximum_extension: f64,
    ) -> Result<Self, ValidatedGeometryError> {
        if !(0..3).all(|axis| plane.origin()[axis].is_finite()) {
            return Err(ValidatedGeometryError::InvalidInput);
        }
        let (a, b) = plane_axes(plane);
        let normal = super::super::cross3(a, b);
        let norm = normal
            .into_iter()
            .fold(super::super::ZERO, |sum, value| sum + value.square());
        if !norm.is_finite() || !normal.into_iter().all(ValidatedInterval::is_finite) {
            return Err(ValidatedGeometryError::ConditioningFailure);
        }
        if norm.lower <= 0.0 {
            return Err(ValidatedGeometryError::Singular);
        }
        let origin = std::array::from_fn(|i| ValidatedInterval::point(plane.origin()[i]));
        Ok(Self {
            domain,
            surface,
            plane,
            normal,
            origin,
            maximum_extension,
        })
    }
}

fn validate_binding_input(
    domain: &PreparedFaceParameterDomain<'_>,
    binding: &ValidatedPlaneTrimBoundaryBinding<'_>,
) -> Result<(), ValidatedGeometryError> {
    let coedge = domain
        .coedges()
        .get(binding.coedge_index)
        .ok_or(ValidatedGeometryError::InvalidInput)?;
    if coedge.identity.edge_id != binding.original_edge_id
        || !binding.boundary_scale.is_finite()
        || binding.boundary_scale <= 0.0
        || !binding.source_chart_error_mm.is_finite()
        || binding.source_chart_error_mm < 0.0
    {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    let retained_range = binding.boundary.range();
    interval_range(binding.boundary_range)?;
    if binding.boundary_range[0] >= binding.boundary_range[1]
        || binding.boundary_range[0] < retained_range[0]
        || binding.boundary_range[1] > retained_range[1]
    {
        return Err(ValidatedGeometryError::ParameterOutOfRange);
    }
    Ok(())
}

fn validate_binding_source(
    domain: &PreparedFaceParameterDomain<'_>,
    binding: &ValidatedPlaneTrimBoundaryBinding<'_>,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<(), ValidatedGeometryError> {
    let coedge = domain
        .coedges()
        .get(binding.coedge_index)
        .ok_or(ValidatedGeometryError::InvalidInput)?;
    let edge = domain
        .face()
        .absolute_boundaries()
        .get(coedge.identity.wire_index)
        .and_then(|wire| wire.get(coedge.identity.coedge_index))
        .ok_or(ValidatedGeometryError::InvalidInput)?;
    if edge.id() != binding.original_edge_id
        || coedge.identity.face_id != domain.face().id()
        || coedge.identity.orientation != (edge.orientation() == domain.face().orientation())
        || !edge.with_curve(|actual| {
            source_identity::retains_original_curve(
                binding.boundary,
                actual,
                binding.boundary_range,
                binding.boundary_scale,
                budget,
                cancel,
            )
        })?
    {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    Ok(())
}

struct TrimRoots {
    chart: Vec<Root>,
    original: Vec<Root>,
    retained_bytes: usize,
}

// The scalar direction contains only exact source-plane coefficients. Any
// interval uncertainty in an axial normal stays in the final scalar factor.
struct OriginalPlaneProjection {
    direction: [f64; 3],
    offset: ValidatedInterval,
    factor: ValidatedInterval,
}

impl OriginalPlaneProjection {
    fn new(normal: [ValidatedInterval; 3], origin: [ValidatedInterval; 3]) -> Option<Self> {
        if normal.into_iter().all(|value| value.lower == value.upper) {
            Some(Self {
                direction: normal.map(|value| value.lower),
                offset: dot3(normal, origin),
                factor: super::super::ONE,
            })
        } else {
            let axis = normal
                .iter()
                .position(|&value| value != super::super::ZERO)?;
            if normal[axis].contains_zero()
                || (0..3).any(|other| other != axis && normal[other] != super::super::ZERO)
            {
                return None;
            }
            let mut direction = [0.0; 3];
            direction[axis] = 1.0;
            Some(Self {
                direction,
                offset: origin[axis],
                factor: normal[axis],
            })
        }
    }

    fn jet(
        &self,
        boundary: &PreparedValidatedCurve<'_>,
        range: [f64; 2],
        scale: ValidatedInterval,
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> Result<ValidatedScalarCurveJet, ValidatedGeometryError> {
        let jet = boundary.projected_jet(range, [0.0; 3], self.direction, budget, cancel)?;
        // Preserve `n dot (scale * C(t) - origin)` in world units. Neither
        // `n * scale` nor `origin / scale` becomes a rounded exact coefficient.
        Ok(ValidatedScalarCurveJet {
            position: (jet.position * scale - self.offset) * self.factor,
            first: (jet.first * scale) * self.factor,
            second: (jet.second * scale) * self.factor,
        })
    }
}

fn isolate_trim_roots(
    context: &FacePlaneContext<'_, '_>,
    index: usize,
    binding: &ValidatedPlaneTrimBoundaryBinding<'_>,
    maximum_bytes: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<TrimRoots, ValidatedGeometryError> {
    let range = context
        .domain
        .coedges()
        .get(index)
        .ok_or(ValidatedGeometryError::InvalidInput)?
        .curve
        .range_tuple();
    let chart = isolate_roots(
        [range.0, range.1],
        maximum_bytes,
        |range, budget| {
            chart_jet(context, index, range, budget, cancel).map_err(|error| {
                coverage_context(
                    error,
                    "original trim inventory / source-chart jet enclosure",
                    None,
                )
            })
        },
        budget,
        cancel,
    )
    .map_err(|error| {
        coverage_context(
            error,
            "original trim inventory / complete source-chart root isolation",
            None,
        )
    })?;
    let chart_bytes = chart
        .capacity()
        .checked_mul(size_of::<Root>())
        .ok_or(ValidatedGeometryError::ResourceLimit)?;
    let remaining = maximum_bytes
        .checked_sub(chart_bytes)
        .ok_or(ValidatedGeometryError::ResourceLimit)?;
    let scale = ValidatedInterval::point(binding.boundary_scale);
    budget.charge(1, cancel)?;
    let projection = OriginalPlaneProjection::new(context.normal, context.origin);
    let original = isolate_roots(
        binding.boundary_range,
        remaining,
        |range, budget| {
            let result = if let Some(projection) = &projection {
                projection.jet(binding.boundary, range, scale, budget, cancel)
            } else {
                // A general interval normal cannot be rounded into the exact
                // coefficients accepted by the correlated scalar evaluator.
                binding
                    .boundary
                    .jet(range, budget, cancel)
                    .map(|jet| ValidatedScalarCurveJet {
                        position: dot3(
                            context.normal,
                            subtract3(jet.position.map(|x| x * scale), context.origin),
                        ),
                        first: dot3(context.normal, jet.first.map(|x| x * scale)),
                        second: dot3(context.normal, jet.second.map(|x| x * scale)),
                    })
            };
            result.map_err(|error| {
                coverage_context(
                    error,
                    "original trim inventory / original-3D jet enclosure",
                    None,
                )
            })
        },
        budget,
        cancel,
    )
    .map_err(|error| {
        coverage_context(
            error,
            "original trim inventory / complete original-3D root isolation",
            None,
        )
    })?;
    if original.len() != chart.len() {
        return Err(coverage_context(
            ValidatedGeometryError::Unresolved,
            "original trim inventory / original-3D and source-chart root cardinality",
            None,
        ));
    }
    for (chart, original) in chart.iter().zip(&original) {
        budget.charge(1, cancel)?;
        if sign(chart.derivative) != sign(original.derivative) {
            return Err(ValidatedGeometryError::NonTransverse);
        }
    }
    let retained_bytes = original
        .capacity()
        .checked_mul(size_of::<Root>())
        .and_then(|bytes| bytes.checked_add(chart_bytes))
        .ok_or(ValidatedGeometryError::ResourceLimit)?;
    Ok(TrimRoots {
        chart,
        original,
        retained_bytes,
    })
}

struct PreparedBoundaryRoots<'a> {
    binding: ValidatedPlaneTrimBoundaryBinding<'a>,
    roots: TrimRoots,
}

/// Complete original-3D and source-chart trim roots for one exact face and plane.
///
/// This query-local inventory is not candidate or full-face authority. It borrows
/// stable original owners, copies binding records, and retains a bounded deep
/// coefficient snapshot because native face carriers can be mutated through
/// topology handles. Its private root vectors are borrowed, never cloned, by
/// endpoint selection and coverage.
pub struct PreparedFacePlaneRoots<'a> {
    domain: &'a PreparedFaceParameterDomain<'a>,
    plane: &'a Plane,
    surface_snapshot: Surface,
    face_orientation: bool,
    boundaries: Vec<Option<PreparedBoundaryRoots<'a>>>,
    retained_bytes: usize,
}

impl fmt::Debug for PreparedFacePlaneRoots<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedFacePlaneRoots")
            .field("face_id", &self.domain.face().id())
            .field("plane", &self.plane)
            .field("retained_bytes", &self.retained_bytes)
            .finish_non_exhaustive()
    }
}

fn same_binding(
    a: &ValidatedPlaneTrimBoundaryBinding<'_>,
    b: &ValidatedPlaneTrimBoundaryBinding<'_>,
) -> bool {
    a.coedge_index == b.coedge_index
        && a.original_edge_id == b.original_edge_id
        && std::ptr::eq(a.boundary, b.boundary)
        && a.boundary_range == b.boundary_range
        && a.boundary_scale == b.boundary_scale
        && a.source_chart_error_mm == b.source_chart_error_mm
}

// This optional theorem concerns the COMPLETE original face, not the existence
// of numerical candidates. Require every directed occurrence's original source
// binding, and retain the unrestricted chart box: material-support clipping may
// otherwise hide nominal pcurve roots still required by this inventory.
fn complete_face_excluded(
    context: &FacePlaneContext<'_, '_>,
    boundaries: &[Option<PreparedBoundaryRoots<'_>>],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<bool, ValidatedGeometryError> {
    let mut source_error = 0.0_f64;
    for boundary in boundaries {
        budget.charge(1, cancel)?;
        let Some(boundary) = boundary else {
            return Ok(false);
        };
        source_error = source_error.max(boundary.binding.source_chart_error_mm);
    }
    let separated = |position: [ValidatedInterval; 3], error: f64| {
        // The physical error ball lies in this outward coordinate cube. The
        // plane's exact stored axes remain enclosed in the interval normal.
        let margin = ValidatedInterval::new(-error, error)?;
        let height = dot3(
            context.normal,
            subtract3(
                position.map(|coordinate| coordinate + margin),
                context.origin,
            ),
        );
        Ok::<_, ValidatedGeometryError>(height.is_finite() && !height.contains_zero())
    };
    let carrier = validated_surface_position_bounds(
        context.surface,
        context.domain.unrestricted_parameter_box(),
        budget,
        cancel,
    )?;
    if !separated(carrier, source_error)? {
        return Ok(false);
    }
    for boundary in boundaries.iter().flatten() {
        budget.charge(1, cancel)?;
        let binding = &boundary.binding;
        let original = binding
            .boundary
            .position_bounds(binding.boundary_range, budget, cancel)?
            .map(|coordinate| coordinate * ValidatedInterval::point(binding.boundary_scale));
        if !separated(original, binding.source_chart_error_mm)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Prepares each selected original coedge's complete trim-root inventories once.
///
/// Bindings contain no candidate endpoint or cut hint. Repeated identical
/// bindings are admitted without repeating isolation. All retained capacities
/// and the carrier coefficient snapshot share `max_retained_bytes`, narrowed
/// by the existing 32 MiB local cap and bounded preparation stack allowance.
/// Call outside original face-carrier and edge-curve mutex callbacks.
///
/// Overlapping captures may identify one root by composing their independently
/// proved, same-sign monotone witnesses in the same original gauge. A looser
/// enclosure over their union does not invalidate those local certificates;
/// each retained witness remains paired with its own derivative enclosure.
///
/// A strict C0 image enclosure may exclude the complete original carrier and
/// every bound original boundary without root isolation. This requires bindings
/// for all directed occurrences and includes chart/joint and physical source
/// uncertainty. Touching or unavailable enclosures retain complete isolation;
/// source/plane validation, cancellation and cumulative resource failures never
/// become empty inventories. Exclusion does not replace downstream face coverage.
pub fn prepared_face_plane_roots<'a>(
    domain: &'a PreparedFaceParameterDomain<'a>,
    plane: &'a Plane,
    boundaries: &[ValidatedPlaneTrimBoundaryBinding<'a>],
    max_retained_bytes: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<PreparedFacePlaneRoots<'a>, ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    let limit = max_retained_bytes
        .min(32 * 1024 * 1024)
        .checked_sub(COVERAGE_STACK_BYTES)
        .ok_or(ValidatedGeometryError::ResourceLimit)?;
    domain.face().with_surface(|surface| {
        let context = FacePlaneContext::new(domain, surface, plane, 0.0)?;
        prepare_roots(
            domain,
            plane,
            &context,
            boundaries.iter().copied(),
            limit,
            budget,
            cancel,
        )
    })
}

pub(super) fn prepare_roots<'a>(
    domain: &'a PreparedFaceParameterDomain<'a>,
    plane: &'a Plane,
    context: &FacePlaneContext<'_, '_>,
    bindings: impl Iterator<Item = ValidatedPlaneTrimBoundaryBinding<'a>>,
    maximum_bytes: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<PreparedFacePlaneRoots<'a>, ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    let count = context.domain.coedges().len();
    let slot_bytes = count
        .checked_mul(size_of::<Option<PreparedBoundaryRoots<'_>>>())
        .and_then(|bytes| bytes.checked_add(size_of::<PreparedFacePlaneRoots<'_>>()))
        .ok_or(ValidatedGeometryError::ResourceLimit)?;
    let remaining = maximum_bytes
        .checked_sub(slot_bytes)
        .ok_or(ValidatedGeometryError::ResourceLimit)?;
    let snapshot_bytes = crate::surface_owned_bytes(context.surface, remaining, budget, cancel)?;
    if !exact_surface(context.surface, context.surface, 0, budget, cancel)? {
        return Err(ValidatedGeometryError::Unsupported);
    }
    budget.charge(count, cancel)?;
    let mut boundaries = Vec::new();
    boundaries
        .try_reserve_exact(count)
        .map_err(|_| ValidatedGeometryError::ResourceLimit)?;
    boundaries.resize_with(count, || None);
    let retained_bytes = boundaries
        .capacity()
        .checked_mul(size_of::<Option<PreparedBoundaryRoots<'_>>>())
        .and_then(|bytes| bytes.checked_add(size_of::<PreparedFacePlaneRoots<'_>>()))
        .and_then(|bytes| bytes.checked_add(snapshot_bytes))
        .filter(|bytes| *bytes <= maximum_bytes)
        .ok_or(ValidatedGeometryError::ResourceLimit)?;
    let mut prepared = PreparedFacePlaneRoots {
        domain,
        plane,
        surface_snapshot: context.surface.clone(),
        face_orientation: context.domain.face().orientation(),
        boundaries,
        retained_bytes,
    };
    for binding in bindings {
        budget.charge(1, cancel)?;
        validate_binding_input(context.domain, &binding)?;
        if let Some(previous) = &prepared.boundaries[binding.coedge_index] {
            if !same_binding(&previous.binding, &binding) {
                return Err(ValidatedGeometryError::InvalidInput);
            }
            continue;
        }
        validate_binding_source(context.domain, &binding, budget, cancel)?;
        prepared.boundaries[binding.coedge_index] = Some(PreparedBoundaryRoots {
            binding,
            roots: TrimRoots {
                chart: Vec::new(),
                original: Vec::new(),
                retained_bytes: 0,
            },
        });
    }
    match complete_face_excluded(context, &prepared.boundaries, budget, cancel) {
        Ok(true) => return Ok(prepared),
        Err(
            error @ (ValidatedGeometryError::ResourceLimit | ValidatedGeometryError::Cancelled),
        ) => {
            return Err(error);
        }
        // This is only a precheck. Unsupported arithmetic or a nonseparated
        // enclosure has no exclusion authority; all work already spent remains
        // charged when the existing complete root algorithm resumes.
        Ok(false) | Err(_) => {}
    }
    for boundary in prepared.boundaries.iter_mut().flatten() {
        budget.charge(1, cancel)?;
        let remaining = maximum_bytes
            .checked_sub(prepared.retained_bytes)
            .ok_or(ValidatedGeometryError::ResourceLimit)?;
        let roots = isolate_trim_roots(
            context,
            boundary.binding.coedge_index,
            &boundary.binding,
            remaining,
            budget,
            cancel,
        )?;
        prepared.retained_bytes = prepared
            .retained_bytes
            .checked_add(roots.retained_bytes)
            .ok_or(ValidatedGeometryError::ResourceLimit)?;
        boundary.roots = roots;
    }
    Ok(prepared)
}

impl PreparedFacePlaneRoots<'_> {
    /// Returns owned object, binding-slot, root-capacity and snapshot bytes.
    ///
    /// Borrowed original curves, source domain metadata and native topology are
    /// excluded; their owners must remain charged by the calling query.
    pub const fn retained_bytes(&self) -> usize { self.retained_bytes }

    pub(super) fn validate_context(
        &self,
        context: &FacePlaneContext<'_, '_>,
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> Result<(), ValidatedGeometryError> {
        budget.charge(1, cancel)?;
        if !std::ptr::eq(self.domain, context.domain)
            || self.plane != context.plane
            || self.face_orientation != context.domain.face().orientation()
            || !exact_surface(&self.surface_snapshot, context.surface, 0, budget, cancel)?
        {
            return Err(ValidatedGeometryError::InvalidInput);
        }
        budget.charge(self.boundaries.len(), cancel)?;
        for boundary in self.boundaries.iter().flatten() {
            budget.charge(1, cancel)?;
            validate_binding_input(context.domain, &boundary.binding)?;
            validate_binding_source(context.domain, &boundary.binding, budget, cancel)?;
        }
        Ok(())
    }

    /// Selects a numerical UV anchor using an original native cut parameter.
    ///
    /// The complete inventories and derivative/crossing witnesses remain
    /// authoritative; this numerical anchor is not candidate endpoint authority.
    /// Original and chart gauges are separately retained and never identified.
    pub fn root_chart_for_cut(
        &self,
        coedge_index: usize,
        source_cut_parameter: f64,
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> Result<[ValidatedInterval; 2], ValidatedGeometryError> {
        budget.charge(1, cancel)?;
        self.domain
            .face()
            .with_surface(|surface| {
                let context = FacePlaneContext::new(self.domain, surface, self.plane, 0.0)?;
                self.validate_context(&context, budget, cancel)?;
                let boundary = self
                    .boundaries
                    .get(coedge_index)
                    .and_then(Option::as_ref)
                    .ok_or(ValidatedGeometryError::InvalidInput)?;
                if !source_cut_parameter.is_finite()
                    || source_cut_parameter < boundary.binding.boundary_range[0]
                    || source_cut_parameter > boundary.binding.boundary_range[1]
                {
                    return Err(ValidatedGeometryError::ParameterOutOfRange);
                }
                let mut selected = None;
                for (&chart, original) in boundary.roots.chart.iter().zip(&boundary.roots.original)
                {
                    budget.charge(1, cancel)?;
                    if source_cut_parameter < original.witness[0]
                        || source_cut_parameter > original.witness[1]
                    {
                        continue;
                    }
                    if selected.is_some() {
                        return Err(ValidatedGeometryError::AmbiguousBranch);
                    }
                    selected = Some(
                        root_chart_enclosure(&context, coedge_index, chart, budget, cancel)?.0,
                    );
                }
                selected.ok_or(ValidatedGeometryError::Unresolved)
            })
            .map_err(|error| {
                coverage_context(
                    error,
                    "original trim chart / generated candidate start",
                    None,
                )
            })
    }
}

pub(super) fn prepare_endpoints<'a>(
    context: &FacePlaneContext<'_, '_>,
    candidates: &[&PreparedValidatedCurve<'_>],
    bindings: &[ValidatedPlaneTrimEndpointBinding<'a>],
    roots: &PreparedFacePlaneRoots<'_>,
    maximum_bytes: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Vec<PreparedEndpoint<'a>>, ValidatedGeometryError> {
    let &FacePlaneContext {
        domain,
        maximum_extension,
        ..
    } = context;
    let mut result: Vec<PreparedEndpoint<'a>> = Vec::new();
    for binding in bindings {
        budget.charge(1, cancel)?;
        if binding.candidate_index >= candidates.len() {
            return Err(ValidatedGeometryError::InvalidInput);
        }
        let source = binding.boundary_binding();
        validate_binding_input(domain, &source)?;
        if !binding.source_cut_parameter.is_finite()
            || binding.source_cut_parameter < source.boundary_range[0]
            || binding.source_cut_parameter > source.boundary_range[1]
        {
            return Err(ValidatedGeometryError::ParameterOutOfRange);
        }
        let original = roots
            .boundaries
            .get(binding.coedge_index)
            .and_then(Option::as_ref)
            .ok_or(ValidatedGeometryError::InvalidInput)?;
        if !same_binding(&source, &original.binding) {
            return Err(ValidatedGeometryError::InvalidInput);
        }
    }
    for index in 0..domain.coedges().len() {
        budget.charge(1, cancel)?;
        let Some(boundary) = &roots.boundaries[index] else {
            // Missing incidence still needs the complete actual uncertainty
            // tube to exclude the plane; cached nominal absence is insufficient.
            prove_no_crossing(context, index, budget, cancel)?;
            continue;
        };
        let first = &boundary.binding;
        let roots = &boundary.roots;
        if roots.chart.is_empty() {
            prove_no_crossing(context, index, budget, cancel)?;
        }
        let scale = ValidatedInterval::point(first.boundary_scale);
        for (&root, &original) in roots.chart.iter().zip(&roots.original) {
            // Native and UV gauges are independently retained. Pair complete
            // root inventories by the original oriented coedge order, never
            // by an invented affine parameter map or nearest-point match.
            let mut selected = None;
            for &binding in bindings {
                budget.charge(1, cancel)?;
                if binding.coedge_index != index
                    || binding.source_cut_parameter < original.witness[0]
                    || binding.source_cut_parameter > original.witness[1]
                {
                    continue;
                }
                if selected.is_some() {
                    return Err(ValidatedGeometryError::AmbiguousBranch);
                }
                if !same_binding(&binding.boundary_binding(), first) {
                    return Err(ValidatedGeometryError::InvalidInput);
                }
                selected = Some(binding);
            }
            let binding = selected.ok_or(ValidatedGeometryError::Unresolved)?;
            let (uv, nominal_position) =
                root_chart_enclosure(context, index, root, budget, cancel)?;
            let original_position = binding
                .boundary
                .position_bounds(original.parameter, budget, cancel)?
                .map(|x| x * scale);
            let candidate = candidates[binding.candidate_index];
            let endpoint = candidate.range[usize::from(binding.endpoint_is_back)];
            let nominal_end = candidate.evaluate(endpoint, budget, cancel)?.error;
            let cut_position = binding
                .boundary
                .evaluate(binding.source_cut_parameter, budget, cancel)?
                .error
                .map(|x| x * scale);
            if norm_upper(std::array::from_fn(|i| cut_position[i] - nominal_end[i]))?
                > maximum_extension
            {
                return Err(ValidatedGeometryError::Unresolved);
            }
            super::super::reserve_native_vec(&mut result, maximum_bytes, 0, budget, cancel)?;
            result.push(PreparedEndpoint {
                binding,
                original,
                root,
                original_position,
                nominal_position,
                uv,
                nominal_end,
            });
        }
    }
    if result.len() != bindings.len() {
        return Err(ValidatedGeometryError::Unresolved);
    }
    Ok(result)
}

fn prove_no_crossing(
    context: &FacePlaneContext<'_, '_>,
    coedge: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<(), ValidatedGeometryError> {
    let &FacePlaneContext {
        domain,
        surface,
        normal,
        origin,
        ..
    } = context;
    for span in domain
        .boundary_parameter_intervals(coedge)
        .map_err(trim_error)?
    {
        let mut pending = [(span, 0usize); 49];
        let mut length = 1;
        while length != 0 {
            budget.charge(1, cancel)?;
            length -= 1;
            let (span, depth) = pending[length];
            let uv = domain
                .boundary_position_enclosure(coedge, span, budget, cancel)
                .map_err(trim_error)?;
            // Absence only needs a fresh C0 image enclosure of the complete
            // source-chart uncertainty tube, not a smooth height jet. In
            // particular, an empty inventory does not make a source crease C1.
            let position = validated_surface_position_bounds(
                surface,
                uv.map(|axis| [axis.lower, axis.upper]),
                budget,
                cancel,
            )?;
            let height = dot3(normal, subtract3(position, origin));
            if !height.is_finite() {
                return Err(ValidatedGeometryError::ConditioningFailure);
            }
            if !height.contains_zero() {
                continue;
            }
            // Preserve the existing coefficient-correlated exclusion for
            // smooth sources when separate coordinate bounds are too loose.
            // At a C0 joint only subdivision of the position proof is valid.
            match chart_height(surface, uv, normal, origin, budget, cancel) {
                Ok(height) if !height.0[0][0].contains_zero() => continue,
                Ok(_) | Err(ValidatedGeometryError::NonSmooth) => {}
                Err(error) => return Err(error),
            }
            let midpoint = span[0] * 0.5 + span[1] * 0.5;
            if depth == 48 || midpoint <= span[0] || midpoint >= span[1] {
                return Err(ValidatedGeometryError::Unresolved);
            }
            budget.charge(2, cancel)?;
            pending[length] = ([midpoint, span[1]], depth + 1);
            pending[length + 1] = ([span[0], midpoint], depth + 1);
            length += 2;
        }
    }
    Ok(())
}

fn chart_jet(
    context: &FacePlaneContext<'_, '_>,
    coedge: usize,
    parameter: [f64; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<ValidatedScalarCurveJet, ValidatedGeometryError> {
    let &FacePlaneContext {
        domain,
        surface,
        normal,
        origin,
        ..
    } = context;
    let boundary = domain
        .coedges()
        .get(coedge)
        .ok_or(ValidatedGeometryError::InvalidInput)?;
    projected_surface_curve_jet(
        surface,
        boundary.curve,
        parameter,
        plane_projection(normal, origin),
        &mut ProjectionExecution { budget, cancel },
    )
}

// Keep the interval plane coefficients; the midpoint is only an algebraic center.
fn plane_projection(
    normal: [ValidatedInterval; 3],
    origin: [ValidatedInterval; 3],
) -> ScalarProjection<3> {
    let center = origin.map(ValidatedInterval::midpoint);
    ScalarProjection {
        direction: normal,
        offset: dot3(
            normal,
            subtract3(center.map(ValidatedInterval::point), origin),
        ),
        center: Some(center),
    }
}

fn chart_height(
    surface: &Surface,
    uv: [ValidatedInterval; 2],
    normal: [ValidatedInterval; 3],
    origin: [ValidatedInterval; 3],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Taylor, ValidatedGeometryError> {
    projected_surface_taylor(
        surface,
        [Taylor::variable(uv[0], 0), Taylor::variable(uv[1], 1)],
        plane_projection(normal, origin),
        0,
        &mut ProjectionExecution { budget, cancel },
    )
}

fn root_chart_enclosure(
    context: &FacePlaneContext<'_, '_>,
    coedge: usize,
    root: Root,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<([ValidatedInterval; 2], [ValidatedInterval; 3]), ValidatedGeometryError> {
    let &FacePlaneContext {
        domain,
        surface,
        normal,
        origin,
        ..
    } = context;
    // Any actual source root belongs to the COMPLETE retained chart tube.
    // Bound its possible parameter displacement by the nominal height error
    // divided by the proved strict nominal height derivative. No derivative
    // of the positional source-chart uncertainty is assumed.
    let witness_uv = domain
        .boundary_position_enclosure(coedge, root.witness, budget, cancel)
        .map_err(trim_error)?;
    let witness_height = chart_height(surface, witness_uv, normal, origin, budget, cancel)?;
    let margin = domain.uncertainty_uv();
    let height_error = (witness_height.0[1][0] * ValidatedInterval::new(-margin[0], margin[0])?
        + witness_height.0[0][1] * ValidatedInterval::new(-margin[1], margin[1])?)
    .magnitude();
    let derivative_lower = if root.derivative.lower > 0.0 {
        root.derivative.lower
    } else {
        -root.derivative.upper
    };
    if derivative_lower <= 0.0 {
        return Err(ValidatedGeometryError::NonTransverse);
    }
    let radius = (height_error / derivative_lower).next_up();
    let parameter = [
        (root.parameter[0] - radius).next_down(),
        (root.parameter[1] + radius).next_up(),
    ];
    if parameter[0] <= root.witness[0] || parameter[1] >= root.witness[1] {
        return Err(ValidatedGeometryError::Unresolved);
    }
    let uv = domain
        .boundary_position_enclosure(coedge, parameter, budget, cancel)
        .map_err(trim_error)?;
    let position =
        validated_surface_jet(surface, uv.map(|x| [x.lower, x.upper]), budget, cancel)?.position;
    Ok((uv, position))
}

// Inventory every oriented original span touching a root cell. An unbound
// second crossing, a joint or a tangent is not an endpoint collar certificate.
fn complete_incidence(
    context: &FacePlaneContext<'_, '_>,
    cell: [[f64; 2]; 2],
    prepared: &PreparedEndpoint<'_>,
    all_roots: &[PreparedEndpoint<'_>],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<bool, ValidatedGeometryError> {
    let &FacePlaneContext {
        domain,
        surface,
        normal,
        origin,
        ..
    } = context;
    let cell = [interval_range(cell[0])?, interval_range(cell[1])?];
    for joint in domain.joints() {
        budget.charge(1, cancel)?;
        if disjoint2(joint.enclosure, cell) {
            continue;
        }
        if chart_height(surface, joint.enclosure, normal, origin, budget, cancel)?.0[0][0]
            .contains_zero()
        {
            return Ok(false);
        }
    }
    // Preparation inventoried the whole original parameter domain of every
    // physical/chart coedge, including opposite seam occurrences. Unbound
    // occurrences proved their entire uncertainty tube zero-free. Consequently
    // these sealed root enclosures, not nominal sampled boundary positions,
    // inventory every possible boundary incidence in this cell.
    for other in all_roots {
        budget.charge(1, cancel)?;
        if std::ptr::eq(other, prepared) {
            continue;
        }
        if !disjoint2(other.uv, cell) {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) enum EndpointCoverage {
    Covered(ValidatedPlaneTrimEndpoint),
    Refused(&'static str, ValidatedGeometryError),
}

fn endpoint_tube_witness(
    candidate: &PreparedValidatedCurve<'_>,
    binding: &ValidatedPlaneTrimEndpointBinding<'_>,
    source_index: usize,
    coordinate: usize,
    tube: ValidatedInterval,
    query: ValidatedInterval,
) -> Option<ValidatedPlaneCoverageWitness> {
    let piece_index = if binding.endpoint_is_back {
        candidate.pieces.len().checked_sub(1)?
    } else {
        0
    };
    // Diagnostic representability must not change the numerical refusal.
    // Query/storage caps already bound these indices; check rather than truncate.
    Some(ValidatedPlaneCoverageWitness::Endpoint(
        ValidatedPlaneEndpointWitness {
            candidate_index: u32::try_from(binding.candidate_index).ok()?,
            piece_index: u32::try_from(piece_index).ok()?,
            coedge_index: u32::try_from(binding.coedge_index).ok()?,
            source_index: u8::try_from(source_index).ok()?,
            endpoint_is_back: binding.endpoint_is_back,
            coordinate: u8::try_from(coordinate).ok()?,
            tube_bounds: [tube.lower, tube.upper],
            query_bounds: [query.lower, query.upper],
        },
    ))
}

fn endpoint_orientation_witness(
    candidate: &PreparedValidatedCurve<'_>,
    prepared: &PreparedEndpoint<'_>,
    source_index: usize,
    coordinate: usize,
    determinant: ValidatedInterval,
    candidate_derivative: ValidatedInterval,
    orientations: [bool; 5],
) -> Option<ValidatedPlaneCoverageWitness> {
    let binding = &prepared.binding;
    let piece_index = if binding.endpoint_is_back {
        candidate.pieces.len().checked_sub(1)?
    } else {
        0
    };
    let [face_orientation, coedge_orientation, absolute_orientation,
        material_positive, candidate_positive] = orientations;
    let bounds = |value: ValidatedInterval| [value.lower, value.upper];
    let indices = |len: usize| {
        std::array::from_fn::<_, 4, _>(|index| {
            if binding.endpoint_is_back {
                len.saturating_sub(1 + index)
            } else {
                index.min(len.saturating_sub(1))
            }
        })
    };
    let guide = candidate.native_curve().and_then(|curve| match curve {
        Curve::IntersectionCurve(curve) => Some(curve.leader().as_ref()),
        _ => None,
    });
    let (guide_end_coefficients, guide_end_knots, guide_coefficient_counts) = match guide {
        Some(Curve::BsplineCurve(curve)) if !curve.control_points().is_empty()
            && !curve.knot_vector().is_empty() =>
        {
            let points = curve.control_points();
            let knots = curve.knot_vector();
            let control_indices = indices(points.len());
            (
                Some(std::array::from_fn(|index| {
                    let point = points[control_indices[index]];
                    [point.x, point.y, point.z, 1.0]
                })),
                Some(indices(knots.len()).map(|index| knots[index])),
                Some([u32::try_from(points.len()).ok()?, u32::try_from(knots.len()).ok()?]),
            )
        }
        Some(Curve::NurbsCurve(curve)) if !curve.control_points().is_empty()
            && !curve.knot_vector().is_empty() =>
        {
            let points = curve.control_points();
            let knots = curve.knot_vector();
            let control_indices = indices(points.len());
            (
                Some(std::array::from_fn(|index| {
                    let point = points[control_indices[index]];
                    [point.x, point.y, point.z, point.w]
                })),
                Some(indices(knots.len()).map(|index| knots[index])),
                Some([u32::try_from(points.len()).ok()?, u32::try_from(knots.len()).ok()?]),
            )
        }
        _ => (None, None, None),
    };
    let (projection_range, projection_planes) = match candidate.pieces[piece_index].projection {
        Projection::BetweenPlanes { parameter, planes } => (
            Some(parameter),
            Some(planes.map(|plane| [
                plane.normal[0], plane.normal[1], plane.normal[2], plane.offset,
            ])),
        ),
        _ => (None, None),
    };
    // Diagnostic representability never changes the original refusal.
    Some(ValidatedPlaneCoverageWitness::EndpointOrientation(
        ValidatedPlaneEndpointOrientationWitness {
            candidate_index: u32::try_from(binding.candidate_index).ok()?,
            piece_index: u32::try_from(piece_index).ok()?,
            coedge_index: u32::try_from(binding.coedge_index).ok()?,
            source_index: u8::try_from(source_index).ok()?,
            endpoint_is_back: binding.endpoint_is_back,
            coordinate: u8::try_from(coordinate).ok()?,
            candidate_derivative: bounds(candidate_derivative),
            material_positive,
            candidate_positive,
            guide_end_coefficients,
            guide_end_knots,
            guide_coefficient_counts,
            projection_range,
            projection_planes,
            candidate_range: candidate.range,
            chart_root: [prepared.root.parameter, prepared.root.witness],
            original_root: [prepared.original.parameter, prepared.original.witness],
            source_cut_parameter: binding.source_cut_parameter,
            chart_derivative: bounds(prepared.root.derivative),
            original_derivative: bounds(prepared.original.derivative),
            determinant: bounds(determinant),
            face_orientation,
            coedge_orientation,
            absolute_orientation,
        },
    ))
}

pub(super) fn endpoint_covers(
    context: &FacePlaneContext<'_, '_>,
    candidates: &[&PreparedValidatedCurve<'_>],
    prepared: &PreparedEndpoint<'_>,
    all_roots: &[PreparedEndpoint<'_>],
    coverage_cell: CoverageCell,
    execution: (&mut ValidatedGeometryBudget, &AtomicBool),
    mut diagnostic: Option<&mut Option<ValidatedPlaneCoverageWitness>>,
) -> Result<EndpointCoverage, ValidatedGeometryError> {
    let (budget, cancel) = execution;
    let &FacePlaneContext {
        domain,
        surface,
        plane,
        normal,
        origin,
        maximum_extension,
    } = context;
    let CoverageCell {
        source_index,
        uv: cell,
        position,
    } = coverage_cell;
    budget.charge(1, cancel)?;
    let binding = prepared.binding;
    let candidate = *candidates
        .get(binding.candidate_index)
        .ok_or(ValidatedGeometryError::InvalidInput)?;
    let boundary = domain
        .coedges()
        .get(binding.coedge_index)
        .ok_or(ValidatedGeometryError::InvalidInput)?;
    let nominal_end = prepared.nominal_end;
    // This is only a search filter. Acceptance still requires the original
    // crossing inventory, material side and a fresh common uniqueness proof.
    if norm_upper(std::array::from_fn(|i| position[i] - nominal_end[i]))? > 4.0 * maximum_extension
    {
        return Ok(EndpointCoverage::Refused(
            "endpoint / search envelope distance",
            ValidatedGeometryError::Unresolved,
        ));
    }
    let endpoint = if binding.endpoint_is_back {
        candidate.range[1]
    } else {
        candidate.range[0]
    };
    let piece = if binding.endpoint_is_back {
        candidate.pieces.last()
    } else {
        candidate.pieces.first()
    }
    .ok_or(ValidatedGeometryError::InvalidInput)?;
    let box4 = flatten(
        piece
            .source_parameters
            .ok_or(ValidatedGeometryError::InvalidInput)?,
    );
    let chart = [interval_range(cell[0])?, interval_range(cell[1])?];
    let plane_chart = plane_parameters(plane, position)?;
    let q = if source_index == 0 {
        [chart[0], chart[1], plane_chart[0], plane_chart[1]]
    } else {
        [plane_chart[0], plane_chart[1], chart[0], chart[1]]
    };
    if (source_index != 0 || candidate.source.graph().is_none())
        && let Some(axis) = (0..4).find(|&i| !contains(box4[i], q[i]))
    {
        if let Some(diagnostic) = diagnostic.as_deref_mut() {
            *diagnostic =
                endpoint_tube_witness(candidate, &binding, source_index, axis, box4[axis], q[axis]);
        }
        return Ok(EndpointCoverage::Refused(
            "endpoint / source cell in original uniqueness tube",
            ValidatedGeometryError::Unresolved,
        ));
    }
    let original = prepared.original;
    let root = prepared.root;
    let nominal_root = prepared.nominal_position;
    let original_root = prepared.original_position;
    let uv = prepared.uv;
    if disjoint2(uv, chart) {
        return Ok(EndpointCoverage::Refused(
            "endpoint / original trim root intersects cell",
            ValidatedGeometryError::Unresolved,
        ));
    }
    let endpoint_jet = candidate.jet([endpoint; 2], budget, cancel)?;
    let axis = (0..3)
        .filter(|&i| sign(endpoint_jet.first[i]).is_some())
        .max_by(|&i, &j| {
            endpoint_jet.first[i]
                .magnitude()
                .total_cmp(&endpoint_jet.first[j].magnitude())
        })
        .ok_or(ValidatedGeometryError::NonTransverse)?;
    let chart_jet = validated_surface_jet(surface, uv.map(|x| [x.lower, x.upper]), budget, cancel)?;
    let hu = dot3(normal, chart_jet.du);
    let hv = dot3(normal, chart_jet.dv);
    let determinant = chart_jet.du[axis] * hv - chart_jet.dv[axis] * hu;
    if determinant.contains_zero() {
        return Ok(EndpointCoverage::Refused(
            "endpoint / chart crossing determinant",
            ValidatedGeometryError::Unresolved,
        ));
    }
    // The stored UV circuit fixes material membership. The prepared identity
    // includes face-use inversion, which reverses winding but cannot change
    // that membership. Recover the absolute stored-wire direction first.
    let absolute_orientation = boundary.identity.orientation == domain.face().orientation();
    let material_positive = sign(-root.derivative * determinant)
        .map(|x| if absolute_orientation { x } else { !x })
        .ok_or(ValidatedGeometryError::NonTransverse)?;
    let candidate_positive =
        sign(endpoint_jet.first[axis]).ok_or(ValidatedGeometryError::NonTransverse)?;
    if material_positive != (candidate_positive != binding.endpoint_is_back) {
        if let Some(diagnostic) = diagnostic.as_deref_mut() {
            *diagnostic = endpoint_orientation_witness(
                candidate,
                prepared,
                source_index,
                axis,
                determinant,
                endpoint_jet.first[axis],
                [
                    domain.face().orientation(),
                    boundary.identity.orientation,
                    absolute_orientation,
                    material_positive,
                    candidate_positive,
                ],
            );
        }
        return Ok(EndpointCoverage::Refused(
            "endpoint / oriented material side",
            ValidatedGeometryError::Unresolved,
        ));
    }
    if !complete_incidence(context, cell, prepared, all_roots, budget, cancel)? {
        return Ok(EndpointCoverage::Refused(
            "endpoint / complete original trim incidence",
            ValidatedGeometryError::Unresolved,
        ));
    }
    let (jet, arc_length) = if let Some(graph) =
        candidate.source.graph().filter(|_| source_index == 0)
    {
        // The original profile stays inside its genuine finite support. The
        // invertible coefficient system proves the unique plane zero for every
        // u here, including the actual original trim root. No extended carrier
        // or fictitious interior Krawczyk box is introduced at a finite end.
        let endpoint_u = graph.profile_endpoint(binding.endpoint_is_back);
        let endpoint_span = uv[0].hull(ValidatedInterval::point(endpoint_u));
        let span = chart[0].hull(endpoint_span);
        let (jet, _) = match graph.profile_jet(candidate.source.curve()?, span, budget, cancel) {
            Ok(jet) => jet,
            Err(error) if splittable(error) => {
                return Ok(EndpointCoverage::Refused(
                    "endpoint / original finite profile graph",
                    error,
                ));
            }
            Err(error) => return Err(error),
        };
        let arc_length = (norm_upper(jet.first)? * endpoint_span.width()).next_up();
        (jet, arc_length)
    } else {
        let coordinate = position[axis]
            .hull(original_root[axis])
            .hull(nominal_root[axis])
            .hull(nominal_end[axis]);
        if coordinate.lower > coordinate.upper {
            return Ok(EndpointCoverage::Refused(
                "endpoint / ordered fixed coordinate",
                ValidatedGeometryError::Unresolved,
            ));
        }
        let mut direction = [0.0; 3];
        direction[axis] = 1.0;
        // A point coordinate has no physical extension. Certify its unique
        // carrier intersection with a constant-plane family, not an invented
        // positive-width coordinate arc or an enlarged original guide.
        let point_coordinate = coordinate.lower == coordinate.upper;
        let coordinate_query = if point_coordinate {
            ValidatedInterval::point(0.0)
        } else {
            coordinate
        };
        let projection = Projection::BetweenPlanes {
            parameter: if point_coordinate {
                [0.0, 1.0]
            } else {
                [coordinate.lower, coordinate.upper]
            },
            planes: [
                CoordinatePlane {
                    normal: direction,
                    offset: coordinate.lower,
                },
                CoordinatePlane {
                    normal: direction,
                    offset: coordinate.upper,
                },
            ],
        };
        // This is a new coordinate on the SAME original carrier intersection. It
        // must certify throughout the old uniqueness tube, containing the actual
        // old endpoint and every original root enclosure. It is not guide growth.
        let root_chart = plane_parameters(
            plane,
            std::array::from_fn(|i| original_root[i].hull(nominal_root[i])),
        )?;
        let root_q = if source_index == 0 {
            [uv[0], uv[1], root_chart[0], root_chart[1]]
        } else {
            [root_chart[0], root_chart[1], uv[0], uv[1]]
        };
        if let Some(axis) = (0..4).find(|&i| !contains(box4[i], root_q[i])) {
            if let Some(diagnostic) = diagnostic {
                *diagnostic = endpoint_tube_witness(
                    candidate,
                    &binding,
                    source_index,
                    axis,
                    box4[axis],
                    root_q[axis],
                );
            }
            return Ok(EndpointCoverage::Refused(
                "endpoint / original trim root in original uniqueness tube",
                ValidatedGeometryError::Unresolved,
            ));
        }
        let mut tube = match certify(
            candidate.source.curve()?,
            projection,
            coordinate_query,
            box4,
            budget,
            cancel,
        ) {
            Ok(tube) => tube,
            Err(error) if splittable(error) => {
                return Ok(EndpointCoverage::Refused(
                    "endpoint / fixed-coordinate uniqueness certification",
                    error,
                ));
            }
            Err(error) => return Err(error),
        };
        for _ in 0..8 {
            budget.charge(1, cancel)?;
            let next = std::array::from_fn(|i| {
                let root = tube.rooted_domain[i];
                let radius = (root.width() * 0.51 + 64.0 * f64::EPSILON * (1.0 + root.magnitude()))
                    .next_up();
                box4[i]
                    .intersection(ValidatedInterval {
                        lower: (root.midpoint() - radius).next_down(),
                        upper: (root.midpoint() + radius).next_up(),
                    })
                    .unwrap_or(box4[i])
            });
            if next == tube.domain {
                break;
            }
            match certify(
                candidate.source.curve()?,
                projection,
                coordinate_query,
                next,
                budget,
                cancel,
            ) {
                Ok(refined) => tube = refined,
                Err(error) if splittable(error) => break,
                Err(error) => return Err(error),
            }
        }
        let jet = implicit_jet(&tube)?;
        let endpoint_coordinate = nominal_end[axis]
            .hull(original_root[axis])
            .hull(nominal_root[axis]);
        let arc_length = (norm_upper(jet.first)? * endpoint_coordinate.width()).next_up();
        (jet, arc_length)
    };
    let source_displacement =
        norm_upper(std::array::from_fn(|i| original_root[i] - nominal_root[i]))?;
    let extension = (arc_length + source_displacement).next_up();
    if !extension.is_finite() || extension > maximum_extension {
        return Ok(EndpointCoverage::Refused(
            "endpoint / achieved extension under frozen allowance",
            ValidatedGeometryError::Unresolved,
        ));
    }
    let position =
        std::array::from_fn(|i| jet.position[i].hull(original_root[i]).hull(nominal_end[i]));
    let normal_squared = normal
        .into_iter()
        .fold(super::super::ZERO, |sum, x| sum + x.square());
    let normal_lower = normal_squared.lower.max(0.0).sqrt().next_down();
    if normal_lower <= 0.0 {
        return Err(ValidatedGeometryError::Singular);
    }
    let plane_error =
        (dot3(normal, subtract3(original_root, origin)).magnitude() / normal_lower).next_up();
    if !plane_error.is_finite() {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    Ok(EndpointCoverage::Covered(ValidatedPlaneTrimEndpoint {
        candidate_index: binding.candidate_index,
        endpoint_is_back: binding.endpoint_is_back,
        coedge_index: binding.coedge_index,
        original_edge_id: binding.original_edge_id,
        source_root_parameter: original.parameter,
        extension_mm: extension,
        arc_length_mm: extension,
        position,
        plane_error_mm: plane_error,
    }))
}

pub(super) fn retain_receipt(
    receipts: &mut Vec<ValidatedPlaneTrimEndpoint>,
    receipt: ValidatedPlaneTrimEndpoint,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<(), ValidatedGeometryError> {
    for previous in receipts.iter_mut() {
        budget.charge(1, cancel)?;
        if previous.candidate_index != receipt.candidate_index
            || previous.endpoint_is_back != receipt.endpoint_is_back
            || previous.coedge_index != receipt.coedge_index
        {
            continue;
        }
        if previous.original_edge_id != receipt.original_edge_id
            || previous.source_root_parameter != receipt.source_root_parameter
        {
            return Err(ValidatedGeometryError::AmbiguousBranch);
        }
        previous.extension_mm = previous.extension_mm.max(receipt.extension_mm);
        previous.arc_length_mm = previous.arc_length_mm.max(receipt.arc_length_mm);
        previous.plane_error_mm = previous.plane_error_mm.max(receipt.plane_error_mm);
        previous.position = std::array::from_fn(|i| previous.position[i].hull(receipt.position[i]));
        return Ok(());
    }
    // Storage was reserved once from the complete original incidence count.
    if receipts.len() == receipts.capacity() {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    budget.charge(1, cancel)?;
    receipts.push(receipt);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::super::{ONE, ZERO, cross3};
    use super::*;
    use crate::{
        BsplineCurve, Invertible, KnotVector, Line, Point3, ValidatedCurveOptions,
        prepared_validated_curve,
    };

    fn encloses(value: ValidatedInterval, expected: f64) {
        assert!(
            value.lower <= expected && expected <= value.upper,
            "{value:?} does not enclose {expected:?}"
        );
    }

    #[test]
    fn original_scalar_world_jets_preserve_units_translation_and_inverse_gauges() {
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(1_000_000);
        let curve = Curve::Line(Line(
            Point3::new(8.0, 8.0, 0.25),
            Point3::new(12.0, 12.0, 2.25),
        ));
        let normal_axial = cross3(
            [2.0, 0.0, 0.0].map(ValidatedInterval::point),
            [0.0, 3.0, 0.0].map(ValidatedInterval::point),
        );
        let origin = [5.0, 5.0, 0.125].map(ValidatedInterval::point);
        for inverted in [false, true] {
            let curve = if inverted {
                curve.inverse()
            } else {
                curve.clone()
            };
            let prepared = prepared_validated_curve(
                &curve,
                [0.125, 0.875],
                &ValidatedCurveOptions::default(),
                &mut budget,
                &cancel,
            )
            .unwrap();
            for (normal, normal_z) in [
                ([0.0, 0.0, 1.0].map(ValidatedInterval::point), 1.0),
                ([2.0, -2.0, 2.0].map(ValidatedInterval::point), 2.0),
                (normal_axial, 6.0),
            ] {
                let projection = OriginalPlaneProjection::new(normal, origin).unwrap();
                for scale in [1.0, 0.125, 0.1] {
                    let jet = projection
                        .jet(
                            &prepared,
                            [0.25, 0.75],
                            ValidatedInterval::point(scale),
                            &mut budget,
                            &cancel,
                        )
                        .unwrap();
                    for parameter in [0.25, 0.5, 0.75] {
                        let z = if inverted {
                            2.25 - 2.0 * parameter
                        } else {
                            0.25 + 2.0 * parameter
                        };
                        encloses(jet.position, normal_z * (scale * z - 0.125));
                    }
                    let slope = if inverted { -2.0 } else { 2.0 };
                    encloses(jet.first, normal_z * scale * slope);
                    encloses(jet.second, 0.0);
                    if normal[0] != ZERO {
                        let spatial = prepared.jet([0.25, 0.75], &mut budget, &cancel).unwrap();
                        let independent = dot3(
                            normal,
                            subtract3(
                                spatial
                                    .position
                                    .map(|value| value * ValidatedInterval::point(scale)),
                                origin,
                            ),
                        );
                        assert!(jet.position.width() < independent.width());
                    }
                }
            }
        }
    }

    #[test]
    fn original_scalar_isolation_keeps_plural_roots_and_both_crossing_directions() {
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(1_000_000);
        // The exact height is `(t - 1/4) * (t - 5/8)`, while the correlated
        // first two coordinates cancel only before interval basis evaluation.
        let curve = Curve::BsplineCurve(BsplineCurve::new(
            KnotVector::from(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            vec![
                Point3::new(8.0, 8.0, 0.15625),
                Point3::new(10.0, 10.0, -0.28125),
                Point3::new(12.0, 12.0, 0.28125),
            ],
        ));
        for inverted in [false, true] {
            let curve = if inverted {
                curve.inverse()
            } else {
                curve.clone()
            };
            let prepared = prepared_validated_curve(
                &curve,
                [0.125, 0.875],
                &ValidatedCurveOptions::default(),
                &mut budget,
                &cancel,
            )
            .unwrap();
            for reversed_normal in [false, true] {
                let direction = if reversed_normal {
                    [-1.0, 1.0, -1.0]
                } else {
                    [1.0, -1.0, 1.0]
                };
                let projection = OriginalPlaneProjection::new(
                    direction.map(ValidatedInterval::point),
                    [5.0, 5.0, 0.0].map(ValidatedInterval::point),
                )
                .unwrap();
                let roots = isolate_roots(
                    [0.125, 0.875],
                    1024 * 1024,
                    |range, budget| {
                        projection.jet(
                            &prepared,
                            range,
                            ValidatedInterval::point(0.125),
                            budget,
                            &cancel,
                        )
                    },
                    &mut budget,
                    &cancel,
                )
                .unwrap();
                let expected = if inverted {
                    [0.375, 0.75]
                } else {
                    [0.25, 0.625]
                };
                assert_eq!(roots.len(), 2);
                for (root, parameter) in roots.iter().zip(expected) {
                    assert!(root.parameter[0] <= parameter && parameter <= root.parameter[1]);
                }
                assert_eq!(sign(roots[0].derivative), Some(reversed_normal));
                assert_eq!(sign(roots[1].derivative), Some(!reversed_normal));
            }
        }
    }

    #[test]
    fn original_scalar_failures_preserve_charged_work_and_fatal_errors() {
        let cancel = AtomicBool::new(false);
        let mut preparation = ValidatedGeometryBudget::new(100_000);
        let curve = Curve::Line(Line(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)));
        let prepared = prepared_validated_curve(
            &curve,
            [0.0, 1.0],
            &ValidatedCurveOptions::default(),
            &mut preparation,
            &cancel,
        )
        .unwrap();
        let projection = OriginalPlaneProjection::new([ZERO, ZERO, ONE], [ZERO; 3]).unwrap();
        let mut exhausted = ValidatedGeometryBudget::new(1);
        assert!(matches!(
            projection.jet(&prepared, [0.0, 1.0], ONE, &mut exhausted, &cancel),
            Err(ValidatedGeometryError::ResourceLimit)
        ));
        assert_eq!(exhausted.consumed(), 1);
        let mut invalid = ValidatedGeometryBudget::new(100);
        assert!(matches!(
            projection.jet(&prepared, [-1.0, 0.0], ONE, &mut invalid, &cancel),
            Err(ValidatedGeometryError::ParameterOutOfRange)
        ));
        assert!(invalid.consumed() > 0);
        let mut cancelled = ValidatedGeometryBudget::new(100);
        assert!(matches!(
            projection.jet(
                &prepared,
                [0.0, 1.0],
                ONE,
                &mut cancelled,
                &AtomicBool::new(true),
            ),
            Err(ValidatedGeometryError::Cancelled)
        ));
        assert_eq!(cancelled.consumed(), 0);
    }

    #[test]
    fn expanded_root_witness_cannot_use_unexpanded_monotonicity() {
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(1_000_000);
        let roots = isolate_roots(
            [0.0, 1.0],
            1024 * 1024,
            |range, budget| {
                budget.charge(1, &cancel)?;
                let t = interval_range(range)?;
                Ok(ValidatedScalarCurveJet {
                    position: (t - ValidatedInterval::point(0.5))
                        * (t - ValidatedInterval::point(0.7)),
                    first: t * ValidatedInterval::point(2.0)
                        - ValidatedInterval::point(0.5)
                        - ValidatedInterval::point(0.7),
                    second: ValidatedInterval::point(2.0),
                })
            },
            &mut budget,
            &cancel,
        )
        .unwrap();
        assert_eq!(roots.len(), 2);
        assert!(roots[0].parameter[0] <= 0.5 && roots[0].parameter[1] >= 0.5);
        assert!(roots[1].parameter[0] <= 0.7 && roots[1].parameter[1] >= 0.7);
    }
}
