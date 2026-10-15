use super::super::{CoordinatePlane, Projection, certify, implicit_jet};
use super::*;
use crate::{BoundedCurve, ValidatedScalarCurveJet, validated_curve2d_jet};
use monstertruck_topology::EdgeId;
use std::collections::HashMap;

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
                let combined = [
                    previous.witness[0].min(witness[0]),
                    previous.witness[1].max(witness[1]),
                ];
                if jet(combined, budget)?.first.contains_zero() {
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

pub(super) fn prepare_endpoints<'a>(
    context: &FacePlaneContext<'_, '_>,
    candidates: &[&PreparedValidatedCurve<'_>],
    bindings: &[ValidatedPlaneTrimEndpointBinding<'a>],
    maximum_bytes: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Vec<PreparedEndpoint<'a>>, ValidatedGeometryError> {
    let &FacePlaneContext {
        domain,
        normal,
        origin,
        maximum_extension,
        ..
    } = context;
    let mut result: Vec<PreparedEndpoint<'a>> = Vec::new();
    let entry_bytes = size_of::<(EdgeId<Curve>, usize)>() + 8;
    let map_admission = domain
        .coedges()
        .len()
        .checked_mul(entry_bytes)
        .and_then(|x| x.checked_mul(8))
        .ok_or(ValidatedGeometryError::ResourceLimit)?;
    if map_admission > maximum_bytes {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    budget.charge(domain.coedges().len(), cancel)?;
    let mut uses = HashMap::<EdgeId<Curve>, usize>::new();
    uses.try_reserve(domain.coedges().len())
        .map_err(|_| ValidatedGeometryError::ResourceLimit)?;
    let map_bytes = uses
        .capacity()
        .checked_mul(entry_bytes)
        .and_then(|x| x.checked_mul(4))
        .ok_or(ValidatedGeometryError::ResourceLimit)?;
    if map_bytes > map_admission {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    let maximum_bytes = maximum_bytes
        .checked_sub(map_bytes)
        .ok_or(ValidatedGeometryError::ResourceLimit)?;
    for coedge in domain.coedges() {
        budget.charge(1, cancel)?;
        uses.entry(coedge.identity.edge_id).or_insert(usize::MAX);
    }
    for (binding_index, binding) in bindings.iter().enumerate() {
        budget.charge(1, cancel)?;
        let boundary = domain
            .coedges()
            .get(binding.coedge_index)
            .ok_or(ValidatedGeometryError::InvalidInput)?;
        if binding.candidate_index >= candidates.len()
            || boundary.identity.edge_id != binding.original_edge_id
            || !binding.boundary_scale.is_finite()
            || binding.boundary_scale <= 0.0
            || !binding.source_cut_parameter.is_finite()
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
            || binding.source_cut_parameter < binding.boundary_range[0]
            || binding.source_cut_parameter > binding.boundary_range[1]
        {
            return Err(ValidatedGeometryError::ParameterOutOfRange);
        }
        let use_ = uses
            .get_mut(&binding.original_edge_id)
            .ok_or(ValidatedGeometryError::InvalidInput)?;
        if *use_ == usize::MAX {
            let edge = domain
                .face()
                .absolute_boundaries()
                .get(boundary.identity.wire_index)
                .and_then(|wire| wire.get(boundary.identity.coedge_index))
                .ok_or(ValidatedGeometryError::InvalidInput)?;
            if edge.id() != binding.original_edge_id
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
            *use_ = binding_index;
        } else {
            let first = &bindings[*use_];
            if !std::ptr::eq(binding.boundary, first.boundary)
                || binding.boundary_range != first.boundary_range
                || binding.boundary_scale != first.boundary_scale
            {
                return Err(ValidatedGeometryError::InvalidInput);
            }
        }
    }
    for (index, boundary) in domain.coedges().iter().enumerate() {
        let range = boundary.curve.range_tuple();
        let &first_index = uses
            .get(&boundary.identity.edge_id)
            .ok_or(ValidatedGeometryError::InvalidInput)?;
        let Some(first) = bindings.get(first_index) else {
            // Missing incidence is not inferred from a nominal pcurve. The
            // COMPLETE source-chart uncertainty tube must exclude the plane.
            prove_no_crossing(context, index, budget, cancel)?;
            continue;
        };
        let occupied = result
            .capacity()
            .checked_mul(size_of::<PreparedEndpoint<'_>>())
            .ok_or(ValidatedGeometryError::ResourceLimit)?;
        let root_allowance = maximum_bytes
            .checked_sub(occupied)
            .ok_or(ValidatedGeometryError::ResourceLimit)?;
        let roots = isolate_roots(
            [range.0, range.1],
            root_allowance,
            |range, budget| Ok(chart_jet(context, index, range, budget, cancel)?.0),
            budget,
            cancel,
        )?;
        let root_bytes = roots
            .capacity()
            .checked_mul(size_of::<Root>())
            .ok_or(ValidatedGeometryError::ResourceLimit)?;
        let original_allowance = root_allowance
            .checked_sub(root_bytes)
            .ok_or(ValidatedGeometryError::ResourceLimit)?;
        let scale = ValidatedInterval::point(first.boundary_scale);
        let original_roots = isolate_roots(
            first.boundary_range,
            original_allowance,
            |range, budget| {
                let jet = first.boundary.jet(range, budget, cancel)?;
                Ok(ValidatedScalarCurveJet {
                    position: dot3(normal, subtract3(jet.position.map(|x| x * scale), origin)),
                    first: dot3(normal, jet.first.map(|x| x * scale)),
                    second: dot3(normal, jet.second.map(|x| x * scale)),
                })
            },
            budget,
            cancel,
        )?;
        if original_roots.len() != roots.len() {
            return Err(ValidatedGeometryError::Unresolved);
        }
        for (&root, &original) in roots.iter().zip(&original_roots) {
            // Native and UV gauges are independently retained. Pair complete
            // root inventories by the original oriented coedge order, never
            // by an invented affine parameter map or nearest-point match.
            if sign(root.derivative) != sign(original.derivative) {
                return Err(ValidatedGeometryError::NonTransverse);
            }
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
                if !std::ptr::eq(binding.boundary, first.boundary)
                    || binding.boundary_scale != first.boundary_scale
                    || binding.boundary_range != first.boundary_range
                {
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
            let temporary = root_bytes
                .checked_add(
                    original_roots
                        .capacity()
                        .checked_mul(size_of::<Root>())
                        .ok_or(ValidatedGeometryError::ResourceLimit)?,
                )
                .ok_or(ValidatedGeometryError::ResourceLimit)?;
            super::super::reserve_native_vec(
                &mut result,
                maximum_bytes,
                temporary,
                budget,
                cancel,
            )?;
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
            let jet =
                validated_surface_jet(surface, uv.map(|x| [x.lower, x.upper]), budget, cancel)?;
            if !dot3(normal, subtract3(jet.position, origin)).contains_zero() {
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
    }
    Ok(())
}

fn chart_jet(
    context: &FacePlaneContext<'_, '_>,
    coedge: usize,
    parameter: [f64; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<
    (
        ValidatedScalarCurveJet,
        [ValidatedInterval; 2],
        [ValidatedInterval; 3],
    ),
    ValidatedGeometryError,
> {
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
    let uv = validated_curve2d_jet(boundary.curve, parameter, budget, cancel)?;
    let jet = validated_surface_jet(
        surface,
        uv.position.map(|x| [x.lower, x.upper]),
        budget,
        cancel,
    )?;
    let first = std::array::from_fn(|i| jet.du[i] * uv.first[0] + jet.dv[i] * uv.first[1]);
    Ok((
        ValidatedScalarCurveJet {
            position: dot3(normal, subtract3(jet.position, origin)),
            first: dot3(normal, first),
            second: super::super::ZERO,
        },
        uv.position,
        jet.position,
    ))
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
        ..
    } = context;
    // Any actual source root belongs to the COMPLETE retained chart tube.
    // Bound its possible parameter displacement by the nominal height error
    // divided by the proved strict nominal height derivative. No derivative
    // of the positional source-chart uncertainty is assumed.
    let witness_uv = domain
        .boundary_position_enclosure(coedge, root.witness, budget, cancel)
        .map_err(trim_error)?;
    let witness_jet = validated_surface_jet(
        surface,
        witness_uv.map(|x| [x.lower, x.upper]),
        budget,
        cancel,
    )?;
    let margin = domain.uncertainty_uv();
    let height_error = (dot3(normal, witness_jet.du)
        * ValidatedInterval::new(-margin[0], margin[0])?
        + dot3(normal, witness_jet.dv) * ValidatedInterval::new(-margin[1], margin[1])?)
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
        let jet = validated_surface_jet(
            surface,
            joint.enclosure.map(|x| [x.lower, x.upper]),
            budget,
            cancel,
        )?;
        if dot3(normal, subtract3(jet.position, origin)).contains_zero() {
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

pub(super) fn endpoint_covers(
    context: &FacePlaneContext<'_, '_>,
    candidates: &[&PreparedValidatedCurve<'_>],
    prepared: &PreparedEndpoint<'_>,
    all_roots: &[PreparedEndpoint<'_>],
    coverage_cell: CoverageCell,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<EndpointCoverage, ValidatedGeometryError> {
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
    if !(0..4).all(|i| contains(box4[i], q[i])) {
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
    let material_positive = sign(-root.derivative * determinant)
        .map(|x| if boundary.identity.orientation { x } else { !x })
        .ok_or(ValidatedGeometryError::NonTransverse)?;
    let candidate_positive =
        sign(endpoint_jet.first[axis]).ok_or(ValidatedGeometryError::NonTransverse)?;
    if material_positive != (candidate_positive != binding.endpoint_is_back) {
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
    let coordinate = position[axis]
        .hull(original_root[axis])
        .hull(nominal_root[axis])
        .hull(nominal_end[axis]);
    if coordinate.lower >= coordinate.upper {
        return Ok(EndpointCoverage::Refused(
            "endpoint / nondegenerate fixed coordinate",
            ValidatedGeometryError::Unresolved,
        ));
    }
    let mut direction = [0.0; 3];
    direction[axis] = 1.0;
    let projection = Projection::BetweenPlanes {
        parameter: [coordinate.lower, coordinate.upper],
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
    if !(0..4).all(|i| contains(box4[i], root_q[i])) {
        return Ok(EndpointCoverage::Refused(
            "endpoint / original trim root in original uniqueness tube",
            ValidatedGeometryError::Unresolved,
        ));
    }
    let mut tube = match certify(
        candidate.source.curve()?,
        projection,
        coordinate,
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
        let next = std::array::from_fn(|i| {
            let root = tube.rooted_domain[i];
            let radius =
                (root.width() * 0.51 + 64.0 * f64::EPSILON * (1.0 + root.magnitude())).next_up();
            box4[i]
                .intersection(ValidatedInterval {
                    lower: (root.midpoint() - radius).next_down(),
                    upper: (root.midpoint() + radius).next_up(),
                })
                .unwrap_or(box4[i])
        });
        match certify(
            candidate.source.curve()?,
            projection,
            coordinate,
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
    use super::*;

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
