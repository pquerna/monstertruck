use super::{
    PreparedValidatedCurve, carriers, dot3, flatten, interval_range, projection_jet, subtract3,
    validated_surface_piecewise_jet as validated_surface_jet,
};
use crate::{
    Conic2D, Curve, Curve2D, ParameterBoxClassification, Plane, PreparedFaceParameterDomain,
    Surface, TrimDomainError, ValidatedGeometryBudget, ValidatedGeometryError, ValidatedInterval,
};
use std::sync::atomic::AtomicBool;

mod endpoints;
mod source_identity;
pub use endpoints::{ValidatedPlaneTrimEndpoint, ValidatedPlaneTrimEndpointBinding};

const COVERAGE_STACK_BYTES: usize = 16 * 1024;

/// Bounded full-face plane-zero subdivision policy.
#[derive(Clone, Copy, Debug)]
pub struct ValidatedPlaneCoverageOptions {
    /// Maximum binary UV subdivision depth, at most 48.
    pub max_depth: usize,
    /// Maximum additional physical endpoint error, zero to disable collars.
    pub maximum_endpoint_extension_mm: f64,
    /// Combined owned matches, original root proofs and endpoint receipts.
    pub max_retained_bytes: usize,
}
impl Default for ValidatedPlaneCoverageOptions {
    fn default() -> Self {
        Self {
            max_depth: 48,
            maximum_endpoint_extension_mm: 0.0,
            max_retained_bytes: 32 * 1024 * 1024,
        }
    }
}

/// A completed full-chart zero-candidate census, not a sampled absence test.
#[derive(Clone, Debug)]
pub struct ValidatedPlaneCoverage {
    /// Number of closed UV cells examined, including unsuccessful parents.
    pub visited_cells: usize,
    /// Number discharged by strict trim exclusion or plane-residual exclusion.
    pub excluded_cells: usize,
    /// Number whose every plane zero is represented by a prepared branch.
    pub covered_cells: usize,
    /// Sealed endpoint-root error receipts, paired with the original candidates.
    pub endpoint_strata: Vec<ValidatedPlaneTrimEndpoint>,
}

/// Proves that all plane zeros of an authoritative trimmed face are represented.
///
/// The complete native covering-chart rectangle is subdivided. Each terminal
/// cell is either strictly outside the material winding, has an interval plane
/// residual excluding zero, or has all its plane roots mapped into existing
/// prepared branch tubes. Coverage requires interval bracketing and strict
/// monotonicity of the actual prepared coordinate, source coefficient identity,
/// and containment in every applicable uniqueness tube. UV-box overlap alone
/// is never coverage. Missing closed interior components therefore cause a typed
/// refusal instead of a false empty section. Boundary, singular, nonsmooth or
/// endpoint cells that cannot be resolved within the allowance also refuse.
///
/// The prepared domain must retain authoritative source-to-native trim
/// correspondence. This routine does not upgrade a sampled pcurve to authority.
/// Candidates must directly retain these exact source carriers, as produced by
/// [`super::prepared_face_plane_curve`]. A spatially mapped standalone binding
/// is not evidence of association with an unrelated world-space face.
///
/// Every original trim-plane crossing must retain an original cut binding.
/// Endpoint closure additionally proves its oriented transverse trim incidence
/// and unique continuation from the actual retained candidate end. The caller
/// must consume every returned endpoint metric in sampling and area bounds.
pub fn validated_face_plane_coverage(
    domain: &PreparedFaceParameterDomain<'_>,
    plane: &Plane,
    candidates: &[&PreparedValidatedCurve<'_>],
    endpoint_bindings: &[ValidatedPlaneTrimEndpointBinding<'_>],
    options: &ValidatedPlaneCoverageOptions,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<ValidatedPlaneCoverage, ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    if options.max_depth > 48
        || !options.maximum_endpoint_extension_mm.is_finite()
        || options.maximum_endpoint_extension_mm < 0.0
    {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    let storage_limit = options
        .max_retained_bytes
        .min(32 * 1024 * 1024)
        .checked_sub(COVERAGE_STACK_BYTES)
        .ok_or(ValidatedGeometryError::ResourceLimit)?;
    let bounds = domain.closed_parameter_box();
    let axes = [interval_range(bounds[0])?, interval_range(bounds[1])?];
    if axes.into_iter().any(|axis| axis.lower >= axis.upper) {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    let origin = plane.origin();
    if !(0..3).all(|i| origin[i].is_finite()) {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    let (a, b) = plane_axes(plane);
    let normal = super::cross3(a, b);
    let norm = normal
        .map(ValidatedInterval::square)
        .into_iter()
        .fold(super::ZERO, |a, b| a + b);
    if !norm.is_finite() || !normal.into_iter().all(ValidatedInterval::is_finite) {
        return Err(ValidatedGeometryError::ConditioningFailure);
    }
    if norm.lower <= 0.0 {
        return Err(ValidatedGeometryError::Singular);
    }
    let origin = std::array::from_fn(|i| ValidatedInterval::point(origin[i]));
    domain.face().with_surface(|surface| {
        let context = endpoints::FacePlaneContext {
            domain,
            surface,
            plane,
            normal,
            origin,
            maximum_extension: options.maximum_endpoint_extension_mm,
        };
        let mut matches = Vec::new();
        for (candidate_index, candidate) in candidates.iter().enumerate() {
            budget.charge(1, cancel)?;
            let Ok((s0, s1, _)) = candidate.source.curve().and_then(carriers) else {
                continue;
            };
            let index = if matches!(s1, Surface::Plane(other) if other == plane)
                && exact_surface(surface, s0, 0, budget, cancel)?
            {
                Some(0)
            } else if matches!(s0, Surface::Plane(other) if other == plane)
                && exact_surface(surface, s1, 0, budget, cancel)?
            {
                Some(1)
            } else {
                None
            };
            if let Some(index) = index {
                budget.charge(1, cancel)?;
                super::reserve_native_vec(&mut matches, storage_limit, 0, budget, cancel)?;
                matches.push((candidate_index, *candidate, index));
            }
        }
        let match_bytes = matches
            .capacity()
            .checked_mul(size_of::<(usize, &PreparedValidatedCurve<'_>, usize)>())
            .ok_or(ValidatedGeometryError::ResourceLimit)?;
        for binding in endpoint_bindings {
            if matching_source_index(&matches, binding.candidate_index, budget, cancel)?.is_none() {
                return Err(ValidatedGeometryError::InvalidInput);
            }
        }
        let receipt_bytes = endpoint_bindings
            .len()
            .checked_mul(size_of::<ValidatedPlaneTrimEndpoint>())
            .ok_or(ValidatedGeometryError::ResourceLimit)?;
        let endpoint_allowance = storage_limit
            .checked_sub(match_bytes)
            .and_then(|x| x.checked_sub(receipt_bytes))
            .ok_or(ValidatedGeometryError::ResourceLimit)?;
        let prepared_endpoints = endpoints::prepare_endpoints(
            &context,
            candidates,
            endpoint_bindings,
            endpoint_allowance,
            budget,
            cancel,
        )
        .map_err(|error| coverage_context(error, "complete original trim-root census"))?;
        budget.charge(endpoint_bindings.len(), cancel)?;
        let mut endpoint_strata = Vec::new();
        endpoint_strata
            .try_reserve_exact(endpoint_bindings.len())
            .map_err(|_| ValidatedGeometryError::ResourceLimit)?;
        if endpoint_strata
            .capacity()
            .checked_mul(size_of::<ValidatedPlaneTrimEndpoint>())
            .is_none_or(|bytes| bytes > receipt_bytes)
        {
            return Err(ValidatedGeometryError::ResourceLimit);
        }
        // Seal every actual original crossing even when a later interior cell
        // happens to bracket an exactly representable endpoint. This prevents
        // an unused, wrong-end binding from bypassing material-side validation.
        for endpoint in &prepared_endpoints {
            let source_index =
                matching_source_index(&matches, endpoint.binding.candidate_index, budget, cancel)?
                    .ok_or(ValidatedGeometryError::InvalidInput)?;
            let cell = endpoint.root_cell();
            let jet = validated_surface_jet(surface, cell, budget, cancel)?;
            let receipt = endpoints::endpoint_covers(
                &context,
                candidates,
                endpoint,
                &prepared_endpoints,
                endpoints::CoverageCell {
                    source_index,
                    uv: cell,
                    position: jet.position,
                },
                budget,
                cancel,
            )
            .map_err(|error| coverage_context(error, "original endpoint unique continuation"))?
            .ok_or(ValidatedGeometryError::PlaneCoverageUnresolved {
                operation: "original endpoint unique continuation",
            })?;
            endpoints::retain_receipt(&mut endpoint_strata, receipt, budget, cancel)?;
        }
        let mut pending = [(bounds, 0usize); 49];
        let mut length = 1;
        let mut result = ValidatedPlaneCoverage {
            visited_cells: 0,
            excluded_cells: 0,
            covered_cells: 0,
            endpoint_strata,
        };
        while length != 0 {
            budget.charge(1, cancel)?;
            length -= 1;
            let (cell, depth) = pending[length];
            result.visited_cells += 1;
            let classification = domain
                .classify_parameter_box(cell, budget, cancel)
                .map_err(trim_error)?;
            if classification == ParameterBoxClassification::Outside {
                result.excluded_cells += 1;
                continue;
            }
            let enclosure = validated_surface_jet(surface, cell, budget, cancel);
            let mut covered = false;
            match enclosure {
                Ok(jet) => {
                    let residual = dot3(normal, subtract3(jet.position, origin));
                    if !residual.is_finite() {
                        return Err(ValidatedGeometryError::ConditioningFailure);
                    }
                    if !residual.contains_zero() {
                        result.excluded_cells += 1;
                        continue;
                    }
                    for &(candidate_index, candidate, index) in &matches {
                        budget.charge(1, cancel)?;
                        if candidate_covers(
                            candidate,
                            index,
                            cell,
                            jet.position,
                            plane,
                            budget,
                            cancel,
                        )? {
                            covered = true;
                            break;
                        }
                        if options.maximum_endpoint_extension_mm > 0.0 {
                            for endpoint in &prepared_endpoints {
                                budget.charge(1, cancel)?;
                                if endpoint.binding.candidate_index != candidate_index {
                                    continue;
                                }
                                if let Some(receipt) = endpoints::endpoint_covers(
                                    &context,
                                    candidates,
                                    endpoint,
                                    &prepared_endpoints,
                                    endpoints::CoverageCell {
                                        source_index: index,
                                        uv: cell,
                                        position: jet.position,
                                    },
                                    budget,
                                    cancel,
                                )? {
                                    endpoints::retain_receipt(
                                        &mut result.endpoint_strata,
                                        receipt,
                                        budget,
                                        cancel,
                                    )?;
                                    covered = true;
                                    break;
                                }
                            }
                            if covered {
                                break;
                            }
                        }
                    }
                }
                Err(error) if splittable(error) => {}
                Err(error) => return Err(error),
            }
            if covered {
                result.covered_cells += 1;
                continue;
            }
            if depth == options.max_depth {
                return Err(ValidatedGeometryError::PlaneCoverageUnresolved {
                    operation: "complete original face cells / subdivision depth",
                });
            }
            let (axis, middle) = split_cell(surface, cell, budget, cancel)?;
            if middle <= cell[axis][0] || middle >= cell[axis][1] {
                return Err(ValidatedGeometryError::PlaneCoverageUnresolved {
                    operation: "complete original face cells / finite split",
                });
            }
            let mut lower = cell;
            let mut upper = cell;
            lower[axis][1] = middle;
            upper[axis][0] = middle;
            budget.charge(2, cancel)?;
            pending[length] = (upper, depth + 1);
            pending[length + 1] = (lower, depth + 1);
            length += 2;
        }
        Ok(result)
    })
}

fn matching_source_index(
    matches: &[(usize, &PreparedValidatedCurve<'_>, usize)],
    candidate_index: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Option<usize>, ValidatedGeometryError> {
    // Matches retain the original candidate enumeration order.
    let (mut low, mut high) = (0, matches.len());
    while low < high {
        budget.charge(1, cancel)?;
        let middle = low + (high - low) / 2;
        if matches[middle].0 == candidate_index {
            return Ok(Some(matches[middle].2));
        }
        if matches[middle].0 < candidate_index {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    Ok(None)
}

fn coverage_context(
    error: ValidatedGeometryError,
    operation: &'static str,
) -> ValidatedGeometryError {
    if error == ValidatedGeometryError::Unresolved {
        ValidatedGeometryError::PlaneCoverageUnresolved { operation }
    } else {
        error
    }
}

fn candidate_covers(
    candidate: &PreparedValidatedCurve<'_>,
    source_index: usize,
    cell: [[f64; 2]; 2],
    position: [ValidatedInterval; 3],
    plane: &Plane,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<bool, ValidatedGeometryError> {
    let (_, _, leader) = carriers(candidate.source.curve()?)?;
    let plane_parameters = plane_parameters(plane, position)?;
    let source = [interval_range(cell[0])?, interval_range(cell[1])?];
    let q = if source_index == 0 {
        [
            source[0],
            source[1],
            plane_parameters[0],
            plane_parameters[1],
        ]
    } else {
        [
            plane_parameters[0],
            plane_parameters[1],
            source[0],
            source[1],
        ]
    };
    for start in 0..candidate.pieces.len() {
        // Up to three neighboring tubes handle cells straddling coordinate joints.
        for count in 1..=3 {
            budget.charge(1, cancel)?;
            if start + count > candidate.pieces.len() {
                break;
            }
            let pieces = &candidate.pieces[start..start + count];
            let mut monotonicity = None;
            let mut contains = true;
            for piece in pieces {
                budget.charge(1, cancel)?;
                let box4 = flatten(
                    piece
                        .source_parameters
                        .ok_or(ValidatedGeometryError::InvalidInput)?,
                );
                if !(0..4).all(|i| box4[i].lower <= q[i].lower && q[i].upper <= box4[i].upper) {
                    contains = false;
                    break;
                }
                let coordinate = projection_jet(
                    leader,
                    piece.projection,
                    interval_range(piece.parameter)?,
                    budget,
                    cancel,
                )?;
                let derivative = dot3(coordinate.first, position) - coordinate.offset_first;
                let sign = if derivative.lower > 0.0 {
                    Some(true)
                } else if derivative.upper < 0.0 {
                    Some(false)
                } else {
                    None
                };
                if sign.is_none() || monotonicity.is_some_and(|previous| Some(previous) != sign) {
                    contains = false;
                    break;
                }
                monotonicity = sign;
            }
            if !contains {
                continue;
            }
            let first = pieces[0];
            let last = pieces[pieces.len() - 1];
            let low = projection_jet(
                leader,
                first.projection,
                ValidatedInterval::point(first.parameter[0]),
                budget,
                cancel,
            )?;
            let high = projection_jet(
                leader,
                last.projection,
                ValidatedInterval::point(last.parameter[1]),
                budget,
                cancel,
            )?;
            let low = dot3(low.normal, position) - low.offset;
            let high = dot3(high.normal, position) - high.offset;
            let bracket = if monotonicity == Some(true) {
                low.upper <= 0.0 && high.lower >= 0.0
            } else {
                low.lower >= 0.0 && high.upper <= 0.0
            };
            if bracket {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

pub(super) fn plane_axes(plane: &Plane) -> ([ValidatedInterval; 3], [ValidatedInterval; 3]) {
    match plane.stored_axes() {
        Some([a, b]) => (
            std::array::from_fn(|i| ValidatedInterval::point(a[i])),
            std::array::from_fn(|i| ValidatedInterval::point(b[i])),
        ),
        None => {
            let [origin, a, b] = plane.defining_points();
            (
                std::array::from_fn(|i| {
                    ValidatedInterval::point(a[i]) - ValidatedInterval::point(origin[i])
                }),
                std::array::from_fn(|i| {
                    ValidatedInterval::point(b[i]) - ValidatedInterval::point(origin[i])
                }),
            )
        }
    }
}

fn plane_parameters(
    plane: &Plane,
    point: [ValidatedInterval; 3],
) -> Result<[ValidatedInterval; 2], ValidatedGeometryError> {
    let origin = plane.origin();
    let (a, b) = plane_axes(plane);
    let delta = subtract3(
        point,
        std::array::from_fn(|i| ValidatedInterval::point(origin[i])),
    );
    let aa = dot3(a, a);
    let ab = dot3(a, b);
    let bb = dot3(b, b);
    let inverse = (aa * bb - ab.square()).reciprocal()?;
    let ar = dot3(a, delta);
    let br = dot3(b, delta);
    Ok([(bb * ar - ab * br) * inverse, (aa * br - ab * ar) * inverse])
}

fn split_cell(
    surface: &Surface,
    cell: [[f64; 2]; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<(usize, f64), ValidatedGeometryError> {
    let knots = match surface {
        Surface::BsplineSurface(surface) => Some(surface.knot_vectors()),
        Surface::NurbsSurface(surface) => Some(surface.knot_vectors()),
        _ => None,
    };
    if let Some(knots) = knots {
        for (axis, vector) in [&knots.0, &knots.1].into_iter().enumerate() {
            for &knot in vector.iter() {
                budget.charge(1, cancel)?;
                if cell[axis][0] < knot && knot < cell[axis][1] {
                    return Ok((axis, knot));
                }
            }
        }
    }
    let width = cell.map(|axis| axis[1] - axis[0]);
    let axis = usize::from(width[1] > width[0]);
    Ok((axis, cell[axis][0] * 0.5 + cell[axis][1] * 0.5))
}

fn trim_error(error: TrimDomainError) -> ValidatedGeometryError {
    match error {
        TrimDomainError::Geometry(error) => error,
        TrimDomainError::ResourceLimit => ValidatedGeometryError::ResourceLimit,
        TrimDomainError::InvalidChart => ValidatedGeometryError::InvalidInput,
        _ => ValidatedGeometryError::Unsupported,
    }
}
fn splittable(error: ValidatedGeometryError) -> bool {
    super::retryable(error) || error == ValidatedGeometryError::NonSmooth
}

fn equal_slice<T: PartialEq>(
    a: &[T],
    b: &[T],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<bool, ValidatedGeometryError> {
    if a.len() != b.len() {
        return Ok(false);
    }
    for (a, b) in a.iter().zip(b) {
        budget.charge(1, cancel)?;
        if a != b {
            return Ok(false);
        }
    }
    Ok(true)
}
fn equal_rows<T: PartialEq>(
    a: &[Vec<T>],
    b: &[Vec<T>],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<bool, ValidatedGeometryError> {
    if a.len() != b.len() {
        return Ok(false);
    }
    for (a, b) in a.iter().zip(b) {
        if !equal_slice(a, b, budget, cancel)? {
            return Ok(false);
        }
    }
    Ok(true)
}
/// Compares stored native surface coefficients without samples or recognition.
///
/// Matrices, orientation, knots, control values and revolution profiles must
/// match exactly. Unsupported T-spline snapshots return `false`. Work and
/// cancellation remain cumulative with the surrounding validated query.
pub fn validated_same_surface_coefficients(
    a: &Surface,
    b: &Surface,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<bool, ValidatedGeometryError> {
    exact_surface(a, b, 0, budget, cancel)
}

fn exact_surface(
    a: &Surface,
    b: &Surface,
    depth: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<bool, ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    if depth > crate::geometry::MAX_NATIVE_GEOMETRY_NESTING {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    Ok(match (a, b) {
        (Surface::Plane(a), Surface::Plane(b)) => a == b,
        (Surface::BsplineSurface(a), Surface::BsplineSurface(b)) => {
            equal_slice(&a.knot_vectors().0, &b.knot_vectors().0, budget, cancel)?
                && equal_slice(&a.knot_vectors().1, &b.knot_vectors().1, budget, cancel)?
                && equal_rows(a.control_points(), b.control_points(), budget, cancel)?
        }
        (Surface::NurbsSurface(a), Surface::NurbsSurface(b)) => {
            equal_slice(&a.knot_vectors().0, &b.knot_vectors().0, budget, cancel)?
                && equal_slice(&a.knot_vectors().1, &b.knot_vectors().1, budget, cancel)?
                && equal_rows(a.control_points(), b.control_points(), budget, cancel)?
        }
        (Surface::SphericalSurface(a), Surface::SphericalSurface(b)) => a == b,
        (Surface::ToroidalSurface(a), Surface::ToroidalSurface(b)) => a == b,
        (Surface::RevolutionSurface(a), Surface::RevolutionSurface(b)) => {
            a.orientation() == b.orientation()
                && a.transform() == b.transform()
                && a.entity().origin() == b.entity().origin()
                && a.entity().axis() == b.entity().axis()
                && exact_curve(
                    a.entity().entity_curve(),
                    b.entity().entity_curve(),
                    depth + 1,
                    budget,
                    cancel,
                )?
        }
        (Surface::ExtrusionSurface(a), Surface::ExtrusionSurface(b)) => {
            a.extruding_vector() == b.extruding_vector()
                && exact_curve(
                    a.entity_curve(),
                    b.entity_curve(),
                    depth + 1,
                    budget,
                    cancel,
                )?
        }
        _ => false,
    })
}

fn exact_curve(
    a: &Curve,
    b: &Curve,
    depth: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<bool, ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    if depth > crate::geometry::MAX_NATIVE_GEOMETRY_NESTING {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    Ok(match (a, b) {
        (Curve::Line(a), Curve::Line(b)) => a == b,
        (Curve::BsplineCurve(a), Curve::BsplineCurve(b)) => {
            equal_slice(a.knot_vector(), b.knot_vector(), budget, cancel)?
                && equal_slice(a.control_points(), b.control_points(), budget, cancel)?
        }
        (Curve::NurbsCurve(a), Curve::NurbsCurve(b)) => {
            equal_slice(a.knot_vector(), b.knot_vector(), budget, cancel)?
                && equal_slice(a.control_points(), b.control_points(), budget, cancel)?
        }
        (Curve::ParameterCurve(a), Curve::ParameterCurve(b)) => {
            exact_curve2d(a.curve(), b.curve(), depth + 1, budget, cancel)?
                && exact_surface(a.surface(), b.surface(), depth + 1, budget, cancel)?
        }
        (Curve::TrimmedCurve(a), Curve::TrimmedCurve(b)) => {
            a.orientation() == b.orientation()
                && a.transform() == b.transform()
                && a.entity().range() == b.entity().range()
                && exact_curve(
                    a.entity().curve(),
                    b.entity().curve(),
                    depth + 1,
                    budget,
                    cancel,
                )?
        }
        _ => false,
    })
}
fn exact_curve2d(
    a: &Curve2D,
    b: &Curve2D,
    depth: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<bool, ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    if depth > crate::geometry::MAX_NATIVE_GEOMETRY_NESTING {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    Ok(match (a, b) {
        (Curve2D::Line(a), Curve2D::Line(b)) => a == b,
        (Curve2D::Polyline(a), Curve2D::Polyline(b)) => {
            equal_slice(a.as_slice(), b.as_slice(), budget, cancel)?
        }
        (Curve2D::BsplineCurve(a), Curve2D::BsplineCurve(b)) => {
            equal_slice(a.knot_vector(), b.knot_vector(), budget, cancel)?
                && equal_slice(a.control_points(), b.control_points(), budget, cancel)?
        }
        (Curve2D::NurbsCurve(a), Curve2D::NurbsCurve(b)) => {
            equal_slice(a.knot_vector(), b.knot_vector(), budget, cancel)?
                && equal_slice(a.control_points(), b.control_points(), budget, cancel)?
        }
        (Curve2D::Conic(a), Curve2D::Conic(b)) => match (a, b) {
            (Conic2D::Ellipse(a), Conic2D::Ellipse(b)) => a == b,
            (Conic2D::Hyperbola(a), Conic2D::Hyperbola(b)) => a == b,
            (Conic2D::Parabola(a), Conic2D::Parabola(b)) => a == b,
            _ => false,
        },
        (Curve2D::TrimmedCurve(a), Curve2D::TrimmedCurve(b)) => {
            a.orientation() == b.orientation()
                && a.transform() == b.transform()
                && a.entity().range() == b.entity().range()
                && exact_curve2d(
                    a.entity().curve(),
                    b.entity().curve(),
                    depth + 1,
                    budget,
                    cancel,
                )?
        }
        _ => false,
    })
}
