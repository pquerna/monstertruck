use super::{
    CoordinatePlane, PreparedValidatedCurve, Projection, ValidatedCurvePiece, carriers, certify,
    dot3, flatten, implicit_jet, interval_range, projection_jet, projection_residual, subtract3,
    unflatten, validated_surface_piecewise_jet as validated_surface_jet,
};
use crate::{
    Conic2D, Curve, Curve2D, ParameterBoxClassification, Plane, PreparedFaceParameterDomain,
    Surface, TrimDomainError, ValidatedGeometryBudget, ValidatedGeometryError, ValidatedInterval,
    ValidatedPlaneCandidateWitness, ValidatedPlaneCoverageRejection, ValidatedPlaneCoverageWitness,
    ValidatedPlaneEndpointWitness, ValidatedPlaneJunctionCondition,
    ValidatedPlaneJunctionRejection, ValidatedPlaneJunctionWitness,
};
use std::sync::atomic::AtomicBool;

mod endpoints;
mod source_identity;
pub use endpoints::{
    ValidatedPlaneTrimEndpoint, ValidatedPlaneTrimEndpointBinding, validated_plane_trim_root_chart,
};

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
/// The complete native covering-chart rectangle is subdivided. Interval Newton
/// contraction first encloses every possible plane zero within each closed
/// cell. Each terminal cell is either strictly outside the material winding,
/// has no plane zero, or has all its plane roots mapped into existing prepared
/// branch tubes. Coverage requires interval bracketing and strict monotonicity
/// of the actual prepared coordinate, source coefficient identity,
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
    domain.face().with_surface(|surface| {
        let context = endpoints::FacePlaneContext::new(
            domain,
            surface,
            plane,
            options.maximum_endpoint_extension_mm,
        )?;
        let normal = context.normal;
        let origin = context.origin;
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
        .map_err(|error| coverage_context(error, "complete original trim-root census", None))?;
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
            let mut witness = None;
            let outcome = endpoints::endpoint_covers(
                &context,
                candidates,
                endpoint,
                &prepared_endpoints,
                endpoints::CoverageCell {
                    source_index,
                    uv: cell,
                    position: jet.position,
                },
                (&mut *budget, cancel),
                Some(&mut witness),
            )
            .map_err(|error| {
                coverage_context(error, "original endpoint unique continuation", Some(cell))
            })?;
            let receipt = match outcome {
                endpoints::EndpointCoverage::Covered(receipt) => receipt,
                endpoints::EndpointCoverage::Refused(operation, error) => {
                    let mut error = coverage_context(error, operation, Some(cell));
                    if let ValidatedGeometryError::PlaneCoverageUnresolved {
                        witness: recorded,
                        ..
                    } = &mut error
                    {
                        *recorded = witness;
                    }
                    return Err(error);
                }
            };
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
            let mut root_cell = cell;
            let enclosure = plane_zero_cell(surface, cell, normal, origin, budget, cancel);
            let mut covered = false;
            let mut witness = None;
            match enclosure {
                Ok(None) => {
                    result.excluded_cells += 1;
                    continue;
                }
                Ok(Some((contracted, position))) => {
                    root_cell = contracted;
                    for &(candidate_index, candidate, index) in &matches {
                        budget.charge(1, cancel)?;
                        if candidate_covers(
                            candidate,
                            index,
                            root_cell,
                            position,
                            plane,
                            (&mut *budget, cancel),
                            (depth == options.max_depth).then_some(CandidateDiagnostics {
                                candidate_index,
                                witness: &mut witness,
                            }),
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
                                if let endpoints::EndpointCoverage::Covered(receipt) =
                                    endpoints::endpoint_covers(
                                        &context,
                                        candidates,
                                        endpoint,
                                        &prepared_endpoints,
                                        endpoints::CoverageCell {
                                            source_index: index,
                                            uv: root_cell,
                                            position,
                                        },
                                        (&mut *budget, cancel),
                                        None,
                                    )?
                                {
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
                    if !covered {
                        covered = junction_covers(
                            &matches,
                            root_cell,
                            position,
                            plane,
                            budget,
                            cancel,
                            (depth == options.max_depth).then_some(&mut witness),
                        )?;
                    }
                }
                Err(error) if splittable(error) => {}
                Err(error) => return Err(error),
            }
            if covered {
                result.covered_cells += 1;
                continue;
            }
            // Winding is needed only when the original whole-cell zero proof
            // cannot discharge this cell. It must not spend the trim subdivision
            // allowance on cells already proved empty or covered.
            if domain
                .classify_parameter_box(root_cell, budget, cancel)
                .map_err(trim_error)?
                == ParameterBoxClassification::Outside
            {
                result.excluded_cells += 1;
                continue;
            }
            if depth == options.max_depth {
                return Err(ValidatedGeometryError::PlaneCoverageUnresolved {
                    operation: "complete original face cells / subdivision depth",
                    parameter_box: Some(root_cell),
                    witness,
                });
            }
            let (axis, middle) = split_cell(surface, root_cell, budget, cancel)?;
            if middle <= root_cell[axis][0] || middle >= root_cell[axis][1] {
                return Err(ValidatedGeometryError::PlaneCoverageUnresolved {
                    operation: "complete original face cells / finite split",
                    parameter_box: Some(root_cell),
                    witness,
                });
            }
            let mut lower = root_cell;
            let mut upper = root_cell;
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
    parameter_box: Option<[[f64; 2]; 2]>,
) -> ValidatedGeometryError {
    if error == ValidatedGeometryError::Unresolved {
        ValidatedGeometryError::PlaneCoverageUnresolved {
            operation,
            parameter_box,
            witness: None,
        }
    } else {
        error
    }
}

type PlaneZeroCell = ([[f64; 2]; 2], [ValidatedInterval; 3]);

fn plane_zero_cell(
    surface: &Surface,
    mut cell: [[f64; 2]; 2],
    normal: [ValidatedInterval; 3],
    origin: [ValidatedInterval; 3],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Option<PlaneZeroCell>, ValidatedGeometryError> {
    let mut jet = validated_surface_jet(surface, cell, budget, cancel)?;
    for axis in 0..2 {
        budget.charge(1, cancel)?;
        let residual = dot3(normal, subtract3(jet.position, origin));
        if !residual.is_finite() {
            return Err(ValidatedGeometryError::ConditioningFailure);
        }
        if !residual.contains_zero() {
            return Ok(None);
        }
        let derivative = dot3(normal, if axis == 0 { jet.du } else { jet.dv });
        if !derivative.is_finite() {
            return Err(ValidatedGeometryError::ConditioningFailure);
        }
        if derivative.contains_zero() || cell[axis][0] == cell[axis][1] {
            continue;
        }
        let range = interval_range(cell[axis])?;
        let midpoint = range.midpoint();
        let mut section = cell;
        section[axis] = [midpoint; 2];
        let section = validated_surface_jet(surface, section, budget, cancel)?;
        let value = dot3(normal, subtract3(section.position, origin));
        if !value.is_finite() {
            return Err(ValidatedGeometryError::ConditioningFailure);
        }
        // For every actual zero, the mean-value theorem places this coordinate
        // in the interval Newton image with the other coordinate still free.
        // This is only necessary-root exclusion, never an existence witness.
        let inverse = derivative.reciprocal()?;
        if !inverse.is_finite() {
            return Err(ValidatedGeometryError::ConditioningFailure);
        }
        let newton = ValidatedInterval::point(midpoint) - value * inverse;
        if !newton.is_finite() {
            return Err(ValidatedGeometryError::ConditioningFailure);
        }
        let Some(contracted) = range.intersection(newton) else {
            return Ok(None);
        };
        if contracted != range {
            cell[axis] = [contracted.lower, contracted.upper];
            jet = validated_surface_jet(surface, cell, budget, cancel)?;
        }
    }
    Ok(Some((cell, jet.position)))
}

struct CandidateDiagnostics<'a> {
    candidate_index: usize,
    witness: &'a mut Option<ValidatedPlaneCoverageWitness>,
}

impl CandidateDiagnostics<'_> {
    fn record(
        &mut self,
        group: (usize, usize),
        rejection: ValidatedPlaneCoverageRejection,
        coordinate: u8,
        bounds: [f64; 2],
    ) {
        use ValidatedPlaneCoverageRejection as R;
        let progress = |stage| match stage {
            R::TubeLower | R::TubeUpper => 0,
            R::CoordinateMonotonicity => 1,
            _ => 2,
        };
        if self.witness.is_some_and(|previous| match previous {
            ValidatedPlaneCoverageWitness::Candidate(previous) => {
                progress(previous.rejection) > progress(rejection)
                    || (progress(previous.rejection) == progress(rejection)
                        && usize::from(previous.piece_count) >= group.1)
            }
            ValidatedPlaneCoverageWitness::Junction(_)
            | ValidatedPlaneCoverageWitness::Endpoint(_) => true,
        }) {
            return;
        }
        *self.witness = Some(ValidatedPlaneCoverageWitness::Candidate(
            ValidatedPlaneCandidateWitness {
                candidate_index: self.candidate_index,
                first_piece_index: group.0,
                piece_count: group.1 as u8,
                rejection,
                coordinate,
                bounds,
            },
        ));
    }
}

struct JunctionDiagnostics<'a> {
    candidate_indices: [u32; 2],
    ends: [Option<bool>; 2],
    uv_separation: Option<(u8, [f64; 2])>,
    witness: &'a mut Option<ValidatedPlaneCoverageWitness>,
}

impl JunctionDiagnostics<'_> {
    fn record(
        &mut self,
        rejection: ValidatedPlaneJunctionRejection,
        coordinate: Option<u8>,
        bounds: Option<[f64; 2]>,
        condition: Option<ValidatedPlaneJunctionCondition>,
    ) {
        use ValidatedPlaneJunctionRejection as R;
        let progress = |stage| match stage {
            R::MissingFirstCoordinate => 0,
            R::MissingSecondCoordinate => 1,
            R::DifferentCoordinates => 2,
            R::TubeLower | R::TubeUpper => 3,
            R::FirstMonotonicity
            | R::SecondMonotonicity
            | R::SameAwayDirection
            | R::DegenerateCoordinate => 4,
            R::OuterLowerBracket | R::OuterUpperBracket => 5,
            R::SharedRootUniqueness => 6,
            R::CoordinateUniqueness => 7,
        };
        if self.witness.is_some_and(|previous| match previous {
            ValidatedPlaneCoverageWitness::Junction(previous) => {
                progress(previous.rejection) >= progress(rejection)
            }
            ValidatedPlaneCoverageWitness::Candidate(_) => false,
            ValidatedPlaneCoverageWitness::Endpoint(_) => true,
        }) {
            return;
        }
        *self.witness = Some(ValidatedPlaneCoverageWitness::Junction(
            ValidatedPlaneJunctionWitness {
                candidate_indices: self.candidate_indices,
                ends: self.ends,
                rejection,
                coordinate,
                bounds,
                condition,
            },
        ));
    }
}

fn junction_condition(error: ValidatedGeometryError) -> Option<ValidatedPlaneJunctionCondition> {
    use ValidatedGeometryError as E;
    use ValidatedPlaneJunctionCondition as C;
    Some(match error {
        E::Unresolved => C::Unresolved,
        E::ConditioningFailure => C::ConditioningFailure,
        E::NonTransverse => C::NonTransverse,
        E::Singular => C::Singular,
        E::DenominatorContainsZero => C::DenominatorContainsZero,
        E::ParameterOutOfRange => C::ParameterOutOfRange,
        E::NonSmooth => C::NonSmooth,
        _ => return None,
    })
}

fn candidate_covers(
    candidate: &PreparedValidatedCurve<'_>,
    source_index: usize,
    cell: [[f64; 2]; 2],
    position: [ValidatedInterval; 3],
    plane: &Plane,
    execution: (&mut ValidatedGeometryBudget, &AtomicBool),
    mut diagnostic: Option<CandidateDiagnostics<'_>>,
) -> Result<bool, ValidatedGeometryError> {
    let (budget, cancel) = execution;
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
                    if let Some(diagnostic) = diagnostic.as_mut()
                        && let Some(axis) = (0..4)
                            .find(|&i| box4[i].lower > q[i].lower || q[i].upper > box4[i].upper)
                    {
                        let (rejection, bounds) = if box4[axis].lower > q[axis].lower {
                            (
                                ValidatedPlaneCoverageRejection::TubeLower,
                                [q[axis].lower, box4[axis].lower],
                            )
                        } else {
                            (
                                ValidatedPlaneCoverageRejection::TubeUpper,
                                [q[axis].upper, box4[axis].upper],
                            )
                        };
                        diagnostic.record((start, count), rejection, axis as u8, bounds);
                    }
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
                    if let Some(diagnostic) = diagnostic.as_mut() {
                        diagnostic.record(
                            (start, count),
                            ValidatedPlaneCoverageRejection::CoordinateMonotonicity,
                            4,
                            [derivative.lower, derivative.upper],
                        );
                    }
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
            let low = projection_residual(
                first.projection,
                ValidatedInterval::point(first.parameter[0]),
                low,
                position,
            );
            let high = projection_residual(
                last.projection,
                ValidatedInterval::point(last.parameter[1]),
                high,
                position,
            );
            let bracket = if monotonicity == Some(true) {
                low.upper <= 0.0 && high.lower >= 0.0
            } else {
                low.lower >= 0.0 && high.upper <= 0.0
            };
            if bracket {
                return Ok(true);
            }
            if let Some(diagnostic) = diagnostic.as_mut() {
                use ValidatedPlaneCoverageRejection as R;
                let (rejection, bounds) = match monotonicity {
                    Some(true) if low.upper > 0.0 => (R::IncreasingLowBracket, low),
                    Some(true) => (R::IncreasingHighBracket, high),
                    _ if low.lower < 0.0 => (R::DecreasingLowBracket, low),
                    _ => (R::DecreasingHighBracket, high),
                };
                diagnostic.record((start, count), rejection, 4, [bounds.lower, bounds.upper]);
            }
        }
    }
    Ok(false)
}
/// An exact shared coordinate proposes a junction, never a proximity weld.
/// First prove both retained ends are the same original root. Then prove a
/// common monotone coordinate on their sealed paths covers the entire cell.
fn junction_covers(
    matches: &[(usize, &PreparedValidatedCurve<'_>, usize)],
    cell: [[f64; 2]; 2],
    position: [ValidatedInterval; 3],
    plane: &Plane,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
    mut diagnostic: Option<&mut Option<ValidatedPlaneCoverageWitness>>,
) -> Result<bool, ValidatedGeometryError> {
    let plane_uv = plane_parameters(plane, position)?;
    let source_uv = [interval_range(cell[0])?, interval_range(cell[1])?];
    for (first_index, &(first_candidate, first, first_source)) in matches.iter().enumerate() {
        for (second_offset, &(second_candidate, second, second_source)) in
            matches[first_index..].iter().enumerate()
        {
            let mut pair_diagnostic = diagnostic.as_deref_mut().and_then(|witness| {
                Some(JunctionDiagnostics {
                    candidate_indices: [
                        u32::try_from(first_candidate).ok()?,
                        u32::try_from(second_candidate).ok()?,
                    ],
                    ends: [None; 2],
                    uv_separation: None,
                    witness,
                })
            });
            for first_back in [false, true] {
                if second_offset == 0 && first_back {
                    continue;
                }
                if let Some(diagnostic) = pair_diagnostic.as_mut() {
                    diagnostic.ends = [Some(first_back), None];
                    diagnostic.uv_separation = None;
                }
                let Some((first_piece, coordinate)) = endpoint_coordinate(first, first_back) else {
                    if let Some(diagnostic) = pair_diagnostic.as_mut() {
                        diagnostic.record(
                            ValidatedPlaneJunctionRejection::MissingFirstCoordinate,
                            None,
                            None,
                            None,
                        );
                    }
                    continue;
                };
                for second_back in [false, true] {
                    // A closed candidate has one actual front/back junction,
                    // not a junction between an end and itself. The identical
                    // fresh original-root proof remains mandatory.
                    if second_offset == 0 && (first_back || !second_back) {
                        continue;
                    }
                    budget.charge(1, cancel)?;
                    if let Some(diagnostic) = pair_diagnostic.as_mut() {
                        diagnostic.ends[1] = Some(second_back);
                    }
                    let Some((second_piece, other)) = endpoint_coordinate(second, second_back)
                    else {
                        if let Some(diagnostic) = pair_diagnostic.as_mut() {
                            diagnostic.record(
                                ValidatedPlaneJunctionRejection::MissingSecondCoordinate,
                                None,
                                None,
                                None,
                            );
                        }
                        continue;
                    };
                    if !coordinate.same_plane(other) {
                        if let Some(diagnostic) = pair_diagnostic.as_mut() {
                            diagnostic.record(
                                ValidatedPlaneJunctionRejection::DifferentCoordinates,
                                None,
                                Some([coordinate.offset, other.offset]),
                                None,
                            );
                        }
                        continue;
                    }
                    let first_box = flatten(
                        first_piece
                            .source_parameters
                            .ok_or(ValidatedGeometryError::InvalidInput)?,
                    );
                    let mut second_box = flatten(
                        second_piece
                            .source_parameters
                            .ok_or(ValidatedGeometryError::InvalidInput)?,
                    );
                    if first_source != second_source {
                        second_box = [second_box[2], second_box[3], second_box[0], second_box[1]];
                    }
                    let domain = std::array::from_fn(|i| first_box[i].hull(second_box[i]));
                    let q = if first_source == 0 {
                        [source_uv[0], source_uv[1], plane_uv[0], plane_uv[1]]
                    } else {
                        [plane_uv[0], plane_uv[1], source_uv[0], source_uv[1]]
                    };
                    if !(0..4)
                        .all(|i| domain[i].lower <= q[i].lower && q[i].upper <= domain[i].upper)
                    {
                        if let Some(diagnostic) = pair_diagnostic.as_mut()
                            && let Some(axis) = (0..4).find(|&i| {
                                domain[i].lower > q[i].lower || q[i].upper > domain[i].upper
                            })
                        {
                            let (rejection, bounds) = if domain[axis].lower > q[axis].lower {
                                (
                                    ValidatedPlaneJunctionRejection::TubeLower,
                                    [q[axis].lower, domain[axis].lower],
                                )
                            } else {
                                (
                                    ValidatedPlaneJunctionRejection::TubeUpper,
                                    [q[axis].upper, domain[axis].upper],
                                )
                            };
                            diagnostic.record(rejection, Some(axis as u8), Some(bounds), None);
                        }
                        continue;
                    }
                    if let Some(diagnostic) = pair_diagnostic.as_mut() {
                        let source_start = if first_source == 0 { 0 } else { 2 };
                        for axis in 0..2 {
                            let first = first_box[source_start + axis];
                            let second = second_box[source_start + axis];
                            let bounds = if first.upper < second.lower {
                                Some([first.upper, second.lower])
                            } else if second.upper < first.lower {
                                Some([second.upper, first.lower])
                            } else {
                                None
                            };
                            if let Some(bounds) = bounds {
                                diagnostic.uv_separation = Some((axis as u8, bounds));
                                break;
                            }
                        }
                    }
                    if joined_neighborhoods_cover(
                        (first, first_piece, first_back),
                        (second, second_piece, second_back),
                        coordinate,
                        &JunctionCell {
                            query: &q,
                            position: &position,
                            domain: &domain,
                            swap_second: first_source != second_source,
                        },
                        (&mut *budget, cancel),
                        pair_diagnostic.as_mut(),
                    )? {
                        return Ok(true);
                    }
                }
            }
        }
    }
    Ok(false)
}

fn endpoint_coordinate<'a>(
    candidate: &'a PreparedValidatedCurve<'_>,
    back: bool,
) -> Option<(&'a ValidatedCurvePiece, CoordinatePlane)> {
    let piece = if back {
        candidate.pieces.last()?
    } else {
        candidate.pieces.first()?
    };
    let Projection::BetweenPlanes { parameter, planes } = piece.projection else {
        return None;
    };
    let end = usize::from(back);
    (piece.parameter[end] == parameter[end]).then_some((piece, planes[end]))
}

struct JunctionCell<'a> {
    query: &'a [ValidatedInterval; 4],
    position: &'a [ValidatedInterval; 3],
    domain: &'a [ValidatedInterval; 4],
    swap_second: bool,
}

fn joined_neighborhoods_cover(
    first: (&PreparedValidatedCurve<'_>, &ValidatedCurvePiece, bool),
    second: (&PreparedValidatedCurve<'_>, &ValidatedCurvePiece, bool),
    coordinate: CoordinatePlane,
    cell: &JunctionCell<'_>,
    execution: (&mut ValidatedGeometryBudget, &AtomicBool),
    mut diagnostic: Option<&mut JunctionDiagnostics<'_>>,
) -> Result<bool, ValidatedGeometryError> {
    let (budget, cancel) = execution;
    if matches!(
        joined_paths_cover(
            first,
            second,
            coordinate,
            *cell.domain,
            *cell.position,
            (&mut *budget, cancel),
            diagnostic.as_deref_mut(),
        )?,
        JoinedPaths::Covered
    ) {
        return Ok(true);
    }
    let (first_curve, first_original, first_back) = first;
    let (second_curve, second_original, second_back) = second;
    let mut first_domain = flatten(
        first_original
            .source_parameters
            .ok_or(ValidatedGeometryError::InvalidInput)?,
    );
    let mut second_domain = flatten(
        second_original
            .source_parameters
            .ok_or(ValidatedGeometryError::InvalidInput)?,
    );
    let mut first_range = first_original.parameter;
    let mut second_range = second_original.parameter;
    let depth = first_curve
        .options
        .max_depth
        .min(second_curve.options.max_depth);
    let iterations = first_curve
        .options
        .max_iterations
        .min(second_curve.options.max_iterations);
    for _ in 0..depth {
        budget.charge(2, cancel)?;
        let first_middle = first_range[0] * 0.5 + first_range[1] * 0.5;
        let second_middle = second_range[0] * 0.5 + second_range[1] * 0.5;
        if first_middle <= first_range[0]
            || first_middle >= first_range[1]
            || second_middle <= second_range[0]
            || second_middle >= second_range[1]
        {
            break;
        }
        first_range[usize::from(!first_back)] = first_middle;
        second_range[usize::from(!second_back)] = second_middle;
        for _ in 0..iterations {
            budget.charge(1, cancel)?;
            // Every later domain is nested inside these known root enclosures.
            // If their hull cannot contain this query, no shorter endpoint
            // subpath can repair it: let the original cell subdivide instead.
            let Some(domain) = junction_domain(
                &first_domain,
                &second_domain,
                cell,
                diagnostic.as_deref_mut(),
            ) else {
                return Ok(false);
            };
            let Some(first) = certified_endpoint_subpath(
                first_curve,
                first_original,
                first_range,
                first_domain,
                budget,
                cancel,
            )?
            else {
                break;
            };
            let Some(second) = certified_endpoint_subpath(
                second_curve,
                second_original,
                second_range,
                second_domain,
                budget,
                cancel,
            )?
            else {
                break;
            };
            let stationary =
                first.next_domain == first_domain && second.next_domain == second_domain;
            // These padded images already enclose the SAME selected roots over
            // the full current spans. Carry them forward, never reset to the
            // original coarse tubes, even when the next operation halves a span.
            first_domain = first.next_domain;
            second_domain = second.next_domain;
            match joined_paths_cover(
                (first_curve, &first.piece, first_back),
                (second_curve, &second.piece, second_back),
                coordinate,
                domain,
                *cell.position,
                (&mut *budget, cancel),
                diagnostic.as_deref_mut(),
            )? {
                JoinedPaths::Covered => return Ok(true),
                // Once a usable strict full-span axis exists, further
                // derivative contraction is unnecessary. The remaining
                // original-root proof needs a smaller path domain instead.
                JoinedPaths::ShorterPaths => break,
                JoinedPaths::SharperJets if stationary => break,
                JoinedPaths::SharperJets => {}
            }
        }
    }
    Ok(false)
}

fn junction_domain(
    first: &[ValidatedInterval; 4],
    second: &[ValidatedInterval; 4],
    cell: &JunctionCell<'_>,
    mut diagnostic: Option<&mut JunctionDiagnostics<'_>>,
) -> Option<[ValidatedInterval; 4]> {
    let second_axis = |axis| {
        if cell.swap_second {
            (axis + 2) % 4
        } else {
            axis
        }
    };
    let domain = std::array::from_fn(|axis| first[axis].hull(second[second_axis(axis)]));
    for axis in 0..4 {
        let second = second[second_axis(axis)];
        let separation = if first[axis].upper < second.lower {
            Some([first[axis].upper, second.lower])
        } else if second.upper < first[axis].lower {
            Some([second.upper, first[axis].lower])
        } else {
            None
        };
        if let Some(bounds) = separation {
            // Both actual endpoint roots satisfy the shared coordinate.
            // Strictly separated retained root enclosures cannot select one
            // common original root, independently of further subdivision.
            if let Some(diagnostic) = diagnostic.as_deref_mut() {
                diagnostic.record(
                    ValidatedPlaneJunctionRejection::SharedRootUniqueness,
                    Some(axis as u8),
                    Some(bounds),
                    None,
                );
            }
            return None;
        }
        let failure = if cell.query[axis].lower < domain[axis].lower {
            Some((
                ValidatedPlaneJunctionRejection::TubeLower,
                [cell.query[axis].lower, domain[axis].lower],
            ))
        } else if domain[axis].upper < cell.query[axis].upper {
            Some((
                ValidatedPlaneJunctionRejection::TubeUpper,
                [domain[axis].upper, cell.query[axis].upper],
            ))
        } else {
            None
        };
        if let Some((rejection, bounds)) = failure {
            if let Some(diagnostic) = diagnostic.as_deref_mut() {
                diagnostic.record(rejection, Some(axis as u8), Some(bounds), None);
            }
            return None;
        }
    }
    Some(domain)
}

struct EndpointNeighborhood {
    piece: ValidatedCurvePiece,
    next_domain: [ValidatedInterval; 4],
}

fn certified_endpoint_subpath(
    candidate: &PreparedValidatedCurve<'_>,
    original: &ValidatedCurvePiece,
    range: [f64; 2],
    domain: [ValidatedInterval; 4],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Option<EndpointNeighborhood>, ValidatedGeometryError> {
    let parameter = interval_range(range)?;
    // The initial domain is the old uniqueness tube. Later domains contain
    // its proved root images and remain nested inside it. Restrict only the
    // native parameter span and retain the original projection and branch.
    let tube = match certify(
        candidate.source.curve()?,
        original.projection,
        parameter,
        domain,
        budget,
        cancel,
    ) {
        Ok(tube) => tube,
        Err(error) if splittable(error) => return Ok(None),
        Err(error) => return Err(error),
    };
    let jet = match implicit_jet(&tube) {
        Ok(jet) => jet,
        Err(error) if splittable(error) => return Ok(None),
        Err(error) => return Err(error),
    };
    let next_domain = std::array::from_fn(|i| {
        let padding = tube.domain[i].width() * 0.0625;
        ValidatedInterval {
            lower: (tube.rooted_domain[i].lower - padding)
                .next_down()
                .max(tube.domain[i].lower),
            upper: (tube.rooted_domain[i].upper + padding)
                .next_up()
                .min(tube.domain[i].upper),
        }
    });
    Ok(Some(EndpointNeighborhood {
        piece: ValidatedCurvePiece {
            parameter: range,
            jet,
            source_parameters: Some(unflatten(tube.domain)),
            contraction_bound: tube.rho,
            surface_residual: subtract3(tube.surfaces[0].position, tube.surfaces[1].position),
            projection: original.projection,
        },
        next_domain,
    }))
}

#[derive(Debug, PartialEq, Eq)]
enum JoinedPaths {
    Covered,
    SharperJets,
    ShorterPaths,
}

fn joined_paths_cover(
    first: (&PreparedValidatedCurve<'_>, &ValidatedCurvePiece, bool),
    second: (&PreparedValidatedCurve<'_>, &ValidatedCurvePiece, bool),
    coordinate: CoordinatePlane,
    domain: [ValidatedInterval; 4],
    position: [ValidatedInterval; 3],
    execution: (&mut ValidatedGeometryBudget, &AtomicBool),
    mut diagnostic: Option<&mut JunctionDiagnostics<'_>>,
) -> Result<JoinedPaths, ValidatedGeometryError> {
    let (budget, cancel) = execution;
    let (first_curve, first_piece, first_back) = first;
    let (second_curve, second_piece, second_back) = second;
    let mut monotone = false;
    for axis in 0..3 {
        budget.charge(1, cancel)?;
        let away = |piece: &ValidatedCurvePiece, back: bool| {
            let derivative = piece.jet.first[axis];
            if derivative.lower > 0.0 {
                Some(!back)
            } else if derivative.upper < 0.0 {
                Some(back)
            } else {
                None
            }
        };
        let first_away = away(first_piece, first_back);
        let second_away = away(second_piece, second_back);
        let (Some(first_positive), Some(second_positive)) = (first_away, second_away) else {
            if let Some(diagnostic) = diagnostic.as_deref_mut() {
                let (rejection, derivative) = if first_away.is_none() {
                    (
                        ValidatedPlaneJunctionRejection::FirstMonotonicity,
                        first_piece.jet.first[axis],
                    )
                } else {
                    (
                        ValidatedPlaneJunctionRejection::SecondMonotonicity,
                        second_piece.jet.first[axis],
                    )
                };
                diagnostic.record(
                    rejection,
                    Some(axis as u8),
                    Some([derivative.lower, derivative.upper]),
                    None,
                );
            }
            continue;
        };
        if first_positive == second_positive {
            if let Some(diagnostic) = diagnostic.as_deref_mut() {
                diagnostic.record(
                    ValidatedPlaneJunctionRejection::SameAwayDirection,
                    Some(axis as u8),
                    None,
                    None,
                );
            }
            continue;
        }
        if position[axis].lower >= position[axis].upper {
            if let Some(diagnostic) = diagnostic.as_deref_mut() {
                diagnostic.record(
                    ValidatedPlaneJunctionRejection::DegenerateCoordinate,
                    Some(axis as u8),
                    Some([position[axis].lower, position[axis].upper]),
                    None,
                );
            }
            continue;
        }
        monotone = true;
        let first_outer = first_curve
            .evaluate(
                first_piece.parameter[usize::from(!first_back)],
                budget,
                cancel,
            )?
            .error[axis];
        let second_outer = second_curve
            .evaluate(
                second_piece.parameter[usize::from(!second_back)],
                budget,
                cancel,
            )?
            .error[axis];
        let (lower, upper) = if first_positive {
            (second_outer, first_outer)
        } else {
            (first_outer, second_outer)
        };
        if lower.upper > position[axis].lower {
            if let Some(diagnostic) = diagnostic.as_deref_mut() {
                diagnostic.record(
                    ValidatedPlaneJunctionRejection::OuterLowerBracket,
                    Some(axis as u8),
                    Some([lower.upper, position[axis].lower]),
                    None,
                );
            }
            continue;
        }
        if position[axis].upper > upper.lower {
            if let Some(diagnostic) = diagnostic.as_deref_mut() {
                diagnostic.record(
                    ValidatedPlaneJunctionRejection::OuterUpperBracket,
                    Some(axis as u8),
                    Some([position[axis].upper, upper.lower]),
                    None,
                );
            }
            continue;
        }
        // Both old endpoint roots lie in this common domain and satisfy these
        // exact shared coefficients. Fresh uniqueness, not box intersection,
        // proves equality even when their numerical enclosures have width.
        let shared = Projection::BetweenPlanes {
            parameter: [0.0, 1.0],
            planes: [coordinate; 2],
        };
        match certify(
            first_curve.source.curve()?,
            shared,
            super::ZERO,
            domain,
            budget,
            cancel,
        ) {
            Ok(_) => {}
            Err(error) if splittable(error) => {
                if let Some(diagnostic) = diagnostic.as_deref_mut() {
                    let (coordinate, bounds) = diagnostic
                        .uv_separation
                        .map_or((None, None), |(axis, bounds)| (Some(axis), Some(bounds)));
                    diagnostic.record(
                        ValidatedPlaneJunctionRejection::SharedRootUniqueness,
                        coordinate,
                        bounds,
                        junction_condition(error),
                    );
                }
                continue;
            }
            Err(error) => return Err(error),
        }
        let mut normal = [0.0; 3];
        normal[axis] = 1.0;
        let range = [position[axis].lower, position[axis].upper];
        let projection = Projection::BetweenPlanes {
            parameter: range,
            planes: range.map(|offset| CoordinatePlane { normal, offset }),
        };
        // The two now-joined sealed paths stay in the common domain. Their
        // strict opposite away-directions and outer bracket represent every
        // coordinate in the cell. Uniqueness in this new coordinate identifies
        // every original cell root with that represented path.
        match certify(
            first_curve.source.curve()?,
            projection,
            position[axis],
            domain,
            budget,
            cancel,
        ) {
            Ok(_) => return Ok(JoinedPaths::Covered),
            Err(error) if splittable(error) => {
                if let Some(diagnostic) = diagnostic.as_deref_mut() {
                    diagnostic.record(
                        ValidatedPlaneJunctionRejection::CoordinateUniqueness,
                        Some(axis as u8),
                        Some([position[axis].lower, position[axis].upper]),
                        junction_condition(error),
                    );
                }
            }
            Err(error) => return Err(error),
        }
    }
    Ok(if monotone {
        JoinedPaths::ShorterPaths
    } else {
        JoinedPaths::SharperJets
    })
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

#[cfg(test)]
mod plane_zero_cell_regressions {
    use super::*;
    use crate::{
        BsplineCurve, BsplineSurface, Edge, Face, Invertible, KnotVector, Line, Point2, Point3,
        Processor, RevolutionSurface, TrimDomainLimits, TrimmedCurve, ValidatedCurveOptions,
        ValidatedFacePlaneSource, ValidatedFacePlaneSourceAnchor, Vector3, Wire, builder,
        prepare_face_parameter_domain_with_uncertainty, prepared_face_plane_curve,
    };
    use monstertruck_core::EuclideanSpace;
    use monstertruck_topology::trimmed::TrimmedFace;
    use std::f64::consts::TAU;

    fn tapered_surface() -> Surface {
        Surface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
            Curve::Line(Line(
                Point3::new(6.0, 0.0, 0.0),
                Point3::new(4.0, 0.0, 12.0),
            )),
            Point3::origin(),
            Vector3::unit_z(),
        )))
    }

    fn axis(index: usize) -> [ValidatedInterval; 3] {
        std::array::from_fn(|i| ValidatedInterval::point(if i == index { 1.0 } else { 0.0 }))
    }

    #[test]
    fn changed_coordinate_normal_retains_its_shared_parameter_coefficient() {
        let guide = Curve::Line(Line(Point3::origin(), Point3::new(1.0, 1.0, 0.0)));
        let projection = Projection::BetweenPlanes {
            parameter: [0.0, 1.0],
            planes: [
                CoordinatePlane {
                    normal: [1.0, 0.0, 0.0],
                    offset: 1.0e6,
                },
                CoordinatePlane {
                    normal: [0.0, 1.0, 0.0],
                    offset: -1.0e6,
                },
            ],
        };
        // Every coordinate plane passes through this exact point. Independent
        // intervalizations of normal(t) and offset(t) lose that identity.
        let point = [1.0e6, -1.0e6, 0.0].map(ValidatedInterval::point);
        let parameter = interval_range([0.0, 1.0]).unwrap();
        let jet = projection_jet(
            &guide,
            projection,
            parameter,
            &mut ValidatedGeometryBudget::new(100),
            &AtomicBool::new(false),
        )
        .unwrap();
        let independent = dot3(jet.normal, point) - jet.offset;
        let correlated = projection_residual(projection, parameter, jet, point);
        assert!(independent.width() > 1.0e6);
        assert!(correlated.contains_zero());
        assert!(correlated.width() < 1.0e-7);
    }

    fn closed_original_candidate(
        back_offset: f64,
    ) -> (
        Surface,
        Plane,
        PreparedValidatedCurve<'static>,
        [[f64; 2]; 2],
    ) {
        // A closed original intersection in a nonperiodic chart. Inversion
        // swaps the independent UV gauges; no angular quotient is admitted.
        let coefficients = [(-1.0, 1.0), (0.0, -1.0), (1.0, 1.0)];
        let surface = Surface::BsplineSurface(BsplineSurface::new(
            (KnotVector::bezier_knot(2), KnotVector::bezier_knot(2)),
            coefficients
                .into_iter()
                .map(|(x, zx)| {
                    coefficients
                        .into_iter()
                        .map(|(y, zy)| Point3::new(x, y, zx + zy))
                        .collect()
                })
                .collect(),
        ))
        .inverse();
        let plane = Plane::new(
            Point3::new(0.0, 0.0, 0.25),
            Point3::new(1.0, 0.0, 0.25),
            Point3::new(0.0, 1.0, 0.25),
        );
        let angle: f64 = 0.35;
        let point = |angle: f64| Point3::new(0.5 * angle.cos(), 0.5 * angle.sin(), 0.25);
        let first = point(angle);
        let guide = Curve::BsplineCurve(BsplineCurve::new(
            KnotVector::from(
                (0_usize..=34)
                    .map(|index| index.saturating_sub(1).min(32) as f64)
                    .collect::<Vec<_>>(),
            ),
            (0..=32)
                .map(|index| {
                    if index == 32 {
                        first
                    } else {
                        point(angle + TAU * index as f64 / 32.0)
                    }
                })
                .collect(),
        ));
        let coordinate = CoordinatePlane {
            normal: [0.0, 1.0, 0.0],
            offset: first.y,
        };
        let chart = [(first.y + 1.0) * 0.5, (first.x + 1.0) * 0.5];
        let curve = prepared_face_plane_curve(
            &guide,
            ValidatedFacePlaneSource {
                surface: &surface,
                anchor: Some(ValidatedFacePlaneSourceAnchor::Front(chart)),
                endpoint_coordinate_planes: [
                    Some(coordinate),
                    Some(CoordinatePlane {
                        offset: coordinate.offset + back_offset,
                        ..coordinate
                    }),
                ],
            },
            &plane,
            [0.0, 32.0],
            &ValidatedCurveOptions::default(),
            &mut ValidatedGeometryBudget::new(2_000_000),
            &AtomicBool::new(false),
        )
        .expect("whole closed original branch on the same inverted source chart");
        let cell = chart.map(|value| [value - 1.0e-12, value + 1.0e-12]);
        (surface, plane, curve, cell)
    }

    #[test]
    fn closed_original_candidate_joins_its_own_front_and_back() {
        let (surface, plane, curve, cell) = closed_original_candidate(0.0);
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(2_000_000);
        let (cell, position) = plane_zero_cell(
            &surface,
            cell,
            axis(2),
            axis(2).map(|value| value * ValidatedInterval::point(0.25)),
            &mut budget,
            &cancel,
        )
        .unwrap()
        .unwrap();
        assert!(
            !candidate_covers(
                &curve,
                0,
                cell,
                position,
                &plane,
                (&mut budget, &cancel),
                None,
            )
            .unwrap(),
            "the cell straddles the finite prepared range's two ends"
        );
        assert!(
            junction_covers(
                &[(0, &curve, 0)],
                cell,
                position,
                &plane,
                &mut budget,
                &cancel,
                None,
            )
            .unwrap(),
            "the two ends require an actual shared original-root proof"
        );
    }

    #[test]
    fn closed_guide_cannot_authorize_a_gap_between_original_candidate_ends() {
        let (surface, plane, curve, cell) = closed_original_candidate(-1.0e-6);
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(2_000_000);
        let front = curve.evaluate(0.0, &mut budget, &cancel).unwrap().error[1];
        let back = curve.evaluate(32.0, &mut budget, &cancel).unwrap().error[1];
        assert!(
            back.upper < front.lower,
            "the closed guide retained a genuine original-root gap"
        );
        let (cell, position) = plane_zero_cell(
            &surface,
            cell,
            axis(2),
            axis(2).map(|value| value * ValidatedInterval::point(0.25)),
            &mut budget,
            &cancel,
        )
        .unwrap()
        .unwrap();
        assert!(
            !junction_covers(
                &[(0, &curve, 0)],
                cell,
                position,
                &plane,
                &mut budget,
                &cancel,
                None,
            )
            .unwrap(),
            "closed guide endpoints are not original continuation authority"
        );
    }

    fn negative_taper_junction(
        second_offset: f64,
    ) -> (Surface, Plane, [PreparedValidatedCurve<'static>; 2]) {
        // Retain an independent raw profile gauge, negative covering-chart
        // angle, and two oppositely parameterized generated incident guides.
        let profile = Curve::TrimmedCurve(Processor::new(TrimmedCurve::new(
            Box::new(Curve::Line(Line(
                Point3::new(6.0, 0.0, 0.0),
                Point3::new(6.0 + 1.0 / 6.0, 0.0, -1.0),
            ))),
            (-12.0, 0.0),
        )));
        let surface = Surface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
            profile,
            Point3::origin(),
            Vector3::unit_z(),
        )))
        .inverse();
        let plane = Plane::new(
            Point3::new(0.0, 0.0, 6.0),
            Point3::new(1.0, 0.0, 6.0),
            Point3::new(0.0, 1.0, 6.0),
        );
        let angle: f64 = -2.77439351226111;
        let point = |angle: f64| Point3::new(5.0 * angle.cos(), 5.0 * angle.sin(), 6.0);
        let cancel = AtomicBool::new(false);
        let curves = [-0.04, 0.04].map(|delta| {
            let guide = Curve::Line(Line(point(angle + delta), point(angle)));
            prepared_face_plane_curve(
                &guide,
                ValidatedFacePlaneSource {
                    surface: &surface,
                    anchor: Some(ValidatedFacePlaneSourceAnchor::Front([angle + delta, -6.0])),
                    endpoint_coordinate_planes: [
                        None,
                        Some(CoordinatePlane {
                            normal: [0.0, 1.0, 0.0],
                            offset: point(angle).y + if delta > 0.0 { second_offset } else { 0.0 },
                        }),
                    ],
                },
                &plane,
                [0.0, 1.0],
                &ValidatedCurveOptions::default(),
                &mut ValidatedGeometryBudget::new(2_000_000),
                &cancel,
            )
            .expect("the original tapered carrier certifies each generated incident path")
        });
        (surface, plane, curves)
    }

    #[test]
    fn negative_taper_cross_candidate_junction_has_constructive_original_coverage() {
        let (surface, plane, curves) = negative_taper_junction(0.0);
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(2_000_000);
        let cell = [[-2.77439351226211, -2.77439351226011], [-6.0; 2]];
        let (_, position) = plane_zero_cell(
            &surface,
            cell,
            axis(2),
            axis(2).map(|x| x * ValidatedInterval::point(6.0)),
            &mut budget,
            &cancel,
        )
        .unwrap()
        .unwrap();
        for curve in &curves {
            assert!(
                !candidate_covers(
                    curve,
                    0,
                    cell,
                    position,
                    &plane,
                    (&mut budget, &cancel),
                    None
                )
                .unwrap(),
                "neither incident finite range represents both sides of the junction"
            );
        }
        let (_, first_coordinate) = endpoint_coordinate(&curves[0], true).unwrap();
        let (_, second_coordinate) = endpoint_coordinate(&curves[1], true).unwrap();
        assert_ne!(first_coordinate, second_coordinate);
        assert!(first_coordinate.same_plane(second_coordinate));
        let matches = [(0, &curves[0], 0), (1, &curves[1], 0)];
        assert!(
            junction_covers(&matches, cell, position, &plane, &mut budget, &cancel, None).unwrap()
        );
        for (allowance, cancelled, expected) in [
            (0, false, ValidatedGeometryError::ResourceLimit),
            (2_000_000, true, ValidatedGeometryError::Cancelled),
        ] {
            assert_eq!(
                junction_covers(
                    &matches,
                    cell,
                    position,
                    &plane,
                    &mut ValidatedGeometryBudget::new(allowance),
                    &AtomicBool::new(cancelled),
                    None,
                )
                .unwrap_err(),
                expected
            );
        }
    }

    #[test]
    fn coarse_curved_endpoint_jets_are_reproved_on_same_original_subpaths() {
        let (surface, plane, mut curves) = negative_taper_junction(0.0);
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(2_000_000);
        let cell = [[-2.77439351226211, -2.77439351226011], [-6.0; 2]];
        let (_, position) = plane_zero_cell(
            &surface,
            cell,
            axis(2),
            axis(2).map(|x| x * ValidatedInterval::point(6.0)),
            &mut budget,
            &cancel,
        )
        .unwrap()
        .unwrap();
        for curve in &mut curves {
            let piece = curve.pieces.last_mut().unwrap();
            // Outward coarsening remains a valid retained derivative enclosure.
            // No source, projection, root tube or parameter range is replaced.
            // Recovery must independently prove the original curved subpaths.
            piece.jet.first = piece.jet.first.map(|bound| bound.hull(super::super::ZERO));
        }
        let (first_piece, coordinate) = endpoint_coordinate(&curves[0], true).unwrap();
        let (second_piece, _) = endpoint_coordinate(&curves[1], true).unwrap();
        let first_box = flatten(first_piece.source_parameters.unwrap());
        let second_box = flatten(second_piece.source_parameters.unwrap());
        let domain = std::array::from_fn(|i| first_box[i].hull(second_box[i]));
        assert_eq!(
            joined_paths_cover(
                (&curves[0], first_piece, true),
                (&curves[1], second_piece, true),
                coordinate,
                domain,
                position,
                (&mut budget, &cancel),
                None,
            )
            .unwrap(),
            JoinedPaths::SharperJets,
            "whole-piece cached slopes cannot establish strict monotonicity"
        );
        let matches = [(0, &curves[0], 0), (1, &curves[1], 0)];
        assert!(
            junction_covers(&matches, cell, position, &plane, &mut budget, &cancel, None).unwrap(),
            "fresh positive-width original subpaths must recover the same junction"
        );
    }

    #[test]
    fn overlapping_original_tubes_do_not_authorize_a_generated_junction_gap() {
        let (surface, plane, curves) = negative_taper_junction(-1.0e-6);
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(2_000_000);
        let cell = [[-2.77439351226211, -2.77439351226011], [-6.0; 2]];
        let (_, position) = plane_zero_cell(
            &surface,
            cell,
            axis(2),
            axis(2).map(|x| x * ValidatedInterval::point(6.0)),
            &mut budget,
            &cancel,
        )
        .unwrap()
        .unwrap();
        let first = flatten(curves[0].pieces.last().unwrap().source_parameters.unwrap());
        let second = flatten(curves[1].pieces.last().unwrap().source_parameters.unwrap());
        assert!((0..4).all(|i| first[i].intersection(second[i]).is_some()));
        assert!(
            !junction_covers(
                &[(0, &curves[0], 0), (1, &curves[1], 0)],
                cell,
                position,
                &plane,
                &mut budget,
                &cancel,
                None,
            )
            .unwrap(),
            "proximity and overlapping uniqueness tubes cannot close an actual gap"
        );
    }

    #[test]
    fn equal_generated_coordinate_does_not_weld_distinct_original_components() {
        let surface = Surface::BsplineSurface(BsplineSurface::new(
            (KnotVector::bezier_knot(2), KnotVector::bezier_knot(1)),
            [(-1.0, 1.0), (0.0, -1.0), (1.0, 1.0)]
                .into_iter()
                .map(|(x, z)| vec![Point3::new(x, -1.0, z), Point3::new(x, 1.0, z)])
                .collect(),
        ));
        let plane = Plane::new(
            Point3::new(0.0, 0.0, 0.25),
            Point3::new(1.0, 0.0, 0.25),
            Point3::new(0.0, 1.0, 0.25),
        );
        let shared = CoordinatePlane {
            normal: [0.0, 1.0, 0.0],
            offset: 0.0,
        };
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(2_000_000);
        let curves = [(0.25, -0.5, true), (0.75, 0.5, false)].map(|(u, x, back)| {
            let guide = Curve::Line(Line(
                Point3::new(x, if back { -0.5 } else { 0.0 }, 0.25),
                Point3::new(x, if back { 0.0 } else { 0.5 }, 0.25),
            ));
            prepared_face_plane_curve(
                &guide,
                ValidatedFacePlaneSource {
                    surface: &surface,
                    anchor: Some(ValidatedFacePlaneSourceAnchor::Front([
                        u,
                        if back { 0.25 } else { 0.5 },
                    ])),
                    endpoint_coordinate_planes: if back {
                        [None, Some(shared)]
                    } else {
                        [Some(shared), None]
                    },
                },
                &plane,
                [0.0, 1.0],
                &ValidatedCurveOptions::default(),
                &mut budget,
                &cancel,
            )
            .unwrap()
        });
        let cell = [[0.25; 2], [0.5 - 1.0e-12, 0.5 + 1.0e-12]];
        let (_, position) = plane_zero_cell(
            &surface,
            cell,
            axis(2),
            axis(2).map(|x| x * ValidatedInterval::point(0.25)),
            &mut budget,
            &cancel,
        )
        .unwrap()
        .unwrap();
        assert!(
            !junction_covers(
                &[(0, &curves[0], 0), (1, &curves[1], 0)],
                cell,
                position,
                &plane,
                &mut budget,
                &cancel,
                None,
            )
            .unwrap(),
            "the common original uniqueness proof must exclude a second component"
        );
    }

    #[test]
    fn original_full_period_taper_contracts_only_the_plane_normal_coordinate() {
        let cancel = AtomicBool::new(false);
        let normal = axis(2);
        let origin = normal.map(|x| x * ValidatedInterval::point(6.0));
        for inverted in [false, true] {
            let surface = if inverted {
                tapered_surface().inverse()
            } else {
                tapered_surface()
            };
            let profile_axis = usize::from(inverted);
            let angle_axis = 1 - profile_axis;
            let mut cell = [[0.0, 1.0]; 2];
            cell[angle_axis] = [0.0, TAU];
            let mut budget = ValidatedGeometryBudget::new(128);
            let (contracted, position) =
                plane_zero_cell(&surface, cell, normal, origin, &mut budget, &cancel)
                    .expect("whole original operation graph fits a fixed small work allowance")
                    .expect("the original full-period section cannot be excluded");
            let profile = interval_range(contracted[profile_axis]).unwrap();
            assert!(profile.lower <= 0.5 && 0.5 <= profile.upper);
            assert!(profile.width() < 1.0e-12);
            assert_eq!(contracted[angle_axis], cell[angle_axis]);
            assert!(position[2].lower <= 6.0 && 6.0 <= position[2].upper);
            assert!(position[2].width() < 1.0e-10);
            assert!(budget.consumed() < 128);
        }
    }

    #[test]
    fn nonmonotone_original_surface_retains_both_unselected_plane_components() {
        let surface = Surface::BsplineSurface(BsplineSurface::new(
            (KnotVector::bezier_knot(2), KnotVector::bezier_knot(1)),
            [(-1.0, 1.0), (0.0, -1.0), (1.0, 1.0)]
                .into_iter()
                .map(|(x, z)| vec![Point3::new(x, -1.0, z), Point3::new(x, 1.0, z)])
                .collect(),
        ));
        let cell = [[0.0, 1.0]; 2];
        let normal = axis(2);
        let origin = normal.map(|x| x * ValidatedInterval::point(0.25));
        let (contracted, _) = plane_zero_cell(
            &surface,
            cell,
            normal,
            origin,
            &mut ValidatedGeometryBudget::new(100_000),
            &AtomicBool::new(false),
        )
        .unwrap()
        .expect("neither distinct original plane-zero component may be excluded");
        for root in [0.25, 0.75] {
            assert!(contracted[0][0] <= root && root <= contracted[0][1]);
        }
        assert_eq!(contracted, cell);
        let plane = Plane::new(
            Point3::new(0.0, 0.0, 0.25),
            Point3::new(1.0, 0.0, 0.25),
            Point3::new(0.0, 1.0, 0.25),
        );
        let guide = Curve::Line(Line(
            Point3::new(0.5, -0.5, 0.25),
            Point3::new(0.5, 0.5, 0.25),
        ));
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(1_000_000);
        let candidate = prepared_face_plane_curve(
            &guide,
            ValidatedFacePlaneSource {
                surface: &surface,
                anchor: Some(ValidatedFacePlaneSourceAnchor::Front([0.75, 0.25])),
                endpoint_coordinate_planes: [None; 2],
            },
            &plane,
            [0.0, 1.0],
            &ValidatedCurveOptions::default(),
            &mut budget,
            &cancel,
        )
        .unwrap();
        for (root, expected) in [(0.75, true), (0.25, false)] {
            let root_cell = [[root; 2], [0.5; 2]];
            let (_, position) =
                plane_zero_cell(&surface, root_cell, normal, origin, &mut budget, &cancel)
                    .unwrap()
                    .unwrap();
            assert_eq!(
                candidate_covers(
                    &candidate,
                    0,
                    root_cell,
                    position,
                    &plane,
                    (&mut budget, &cancel),
                    None,
                )
                .unwrap(),
                expected
            );
        }
    }

    #[test]
    fn empty_original_plane_cell_does_not_spend_work_on_trim_contact() {
        let points = [
            Point3::new(-3.0, -2.0, 0.0),
            Point3::new(3.0, -2.0, 0.0),
            Point3::new(3.0, 2.0, 0.0),
            Point3::new(-3.0, 2.0, 0.0),
        ];
        let vertices = points.map(builder::vertex);
        let wire: Wire = (0..4)
            .map(|i| {
                Edge::new(
                    &vertices[i],
                    &vertices[(i + 1) % 4],
                    Curve::Line(Line(points[i], points[(i + 1) % 4])),
                )
            })
            .collect();
        let trims = (0..4)
            .map(|i| {
                let a = points[i];
                let b = points[(i + 1) % 4];
                Some(Curve2D::Line(Line(
                    Point2::new(a.x, a.y),
                    Point2::new(b.x, b.y),
                )))
            })
            .collect();
        let trimmed = TrimmedFace::new(
            Face::new(
                vec![wire],
                Surface::Plane(Plane::new(
                    Point3::origin(),
                    Point3::new(1.0, 0.0, 0.0),
                    Point3::new(0.0, 1.0, 0.0),
                )),
            ),
            vec![trims],
        );
        let plane = Plane::new(
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
        );
        let cancel = AtomicBool::new(false);
        for uncertainty in [0.0, 1.0e-7] {
            let domain = prepare_face_parameter_domain_with_uncertainty(
                &trimmed,
                [uncertainty; 2],
                TrimDomainLimits {
                    max_coedges: 16,
                    max_subdivisions: 65_536,
                    max_retained_bytes: 32 * 1024 * 1024,
                },
                &mut ValidatedGeometryBudget::new(1_000_000),
                &cancel,
            )
            .unwrap();
            assert_eq!(domain.uncertainty_uv(), [uncertainty; 2]);
            let mut budget = ValidatedGeometryBudget::new(256);
            let coverage = validated_face_plane_coverage(
                &domain,
                &plane,
                &[],
                &[],
                &ValidatedPlaneCoverageOptions::default(),
                &mut budget,
                &cancel,
            )
            .expect("plane exclusion precedes whole-boundary trim winding");
            assert_eq!(coverage.visited_cells, 1);
            assert_eq!(coverage.excluded_cells, 1);
            assert_eq!(coverage.covered_cells, 0);
            assert!(coverage.endpoint_strata.is_empty());
            assert!(budget.consumed() < 256);
        }
    }

    #[test]
    fn root_exclusion_keeps_work_and_cancellation_failures_typed() {
        let surface = tapered_surface();
        let cell = [[0.0, 1.0], [0.0, TAU]];
        let normal = axis(2);
        let origin = normal.map(|x| x * ValidatedInterval::point(13.0));
        assert!(
            plane_zero_cell(
                &surface,
                cell,
                normal,
                origin,
                &mut ValidatedGeometryBudget::new(128),
                &AtomicBool::new(false),
            )
            .unwrap()
            .is_none()
        );
        for (allowance, cancelled, expected) in [
            (0, false, ValidatedGeometryError::ResourceLimit),
            (128, true, ValidatedGeometryError::Cancelled),
        ] {
            assert_eq!(
                plane_zero_cell(
                    &surface,
                    cell,
                    normal,
                    origin,
                    &mut ValidatedGeometryBudget::new(allowance),
                    &AtomicBool::new(cancelled),
                )
                .unwrap_err(),
                expected
            );
        }
    }
}
