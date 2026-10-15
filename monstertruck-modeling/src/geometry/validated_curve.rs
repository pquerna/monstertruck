//! Certified, immutable native-parameter intersection branches.
//!
//! A prepared intersection is a unique regular source-surface root in a
//! validated connected chart tube. C2 leaders retain the native equation
//! `L'(t) dot ((S0 + S1)/2 - L(t)) = 0`. Original nonsmooth leaders instead
//! propose fixed planes at native knots; generated face-plane queries may use
//! conditioned groups of those knot coordinates. Neighboring equations share
//! the identical joint plane and interpolate linearly in native `t`, so
//! continuation proves a common exact root rather than overlapping samples.
//! Guide geometry only proposes coordinates, never intersection containment.
//! This immutable prepared representation never snaps endpoints, switches to
//! nearest-point projection, or returns a leader on Newton failure.
//! Numerical iterations only propose centers. Parametric Krawczyk inclusion,
//! with a strict contraction bound, supplies existence and local uniqueness for
//! every real parameter in each closed piece. Continuous piecewise-smooth
//! sources use the same bound as a Banach contraction through their bounded
//! one-sided Jacobians; this does not assert C1 or validate a Hessian Taylor
//! estimate. Consecutive tubes retain a common certified endpoint root.
//! Other components outside that tube, and trimmed-face membership, are not
//! asserted absent; callers must retain their independent trim certificates.

use super::validated_interval::{
    ONE, Taylor, ValidatedCurveJet, ValidatedGeometryBudget, ValidatedGeometryError,
    ValidatedInterval, ValidatedScalarCurveJet, ValidatedSurfaceJet, ZERO, curve_span_smooth,
    interval_range, processed_parameter, surface_box_smooth, validated_curve_piecewise_jet,
    validated_curve_position_bounds, validated_curve_projected_jet,
    validated_projection_leader_jet, validated_surface_piecewise_jet,
};
use crate::{
    Curve, Curve2D, KnotVector, Matrix4, ParametricCurve, ParametricSurface, Plane, Point3,
    Processor, Surface, SurfaceCurve, TrimmedCurve, curve_owned_bytes, surface_owned_bytes,
};
use std::mem::size_of;
use std::sync::atomic::AtomicBool;

mod coverage;
mod extrusion;
pub use coverage::{
    ValidatedPlaneCoverage, ValidatedPlaneCoverageOptions, ValidatedPlaneTrimEndpoint,
    ValidatedPlaneTrimEndpointBinding, validated_face_plane_coverage,
    validated_plane_trim_root_chart, validated_same_surface_coefficients,
};

/// Bounded continuation and certified endpoint refinement policy.
#[derive(Clone, Copy, Debug)]
pub struct ValidatedCurveOptions {
    /// Maximum binary subdivision depth, at most 48.
    pub max_depth: usize,
    /// Maximum numerical or interval refinement iterations, at most 128.
    pub max_iterations: usize,
    /// Initial chart radius relative to the magnitude of a proposed coordinate.
    pub parameter_radius: f64,
    /// Maximum spatial interval width admitted by point evaluation.
    pub endpoint_tolerance: f64,
    /// Maximum owned source and proof storage, checked before allocations.
    pub max_retained_bytes: usize,
}
impl Default for ValidatedCurveOptions {
    fn default() -> Self {
        Self {
            max_depth: 48,
            max_iterations: 24,
            parameter_radius: 0.125,
            endpoint_tolerance: 1.0e-10,
            max_retained_bytes: 32 * 1024 * 1024,
        }
    }
}

/// A closed parameter piece of a single validated native branch.
#[derive(Clone, Copy, Debug)]
pub struct ValidatedCurvePiece {
    /// Closed native parameter coverage, with no rescaling.
    pub parameter: [f64; 2],
    /// Position and piecewise/one-sided first and second derivative bounds.
    /// Taylor use requires [`PreparedValidatedCurve::is_smooth_on`].
    pub jet: ValidatedCurveJet,
    /// Direct source chart boxes; mapped/explicit views retain no single tube here.
    pub source_parameters: Option<[[ValidatedInterval; 2]; 2]>,
    /// Outward bound on the preconditioned Jacobian defect infinity norm.
    /// Constructive algebraic elimination reports zero, without claiming an
    /// interior four-variable Krawczyk certificate.
    pub contraction_bound: f64,
    /// Source-surface difference enclosure at the enclosed branch roots.
    pub surface_residual: [ValidatedInterval; 3],
    projection: Projection,
}

/// A certified approximate point on an immutable prepared branch.
#[derive(Clone, Copy, Debug)]
pub struct ValidatedCurvePoint {
    /// Representative spatial point, not an assertion of exact arithmetic.
    pub position: Point3,
    /// Absolute spatial enclosure of the exact prepared-branch point.
    pub error: [ValidatedInterval; 3],
}

/// An immutable source binding and a complete certified branch partition.
///
/// Prepare once over the whole requested native range, then reuse this binding
/// for all adaptive children. Re-preparing independently is a new branch
/// selection and is not a substitute for continuation of an existing binding.
///
/// Retained trim wrappers prepare their implicit basis only once, keep each
/// original parameter reflection, and apply the stored homogeneous spatial
/// transform after root certification. They never reproject a transformed
/// guide in the world Euclidean metric. Mapped pieces report world-space jets;
/// direct UV coverage must instead use [`prepared_face_plane_curve`] with the
/// actual face carrier and plane.
#[derive(Debug)]
pub struct PreparedValidatedCurve<'a> {
    source: PreparedSource<'a>,
    range: [f64; 2],
    pieces: Vec<ValidatedCurvePiece>,
    options: ValidatedCurveOptions,
    mapped: Option<Box<PreparedValidatedCurve<'a>>>,
}

#[derive(Debug)]
enum PreparedSource<'a> {
    Borrowed(&'a Curve),
    Owned {
        curve: Box<Curve>,
        bytes: usize,
        authority: OwnedSourceAuthority,
    },
    Mapped {
        transform: Matrix4,
        range: (f64, f64),
        orientation: bool,
    },
}
#[derive(Debug)]
enum OwnedSourceAuthority {
    Original,
    Extrusion(Box<extrusion::Graph>),
}

impl PreparedSource<'_> {
    fn graph(&self) -> Option<&extrusion::Graph> {
        match self {
            Self::Owned {
                authority: OwnedSourceAuthority::Extrusion(graph),
                ..
            } => Some(graph),
            _ => None,
        }
    }
    fn curve(&self) -> Result<&Curve, ValidatedGeometryError> {
        match self {
            Self::Borrowed(curve) => Ok(curve),
            Self::Owned { curve, .. } => Ok(curve),
            Self::Mapped { .. } => Err(ValidatedGeometryError::InvalidInput),
        }
    }
    fn mapping(&self) -> Result<(&Matrix4, (f64, f64), bool), ValidatedGeometryError> {
        match self {
            Self::Mapped {
                transform,
                range,
                orientation,
            } => Ok((transform, *range, *orientation)),
            _ => Err(ValidatedGeometryError::InvalidInput),
        }
    }
    fn owned_bytes(&self) -> usize {
        match self {
            Self::Borrowed(_) | Self::Mapped { .. } => 0,
            Self::Owned { bytes, .. } => *bytes,
        }
    }
}

/// A fresh certificate borrows its existing immutable source authority.
#[derive(Clone, Copy)]
enum Certification<'a, 'source> {
    Original(&'a Curve),
    Prepared(&'a PreparedValidatedCurve<'source>),
}
impl Certification<'_, '_> {
    fn certify(
        self,
        projection: Projection,
        parameter: ValidatedInterval,
        domain: [ValidatedInterval; 4],
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> Result<Tube, ValidatedGeometryError> {
        match (self, projection) {
            (Self::Prepared(prepared), Projection::SourceProfile) => prepared
                .source
                .graph()
                .ok_or(ValidatedGeometryError::InvalidInput)?
                .certify(prepared.source.curve()?, parameter, budget, cancel),
            (Self::Original(_), Projection::SourceProfile) => {
                Err(ValidatedGeometryError::InvalidInput)
            }
            (Self::Original(curve), _) => {
                certify(curve, projection, parameter, domain, budget, cancel)
            }
            (Self::Prepared(prepared), _) => certify(
                prepared.source.curve()?,
                projection,
                parameter,
                domain,
                budget,
                cancel,
            ),
        }
    }
}

/// A stored coordinate proposal for an original carrier intersection.
///
/// Equal coefficients at incident generated ends propose a common junction;
/// only fresh original-carrier uniqueness can certify that junction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoordinatePlane {
    /// Finite nonzero normal. Preparation may negate all coefficients together.
    pub normal: [f64; 3],
    /// Finite plane offset in the same native spatial gauge as the carriers.
    pub offset: f64,
}

impl CoordinatePlane {
    fn aligned_to(self, reference: Self) -> Self {
        if (0..3)
            .map(|i| self.normal[i] * reference.normal[i])
            .sum::<f64>()
            < 0.0
        {
            self.negated()
        } else {
            self
        }
    }

    fn negated(self) -> Self {
        Self {
            normal: self.normal.map(|value| -value),
            offset: -self.offset,
        }
    }

    fn same_plane(self, other: Self) -> bool { self == other || self == other.negated() }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Projection {
    Native,
    SourceProfile,
    BetweenPlanes {
        parameter: [f64; 2],
        planes: [CoordinatePlane; 2],
    },
}

#[derive(Clone, Copy)]
struct PlaneJet {
    normal: [ValidatedInterval; 3],
    first: [ValidatedInterval; 3],
    second: [ValidatedInterval; 3],
    offset: ValidatedInterval,
    offset_first: ValidatedInterval,
    offset_second: ValidatedInterval,
}

impl PreparedValidatedCurve<'_> {
    /// Returns the complete admitted native range.
    pub const fn range(&self) -> [f64; 2] { self.range }
    /// Returns the ordered, gap-free, closed branch partition.
    pub fn pieces(&self) -> &[ValidatedCurvePiece] { &self.pieces }

    /// Borrows this binding's already prepared basis, without root reselection.
    ///
    /// Its range is exactly the retained active interval mapped through the
    /// original trim reflection, not an unused natural-domain extension.
    pub fn mapped_basis(&self) -> Option<&Self> { self.mapped.as_deref() }

    /// Borrows the unchanged native source of an unmapped binding.
    ///
    /// Mapped nodes retain their original operators and prepared basis rather
    /// than a cloned recursive curve. Follow [`Self::mapped_basis`] to inspect
    /// the original guide and carriers at the unmapped leaf.
    pub fn native_curve(&self) -> Option<&Curve> { self.source.curve().ok() }

    /// Returns owned source and proof-capacity bytes, including this binding.
    pub fn retained_bytes(&self) -> usize {
        size_of::<Self>()
            .saturating_add(self.source.owned_bytes())
            .saturating_add(
                self.pieces
                    .capacity()
                    .saturating_mul(size_of::<ValidatedCurvePiece>()),
            )
            .saturating_add(self.mapped.as_deref().map_or(0, Self::retained_bytes))
    }

    /// Proves C1 with bounded piecewise second derivatives on this same binding.
    ///
    /// Positive-width intervals admit one-sided endpoint derivatives. An
    /// unproved C0 crossing, uncertain mapped knot band, or projection-chart
    /// joint returns false; finite unions of Hessians are not proof. Polynomial
    /// and rational curve/profile joins can qualify through exact homogeneous
    /// first-jet coefficient identities, without family recognition.
    /// False never invalidates the binding's zeroth-order point enclosure.
    pub fn is_smooth_on(
        &self,
        range: [f64; 2],
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> Result<bool, ValidatedGeometryError> {
        budget.charge(1, cancel)?;
        let range = interval_range(range)?;
        self.check_range(range)?;
        if let Some(child) = &self.mapped {
            let (_, active, orientation) = self.source.mapping()?;
            let mapped = reflected_trim_range(orientation, active, range)?;
            return child.is_smooth_on([mapped.lower, mapped.upper], budget, cancel);
        }
        if let Some(graph) = self.source.graph() {
            return graph.is_smooth(self.source.curve()?, range, budget, cancel);
        }
        if !matches!(self.source.curve()?, Curve::IntersectionCurve(_)) {
            return curve_span_smooth(self.source.curve()?, range, 1, false, budget, cancel);
        }
        let (first, second, leader) = carriers(self.source.curve()?)?;
        let mut projection = None;
        let mut covered = false;
        for piece in &self.pieces {
            budget.charge(1, cancel)?;
            let Some(part) = range.intersection(ValidatedInterval {
                lower: piece.parameter[0],
                upper: piece.parameter[1],
            }) else {
                continue;
            };
            if part.lower == part.upper && range.lower < range.upper {
                continue;
            }
            if projection.is_some_and(|previous| previous != piece.projection) {
                return Ok(false);
            }
            projection = Some(piece.projection);
            let original = flatten(
                piece
                    .source_parameters
                    .ok_or(ValidatedGeometryError::InvalidInput)?,
            );
            let tube = Certification::Prepared(self).certify(
                piece.projection,
                part,
                original,
                budget,
                cancel,
            )?;
            let charts = unflatten(tube.rooted_domain);
            if !surface_box_smooth(first, charts[0][0], charts[0][1], budget, cancel)?
                || !surface_box_smooth(second, charts[1][0], charts[1][1], budget, cancel)?
            {
                return Ok(false);
            }
            if matches!(piece.projection, Projection::Native)
                && !curve_span_smooth(leader, part, 2, true, budget, cancel)?
            {
                return Ok(false);
            }
            covered = true;
        }
        Ok(covered)
    }

    /// Encloses the whole image without assuming C1 or using a Taylor bound.
    ///
    /// This preserves certified zeroth-order queries across finite continuous
    /// C0 source or coordinate joints. Subranges refine the same retained root
    /// tubes; they never select another branch or substitute a leader.
    pub fn position_bounds(
        &self,
        range: [f64; 2],
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> Result<[ValidatedInterval; 3], ValidatedGeometryError> {
        budget.charge(1, cancel)?;
        let range = interval_range(range)?;
        self.check_range(range)?;
        if range.lower == range.upper {
            return Ok(self.evaluate(range.lower, budget, cancel)?.error);
        }
        let mut result: Option<[ValidatedInterval; 3]> = None;
        for piece in &self.pieces {
            budget.charge(1, cancel)?;
            let Some(part) = range.intersection(ValidatedInterval {
                lower: piece.parameter[0],
                upper: piece.parameter[1],
            }) else {
                continue;
            };
            if part.lower == part.upper {
                continue;
            }
            let position = if let Some(child) = &self.mapped {
                let (transform, active, orientation) = self.source.mapping()?;
                let mapped = reflected_trim_range(orientation, active, part)?;
                transform_prepared_position(
                    *transform,
                    child.position_bounds([mapped.lower, mapped.upper], budget, cancel)?,
                )?
            } else if matches!(self.source.curve()?, Curve::IntersectionCurve(_)) {
                let original = flatten(
                    piece
                        .source_parameters
                        .ok_or(ValidatedGeometryError::InvalidInput)?,
                );
                let mut domain = original;
                let mut position = piece.jet.position;
                for _ in 0..self.options.max_iterations {
                    budget.charge(1, cancel)?;
                    let tube = match Certification::Prepared(self).certify(
                        piece.projection,
                        part,
                        domain,
                        budget,
                        cancel,
                    ) {
                        Ok(tube) => tube,
                        // The retained enclosure remains a proof if further
                        // interval contraction is arithmetically inconclusive.
                        Err(error) if retryable(error) => break,
                        Err(error) => return Err(error),
                    };
                    position = intersect3(position, tube.position()?)?;
                    let next = root_image_domain(&tube)?;
                    if next == domain {
                        break;
                    }
                    domain = next;
                }
                position
            } else {
                validated_curve_position_bounds(
                    self.source.curve()?,
                    [part.lower, part.upper],
                    budget,
                    cancel,
                )?
            };
            result = Some(match result {
                None => position,
                Some(previous) => std::array::from_fn(|i| previous[i].hull(position[i])),
            });
        }
        result.ok_or(ValidatedGeometryError::ParameterOutOfRange)
    }

    /// Encloses a subrange without selecting another intersection branch.
    ///
    /// Original intersections receive fresh whole-subrange certificates inside
    /// the retained uniqueness tube. Nested root-image contraction sharpens
    /// their jets without replacing the original projection or selecting a new
    /// root. Every tighter jet is independently certified over the entire span,
    /// under the existing iteration, work and cancellation limits. Refinement
    /// stops at the root-image arithmetic margin; point jets are not substituted.
    ///
    /// Implicit third derivatives are not supplied: their intervals are
    /// unbounded. Other fields enclose finite piecewise/one-sided derivatives,
    /// not a Taylor theorem across C0. Taylor/chord use requires
    /// [`Self::is_smooth_on`]. A range crossing a coordinate joint returns
    /// [`ValidatedGeometryError::NonSmooth`]; [`Self::position_bounds`] remains
    /// available for its whole image.
    pub fn jet(
        &self,
        range: [f64; 2],
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> Result<ValidatedCurveJet, ValidatedGeometryError> {
        let range = interval_range(range)?;
        self.check_range(range)?;
        if let Some(child) = &self.mapped {
            let (transform, active, orientation) = self.source.mapping()?;
            let mut result: Option<ValidatedCurveJet> = None;
            let mut projection = None;
            for piece in &self.pieces {
                budget.charge(1, cancel)?;
                let Some(part) = range.intersection(ValidatedInterval {
                    lower: piece.parameter[0],
                    upper: piece.parameter[1],
                }) else {
                    continue;
                };
                if part.lower == part.upper && range.lower < range.upper {
                    continue;
                }
                if projection.is_some_and(|previous| previous != piece.projection) {
                    return Err(ValidatedGeometryError::NonSmooth);
                }
                projection = Some(piece.projection);
                let parameter = reflected_trim_range(orientation, active, part)?;
                let jet = child.jet([parameter.lower, parameter.upper], budget, cancel)?;
                let mut jet = transform_prepared_jet(*transform, jet)?;
                if !orientation {
                    jet.first = jet.first.map(std::ops::Neg::neg);
                }
                result = Some(match result {
                    None => jet,
                    Some(previous) => hull_jet(previous, jet),
                });
            }
            return result.ok_or(ValidatedGeometryError::ParameterOutOfRange);
        }
        if !matches!(self.source.curve()?, Curve::IntersectionCurve(_)) {
            return validated_curve_piecewise_jet(
                self.source.curve()?,
                [range.lower, range.upper],
                budget,
                cancel,
            );
        }
        let mut result: Option<ValidatedCurveJet> = None;
        let mut projection = None;
        for piece in &self.pieces {
            budget.charge(1, cancel)?;
            let Some(part) = range.intersection(ValidatedInterval {
                lower: piece.parameter[0],
                upper: piece.parameter[1],
            }) else {
                continue;
            };
            if part.lower == part.upper && range.lower < range.upper {
                continue;
            }
            if projection.is_some_and(|previous| previous != piece.projection) {
                return Err(ValidatedGeometryError::NonSmooth);
            }
            projection = Some(piece.projection);
            let parameters = flatten(
                piece
                    .source_parameters
                    .ok_or(ValidatedGeometryError::InvalidInput)?,
            );
            let jet = refine_implicit_jet(
                Certification::Prepared(self),
                piece.projection,
                part,
                parameters,
                &self.options,
                budget,
                cancel,
            )?;
            result = Some(match result {
                None => jet,
                Some(previous) => hull_jet(previous, jet),
            });
        }
        result.ok_or(ValidatedGeometryError::ParameterOutOfRange)
    }

    /// Encloses `direction dot (curve(t) - origin)` on this prepared branch.
    ///
    /// Explicit coefficients retain scalar correlation before source basis
    /// evaluation, including raw plane-carried parameter curves. Implicit
    /// roots conservatively project their certified jets and intersect position
    /// with original plane-carrier coefficient bounds. Proven constant carrier
    /// functionals retain exact zero derivatives. Binding and joint restrictions
    /// are unchanged; mapped roots retain their actual post-root spatial map.
    pub fn projected_jet(
        &self,
        range: [f64; 2],
        origin: [f64; 3],
        direction: [f64; 3],
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> Result<ValidatedScalarCurveJet, ValidatedGeometryError> {
        budget.charge(1, cancel)?;
        let interval = interval_range(range)?;
        self.check_range(interval)?;
        if !origin.into_iter().chain(direction).all(f64::is_finite) {
            return Err(ValidatedGeometryError::InvalidInput);
        }
        if self.mapped.is_none() && !matches!(self.source.curve()?, Curve::IntersectionCurve(_)) {
            match validated_curve_projected_jet(
                self.source.curve()?,
                range,
                origin,
                direction,
                budget,
                cancel,
            ) {
                Ok(jet) => return Ok(jet),
                Err(ValidatedGeometryError::NonSmooth) => {}
                Err(error) => return Err(error),
            }
        }
        let jet = self.jet(range, budget, cancel)?;
        let mut result = ValidatedScalarCurveJet {
            position: ZERO,
            first: ZERO,
            second: ZERO,
        };
        for axis in 0..3 {
            let factor = ValidatedInterval::point(direction[axis]);
            result.position = result.position
                + factor * (jet.position[axis] - ValidatedInterval::point(origin[axis]));
            result.first = result.first + factor * jet.first[axis];
            result.second = result.second + factor * jet.second[axis];
        }
        if self.mapped.is_none() && matches!(self.source.curve()?, Curve::IntersectionCurve(_)) {
            let (first, second, _) = carriers(self.source.curve()?)?;
            let direction = direction.map(ValidatedInterval::point);
            let origin = origin.map(ValidatedInterval::point);
            let mut coefficients = [None; 2];
            for (index, surface) in [first, second].into_iter().enumerate() {
                budget.charge(1, cancel)?;
                if let Surface::Plane(plane) = surface {
                    let (u, v) = coverage::plane_axes(plane);
                    let p = std::array::from_fn(|i| ValidatedInterval::point(plane.origin()[i]));
                    coefficients[index] = Some([
                        dot3(direction, subtract3(p, origin)),
                        dot3(direction, u),
                        dot3(direction, v),
                    ]);
                }
            }
            let mut position: Option<ValidatedInterval> = None;
            for piece in &self.pieces {
                budget.charge(1, cancel)?;
                let Some(part) = interval.intersection(ValidatedInterval {
                    lower: piece.parameter[0],
                    upper: piece.parameter[1],
                }) else {
                    continue;
                };
                if part.lower == part.upper && interval.lower < interval.upper {
                    continue;
                }
                let charts = piece
                    .source_parameters
                    .ok_or(ValidatedGeometryError::InvalidInput)?;
                let mut bound = result.position;
                for index in 0..2 {
                    if let Some([offset, u, v]) = coefficients[index] {
                        let value = offset + u * charts[index][0] + v * charts[index][1];
                        bound = bound
                            .intersection(value)
                            .ok_or(ValidatedGeometryError::ConditioningFailure)?;
                    }
                }
                position = Some(position.map_or(bound, |previous| previous.hull(bound)));
            }
            result.position = position.ok_or(ValidatedGeometryError::ParameterOutOfRange)?;
            if coefficients
                .into_iter()
                .flatten()
                .any(|[_, u, v]| u == ZERO && v == ZERO)
            {
                // This is a source-coefficient identity on an entire carrier,
                // not a conclusion drawn from sampled residuals.
                result.first = result
                    .first
                    .intersection(ZERO)
                    .ok_or(ValidatedGeometryError::ConditioningFailure)?;
                result.second = result
                    .second
                    .intersection(ZERO)
                    .ok_or(ValidatedGeometryError::ConditioningFailure)?;
            }
        }
        if ![result.position, result.first, result.second]
            .into_iter()
            .all(ValidatedInterval::is_finite)
        {
            return Err(ValidatedGeometryError::ConditioningFailure);
        }
        Ok(result)
    }

    /// Evaluates only the already certified branch, returning its error box.
    ///
    /// No native legacy Newton, leader fallback, or endpoint snap is used.
    pub fn evaluate(
        &self,
        parameter: f64,
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> Result<ValidatedCurvePoint, ValidatedGeometryError> {
        let t = interval_range([parameter, parameter])?;
        self.check_range(t)?;
        let enclosure =
            self.position_enclosure(t, self.options.endpoint_tolerance, budget, cancel)?;
        spatial_point(enclosure, self.options.endpoint_tolerance)
    }

    fn position_enclosure(
        &self,
        range: ValidatedInterval,
        tolerance: f64,
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> Result<[ValidatedInterval; 3], ValidatedGeometryError> {
        budget.charge(1, cancel)?;
        self.check_range(range)?;
        if let Some(child) = &self.mapped {
            let (transform, active, orientation) = self.source.mapping()?;
            let parameter = reflected_trim_range(orientation, active, range)?;
            let mut child_tolerance = tolerance;
            for _ in 0..self.options.max_iterations {
                budget.charge(1, cancel)?;
                let position =
                    child.position_enclosure(parameter, child_tolerance, budget, cancel)?;
                let position = transform_prepared_position(*transform, position)?;
                if position.into_iter().all(|value| value.width() <= tolerance) {
                    return Ok(position);
                }
                child_tolerance *= 0.125;
                if child_tolerance <= 0.0 {
                    return Err(ValidatedGeometryError::ConditioningFailure);
                }
            }
            return Err(ValidatedGeometryError::Unresolved);
        }
        if !matches!(self.source.curve()?, Curve::IntersectionCurve(_)) {
            return validated_curve_position_bounds(
                self.source.curve()?,
                [range.lower, range.upper],
                budget,
                cancel,
            );
        }
        let mut options = self.options;
        options.endpoint_tolerance = tolerance;
        let mut result: Option<[ValidatedInterval; 3]> = None;
        for piece in &self.pieces {
            budget.charge(1, cancel)?;
            let Some(part) = range.intersection(ValidatedInterval {
                lower: piece.parameter[0],
                upper: piece.parameter[1],
            }) else {
                continue;
            };
            if part.lower == part.upper && range.lower < range.upper {
                continue;
            }
            let initial = flatten(
                piece
                    .source_parameters
                    .ok_or(ValidatedGeometryError::InvalidInput)?,
            );
            let (_, enclosure) = refine_endpoint(
                Certification::Prepared(self),
                piece.projection,
                part,
                initial,
                &options,
                budget,
                cancel,
            )?;
            result = Some(match result {
                None => enclosure,
                Some(previous) => std::array::from_fn(|i| previous[i].hull(enclosure[i])),
            });
        }
        result.ok_or(ValidatedGeometryError::ParameterOutOfRange)
    }

    fn check_range(&self, range: ValidatedInterval) -> Result<(), ValidatedGeometryError> {
        if range.lower < self.range[0] || range.upper > self.range[1] {
            Err(ValidatedGeometryError::ParameterOutOfRange)
        } else {
            Ok(())
        }
    }
}

/// Prepares a complete regular projected branch over a finite native range.
///
/// Input coefficients, intervals, options, work, and cancellation are checked
/// before allocation or iteration. Nonsmooth leaders receive shared joint-plane
/// coordinates, with separate smooth derivative bounds on each native span.
/// Ambiguous continuation, singular charts, nontransverse carriers, unseparated
/// rational denominators, and exhausted subdivision are typed refusals. The
/// prepared representation is distinct from the legacy infallible evaluator's
/// endpoint snapping and fallback behavior.
pub fn prepared_validated_curve<'a>(
    curve: &'a Curve,
    range: [f64; 2],
    options: &ValidatedCurveOptions,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<PreparedValidatedCurve<'a>, ValidatedGeometryError> {
    prepare_validated_curve(curve, range, options, false, 0, budget, cancel)
}

/// Consumes an original native curve and retains one immutable prepared branch.
///
/// This uses the same continuation, coordinate choices and closed-piece proof
/// as [`prepared_validated_curve`], not a face/plane substitute. Original guides
/// and carriers are moved, not cloned. Retained trim operators keep their exact
/// active intervals, reflection anchors and spatial matrices; their owned basis
/// boxes are reused. Queries never prepare or select another branch.
///
/// The complete owned source is accounted, with the native structural-depth
/// limit and cancellation/work charges, before proof or capture allocation.
/// Source payload and proof capacities share `options.max_retained_bytes`.
/// As with the borrowed API, uniqueness is local to the certified connected
/// tube; no assertion excludes other components outside it.
pub fn prepared_validated_curve_owned(
    curve: Curve,
    range: [f64; 2],
    options: &ValidatedCurveOptions,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<PreparedValidatedCurve<'static>, ValidatedGeometryError> {
    let bytes = curve_owned_bytes(&curve, options.max_retained_bytes, budget, cancel)?;
    let proof_capacity = options
        .max_retained_bytes
        .checked_sub(bytes)
        .ok_or(ValidatedGeometryError::ResourceLimit)?;
    if proof_capacity < size_of::<PreparedValidatedCurve<'static>>() {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    validate_preparation_range(&curve, range, options, false, 0, budget, cancel)?;
    match curve {
        Curve::TrimmedCurve(wrapper)
            if retained_implicit_basis(wrapper.entity().curve(), 1, budget, cancel)? =>
        {
            prepare_owned_mapped(wrapper, range, options, 0, budget, cancel)
        }
        curve => prepare_owned_leaf(Box::new(curve), Some(bytes), range, options, budget, cancel),
    }
}

fn prepare_owned_box(
    curve: Box<Curve>,
    range: [f64; 2],
    options: &ValidatedCurveOptions,
    depth: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<PreparedValidatedCurve<'static>, ValidatedGeometryError> {
    validate_preparation_range(&curve, range, options, true, depth, budget, cancel)?;
    if let Curve::TrimmedCurve(wrapper) = curve.as_ref()
        && retained_implicit_basis(wrapper.entity().curve(), depth + 1, budget, cancel)?
    {
        let Curve::TrimmedCurve(wrapper) = *curve else {
            return Err(ValidatedGeometryError::InvalidInput);
        };
        prepare_owned_mapped(wrapper, range, options, depth, budget, cancel)
    } else {
        prepare_owned_leaf(curve, None, range, options, budget, cancel)
    }
}

fn prepare_owned_leaf(
    curve: Box<Curve>,
    bytes: Option<usize>,
    range: [f64; 2],
    options: &ValidatedCurveOptions,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<PreparedValidatedCurve<'static>, ValidatedGeometryError> {
    let bytes = match bytes {
        Some(bytes) => bytes,
        None => curve_owned_bytes(&curve, options.max_retained_bytes, budget, cancel)?,
    };
    let mut proof_options = *options;
    proof_options.max_retained_bytes = options
        .max_retained_bytes
        .checked_sub(bytes)
        .ok_or(ValidatedGeometryError::ResourceLimit)?;
    if proof_options.max_retained_bytes < size_of::<PreparedValidatedCurve<'static>>() {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    // Unmapped preparation retains no reference outside its private source
    // slot. Move the proven pieces unchanged, then replace that borrowed slot
    // with the very same owned native object, without another root search.
    let (range, pieces) = {
        let prepared = prepare_unmapped_curve(
            &curve,
            range,
            &proof_options,
            SourcePreparation::default(),
            budget,
            cancel,
        )?;
        (prepared.range, prepared.pieces)
    };
    Ok(PreparedValidatedCurve {
        source: PreparedSource::Owned {
            curve,
            bytes,
            authority: OwnedSourceAuthority::Original,
        },
        range,
        pieces,
        options: *options,
        mapped: None,
    })
}

fn prepare_owned_mapped(
    wrapper: Processor<TrimmedCurve<Box<Curve>>, Matrix4>,
    range: [f64; 2],
    options: &ValidatedCurveOptions,
    depth: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<PreparedValidatedCurve<'static>, ValidatedGeometryError> {
    if !(0..4).all(|i| (0..4).all(|j| wrapper.transform()[i][j].is_finite())) {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    let (trim, transform, orientation) = wrapper.into_parts();
    let (basis, active) = trim.into_parts();
    let native = reflected_trim_range(orientation, active, interval_range(range)?)?;
    let child = prepare_owned_box(
        basis,
        [native.lower, native.upper],
        options,
        depth + 1,
        budget,
        cancel,
    )?;
    finish_mapped_curve(
        PreparedSource::Mapped {
            transform,
            range: active,
            orientation,
        },
        child,
        range,
        options,
        budget,
        cancel,
    )
}

fn validate_preparation_range(
    curve: &Curve,
    range: [f64; 2],
    options: &ValidatedCurveOptions,
    conventional_extension: bool,
    depth: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<(), ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    if depth > crate::geometry::MAX_NATIVE_GEOMETRY_NESTING {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    let interval = interval_range(range)?;
    if interval.lower >= interval.upper
        || options.max_depth > 48
        || options.max_iterations == 0
        || options.max_iterations > 128
        || !options.parameter_radius.is_finite()
        || options.parameter_radius <= 0.0
        || !options.endpoint_tolerance.is_finite()
        || options.endpoint_tolerance <= 0.0
    {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    if options.max_retained_bytes < size_of::<PreparedValidatedCurve<'_>>() {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    let native = curve
        .try_range_tuple()
        .ok_or(ValidatedGeometryError::InvalidInput)?;
    if !native.0.is_finite() || !native.1.is_finite() || native.0 > native.1 {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    if (range[0] < native.0 || range[1] > native.1)
        && !(conventional_extension && conventional_projection_domain(curve))
    {
        return Err(ValidatedGeometryError::ParameterOutOfRange);
    }
    Ok(())
}

fn prepare_validated_curve<'a>(
    curve: &'a Curve,
    range: [f64; 2],
    options: &ValidatedCurveOptions,
    conventional_extension: bool,
    depth: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<PreparedValidatedCurve<'a>, ValidatedGeometryError> {
    // Guard native recursive range/proposal evaluation before entering it.
    if depth == 0 {
        curve_owned_bytes(curve, usize::MAX, budget, cancel)?;
    }
    validate_preparation_range(
        curve,
        range,
        options,
        conventional_extension,
        depth,
        budget,
        cancel,
    )?;
    if let Curve::TrimmedCurve(wrapper) = curve
        && retained_implicit_basis(wrapper.entity().curve(), depth + 1, budget, cancel)?
    {
        return prepare_mapped_curve(curve, range, options, depth, budget, cancel);
    }
    prepare_unmapped_curve(
        curve,
        range,
        options,
        SourcePreparation::default(),
        budget,
        cancel,
    )
}

#[derive(Clone, Copy, Default)]
struct SourcePreparation {
    anchor: Option<ValidatedFacePlaneSourceAnchor>,
    endpoint_coordinate_planes: [Option<CoordinatePlane>; 2],
    coalesce_coordinates: bool,
}

type CoordinateSpans = Vec<([f64; 2], Projection)>;

fn prepare_unmapped_curve<'a>(
    curve: &'a Curve,
    range: [f64; 2],
    options: &ValidatedCurveOptions,
    source_preparation: SourcePreparation,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<PreparedValidatedCurve<'a>, ValidatedGeometryError> {
    if !matches!(curve, Curve::IntersectionCurve(_)) {
        let jet = validated_curve_piecewise_jet(curve, range, budget, cancel)?;
        budget.charge(1, cancel)?;
        let mut pieces = Vec::new();
        reserve_native_vec(
            &mut pieces,
            options.max_retained_bytes,
            size_of::<PreparedValidatedCurve<'_>>(),
            budget,
            cancel,
        )?;
        pieces.push(ValidatedCurvePiece {
            parameter: range,
            jet,
            source_parameters: None,
            contraction_bound: 0.0,
            surface_residual: [ZERO; 3],
            projection: Projection::Native,
        });
        return Ok(PreparedValidatedCurve {
            source: PreparedSource::Borrowed(curve),
            range,
            options: *options,
            pieces,
            mapped: None,
        });
    }
    let (surface0, surface1, leader) = carriers(curve)?;
    let (coordinate_spans, coalesced_coordinates) = coordinate_spans(
        leader,
        range,
        options.max_retained_bytes - size_of::<PreparedValidatedCurve<'_>>(),
        &source_preparation,
        options.max_depth,
        budget,
        cancel,
    )?;
    let from_back = matches!(
        source_preparation.anchor,
        Some(ValidatedFacePlaneSourceAnchor::Back(_))
    );
    let anchor_parameter = range[usize::from(from_back)];
    let first_projection = coordinate_spans[if from_back {
        coordinate_spans.len() - 1
    } else {
        0
    }]
    .1;
    let leader_anchor =
        validated_projection_leader_jet(leader, [anchor_parameter; 2], budget, cancel)?
            .position
            .map(ValidatedInterval::midpoint);
    let seed0 = match source_preparation.anchor {
        Some(anchor) => anchor.chart(),
        None => nearest_seed(surface0, leader_anchor, options, budget, cancel)?,
    };
    let seed1 = nearest_seed(surface1, leader_anchor, options, budget, cancel)?;
    let center = numerical_root(
        curve,
        first_projection,
        anchor_parameter,
        [seed0[0], seed0[1], seed1[0], seed1[1]],
        options,
        budget,
        cancel,
    )?;
    let mut initial = None;
    let mut last_error = ValidatedGeometryError::Unresolved;
    for iteration in 0..options.max_iterations {
        budget.charge(1, cancel)?;
        let radius = options.parameter_radius * 0.5_f64.powi(iteration as i32);
        let domain = std::array::from_fn(|i| inflated(center[i], radius * (1.0 + center[i].abs())));
        match certify(
            curve,
            first_projection,
            ValidatedInterval::point(anchor_parameter),
            domain,
            budget,
            cancel,
        ) {
            Ok(tube) => {
                initial = Some(tube);
                break;
            }
            Err(error) if retryable(error) => last_error = error,
            Err(error) => return Err(error),
        }
    }
    let initial = initial.ok_or(last_error)?;
    let (mut anchor, _) = refine_endpoint(
        Certification::Original(curve),
        first_projection,
        ValidatedInterval::point(anchor_parameter),
        initial.domain,
        options,
        budget,
        cancel,
    )?;
    // The fixed stack bounds both allocation and traversal independently of input.
    let mut pending = [([0.0; 2], 0usize); 49];
    let mut pieces = Vec::new();
    let coordinate_bytes = coordinate_spans
        .capacity()
        .saturating_mul(size_of::<([f64; 2], Projection)>());
    let mut coordinate_spans = coordinate_spans.into_iter();
    while let Some((coordinate_range, projection)) = if from_back {
        coordinate_spans.next_back()
    } else {
        coordinate_spans.next()
    } {
        pending[0] = (coordinate_range, 0);
        let mut pending_len = 1;
        while pending_len != 0 {
            budget.charge(1, cancel)?;
            pending_len -= 1;
            let (part, depth) = pending[pending_len];
            let t = interval_range(part)?;
            let middle = t.midpoint();
            let proposed = numerical_root(
                curve,
                projection,
                middle,
                anchor.map(ValidatedInterval::midpoint),
                options,
                budget,
                cancel,
            );
            let attempt = proposed.and_then(|center| {
                let domain = std::array::from_fn(|i| {
                    let relative_width =
                        (part[1] - part[0]) / (coordinate_range[1] - coordinate_range[0]);
                    let displacement = (center[i] - anchor[i].midpoint()).abs();
                    let radius =
                        options.parameter_radius * (1.0 + center[i].abs()) * relative_width
                            + 2.0 * displacement
                            + 2.0 * anchor[i].width();
                    inflated(center[i], radius).hull(anchor[i])
                });
                let tube = certify(curve, projection, t, domain, budget, cancel)?;
                let jet = if coalesced_coordinates {
                    let jet = implicit_jet(&tube)?;
                    if !jet
                        .first
                        .into_iter()
                        .any(|value| value.lower > 0.0 || value.upper < 0.0)
                    {
                        return Err(ValidatedGeometryError::NonTransverse);
                    }
                    Some(jet)
                } else {
                    None
                };
                Ok((tube, jet))
            });
            match attempt {
                Ok((tube, jet)) => {
                    if !(0..4).all(|i| {
                        tube.domain[i].lower <= anchor[i].lower
                            && anchor[i].upper <= tube.domain[i].upper
                    }) {
                        return Err(ValidatedGeometryError::AmbiguousBranch);
                    }
                    let jet = match jet {
                        Some(jet) => jet,
                        None => implicit_jet(&tube)?,
                    };
                    let (next, _) = refine_endpoint(
                        Certification::Original(curve),
                        projection,
                        ValidatedInterval::point(part[usize::from(!from_back)]),
                        tube.domain,
                        options,
                        budget,
                        cancel,
                    )?;
                    budget.charge(1, cancel)?;
                    reserve_native_vec(
                        &mut pieces,
                        options.max_retained_bytes,
                        size_of::<PreparedValidatedCurve<'_>>().saturating_add(coordinate_bytes),
                        budget,
                        cancel,
                    )?;
                    pieces.push(ValidatedCurvePiece {
                        parameter: part,
                        jet,
                        source_parameters: Some(unflatten(tube.domain)),
                        contraction_bound: tube.contraction_bound(),
                        surface_residual: tube.surface_residual(),
                        projection,
                    });
                    anchor = next;
                }
                Err(error) if retryable(error) => {
                    if depth == options.max_depth || middle <= part[0] || middle >= part[1] {
                        return Err(error);
                    }
                    budget.charge(2, cancel)?;
                    let halves = [[part[0], middle], [middle, part[1]]];
                    pending[pending_len] = (halves[usize::from(!from_back)], depth + 1);
                    pending[pending_len + 1] = (halves[usize::from(from_back)], depth + 1);
                    pending_len += 2;
                }
                Err(error) => return Err(error),
            }
        }
    }
    if from_back {
        budget.charge(pieces.len(), cancel)?;
        pieces.reverse();
    }
    Ok(PreparedValidatedCurve {
        source: PreparedSource::Borrowed(curve),
        range,
        pieces,
        options: *options,
        mapped: None,
    })
}

fn retained_implicit_basis(
    curve: &Curve,
    depth: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<bool, ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    if depth > crate::geometry::MAX_NATIVE_GEOMETRY_NESTING {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    match curve {
        Curve::IntersectionCurve(_) => Ok(true),
        Curve::TrimmedCurve(wrapper) => {
            retained_implicit_basis(wrapper.entity().curve(), depth + 1, budget, cancel)
        }
        _ => Ok(false),
    }
}

fn conventional_projection_domain(curve: &Curve) -> bool {
    let Curve::IntersectionCurve(curve) = curve else {
        return false;
    };
    match curve.leader().as_ref() {
        Curve::Line(_) => true,
        Curve::ParameterCurve(curve) => {
            matches!(curve.curve(), Curve2D::Line(_) | Curve2D::Conic(_))
        }
        _ => false,
    }
}

fn prepare_mapped_curve<'a>(
    curve: &'a Curve,
    range: [f64; 2],
    options: &ValidatedCurveOptions,
    depth: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<PreparedValidatedCurve<'a>, ValidatedGeometryError> {
    let Curve::TrimmedCurve(wrapper) = curve else {
        return Err(ValidatedGeometryError::InvalidInput);
    };
    if !(0..4).all(|i| (0..4).all(|j| wrapper.transform()[i][j].is_finite())) {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    let native = reflected_trim_range(
        wrapper.orientation(),
        wrapper.entity().range(),
        interval_range(range)?,
    )?;
    let child = prepare_validated_curve(
        wrapper.entity().curve(),
        [native.lower, native.upper],
        options,
        true,
        depth + 1,
        budget,
        cancel,
    )?;
    finish_mapped_curve(
        PreparedSource::Mapped {
            transform: *wrapper.transform(),
            range: wrapper.entity().range(),
            orientation: wrapper.orientation(),
        },
        child,
        range,
        options,
        budget,
        cancel,
    )
}

fn finish_mapped_curve<'a>(
    source: PreparedSource<'a>,
    child: PreparedValidatedCurve<'a>,
    range: [f64; 2],
    options: &ValidatedCurveOptions,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<PreparedValidatedCurve<'a>, ValidatedGeometryError> {
    let (transform, active, orientation) = source.mapping()?;
    let overhead = size_of::<PreparedValidatedCurve<'_>>()
        .checked_add(child.retained_bytes())
        .ok_or(ValidatedGeometryError::ResourceLimit)?;
    if overhead > options.max_retained_bytes {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    let mut pieces = Vec::new();
    let mut start = range[0];
    budget.charge(child.pieces.len(), cancel)?;
    let rho = child
        .pieces
        .iter()
        .map(|piece| piece.contraction_bound)
        .fold(0.0, f64::max);
    let mut residual = [ZERO; 3];
    for piece in &child.pieces {
        residual = std::array::from_fn(|i| residual[i].hull(piece.surface_residual[i]));
    }
    for index in 0..child.pieces.len() {
        budget.charge(1, cancel)?;
        let child_index = if orientation {
            index
        } else {
            child.pieces.len() - 1 - index
        };
        let piece = &child.pieces[child_index];
        let end = if index + 1 == child.pieces.len() {
            range[1]
        } else {
            let boundary = if orientation {
                piece.parameter[1]
            } else {
                piece.parameter[0]
            };
            let mapped =
                reflected_trim_range(orientation, active, ValidatedInterval::point(boundary))?;
            if mapped.upper <= start || mapped.lower >= range[1] {
                continue;
            }
            if mapped.lower != mapped.upper {
                let next = if orientation {
                    child_index + 1
                } else {
                    child_index - 1
                };
                if piece.projection == child.pieces[next].projection {
                    continue;
                }
                return Err(ValidatedGeometryError::NonSmooth);
            }
            mapped.lower
        };
        let mut pending = [([0.0; 2], 0usize); 49];
        pending[0] = ([start, end], 0);
        let mut length = 1;
        while length != 0 {
            budget.charge(1, cancel)?;
            length -= 1;
            let (part, subdivision) = pending[length];
            let mapped = reflected_trim_range(orientation, active, interval_range(part)?)?;
            let attempt = child
                .jet([mapped.lower, mapped.upper], budget, cancel)
                .and_then(|jet| transform_prepared_jet(*transform, jet));
            match attempt {
                Ok(mut jet) => {
                    if !orientation {
                        jet.first = jet.first.map(std::ops::Neg::neg);
                    }
                    reserve_native_vec(
                        &mut pieces,
                        options.max_retained_bytes,
                        overhead,
                        budget,
                        cancel,
                    )?;
                    pieces.push(ValidatedCurvePiece {
                        parameter: part,
                        jet,
                        source_parameters: None,
                        contraction_bound: rho,
                        surface_residual: residual,
                        projection: piece.projection,
                    });
                }
                Err(error)
                    if retryable(error)
                        || error == ValidatedGeometryError::DenominatorContainsZero =>
                {
                    if subdivision >= options.max_depth {
                        return Err(error);
                    }
                    let middle = mapped_midpoint(part)?;
                    budget.charge(2, cancel)?;
                    pending[length] = ([middle, part[1]], subdivision + 1);
                    pending[length + 1] = ([part[0], middle], subdivision + 1);
                    length += 2;
                }
                Err(error) => return Err(error),
            }
        }
        start = end;
    }
    if start != range[1] || pieces.is_empty() {
        return Err(ValidatedGeometryError::Unresolved);
    }
    budget.charge(1, cancel)?;
    Ok(PreparedValidatedCurve {
        source,
        range,
        pieces,
        options: *options,
        mapped: Some(Box::new(child)),
    })
}

fn mapped_midpoint(range: [f64; 2]) -> Result<f64, ValidatedGeometryError> {
    let middle = interval_range(range)?.midpoint();
    if range[0] < middle && middle < range[1] {
        Ok(middle)
    } else {
        Err(ValidatedGeometryError::Unresolved)
    }
}

fn transform_prepared_position(
    matrix: Matrix4,
    position: [ValidatedInterval; 3],
) -> Result<[ValidatedInterval; 3], ValidatedGeometryError> {
    if !(0..4).all(|i| (0..4).all(|j| matrix[i][j].is_finite())) {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    let homogeneous: [ValidatedInterval; 4] = std::array::from_fn(|i| {
        (0..3)
            .map(|j| ValidatedInterval::point(matrix[j][i]) * position[j])
            .fold(ValidatedInterval::point(matrix[3][i]), |a, b| a + b)
    });
    let inverse = homogeneous[3].reciprocal()?;
    let result: [ValidatedInterval; 3] = std::array::from_fn(|i| homogeneous[i] * inverse);
    if result.into_iter().all(ValidatedInterval::is_finite) {
        Ok(result)
    } else {
        Err(ValidatedGeometryError::ConditioningFailure)
    }
}

fn transform_prepared_jet(
    matrix: Matrix4,
    jet: ValidatedCurveJet,
) -> Result<ValidatedCurveJet, ValidatedGeometryError> {
    if !(0..4).all(|i| (0..4).all(|j| matrix[i][j].is_finite())) {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    let value: [ValidatedInterval; 4] = std::array::from_fn(|i| {
        (0..3)
            .map(|j| ValidatedInterval::point(matrix[j][i]) * jet.position[j])
            .fold(ValidatedInterval::point(matrix[3][i]), |a, b| a + b)
    });
    let first: [ValidatedInterval; 4] = std::array::from_fn(|i| {
        (0..3)
            .map(|j| ValidatedInterval::point(matrix[j][i]) * jet.first[j])
            .fold(ZERO, |a, b| a + b)
    });
    let second: [ValidatedInterval; 4] = std::array::from_fn(|i| {
        (0..3)
            .map(|j| ValidatedInterval::point(matrix[j][i]) * jet.second[j])
            .fold(ZERO, |a, b| a + b)
    });
    let inverse = value[3].reciprocal()?;
    let position: [ValidatedInterval; 3] = std::array::from_fn(|i| value[i] * inverse);
    let derivative: [ValidatedInterval; 3] =
        std::array::from_fn(|i| (first[i] - position[i] * first[3]) * inverse);
    let derivative2: [ValidatedInterval; 3] = std::array::from_fn(|i| {
        (second[i]
            - ValidatedInterval::point(2.0) * derivative[i] * first[3]
            - position[i] * second[3])
            * inverse
    });
    if !position
        .into_iter()
        .chain(derivative)
        .chain(derivative2)
        .all(ValidatedInterval::is_finite)
    {
        return Err(ValidatedGeometryError::ConditioningFailure);
    }
    Ok(ValidatedCurveJet {
        position,
        first: derivative,
        second: derivative2,
        third: [ValidatedInterval {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
        }; 3],
    })
}

/// Numerical source-chart proposal at an actual native candidate endpoint.
///
/// This selects where continuation starts, not a certified branch or a period
/// shift. Full original-source continuation and endpoint coverage still prove
/// the selection. Native parameter direction and derivative signs never change.
#[derive(Clone, Copy, Debug)]
pub enum ValidatedFacePlaneSourceAnchor {
    /// Start at the ascending native parameter range's front.
    Front([f64; 2]),
    /// Start at its back and continue toward the front in the same native gauge.
    Back([f64; 2]),
    /// Both actual original endpoint charts, still proposals rather than proof.
    Both {
        /// Original chart at the ascending native range's front.
        front: [f64; 2],
        /// Original chart at its back, without a gauge or period shift.
        back: [f64; 2],
    },
}

impl ValidatedFacePlaneSourceAnchor {
    fn finite(self) -> bool {
        match self {
            Self::Front(chart) | Self::Back(chart) => chart.into_iter().all(f64::is_finite),
            Self::Both { front, back } => front.into_iter().chain(back).all(f64::is_finite),
        }
    }
    fn chart(self) -> [f64; 2] {
        match self {
            Self::Front(chart) | Self::Back(chart) => chart,
            Self::Both { front, .. } => front,
        }
    }
}

/// Borrowed original carrier and optional numerical start for a generated query.
#[derive(Clone, Copy, Debug)]
pub struct ValidatedFacePlaneSource<'a> {
    /// The actual immutable source-face carrier.
    pub surface: &'a Surface,
    /// Original trim-root chart proposal at the indicated native endpoint.
    ///
    /// This does not certify a branch or replace full continuation and coverage.
    /// In particular it does not modify any retained original master selection.
    pub anchor: Option<ValidatedFacePlaneSourceAnchor>,
    /// Optional exact shared coordinates at the generated front and back.
    ///
    /// Incident guides must retain identical coefficients. These proposals
    /// never replace original trim incidence or a common uniqueness proof.
    pub endpoint_coordinate_planes: [Option<CoordinatePlane>; 2],
}

/// Binds a native guide to its authoritative source-face and section-plane roots.
///
/// The guide supplies native parameters and coordinate proposals, not geometric
/// containment. The returned binding owns bounded coefficient clones, so it can
/// escape a topology carrier-lock callback. All allocation sizes are counted
/// before cloning. Sampling and all-domain coverage must share this same binding.
///
/// An actual original trim-root chart proposal avoids choosing an unrelated
/// periodic sheet through nearest-point inversion at either native endpoint.
/// The proposal never substitutes for the same whole-range interval proof.
///
/// Generated coordinate proposals may combine adjacent guide-knot spans.
/// Guide jets only condition those proposals: original-carrier inclusion,
/// anchor continuity and final whole-face coverage remain mandatory. A refused
/// coarse continuation retries the original knot coordinates with the same
/// budget; original-master preparation is unchanged.
///
/// For a transverse original extrusion with genuinely bounded profile support,
/// two actual endpoint chart proposals can instead certify a complete
/// affine-profile graph by invertible plane
/// coefficients. Its jets and coverage borrow the same owned source; this is
/// distinct authority, not a fictitious interior four-variable inclusion.
pub fn prepared_face_plane_curve(
    guide: &Curve,
    source: ValidatedFacePlaneSource<'_>,
    plane: &Plane,
    range: [f64; 2],
    options: &ValidatedCurveOptions,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<PreparedValidatedCurve<'static>, ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    let ValidatedFacePlaneSource {
        surface,
        anchor,
        endpoint_coordinate_planes,
    } = source;
    let interval = interval_range(range)?;
    if interval.lower >= interval.upper
        || options.max_depth > 48
        || options.max_iterations == 0
        || options.max_iterations > 128
        || !options.parameter_radius.is_finite()
        || options.parameter_radius <= 0.0
        || !options.endpoint_tolerance.is_finite()
        || options.endpoint_tolerance <= 0.0
        || anchor.is_some_and(|anchor| !anchor.finite())
        || endpoint_coordinate_planes
            .into_iter()
            .flatten()
            .any(|coordinate| {
                !coordinate.offset.is_finite()
                    || !coordinate.normal.into_iter().all(f64::is_finite)
                    || coordinate.normal.into_iter().all(|value| value == 0.0)
            })
    {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    let guide = if let Curve::IntersectionCurve(curve) = guide {
        curve.leader().as_ref()
    } else {
        guide
    };
    let native = guide
        .try_range_tuple()
        .ok_or(ValidatedGeometryError::InvalidInput)?;
    if range[0] < native.0 || range[1] > native.1 {
        return Err(ValidatedGeometryError::ParameterOutOfRange);
    }
    let bytes = size_of::<Curve>()
        .checked_add(curve_owned_bytes(
            guide,
            options.max_retained_bytes,
            budget,
            cancel,
        )?)
        .and_then(|value| value.checked_add(size_of::<Surface>()))
        .ok_or(ValidatedGeometryError::ResourceLimit)?;
    let bytes = bytes
        .checked_add(surface_owned_bytes(
            surface,
            options.max_retained_bytes,
            budget,
            cancel,
        )?)
        .ok_or(ValidatedGeometryError::ResourceLimit)?;
    let mut bounded = *options;
    bounded.max_retained_bytes = bounded
        .max_retained_bytes
        .checked_sub(bytes)
        .ok_or(ValidatedGeometryError::ResourceLimit)?;
    let mut minimum = size_of::<PreparedValidatedCurve<'_>>();
    if matches!(surface, Surface::ExtrusionSurface(source)
        if extrusion::bounded_profile_domain(source.entity_curve()).is_ok())
        && matches!(anchor, Some(ValidatedFacePlaneSourceAnchor::Both { .. }))
    {
        minimum = minimum
            .checked_add(size_of::<extrusion::Graph>())
            .and_then(|bytes| bytes.checked_add(size_of::<ValidatedCurvePiece>()))
            .ok_or(ValidatedGeometryError::ResourceLimit)?;
    }
    if bounded.max_retained_bytes < minimum {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    budget.charge(4, cancel)?;
    let owned = Curve::IntersectionCurve(SurfaceCurve::with_boundaries(
        Box::new(surface.clone()),
        Box::new(Surface::Plane(*plane)),
        Box::new(clone_coordinate_guide(guide, 0, budget, cancel)?),
        None,
        None,
    ));
    // Preserve the generic source/range admission before supplying the
    // generated query's numerical proposal to the unchanged continuation.
    curve_owned_bytes(&owned, usize::MAX, budget, cancel)?;
    validate_preparation_range(&owned, range, &bounded, false, 0, budget, cancel)?;
    if let Some((graph, pieces)) =
        extrusion::prepare(&owned, range, anchor, &bounded, budget, cancel)?
    {
        let bytes = bytes
            .checked_add(size_of::<extrusion::Graph>())
            .ok_or(ValidatedGeometryError::ResourceLimit)?;
        budget.charge(2, cancel)?;
        return Ok(PreparedValidatedCurve {
            source: PreparedSource::Owned {
                curve: Box::new(owned),
                bytes,
                authority: OwnedSourceAuthority::Extrusion(Box::new(graph)),
            },
            range,
            pieces,
            options: *options,
            mapped: None,
        });
    }
    let preparation = SourcePreparation {
        anchor,
        endpoint_coordinate_planes,
        coalesce_coordinates: true,
    };
    let prepared =
        match prepare_unmapped_curve(&owned, range, &bounded, preparation, budget, cancel) {
            Ok(prepared) => prepared,
            Err(error)
                if retryable(error)
                    || matches!(
                        error,
                        ValidatedGeometryError::AmbiguousBranch | ValidatedGeometryError::NonSmooth
                    ) =>
            {
                prepare_unmapped_curve(
                    &owned,
                    range,
                    &bounded,
                    SourcePreparation {
                        coalesce_coordinates: false,
                        ..preparation
                    },
                    budget,
                    cancel,
                )?
            }
            Err(error) => return Err(error),
        };
    let PreparedValidatedCurve {
        source,
        pieces,
        mapped,
        ..
    } = prepared;
    drop(mapped);
    drop(source);
    budget.charge(1, cancel)?;
    Ok(PreparedValidatedCurve {
        source: PreparedSource::Owned {
            curve: Box::new(owned),
            bytes,
            authority: OwnedSourceAuthority::Original,
        },
        range,
        pieces,
        options: *options,
        mapped: None,
    })
}

fn clone_coordinate_guide(
    curve: &Curve,
    depth: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Curve, ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    if depth > crate::geometry::MAX_NATIVE_GEOMETRY_NESTING {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    match curve {
        Curve::IntersectionCurve(curve) => {
            clone_coordinate_guide(curve.leader(), depth + 1, budget, cancel)
        }
        Curve::TrimmedCurve(wrapper) => {
            let basis =
                clone_coordinate_guide(wrapper.entity().curve(), depth + 1, budget, cancel)?;
            budget.charge(1, cancel)?;
            Ok(Curve::TrimmedCurve(wrapper.map_ref(|trim| {
                TrimmedCurve::new(Box::new(basis), trim.range())
            })))
        }
        _ => Ok(curve.clone()),
    }
}

#[derive(Clone, Copy)]
struct System {
    value: [ValidatedInterval; 4],
    jacobian: [[ValidatedInterval; 4]; 4],
    surfaces: [ValidatedSurfaceJet; 2],
    plane: PlaneJet,
}
struct Tube {
    authority: TubeAuthority,
    domain: [ValidatedInterval; 4],
    rooted_domain: [ValidatedInterval; 4],
}

// Proof alternatives stay on the stack; boxing would allocate on every query.
#[allow(clippy::large_enum_variant)]
enum TubeAuthority {
    Krawczyk {
        system: System,
        inverse: [[f64; 4]; 4],
        defect: [[ValidatedInterval; 4]; 4],
        rho: f64,
    },
    Constructive(ValidatedCurveJet),
}
impl Tube {
    fn position(&self) -> Result<[ValidatedInterval; 3], ValidatedGeometryError> {
        match &self.authority {
            TubeAuthority::Krawczyk { system, .. } => {
                intersect3(system.surfaces[0].position, system.surfaces[1].position)
            }
            TubeAuthority::Constructive(jet) => Ok(jet.position),
        }
    }
    fn contraction_bound(&self) -> f64 {
        match &self.authority {
            TubeAuthority::Krawczyk { rho, .. } => *rho,
            TubeAuthority::Constructive(_) => 0.0,
        }
    }
    fn surface_residual(&self) -> [ValidatedInterval; 3] {
        match &self.authority {
            TubeAuthority::Krawczyk { system, .. } => {
                subtract3(system.surfaces[0].position, system.surfaces[1].position)
            }
            TubeAuthority::Constructive(_) => [ZERO; 3],
        }
    }
}

fn carriers(curve: &Curve) -> Result<(&Surface, &Surface, &Curve), ValidatedGeometryError> {
    match curve {
        Curve::IntersectionCurve(curve) => Ok((
            curve.surface0().as_ref(),
            curve.surface1().as_ref(),
            curve.leader().as_ref(),
        )),
        _ => Err(ValidatedGeometryError::Unsupported),
    }
}

fn system(
    curve: &Curve,
    projection: Projection,
    parameter: ValidatedInterval,
    domain: [ValidatedInterval; 4],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<System, ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    let (surface0, surface1, leader) = carriers(curve)?;
    let plane_jet = projection_jet(leader, projection, parameter, budget, cancel)?;
    let surface0 = validated_surface_piecewise_jet(
        surface0,
        [
            [domain[0].lower, domain[0].upper],
            [domain[1].lower, domain[1].upper],
        ],
        budget,
        cancel,
    )?;
    let surface1 = validated_surface_piecewise_jet(
        surface1,
        [
            [domain[2].lower, domain[2].upper],
            [domain[3].lower, domain[3].upper],
        ],
        budget,
        cancel,
    )?;
    let difference = subtract3(surface0.position, surface1.position);
    let point = scale3(add3(surface0.position, surface1.position), 0.5);
    let plane = projection_residual(projection, parameter, plane_jet, point);
    let columns = [
        surface0.du,
        surface0.dv,
        negate3(surface1.du),
        negate3(surface1.dv),
    ];
    let chart_columns = [surface0.du, surface0.dv, surface1.du, surface1.dv];
    let jacobian = std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            if row < 3 {
                columns[column][row]
            } else {
                dot3(plane_jet.normal, scale3(chart_columns[column], 0.5))
            }
        })
    });
    let value = [difference[0], difference[1], difference[2], plane];
    if !value.into_iter().all(ValidatedInterval::is_finite)
        || !jacobian
            .into_iter()
            .flatten()
            .all(ValidatedInterval::is_finite)
    {
        return Err(ValidatedGeometryError::ConditioningFailure);
    }
    Ok(System {
        value,
        jacobian,
        surfaces: [surface0, surface1],
        plane: plane_jet,
    })
}

fn certify(
    curve: &Curve,
    projection: Projection,
    parameter: ValidatedInterval,
    domain: [ValidatedInterval; 4],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Tube, ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    if !domain.into_iter().all(ValidatedInterval::is_finite) {
        return Err(ValidatedGeometryError::ConditioningFailure);
    }
    let enclosure = system(curve, projection, parameter, domain, budget, cancel)?;
    let middle = domain.map(ValidatedInterval::midpoint);
    let center = system(
        curve,
        projection,
        parameter,
        middle.map(ValidatedInterval::point),
        budget,
        cancel,
    )?;
    let inverse = inverse4(
        enclosure
            .jacobian
            .map(|row| row.map(ValidatedInterval::midpoint)),
    )?;
    let defect = defect_matrix(inverse, enclosure.jacobian);
    let rho = matrix_norm(defect);
    if !rho.is_finite() || rho >= 1.0 {
        return Err(ValidatedGeometryError::ConditioningFailure);
    }
    let correction = matrix_vector(inverse, center.value);
    let delta = std::array::from_fn(|i| domain[i] - ValidatedInterval::point(middle[i]));
    let error = interval_matrix_vector(defect, delta);
    let krawczyk: [ValidatedInterval; 4] =
        std::array::from_fn(|i| ValidatedInterval::point(middle[i]) - correction[i] + error[i]);
    if !(0..4).all(|i| domain[i].strictly_contains(krawczyk[i])) {
        return Err(ValidatedGeometryError::Unresolved);
    }
    let n0 = cross3(enclosure.surfaces[0].du, enclosure.surfaces[0].dv);
    let n1 = cross3(enclosure.surfaces[1].du, enclosure.surfaces[1].dv);
    let tangent = cross3(n0, n1);
    let transverse = tangent
        .map(ValidatedInterval::square)
        .into_iter()
        .fold(ZERO, |a, b| a + b);
    if transverse.lower <= 0.0 {
        return Err(ValidatedGeometryError::NonTransverse);
    }
    // Inclusion encloses the exact branch in the Krawczyk image as well.
    let rooted = system(curve, projection, parameter, krawczyk, budget, cancel)?;
    Ok(Tube {
        authority: TubeAuthority::Krawczyk {
            system: rooted,
            inverse,
            defect,
            rho,
        },
        domain,
        rooted_domain: krawczyk,
    })
}

fn root_image_padding(image: ValidatedInterval) -> f64 {
    64.0 * f64::EPSILON * (1.0 + image.magnitude())
}

fn root_image_domain(tube: &Tube) -> Result<[ValidatedInterval; 4], ValidatedGeometryError> {
    if matches!(tube.authority, TubeAuthority::Constructive(_)) {
        return Ok(tube.rooted_domain);
    }
    let mut next = tube.domain;
    for (i, bound) in next.iter_mut().enumerate() {
        let image = tube.rooted_domain[i];
        let radius = 0.51 * image.width() + root_image_padding(image);
        *bound = tube.domain[i]
            .intersection(inflated(image.midpoint(), radius))
            .ok_or(ValidatedGeometryError::ConditioningFailure)?;
    }
    Ok(next)
}

// A centered domain is only a proposal. Fresh inclusion inside the previous
// uniqueness domain must prove the identical whole-parameter root inventory.
fn centered_root_domain(tube: &Tube) -> [ValidatedInterval; 4] {
    std::array::from_fn(|i| {
        let image = tube.rooted_domain[i];
        let radius = (tube.domain[i].width() * 0.25)
            .min(image.width() * 0.51)
            .max(root_image_padding(image));
        let centered = inflated(image.midpoint(), radius);
        ValidatedInterval {
            lower: centered.lower.max(tube.domain[i].lower),
            upper: centered.upper.min(tube.domain[i].upper),
        }
    })
}

fn refine_implicit_jet(
    source: Certification<'_, '_>,
    projection: Projection,
    parameter: ValidatedInterval,
    domain: [ValidatedInterval; 4],
    options: &ValidatedCurveOptions,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<ValidatedCurveJet, ValidatedGeometryError> {
    let mut tube = source.certify(projection, parameter, domain, budget, cancel)?;
    let mut jet = implicit_jet(&tube)?;
    // Constructive charts already evaluate their prescribed source range;
    // there is no implicit uniqueness domain to contract.
    if matches!(tube.authority, TubeAuthority::Constructive(_)) {
        return Ok(jet);
    }
    for _ in 1..options.max_iterations {
        budget.charge(1, cancel)?;
        let next = root_image_domain(&tube)?;
        let proposed = centered_root_domain(&tube);
        let mut refined = None;
        // Near-unit contraction can leave one nonlinear chart coordinate broad
        // even when the requested parameter span is tiny. Tighten coordinates
        // separately: another coordinate may genuinely vary over the full span.
        // No centered proposal, guide, endpoint or midpoint supplies authority.
        for axis in 0..4 {
            let padding = root_image_padding(tube.rooted_domain[axis]);
            if proposed[axis].lower - next[axis].lower <= padding
                && next[axis].upper - proposed[axis].upper <= padding
            {
                continue;
            }
            let mut candidate = next;
            candidate[axis] = proposed[axis];
            match source.certify(projection, parameter, candidate, budget, cancel) {
                Ok(tube) => {
                    refined = Some(tube);
                    break;
                }
                Err(error) if retryable(error) => {}
                Err(error) => return Err(error),
            }
        }
        let refined = match refined {
            Some(tube) => tube,
            None => {
                // An unsuccessful proposal cannot discard the old valid jet.
                // Retain the image-containing contraction when it progresses.
                if !(0..4).any(|i| {
                    let padding = root_image_padding(tube.rooted_domain[i]);
                    next[i].lower - tube.domain[i].lower > padding
                        || tube.domain[i].upper - next[i].upper > padding
                }) {
                    break;
                }
                match source.certify(projection, parameter, next, budget, cancel) {
                    Ok(tube) => tube,
                    Err(error) if retryable(error) => break,
                    Err(error) => return Err(error),
                }
            }
        };
        let refined_jet = match implicit_jet(&refined) {
            Ok(jet) => jet,
            Err(error) if retryable(error) => break,
            Err(error) => return Err(error),
        };
        jet.position = intersect3(jet.position, refined_jet.position)?;
        jet.first = intersect3(jet.first, refined_jet.first)?;
        jet.second = intersect3(jet.second, refined_jet.second)?;
        tube = refined;
    }
    Ok(jet)
}

fn refine_endpoint(
    source: Certification<'_, '_>,
    projection: Projection,
    parameter: ValidatedInterval,
    domain: [ValidatedInterval; 4],
    options: &ValidatedCurveOptions,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<([ValidatedInterval; 4], [ValidatedInterval; 3]), ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    let mut tube = source.certify(projection, parameter, domain, budget, cancel)?;
    for iteration in 0..options.max_iterations {
        // This box contains the proven image and stays inside its owned
        // uniqueness tube. Preserve the existing arithmetic padding floor
        // while nonlinear coordinates are still contracting.
        let enclosed = std::array::from_fn(|i| {
            let image = tube.rooted_domain[i];
            let padding = (tube.domain[i].width() * 0.0625).max(root_image_padding(image));
            ValidatedInterval {
                lower: (image.lower - padding)
                    .next_down()
                    .max(tube.domain[i].lower),
                upper: (image.upper + padding).next_up().min(tube.domain[i].upper),
            }
        });
        let position = tube.position()?;
        if position
            .into_iter()
            .all(|interval| interval.width() <= options.endpoint_tolerance)
        {
            return Ok((enclosed, position));
        }
        if iteration + 1 == options.max_iterations {
            break;
        }
        budget.charge(1, cancel)?;
        // A nearly unit contraction bound can leave a nonlinear coordinate
        // broad while affine coordinates reach arithmetic width. Propose a
        // smaller centered box instead of assuming the old image must fit it.
        // Fresh inclusion for the identical parameter interval proves roots
        // inside the old uniqueness tube, hence the SAME selected branch.
        let proposed = centered_root_domain(&tube);
        tube = match source.certify(projection, parameter, proposed, budget, cancel) {
            Ok(next) => next,
            // The proposal carries no authority. Its refusal retains the
            // original image-containing contraction, without refunding work.
            Err(error) if retryable(error) => {
                source.certify(projection, parameter, enclosed, budget, cancel)?
            }
            Err(error) => return Err(error),
        };
    }
    Err(ValidatedGeometryError::Unresolved)
}

fn implicit_jet(tube: &Tube) -> Result<ValidatedCurveJet, ValidatedGeometryError> {
    let system = match &tube.authority {
        TubeAuthority::Krawczyk { system, .. } => system,
        TubeAuthority::Constructive(jet) => return Ok(*jet),
    };
    let [s0, s1] = system.surfaces;
    let plane = system.plane;
    let position = intersect3(s0.position, s1.position)?;
    let ft = dot3(plane.first, position) - plane.offset_first;
    let q1 = solve_interval(tube, [ZERO, ZERO, ZERO, -ft])?;
    let first0 = add3(multiply3(s0.du, q1[0]), multiply3(s0.dv, q1[1]));
    let first1 = add3(multiply3(s1.du, q1[2]), multiply3(s1.dv, q1[3]));
    let first = intersect3(first0, first1)?;
    let h0 = second_composition(s0, q1[0], q1[1]);
    let h1 = second_composition(s1, q1[2], q1[3]);
    let hposition = scale3(add3(h0, h1), 0.5);
    let ftt = dot3(plane.second, position) - plane.offset_second;
    let mixed = dot3(plane.first, first) * ValidatedInterval::point(2.0);
    let last = ftt + mixed + dot3(plane.normal, hposition);
    let rhs = subtract3(h1, h0);
    let q2 = solve_interval(tube, [rhs[0], rhs[1], rhs[2], -last])?;
    let second0 = add3(add3(multiply3(s0.du, q2[0]), multiply3(s0.dv, q2[1])), h0);
    let second1 = add3(add3(multiply3(s1.du, q2[2]), multiply3(s1.dv, q2[3])), h1);
    let second = intersect3(second0, second1)?;
    if !position
        .into_iter()
        .chain(first)
        .chain(second)
        .all(ValidatedInterval::is_finite)
    {
        return Err(ValidatedGeometryError::ConditioningFailure);
    }
    Ok(ValidatedCurveJet {
        position,
        first,
        second,
        third: [ValidatedInterval {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
        }; 3],
    })
}

fn intersect3(
    a: [ValidatedInterval; 3],
    b: [ValidatedInterval; 3],
) -> Result<[ValidatedInterval; 3], ValidatedGeometryError> {
    let mut result = [ZERO; 3];
    for i in 0..3 {
        result[i] = a[i]
            .intersection(b[i])
            .ok_or(ValidatedGeometryError::ConditioningFailure)?;
    }
    Ok(result)
}

fn second_composition(
    surface: ValidatedSurfaceJet,
    u: ValidatedInterval,
    v: ValidatedInterval,
) -> [ValidatedInterval; 3] {
    add3(
        add3(
            multiply3(surface.duu, u.square()),
            multiply3(surface.duv, u * v * ValidatedInterval::point(2.0)),
        ),
        multiply3(surface.dvv, v.square()),
    )
}

fn solve_interval(
    tube: &Tube,
    rhs: [ValidatedInterval; 4],
) -> Result<[ValidatedInterval; 4], ValidatedGeometryError> {
    let TubeAuthority::Krawczyk {
        inverse,
        defect,
        rho,
        ..
    } = &tube.authority
    else {
        return Err(ValidatedGeometryError::InvalidInput);
    };
    let base = matrix_vector(*inverse, rhs);
    let maximum = base
        .into_iter()
        .map(ValidatedInterval::magnitude)
        .fold(0.0, f64::max);
    let correction = (ValidatedInterval::point(*rho)
        * ValidatedInterval::point(maximum)
        * (ONE - ValidatedInterval::point(*rho)).reciprocal()?)
    .upper;
    let error = ValidatedInterval {
        lower: -correction,
        upper: correction,
    };
    let mut result = base.map(|entry| entry + error);
    for _ in 0..3 {
        let residual = interval_matrix_vector(*defect, result);
        result = std::array::from_fn(|i| {
            result[i]
                .intersection(base[i] + residual[i])
                .unwrap_or(result[i])
        });
    }
    if result.into_iter().all(ValidatedInterval::is_finite) {
        Ok(result)
    } else {
        Err(ValidatedGeometryError::ConditioningFailure)
    }
}

fn numerical_root(
    curve: &Curve,
    projection: Projection,
    parameter: f64,
    mut center: [f64; 4],
    options: &ValidatedCurveOptions,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<[f64; 4], ValidatedGeometryError> {
    for _ in 0..options.max_iterations {
        budget.charge(1, cancel)?;
        let value = system(
            curve,
            projection,
            ValidatedInterval::point(parameter),
            center.map(ValidatedInterval::point),
            budget,
            cancel,
        )?;
        let inverse = inverse4(
            value
                .jacobian
                .map(|row| row.map(ValidatedInterval::midpoint)),
        )?;
        let delta: [f64; 4] = std::array::from_fn(|i| {
            (0..4)
                .map(|j| inverse[i][j] * value.value[j].midpoint())
                .sum()
        });
        center = std::array::from_fn(|i| center[i] - delta[i]);
        if !center.into_iter().all(f64::is_finite) {
            return Err(ValidatedGeometryError::ConditioningFailure);
        }
        if delta.into_iter().map(f64::abs).fold(0.0, f64::max) < 1.0e-12 {
            break;
        }
    }
    Ok(center)
}

fn nearest_seed(
    surface: &Surface,
    point: [f64; 3],
    options: &ValidatedCurveOptions,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<[f64; 2], ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    if let Surface::Plane(plane) = surface {
        let origin = plane.origin();
        let a = plane.axis_u();
        let b = plane.axis_v();
        let residual: [f64; 3] = std::array::from_fn(|i| point[i] - origin[i]);
        let aa: f64 = (0..3).map(|i| a[i] * a[i]).sum();
        let ab: f64 = (0..3).map(|i| a[i] * b[i]).sum();
        let bb: f64 = (0..3).map(|i| b[i] * b[i]).sum();
        let ar: f64 = (0..3).map(|i| a[i] * residual[i]).sum();
        let br: f64 = (0..3).map(|i| b[i] * residual[i]).sum();
        let determinant = aa * bb - ab * ab;
        if !determinant.is_finite() || determinant <= 0.0 {
            return Err(ValidatedGeometryError::Singular);
        }
        let result = [
            (bb * ar - ab * br) / determinant,
            (aa * br - ab * ar) / determinant,
        ];
        return if result.into_iter().all(f64::is_finite) {
            Ok(result)
        } else {
            Err(ValidatedGeometryError::ConditioningFailure)
        };
    }
    let ranges = surface.try_range_tuple();
    let u = ranges.0.ok_or(ValidatedGeometryError::Unsupported)?;
    let v = ranges.1.ok_or(ValidatedGeometryError::Unsupported)?;
    if ![u.0, u.1, v.0, v.1].into_iter().all(f64::is_finite) || u.0 >= u.1 || v.0 >= v.1 {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    let mut best: Option<(f64, [f64; 2])> = None;
    for i in 0..=4 {
        for j in 0..=8 {
            budget.charge(1, cancel)?;
            let mut uv = [
                u.0 * (1.0 - i as f64 / 4.0) + u.1 * (i as f64 / 4.0),
                v.0 * (1.0 - j as f64 / 8.0) + v.1 * (j as f64 / 8.0),
            ];
            for _ in 0..options.max_iterations.min(8) {
                budget.charge(1, cancel)?;
                let jet = match validated_surface_piecewise_jet(
                    surface,
                    [[uv[0], uv[0]], [uv[1], uv[1]]],
                    budget,
                    cancel,
                ) {
                    Ok(value) => value,
                    Err(error) if retryable(error) => break,
                    Err(error) => return Err(error),
                };
                let residual: [f64; 3] =
                    std::array::from_fn(|k| jet.position[k].midpoint() - point[k]);
                let distance: f64 = residual.into_iter().map(|r| r * r).sum();
                if distance.is_finite() && best.is_none_or(|(previous, _)| distance < previous) {
                    best = Some((distance, uv));
                }
                let a = jet.du.map(ValidatedInterval::midpoint);
                let b = jet.dv.map(ValidatedInterval::midpoint);
                let aa: f64 = a.into_iter().map(|x| x * x).sum();
                let ab: f64 = (0..3).map(|k| a[k] * b[k]).sum();
                let bb: f64 = b.into_iter().map(|x| x * x).sum();
                let ar: f64 = (0..3).map(|k| a[k] * residual[k]).sum();
                let br: f64 = (0..3).map(|k| b[k] * residual[k]).sum();
                let determinant = aa * bb - ab * ab;
                if !determinant.is_finite() || determinant <= 0.0 {
                    break;
                }
                uv = [
                    uv[0] - (bb * ar - ab * br) / determinant,
                    uv[1] - (aa * br - ab * ar) / determinant,
                ];
                if !uv.into_iter().all(f64::is_finite) {
                    break;
                }
            }
        }
    }
    best.map(|(_, uv)| uv)
        .ok_or(ValidatedGeometryError::Singular)
}

fn inverse4(matrix: [[f64; 4]; 4]) -> Result<[[f64; 4]; 4], ValidatedGeometryError> {
    let mut augmented: [[f64; 8]; 4] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if j < 4 {
                matrix[i][j]
            } else if i == j - 4 {
                1.0
            } else {
                0.0
            }
        })
    });
    for column in 0..4 {
        let pivot = (column..4)
            .max_by(|&a, &b| {
                augmented[a][column]
                    .abs()
                    .total_cmp(&augmented[b][column].abs())
            })
            .ok_or(ValidatedGeometryError::Singular)?;
        if !augmented[pivot][column].is_finite() || augmented[pivot][column] == 0.0 {
            return Err(ValidatedGeometryError::Singular);
        }
        augmented.swap(column, pivot);
        let scale = augmented[column][column];
        let (before, tail) = augmented.split_at_mut(column);
        let (pivot_rows, after) = tail.split_at_mut(1);
        let pivot_row = &mut pivot_rows[0];
        for value in pivot_row.iter_mut() {
            *value /= scale;
        }
        for row in before.iter_mut().chain(after.iter_mut()) {
            let factor = row[column];
            for (value, pivot_value) in row.iter_mut().zip(pivot_row.iter()) {
                *value -= factor * *pivot_value;
            }
        }
    }
    let result: [[f64; 4]; 4] =
        std::array::from_fn(|i| std::array::from_fn(|j| augmented[i][j + 4]));
    if result.into_iter().flatten().all(f64::is_finite) {
        Ok(result)
    } else {
        Err(ValidatedGeometryError::ConditioningFailure)
    }
}
fn defect_matrix(
    inverse: [[f64; 4]; 4],
    matrix: [[ValidatedInterval; 4]; 4],
) -> [[ValidatedInterval; 4]; 4] {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let product = (0..4)
                .map(|k| ValidatedInterval::point(inverse[i][k]) * matrix[k][j])
                .fold(ZERO, |a, b| a + b);
            ValidatedInterval::point(if i == j { 1.0 } else { 0.0 }) - product
        })
    })
}
fn matrix_norm(matrix: [[ValidatedInterval; 4]; 4]) -> f64 {
    matrix
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|entry| ValidatedInterval::point(entry.magnitude()))
                .fold(ZERO, |a, b| a + b)
                .upper
        })
        .fold(0.0, f64::max)
}
fn matrix_vector(matrix: [[f64; 4]; 4], vector: [ValidatedInterval; 4]) -> [ValidatedInterval; 4] {
    std::array::from_fn(|i| {
        (0..4)
            .map(|j| ValidatedInterval::point(matrix[i][j]) * vector[j])
            .fold(ZERO, |a, b| a + b)
    })
}
fn interval_matrix_vector(
    matrix: [[ValidatedInterval; 4]; 4],
    vector: [ValidatedInterval; 4],
) -> [ValidatedInterval; 4] {
    std::array::from_fn(|i| {
        (0..4)
            .map(|j| matrix[i][j] * vector[j])
            .fold(ZERO, |a, b| a + b)
    })
}
fn spatial_point(
    enclosure: [ValidatedInterval; 3],
    tolerance: f64,
) -> Result<ValidatedCurvePoint, ValidatedGeometryError> {
    if enclosure
        .into_iter()
        .all(|entry| entry.is_finite() && entry.width() <= tolerance)
    {
        Ok(ValidatedCurvePoint {
            position: Point3::new(
                enclosure[0].midpoint(),
                enclosure[1].midpoint(),
                enclosure[2].midpoint(),
            ),
            error: enclosure,
        })
    } else {
        Err(ValidatedGeometryError::Unresolved)
    }
}
fn retryable(error: ValidatedGeometryError) -> bool {
    use ValidatedGeometryError as E;
    match error {
        E::Unresolved
        | E::ConditioningFailure
        | E::NonTransverse
        | E::Singular
        | E::DenominatorContainsZero
        | E::ParameterOutOfRange => true,
        // A completed coverage operation already exhausted its bounded proof;
        // starting a new local branch search cannot repair that certificate.
        E::PlaneCoverageUnresolved { .. }
        | E::Cancelled
        | E::ResourceLimit
        | E::AmbiguousBranch
        | E::InvalidInput
        | E::Unsupported
        | E::NonSmooth => false,
    }
}
fn inflated(center: f64, radius: f64) -> ValidatedInterval {
    ValidatedInterval {
        lower: (center - radius).next_down(),
        upper: (center + radius).next_up(),
    }
}
fn flatten(parameters: [[ValidatedInterval; 2]; 2]) -> [ValidatedInterval; 4] {
    [
        parameters[0][0],
        parameters[0][1],
        parameters[1][0],
        parameters[1][1],
    ]
}
fn unflatten(parameters: [ValidatedInterval; 4]) -> [[ValidatedInterval; 2]; 2] {
    [
        [parameters[0], parameters[1]],
        [parameters[2], parameters[3]],
    ]
}
fn hull_jet(a: ValidatedCurveJet, b: ValidatedCurveJet) -> ValidatedCurveJet {
    ValidatedCurveJet {
        position: std::array::from_fn(|i| a.position[i].hull(b.position[i])),
        first: std::array::from_fn(|i| a.first[i].hull(b.first[i])),
        second: std::array::from_fn(|i| a.second[i].hull(b.second[i])),
        third: std::array::from_fn(|i| a.third[i].hull(b.third[i])),
    }
}
fn add3(a: [ValidatedInterval; 3], b: [ValidatedInterval; 3]) -> [ValidatedInterval; 3] {
    std::array::from_fn(|i| a[i] + b[i])
}
fn subtract3(a: [ValidatedInterval; 3], b: [ValidatedInterval; 3]) -> [ValidatedInterval; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}
fn negate3(a: [ValidatedInterval; 3]) -> [ValidatedInterval; 3] { a.map(|entry| -entry) }
fn multiply3(a: [ValidatedInterval; 3], b: ValidatedInterval) -> [ValidatedInterval; 3] {
    a.map(|entry| entry * b)
}
fn scale3(a: [ValidatedInterval; 3], b: f64) -> [ValidatedInterval; 3] {
    multiply3(a, ValidatedInterval::point(b))
}
fn dot3(a: [ValidatedInterval; 3], b: [ValidatedInterval; 3]) -> ValidatedInterval {
    (0..3).map(|i| a[i] * b[i]).fold(ZERO, |a, b| a + b)
}
fn cross3(a: [ValidatedInterval; 3], b: [ValidatedInterval; 3]) -> [ValidatedInterval; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn check_leader_knots(
    curve: &Curve,
    range: ValidatedInterval,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<(), ValidatedGeometryError> {
    check_curve_knots(curve, range, 0, budget, cancel)
}

fn reflected_trim_range(
    orientation: bool,
    domain: (f64, f64),
    range: ValidatedInterval,
) -> Result<ValidatedInterval, ValidatedGeometryError> {
    Ok(processed_parameter(orientation, Some(domain), Taylor::variable(range, 0))?.0[0][0])
}

fn check_curve_knots(
    curve: &Curve,
    range: ValidatedInterval,
    depth: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<(), ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    if depth > crate::geometry::MAX_NATIVE_GEOMETRY_NESTING {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    match curve {
        Curve::Line(_) => Ok(()),
        Curve::BsplineCurve(curve) => check_knots(
            curve.knot_vector(),
            curve.control_points().len(),
            range,
            budget,
            cancel,
        ),
        Curve::NurbsCurve(curve) => check_knots(
            curve.knot_vector(),
            curve.control_points().len(),
            range,
            budget,
            cancel,
        ),
        Curve::TrimmedCurve(curve) => check_curve_knots(
            curve.entity().curve(),
            reflected_trim_range(curve.orientation(), curve.entity().range(), range)?,
            depth + 1,
            budget,
            cancel,
        ),
        Curve::ParameterCurve(curve) => {
            check_curve2d_knots(curve.curve(), range, depth + 1, budget, cancel)?;
            match curve.surface().as_ref() {
                Surface::Plane(_) | Surface::SphericalSurface(_) | Surface::ToroidalSurface(_) => {
                    Ok(())
                }
                _ => Err(ValidatedGeometryError::Unsupported),
            }
        }
        Curve::IntersectionCurve(_) => Err(ValidatedGeometryError::Unsupported),
    }
}

fn check_curve2d_knots(
    curve: &Curve2D,
    range: ValidatedInterval,
    depth: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<(), ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    if depth > crate::geometry::MAX_NATIVE_GEOMETRY_NESTING {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    match curve {
        Curve2D::Line(_) | Curve2D::Conic(_) => Ok(()),
        Curve2D::BsplineCurve(curve) => check_knots(
            curve.knot_vector(),
            curve.control_points().len(),
            range,
            budget,
            cancel,
        ),
        Curve2D::NurbsCurve(curve) => check_knots(
            curve.knot_vector(),
            curve.control_points().len(),
            range,
            budget,
            cancel,
        ),
        Curve2D::Polyline(_) => Err(ValidatedGeometryError::NonSmooth),
        Curve2D::TrimmedCurve(curve) => check_curve2d_knots(
            curve.entity().curve(),
            reflected_trim_range(curve.orientation(), curve.entity().range(), range)?,
            depth + 1,
            budget,
            cancel,
        ),
    }
}

fn check_knots(
    knots: &KnotVector,
    count: usize,
    range: ValidatedInterval,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<(), ValidatedGeometryError> {
    if count == 0 || knots.len() <= count {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    let degree = knots.len() - count - 1;
    let mut index = 0;
    while index < knots.len() {
        budget.charge(1, cancel)?;
        let start = index;
        let knot = knots[index];
        if !knot.is_finite() || index > 0 && knot < knots[index - 1] {
            return Err(ValidatedGeometryError::InvalidInput);
        }
        while index < knots.len() && knots[index] == knot {
            budget.charge(1, cancel)?;
            index += 1;
        }
        if knots[0] < knot
            && knot < knots[knots.len() - 1]
            && range.lower <= knot
            && knot <= range.upper
            && degree < index - start + 2
        {
            return Err(ValidatedGeometryError::NonSmooth);
        }
    }
    Ok(())
}

fn append_mapped_knot(
    value: f64,
    offset: ValidatedInterval,
    reversed: bool,
    range: [f64; 2],
    budget: &mut ValidatedGeometryBudget,
    append: &mut impl FnMut(f64, &mut ValidatedGeometryBudget) -> Result<(), ValidatedGeometryError>,
) -> Result<(), ValidatedGeometryError> {
    if !value.is_finite() {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    let mapped = if reversed {
        offset - ValidatedInterval::point(value)
    } else {
        ValidatedInterval::point(value) - offset
    };
    if !mapped.is_finite() {
        return Err(ValidatedGeometryError::ConditioningFailure);
    }
    if mapped.upper <= range[0] || mapped.lower >= range[1] {
        return Ok(());
    }
    if mapped.lower != mapped.upper {
        return Err(ValidatedGeometryError::NonSmooth);
    }
    append(mapped.lower, budget)
}

fn append_source_knots(
    knots: &KnotVector,
    offset: ValidatedInterval,
    reversed: bool,
    range: [f64; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
    append: &mut impl FnMut(f64, &mut ValidatedGeometryBudget) -> Result<(), ValidatedGeometryError>,
) -> Result<(), ValidatedGeometryError> {
    for index in 0..knots.len() {
        budget.charge(1, cancel)?;
        let index = if reversed {
            knots.len() - 1 - index
        } else {
            index
        };
        append_mapped_knot(knots[index], offset, reversed, range, budget, append)?;
    }
    Ok(())
}

/// Retained source-to-root scalar chart for knot traversal.
#[derive(Clone, Copy)]
struct KnotParameterMap {
    offset: ValidatedInterval,
    reversed: bool,
}

fn append_curve_knots(
    curve: &Curve,
    chart: KnotParameterMap,
    depth: usize,
    range: [f64; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
    append: &mut impl FnMut(f64, &mut ValidatedGeometryBudget) -> Result<(), ValidatedGeometryError>,
) -> Result<(), ValidatedGeometryError> {
    let KnotParameterMap { offset, reversed } = chart;
    budget.charge(1, cancel)?;
    if depth > crate::geometry::MAX_NATIVE_GEOMETRY_NESTING {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    match curve {
        Curve::Line(_) => Ok(()),
        Curve::BsplineCurve(curve) => append_source_knots(
            curve.knot_vector(),
            offset,
            reversed,
            range,
            budget,
            cancel,
            append,
        ),
        Curve::NurbsCurve(curve) => append_source_knots(
            curve.knot_vector(),
            offset,
            reversed,
            range,
            budget,
            cancel,
            append,
        ),
        Curve::ParameterCurve(curve) => append_curve2d_knots(
            curve.curve(),
            chart,
            depth + 1,
            range,
            budget,
            cancel,
            append,
        ),
        Curve::TrimmedCurve(curve) => {
            let (a, b) = curve.entity().range();
            let (offset, reversed) = if curve.orientation() {
                (offset, reversed)
            } else {
                (
                    ValidatedInterval::point(a) + ValidatedInterval::point(b) - offset,
                    !reversed,
                )
            };
            append_curve_knots(
                curve.entity().curve(),
                KnotParameterMap { offset, reversed },
                depth + 1,
                range,
                budget,
                cancel,
                append,
            )
        }
        Curve::IntersectionCurve(_) => Err(ValidatedGeometryError::Unsupported),
    }
}

fn append_curve2d_knots(
    curve: &Curve2D,
    chart: KnotParameterMap,
    depth: usize,
    range: [f64; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
    append: &mut impl FnMut(f64, &mut ValidatedGeometryBudget) -> Result<(), ValidatedGeometryError>,
) -> Result<(), ValidatedGeometryError> {
    let KnotParameterMap { offset, reversed } = chart;
    budget.charge(1, cancel)?;
    if depth > crate::geometry::MAX_NATIVE_GEOMETRY_NESTING {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    match curve {
        Curve2D::Line(_) | Curve2D::Conic(_) => Ok(()),
        Curve2D::BsplineCurve(curve) => append_source_knots(
            curve.knot_vector(),
            offset,
            reversed,
            range,
            budget,
            cancel,
            append,
        ),
        Curve2D::NurbsCurve(curve) => append_source_knots(
            curve.knot_vector(),
            offset,
            reversed,
            range,
            budget,
            cancel,
            append,
        ),
        Curve2D::Polyline(curve) => {
            for index in 0..curve.len() {
                budget.charge(1, cancel)?;
                let index = if reversed {
                    curve.len() - 1 - index
                } else {
                    index
                };
                append_mapped_knot(index as f64, offset, reversed, range, budget, append)?;
            }
            Ok(())
        }
        Curve2D::TrimmedCurve(curve) => {
            let (a, b) = curve.entity().range();
            let (offset, reversed) = if curve.orientation() {
                (offset, reversed)
            } else {
                (
                    ValidatedInterval::point(a) + ValidatedInterval::point(b) - offset,
                    !reversed,
                )
            };
            append_curve2d_knots(
                curve.entity().curve(),
                KnotParameterMap { offset, reversed },
                depth + 1,
                range,
                budget,
                cancel,
                append,
            )
        }
    }
}

fn projection_jet(
    leader: &Curve,
    projection: Projection,
    parameter: ValidatedInterval,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<PlaneJet, ValidatedGeometryError> {
    match projection {
        Projection::Native => {
            let guide = validated_projection_leader_jet(
                leader,
                [parameter.lower, parameter.upper],
                budget,
                cancel,
            )?;
            Ok(PlaneJet {
                normal: guide.first,
                first: guide.second,
                second: guide.third,
                offset: dot3(guide.first, guide.position),
                offset_first: dot3(guide.second, guide.position) + dot3(guide.first, guide.first),
                offset_second: dot3(guide.third, guide.position)
                    + dot3(guide.first, guide.second) * ValidatedInterval::point(3.0),
            })
        }
        Projection::SourceProfile => Err(ValidatedGeometryError::InvalidInput),
        Projection::BetweenPlanes {
            parameter: range,
            planes: [first, last],
        } => {
            let inverse = (ValidatedInterval::point(range[1]) - ValidatedInterval::point(range[0]))
                .reciprocal()?;
            let ratio = (parameter - ValidatedInterval::point(range[0])) * inverse;
            let delta: [ValidatedInterval; 3] = std::array::from_fn(|i| {
                ValidatedInterval::point(last.normal[i]) - ValidatedInterval::point(first.normal[i])
            });
            let offset_delta =
                ValidatedInterval::point(last.offset) - ValidatedInterval::point(first.offset);
            Ok(PlaneJet {
                normal: std::array::from_fn(|i| {
                    ValidatedInterval::point(first.normal[i]) + ratio * delta[i]
                }),
                first: delta.map(|entry| entry * inverse),
                second: [ZERO; 3],
                offset: ValidatedInterval::point(first.offset) + ratio * offset_delta,
                offset_first: offset_delta * inverse,
                offset_second: ZERO,
            })
        }
    }
}

fn projection_residual(
    projection: Projection,
    parameter: ValidatedInterval,
    jet: PlaneJet,
    point: [ValidatedInterval; 3],
) -> ValidatedInterval {
    let Projection::BetweenPlanes {
        parameter: range,
        planes: [first, last],
    } = projection
    else {
        return dot3(jet.normal, point) - jet.offset;
    };
    let value = |plane: CoordinatePlane| {
        dot3(plane.normal.map(ValidatedInterval::point), point)
            - ValidatedInterval::point(plane.offset)
    };
    if parameter.lower == parameter.upper {
        if parameter.lower == range[0] {
            return value(first);
        }
        if parameter.lower == range[1] {
            return value(last);
        }
    }
    // The SAME interpolation parameter multiplies normal and offset. Form
    // their combined coefficient before multiplying its interval; otherwise
    // a short path with a changed coordinate normal can incur an irreducible
    // overestimate proportional to its absolute world position.
    value(first)
        + (parameter - ValidatedInterval::point(range[0]))
            * (dot3(jet.first, point) - jet.offset_first)
}

fn coordinate_spans(
    leader: &Curve,
    range: [f64; 2],
    memory_limit: usize,
    source_preparation: &SourcePreparation,
    max_depth: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<(CoordinateSpans, bool), ValidatedGeometryError> {
    let endpoint_coordinate_planes = &source_preparation.endpoint_coordinate_planes;
    match check_leader_knots(leader, interval_range(range)?, budget, cancel) {
        Ok(()) if endpoint_coordinate_planes.iter().all(Option::is_none) => {
            budget.charge(1, cancel)?;
            let mut result = Vec::new();
            reserve_native_vec(&mut result, memory_limit, 0, budget, cancel)?;
            result.push((range, Projection::Native));
            return Ok((result, false));
        }
        Err(ValidatedGeometryError::NonSmooth) => {}
        Ok(()) => {}
        Err(error) => return Err(error),
    }
    budget.charge(1, cancel)?;
    let mut knots = Vec::new();
    reserve_native_vec(&mut knots, memory_limit, 0, budget, cancel)?;
    knots.push(range[0]);
    let mut append =
        |value: f64, budget: &mut ValidatedGeometryBudget| -> Result<(), ValidatedGeometryError> {
            budget.charge(1, cancel)?;
            if !value.is_finite() {
                return Err(ValidatedGeometryError::InvalidInput);
            }
            if range[0] < value
                && value < range[1]
                && knots.last().is_none_or(|previous| *previous != value)
            {
                reserve_native_vec(&mut knots, memory_limit, 0, budget, cancel)?;
                knots.push(value);
            }
            Ok(())
        };
    append_curve_knots(
        leader,
        KnotParameterMap {
            offset: ZERO,
            reversed: false,
        },
        0,
        range,
        budget,
        cancel,
        &mut append,
    )?;
    budget.charge(1, cancel)?;
    reserve_native_vec(&mut knots, memory_limit, 0, budget, cancel)?;
    knots.push(range[1]);
    let mut result = Vec::new();
    let natural_start = coordinate_plane(leader, knots[0], None, budget, cancel)?;
    let mut previous = endpoint_coordinate_planes[0].map_or(natural_start, |coordinate| {
        coordinate.aligned_to(natural_start)
    });
    for pair in knots.windows(2) {
        budget.charge(1, cancel)?;
        let next = match endpoint_coordinate_planes[1].filter(|_| pair[1] == range[1]) {
            Some(coordinate) => coordinate.aligned_to(previous),
            None => coordinate_plane(leader, pair[1], Some(previous), budget, cancel)?,
        };
        let parameter = [pair[0], pair[1]];
        if parameter[0] >= parameter[1] {
            return Err(ValidatedGeometryError::InvalidInput);
        }
        budget.charge(1, cancel)?;
        reserve_native_vec(
            &mut result,
            memory_limit,
            knots.capacity().saturating_mul(size_of::<f64>()),
            budget,
            cancel,
        )?;
        result.push((
            parameter,
            Projection::BetweenPlanes {
                parameter,
                planes: [previous, next],
            },
        ));
        previous = next;
    }
    let previous_count = result.len();
    if source_preparation.coalesce_coordinates && previous_count > 1 {
        let live = knots
            .capacity()
            .checked_mul(size_of::<f64>())
            .and_then(|bytes| {
                result
                    .capacity()
                    .checked_mul(size_of::<([f64; 2], Projection)>())
                    .and_then(|spans| bytes.checked_add(spans))
            })
            .and_then(|bytes| bytes.checked_add(COARSE_COORDINATE_SCRATCH_BYTES));
        // Coalescing is optional: retain the original admissible partition if
        // its additional bounded stack cannot coexist with these same buffers.
        if live.is_some_and(|bytes| bytes <= memory_limit) {
            coalesce_coordinate_spans(leader, &mut result, max_depth, budget, cancel)?;
        }
    }
    let coalesced = result.len() < previous_count;
    Ok((result, coalesced))
}

const COARSE_COORDINATE_SCRATCH_BYTES: usize = size_of::<[(usize, usize, usize); 49]>()
    + size_of::<(usize, usize, usize)>()
    + 2 * size_of::<ValidatedCurveJet>()
    + 4 * size_of::<CoordinatePlane>()
    + 2 * size_of::<([f64; 2], Projection)>()
    + 3 * size_of::<Projection>()
    + size_of::<[f64; 2]>();

fn coalesce_coordinate_spans(
    leader: &Curve,
    spans: &mut CoordinateSpans,
    max_depth: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<(), ValidatedGeometryError> {
    let mut pending = [(0usize, 0usize, 0usize); 49];
    pending[0] = (0, spans.len(), 0);
    let mut length = 1;
    let mut output = 0;
    while length != 0 {
        budget.charge(1, cancel)?;
        length -= 1;
        let (start, end, depth) = pending[length];
        if end - start == 1 || depth == max_depth {
            budget.charge(end - start, cancel)?;
            if output != start {
                spans.copy_within(start..end, output);
            }
            output += end - start;
            continue;
        }
        let first = spans[start];
        let last = spans[end - 1];
        let range = [first.0[0], last.0[1]];
        let proposal = coarse_coordinate_proposal(leader, range, [first.1, last.1], budget, cancel);
        match proposal {
            Ok(Some(projection)) => {
                spans[output] = (range, projection);
                output += 1;
            }
            Err(
                error @ (ValidatedGeometryError::Cancelled | ValidatedGeometryError::ResourceLimit),
            ) => {
                return Err(error);
            }
            Ok(None) | Err(_) => {
                let middle = start + (end - start) / 2;
                budget.charge(2, cancel)?;
                pending[length] = (middle, end, depth + 1);
                pending[length + 1] = (start, middle, depth + 1);
                length += 2;
            }
        }
    }
    budget.charge(1, cancel)?;
    spans.truncate(output);
    Ok(())
}

fn coarse_coordinate_proposal(
    leader: &Curve,
    range: [f64; 2],
    projections: [Projection; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Option<Projection>, ValidatedGeometryError> {
    let [
        Projection::BetweenPlanes {
            planes: [first, _], ..
        },
        Projection::BetweenPlanes {
            planes: [_, last], ..
        },
    ] = projections
    else {
        return Ok(None);
    };
    let positive_dot = |a: [f64; 3], b: [f64; 3]| {
        let dot = (0..3).map(|axis| a[axis] * b[axis]).sum::<f64>();
        dot.is_finite() && dot > 0.0
    };
    if first == last || !positive_dot(first.normal, last.normal) {
        return Ok(None);
    }
    // Do not turn opposite natural tangent directions into an apparently
    // mergeable proposal merely by negating equivalent plane coefficients.
    let natural_first = coordinate_plane(leader, range[0], None, budget, cancel)?;
    let natural_last = coordinate_plane(leader, range[1], None, budget, cancel)?;
    if !positive_dot(natural_first.normal, natural_last.normal) {
        return Ok(None);
    }
    let guide = validated_curve_piecewise_jet(leader, range, budget, cancel)?;
    if !guide
        .first
        .into_iter()
        .any(|value| value.lower > 0.0 || value.upper < 0.0)
    {
        return Ok(None);
    }
    Ok(Some(Projection::BetweenPlanes {
        parameter: range,
        planes: [first, last],
    }))
}

fn coordinate_plane(
    leader: &Curve,
    parameter: f64,
    previous: Option<CoordinatePlane>,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<CoordinatePlane, ValidatedGeometryError> {
    let guide = validated_projection_leader_jet(leader, [parameter, parameter], budget, cancel)?;
    let point = guide.position.map(ValidatedInterval::midpoint);
    let mut normal = guide.first.map(ValidatedInterval::midpoint);
    let scale = normal.into_iter().map(f64::abs).fold(0.0, f64::max);
    if !scale.is_finite() || scale == 0.0 {
        return Err(ValidatedGeometryError::NonTransverse);
    }
    normal = normal.map(|entry| entry / scale);
    if previous.is_some_and(|plane| (0..3).map(|i| plane.normal[i] * normal[i]).sum::<f64>() < 0.0)
    {
        normal = normal.map(|entry| -entry);
    }
    let offset = (0..3).map(|i| normal[i] * point[i]).sum();
    if !normal.into_iter().chain([offset]).all(f64::is_finite) {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    Ok(CoordinatePlane { normal, offset })
}

fn reserve_native_vec<T>(
    values: &mut Vec<T>,
    limit: usize,
    other_bytes: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<(), ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    if values.len() < values.capacity() {
        return Ok(());
    }
    let capacity = values
        .capacity()
        .max(2)
        .checked_mul(2)
        .ok_or(ValidatedGeometryError::ResourceLimit)?;
    let bytes = capacity
        .checked_add(values.capacity())
        .and_then(|count| count.checked_mul(size_of::<T>()))
        .and_then(|bytes| bytes.checked_add(other_bytes))
        .ok_or(ValidatedGeometryError::ResourceLimit)?;
    if bytes > limit {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    budget.charge(values.len().saturating_add(capacity), cancel)?;
    values
        .try_reserve_exact(capacity - values.len())
        .map_err(|_| ValidatedGeometryError::ResourceLimit)
}

#[cfg(test)]
mod continuation_regressions {
    use super::*;
    use crate::{
        BsplineSurface, Invertible, Line, NurbsSurface, Processor, RevolutionSurface, TrimmedCurve,
        Vector4,
    };

    #[test]
    fn continuous_c0_sources_retain_whole_image_without_a_false_taylor_receipt() {
        for rational in [false, true] {
            let bend = 1.0 / 1024.0;
            let knots = (
                KnotVector::from(vec![0.0, 0.0, 1.0, 2.0, 2.0]),
                KnotVector::from(vec![0.0, 0.0, 1.0, 1.0]),
            );
            let controls = vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 1.0)],
                vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 0.0, 1.0)],
                vec![Point3::new(2.0, bend, 0.0), Point3::new(2.0, bend, 1.0)],
            ];
            let surface = if rational {
                Surface::NurbsSurface(NurbsSurface::new(BsplineSurface::new(
                    knots,
                    controls
                        .iter()
                        .map(|row| {
                            row.iter()
                                .map(|p| Vector4::new(p.x, p.y, p.z, 1.0))
                                .collect()
                        })
                        .collect(),
                )))
            } else {
                Surface::BsplineSurface(BsplineSurface::new(knots, controls))
            };
            let plane = Plane::new(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            );
            let curve = Curve::IntersectionCurve(SurfaceCurve::with_boundaries(
                Box::new(surface),
                Box::new(Surface::Plane(plane)),
                Box::new(Curve::Line(Line(
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(2.0, 0.0, 0.0),
                ))),
                None,
                None,
            ));
            let cancel = AtomicBool::new(false);
            let mut budget = ValidatedGeometryBudget::new(4_000_000);
            let prepared = prepared_validated_curve(
                &curve,
                [0.0, 1.0],
                &ValidatedCurveOptions::default(),
                &mut budget,
                &cancel,
            )
            .expect(
                "A small continuous source kink admits a contraction root without a C1 assertion.",
            );
            assert!(
                !prepared
                    .is_smooth_on([0.0, 1.0], &mut budget, &cancel)
                    .unwrap()
            );
            assert!(
                prepared
                    .is_smooth_on([0.1, 0.2], &mut budget, &cancel)
                    .unwrap()
            );
            let image = prepared
                .position_bounds([0.0, 1.0], &mut budget, &cancel)
                .unwrap();
            assert!(image[0].lower <= 0.0 && image[0].upper >= 2.0);
            assert!(image[1].lower <= 0.0 && image[1].upper >= bend);
            let small = prepared
                .position_bounds([0.49, 0.51], &mut budget, &cancel)
                .unwrap();
            assert!(small[0].width() < image[0].width() * 0.1);
            for parameter in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let point = prepared.evaluate(parameter, &mut budget, &cancel).unwrap();
                for (axis, expected) in [
                    2.0 * parameter,
                    bend * (2.0 * parameter - 1.0).max(0.0),
                    0.0,
                ]
                .into_iter()
                .enumerate()
                {
                    assert!(
                        point.error[axis].lower <= expected && expected <= point.error[axis].upper
                    );
                }
            }
            assert!(matches!(
                prepared.position_bounds([0.4, 0.6], &mut ValidatedGeometryBudget::new(0), &cancel),
                Err(ValidatedGeometryError::ResourceLimit)
            ));
            assert!(matches!(
                prepared.is_smooth_on([0.4, 0.6], &mut budget, &AtomicBool::new(true)),
                Err(ValidatedGeometryError::Cancelled)
            ));
        }
    }

    fn plane_intersection(parallel: bool) -> Curve {
        let first = Plane::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );
        let second = if parallel {
            first
        } else {
            Plane::new(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(0.0, 0.0, 1.0),
            )
        };
        Curve::IntersectionCurve(SurfaceCurve::with_boundaries(
            Box::new(Surface::Plane(first)),
            Box::new(Surface::Plane(second)),
            Box::new(Curve::Line(Line(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ))),
            None,
            None,
        ))
    }

    #[test]
    fn implicit_scalar_projection_preserves_a_source_plane_identity() {
        let curve = Curve::IntersectionCurve(SurfaceCurve::with_boundaries(
            Box::new(Surface::Plane(Plane::new(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
                Point3::new(0.0, 0.0, 1.0),
            ))),
            Box::new(Surface::Plane(Plane::new(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ))),
            Box::new(Curve::Line(Line(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
            ))),
            None,
            None,
        ));
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(1_000_000);
        let prepared = prepared_validated_curve(
            &curve,
            [0.0, 1.0],
            &ValidatedCurveOptions::default(),
            &mut budget,
            &cancel,
        )
        .unwrap();
        let scalar = prepared
            .projected_jet([0.0, 1.0], [0.0; 3], [1.0, -1.0, 0.0], &mut budget, &cancel)
            .unwrap();
        assert_eq!([scalar.position, scalar.first, scalar.second], [ZERO; 3]);
    }

    #[test]
    fn nonuniform_mapped_root_does_not_reproject_the_world_guide() {
        let delta = 1.0 / 1024.0;
        let original = Curve::IntersectionCurve(SurfaceCurve::with_boundaries(
            Box::new(Surface::Plane(Plane::new(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 0.0, 1.0),
            ))),
            Box::new(Surface::Plane(Plane::new(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ))),
            Box::new(Curve::Line(Line(
                Point3::new(0.0, delta, 0.0),
                Point3::new(1.0, 2.0 * delta, 0.0),
            ))),
            None,
            None,
        ));
        let pose = Matrix4::new(
            2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 7.0, 11.0, 13.0, 1.0,
        );
        let mut wrapper =
            Processor::with_transform(TrimmedCurve::new(Box::new(original), (0.25, 0.75)), pose);
        wrapper.invert();
        let curve = Curve::TrimmedCurve(wrapper);
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(1_000_000);
        let prepared = prepared_validated_curve(
            &curve,
            [0.25, 0.75],
            &ValidatedCurveOptions::default(),
            &mut budget,
            &cancel,
        )
        .unwrap();
        let jet = prepared.jet([0.25, 0.75], &mut budget, &cancel).unwrap();
        let expected_speed = -2.0 * (1.0 + delta * delta);
        assert!(jet.first[0].lower <= expected_speed && expected_speed <= jet.first[0].upper);
        assert!(jet.first[0].width() < 1.0e-8);
        for parameter in [0.25, 0.5, 0.75] {
            let native = 1.0 - parameter;
            let expected_x = 7.0 + 2.0 * ((1.0 + delta * delta) * native + delta * delta);
            let illegally_reprojected_x = 7.0 + 2.0 * native + 4.5 * delta * delta * (native + 1.0);
            let point = prepared.evaluate(parameter, &mut budget, &cancel).unwrap();
            assert!(point.error[0].lower <= expected_x && expected_x <= point.error[0].upper);
            assert!(
                illegally_reprojected_x < point.error[0].lower
                    || illegally_reprojected_x > point.error[0].upper
            );
            for (axis, expected) in [(1, 11.0), (2, 13.0)] {
                assert!(point.error[axis].lower <= expected && expected <= point.error[axis].upper);
            }
        }
    }

    #[test]
    fn retained_implicit_trim_transforms_the_same_prepared_root() {
        let pose = Matrix4::new(
            2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 1.0, 2.0, 3.0, 1.0,
        );
        let mut wrapper = Processor::with_transform(
            TrimmedCurve::new(Box::new(plane_intersection(false)), (0.25, 0.75)),
            pose,
        );
        wrapper.invert();
        let curve = Curve::TrimmedCurve(wrapper);
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(1_000_000);
        let prepared = prepared_validated_curve(
            &curve,
            [0.25, 0.75],
            &ValidatedCurveOptions::default(),
            &mut budget,
            &cancel,
        )
        .unwrap();
        assert!(prepared.mapped.is_some());
        let jet = prepared.jet([0.25, 0.75], &mut budget, &cancel).unwrap();
        for (axis, speed) in [0.0, -3.0, 0.0].into_iter().enumerate() {
            assert!(jet.first[axis].lower <= speed && speed <= jet.first[axis].upper);
            assert!(jet.second[axis].contains_zero());
        }
        for parameter in [0.25, 0.5, 0.75] {
            let point = prepared.evaluate(parameter, &mut budget, &cancel).unwrap();
            for (axis, expected) in [1.0, 2.0 + 3.0 * (1.0 - parameter), 3.0]
                .into_iter()
                .enumerate()
            {
                assert!(point.error[axis].lower <= expected && expected <= point.error[axis].upper);
            }
        }
        assert!(prepared.retained_bytes() > prepared.mapped.as_ref().unwrap().retained_bytes());
    }

    #[test]
    fn retained_implicit_trim_continues_an_analytic_guide_without_clamping() {
        let curve = Curve::TrimmedCurve(Processor::new(TrimmedCurve::new(
            Box::new(plane_intersection(false)),
            (-1.0, 2.0),
        )));
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(1_000_000);
        let prepared = prepared_validated_curve(
            &curve,
            [-1.0, 2.0],
            &ValidatedCurveOptions::default(),
            &mut budget,
            &cancel,
        )
        .unwrap();
        let jet = prepared.jet([-1.0, 2.0], &mut budget, &cancel).unwrap();
        assert!(jet.position[1].lower <= -1.0 && jet.position[1].upper >= 2.0);
        for parameter in [-1.0, 0.5, 2.0] {
            let point = prepared.evaluate(parameter, &mut budget, &cancel).unwrap();
            assert!(point.error[1].lower <= parameter && parameter <= point.error[1].upper);
        }
    }

    #[test]
    fn retained_implicit_trim_uses_homogeneous_quotients_and_refuses_poles() {
        let pose = |constant| {
            Matrix4::new(
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.25, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                constant,
            )
        };
        let curve = Curve::TrimmedCurve(Processor::with_transform(
            TrimmedCurve::new(Box::new(plane_intersection(false)), (0.0, 1.0)),
            pose(1.0),
        ));
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(1_000_000);
        let prepared = prepared_validated_curve(
            &curve,
            [0.0, 1.0],
            &ValidatedCurveOptions::default(),
            &mut budget,
            &cancel,
        )
        .unwrap();
        let point = prepared.evaluate(0.5, &mut budget, &cancel).unwrap();
        assert!(point.error[1].lower <= 4.0 / 9.0 && 4.0 / 9.0 <= point.error[1].upper);
        let jet = prepared.jet([0.0, 1.0], &mut budget, &cancel).unwrap();
        assert!(jet.first[1].lower <= 0.64 && jet.first[1].upper >= 1.0);
        assert!(jet.second[1].lower <= -0.5 && jet.second[1].upper >= -0.256);
        let pole = Curve::TrimmedCurve(Processor::with_transform(
            TrimmedCurve::new(Box::new(plane_intersection(false)), (0.0, 1.0)),
            pose(-0.125),
        ));
        let options = ValidatedCurveOptions {
            max_depth: 6,
            ..ValidatedCurveOptions::default()
        };
        assert!(matches!(
            prepared_validated_curve(&pole, [0.0, 1.0], &options, &mut budget, &cancel),
            Err(ValidatedGeometryError::DenominatorContainsZero)
        ));
    }

    #[test]
    fn restricted_original_cylinder_jets_resolve_green_sign_from_a_coarse_tube() {
        let profile = Curve::TrimmedCurve(Processor::new(TrimmedCurve::new(
            Box::new(Curve::Line(Line(
                Point3::new(2.0, 0.0, -1.0),
                Point3::new(2.0, 0.0, 0.0),
            ))),
            (1.0, 13.0),
        )));
        let surface = Surface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
            profile,
            Point3::new(0.0, 0.0, 0.0),
            crate::Vector3::unit_z(),
        )))
        .inverse();
        let plane = Plane::new(
            Point3::new(0.0, 0.0, 6.0),
            Point3::new(1.0, 0.0, 6.0),
            Point3::new(0.0, 1.0, 6.0),
        );
        let point = |angle: f64| Point3::new(2.0 * angle.cos(), 2.0 * angle.sin(), 6.0);
        let endpoints = [point(0.69), point(0.71)];
        let curve = Curve::IntersectionCurve(SurfaceCurve::with_boundaries(
            Box::new(surface),
            Box::new(Surface::Plane(plane)),
            Box::new(Curve::Line(Line(endpoints[0], endpoints[1]))),
            None,
            None,
        ));
        let projection = Projection::BetweenPlanes {
            parameter: [0.0, 1.0],
            planes: endpoints.map(|point| CoordinatePlane {
                normal: [0.0, 1.0, 0.0],
                offset: point.y,
            }),
        };
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(2_000_000);
        let options = ValidatedCurveOptions::default();
        let requested = [0.49, 0.51];
        let green = |jet: &ValidatedCurveJet| {
            jet.position[0] * jet.first[1] - jet.position[1] * jet.first[0]
        };
        // A genuine original uniqueness tube, substantially wider than the
        // requested subspan. Its coefficients and cached jets are unmodified.
        let domain = [
            ValidatedInterval::new(0.475, 0.925).unwrap(),
            ValidatedInterval::new(6.0, 8.0).unwrap(),
            ValidatedInterval::new(-1.0, 4.0).unwrap(),
            ValidatedInterval::new(-1.0, 4.0).unwrap(),
        ];
        let tube = certify(
            &curve,
            projection,
            interval_range([0.0, 1.0]).unwrap(),
            domain,
            &mut budget,
            &cancel,
        )
        .unwrap();
        let old_jet = implicit_jet(&tube).unwrap();
        let prepared = PreparedValidatedCurve {
            source: PreparedSource::Borrowed(&curve),
            range: [0.0, 1.0],
            pieces: vec![ValidatedCurvePiece {
                parameter: [0.0, 1.0],
                jet: old_jet,
                source_parameters: Some(unflatten(tube.domain)),
                contraction_bound: tube.contraction_bound(),
                surface_residual: tube.surface_residual(),
                projection,
            }],
            options,
            mapped: None,
        };
        let jet = prepared.jet(requested, &mut budget, &cancel).unwrap();
        assert!(
            green(&jet).lower > 0.0,
            "fresh whole-span Green sign must resolve"
        );
        let dy = endpoints[1].y - endpoints[0].y;
        for t in [requested[0], 0.5, requested[1]] {
            let y = endpoints[0].y + t * dy;
            let x = (4.0 - y * y).sqrt();
            let position = [x, y, 6.0];
            let first = [-y * dy / x, dy, 0.0];
            let second = [-4.0 * dy * dy / (x * x * x), 0.0, 0.0];
            for axis in 0..3 {
                for (bound, value) in [
                    (jet.position[axis], position[axis]),
                    (jet.first[axis], first[axis]),
                    (jet.second[axis], second[axis]),
                ] {
                    assert!(
                        bound.lower <= value && value <= bound.upper,
                        "{bound:?} misses {value}"
                    );
                }
            }
        }
    }

    #[test]
    fn transverse_planes_cover_native_parameter_and_derivatives() {
        let curve = plane_intersection(false);
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(1_000_000);
        let options = ValidatedCurveOptions::default();
        let prepared = prepared_validated_curve(&curve, [0.0, 1.0], &options, &mut budget, &cancel)
            .expect("Two transverse affine carriers have a validated native branch.");
        let mut covered_until = 0.0;
        for piece in prepared.pieces() {
            assert_eq!(piece.parameter[0], covered_until);
            assert!(piece.parameter[0] < piece.parameter[1]);
            assert!(piece.contraction_bound < 1.0);
            assert!(piece.source_parameters.is_some());
            assert!(
                piece
                    .surface_residual
                    .into_iter()
                    .all(ValidatedInterval::contains_zero)
            );
            covered_until = piece.parameter[1];
        }
        assert_eq!(covered_until, 1.0);
        let jet = prepared
            .jet([0.0, 1.0], &mut budget, &cancel)
            .expect("The complete smooth branch has whole-interval jets.");
        for (axis, speed) in [0.0, 1.0, 0.0].into_iter().enumerate() {
            assert!(jet.position[axis].is_finite());
            assert!(jet.first[axis].lower <= speed && speed <= jet.first[axis].upper);
            assert!(jet.first[axis].width() < 1.0e-8);
            assert!(jet.second[axis].contains_zero() && jet.second[axis].magnitude() < 1.0e-8);
        }
        for parameter in [0.0, 0.125, 0.5, 0.875, 1.0] {
            let point = prepared.evaluate(parameter, &mut budget, &cancel)
                .expect("Prepared evaluation retains the certified root at endpoints and interior parameters.");
            for (axis, value) in [0.0, parameter, 0.0].into_iter().enumerate() {
                assert!(point.error[axis].lower <= value && value <= point.error[axis].upper);
                assert!(
                    point.error[axis].lower <= point.position[axis]
                        && point.position[axis] <= point.error[axis].upper
                );
                assert!(point.error[axis].width() <= options.endpoint_tolerance);
            }
        }
    }

    #[test]
    fn continuation_reports_degeneracy_cancellation_and_work_exhaustion() {
        let singular = plane_intersection(true);
        let options = ValidatedCurveOptions::default();
        assert!(matches!(
            prepared_validated_curve(
                &singular,
                [0.0, 1.0],
                &options,
                &mut ValidatedGeometryBudget::new(1_000_000),
                &AtomicBool::new(false)
            ),
            Err(ValidatedGeometryError::Singular | ValidatedGeometryError::NonTransverse)
        ));
        let regular = plane_intersection(false);
        assert!(matches!(
            prepared_validated_curve(
                &regular,
                [0.0, 1.0],
                &options,
                &mut ValidatedGeometryBudget::new(1_000_000),
                &AtomicBool::new(true)
            ),
            Err(ValidatedGeometryError::Cancelled)
        ));
        assert!(matches!(
            prepared_validated_curve(
                &regular,
                [0.0, 1.0],
                &options,
                &mut ValidatedGeometryBudget::new(0),
                &AtomicBool::new(false)
            ),
            Err(ValidatedGeometryError::ResourceLimit)
        ));
    }

    #[test]
    fn owned_original_nonlinear_binding_preserves_connected_branch_and_direct_uvs() {
        // A genuine two-surface intersection, not a face/plane replacement:
        // S(u,v) = (2u-1, 2v-1, (2u-1)^2), intersected with z=1/4.
        // There are two components x=+/-1/2. The original guide selects +1/2;
        // the retained connected tube must never reselect the other component.
        let controls = [(-1.0, 1.0), (0.0, -1.0), (1.0, 1.0)]
            .into_iter()
            .map(|(x, z)| vec![Vector4::new(x, -1.0, z, 1.0), Vector4::new(x, 1.0, z, 1.0)])
            .collect();
        let curved = Surface::NurbsSurface(NurbsSurface::new(BsplineSurface::new(
            (KnotVector::bezier_knot(2), KnotVector::bezier_knot(1)),
            controls,
        )));
        let plane = Surface::Plane(Plane::new(
            Point3::new(-1.0, -1.0, 0.25),
            Point3::new(1.0, -1.0, 0.25),
            Point3::new(-1.0, 1.0, 0.25),
        ));
        for (u, x) in [(0.25, -0.5), (0.75, 0.5)] {
            for v in [0.25, 0.5, 0.75] {
                let expected = Point3::new(x, 2.0 * v - 1.0, 0.25);
                assert_eq!(curved.subs(u, v), expected);
                assert_eq!(plane.subs(u, v), expected);
            }
        }
        let curve = Curve::IntersectionCurve(SurfaceCurve::with_boundaries(
            Box::new(curved),
            Box::new(plane),
            Box::new(Curve::Line(Line(
                Point3::new(0.5, -0.5, 0.25),
                Point3::new(0.5, 0.5, 0.25),
            ))),
            None,
            None,
        ));
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(8_000_000);
        let options = ValidatedCurveOptions::default();
        let borrowed =
            prepared_validated_curve(&curve, [0.0, 1.0], &options, &mut budget, &cancel).unwrap();
        let owned: PreparedValidatedCurve<'static> = prepared_validated_curve_owned(
            curve.clone(),
            [0.0, 1.0],
            &options,
            &mut budget,
            &cancel,
        )
        .unwrap();
        for prepared in [&borrowed, &owned] {
            // Each carrier's actual UV trace is (3/4, 1/4+t/2).
            // Check whole-piece UV coverage as well as independent world P/D;
            // a nearby component or a copied guide cannot satisfy this receipt.
            for piece in prepared.pieces() {
                let parameters = piece.source_parameters.expect("Original carrier UV tube");
                for carrier in parameters {
                    assert!(carrier[0].lower > 0.5 && carrier[0].lower <= 0.75);
                    assert!(carrier[0].upper >= 0.75);
                    for t in piece.parameter {
                        let v = 0.25 + 0.5 * t;
                        assert!(carrier[1].lower <= v && v <= carrier[1].upper);
                    }
                }
            }
            for parameter in [0.0, 0.125, 0.5, 0.875, 1.0] {
                let point = prepared.evaluate(parameter, &mut budget, &cancel).unwrap();
                for (axis, expected) in [0.5, parameter - 0.5, 0.25].into_iter().enumerate() {
                    assert!(
                        point.error[axis].lower <= expected && expected <= point.error[axis].upper
                    );
                    assert!(point.error[axis].width() <= options.endpoint_tolerance);
                }
            }
            let image = prepared
                .position_bounds([0.125, 0.875], &mut budget, &cancel)
                .unwrap();
            assert!(image[0].lower > 0.0);
            assert!(image[1].lower <= -0.375 && image[1].upper >= 0.375);
            let jet = prepared.jet([0.25, 0.75], &mut budget, &cancel).unwrap();
            for (axis, derivative) in [0.0, 1.0, 0.0].into_iter().enumerate() {
                assert!(jet.first[axis].lower <= derivative && derivative <= jet.first[axis].upper);
                assert!(jet.second[axis].lower <= 0.0 && jet.second[axis].upper >= 0.0);
            }
        }
    }

    #[test]
    fn owned_original_nested_trim_reuses_the_same_reflections_and_pose() {
        let mut inner = Processor::with_transform(
            TrimmedCurve::new(Box::new(plane_intersection(false)), (0.125, 0.875)),
            Matrix4::from_translation(crate::Vector3::new(2.0, 3.0, 5.0)),
        );
        inner.invert();
        let mut outer = Processor::with_transform(
            TrimmedCurve::new(Box::new(Curve::TrimmedCurve(inner)), (0.25, 0.75)),
            Matrix4::from_nonuniform_scale(2.0, 3.0, 4.0),
        );
        outer.invert();
        let curve = Curve::TrimmedCurve(outer);
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(4_000_000);
        let options = ValidatedCurveOptions::default();
        let owned = prepared_validated_curve_owned(
            curve.clone(),
            [0.25, 0.75],
            &options,
            &mut budget,
            &cancel,
        )
        .unwrap();
        assert!(owned.retained_bytes() <= options.max_retained_bytes);
        let basis = owned.mapped_basis().unwrap().mapped_basis().unwrap();
        for parameter in [0.25, 0.5, 0.75] {
            let native = basis.native_curve().unwrap().subs(parameter);
            let point = basis.evaluate(parameter, &mut budget, &cancel).unwrap();
            for (axis, expected) in [0.0, parameter, 0.0].into_iter().enumerate() {
                assert!((native[axis] - expected).abs() < 1.0e-9);
                assert!(point.error[axis].lower <= expected && expected <= point.error[axis].upper);
            }
        }
        assert!(matches!(
            basis.evaluate(0.125, &mut budget, &cancel),
            Err(ValidatedGeometryError::ParameterOutOfRange)
        ));
        // Both original inverse anchors sum to one, so the two reflections
        // compose to t. The two spatial maps give (4, 9+3t, 20).
        for parameter in [0.25, 0.375, 0.5, 0.75] {
            let native = curve.subs(parameter);
            let point = owned.evaluate(parameter, &mut budget, &cancel).unwrap();
            for (axis, expected) in [4.0, 9.0 + 3.0 * parameter, 20.0].into_iter().enumerate() {
                assert!((native[axis] - expected).abs() < 1.0e-9);
                assert!(point.error[axis].lower <= expected && expected <= point.error[axis].upper);
            }
        }
        let jet = owned.jet([0.25, 0.75], &mut budget, &cancel).unwrap();
        for (axis, derivative) in [0.0, 3.0, 0.0].into_iter().enumerate() {
            assert!(jet.first[axis].lower <= derivative && derivative <= jet.first[axis].upper);
            assert!(jet.second[axis].lower <= 0.0 && jet.second[axis].upper >= 0.0);
        }
        assert!(
            owned
                .is_smooth_on([0.25, 0.75], &mut budget, &cancel)
                .unwrap()
        );
        assert!(matches!(
            owned.evaluate(0.125, &mut budget, &cancel),
            Err(ValidatedGeometryError::ParameterOutOfRange)
        ));
    }

    #[test]
    fn owned_original_binding_refuses_nonunique_roots_and_resource_exhaustion() {
        // Coincident xy carriers and the guide's y=t plane leave every x a
        // solution: two independent native surface points witness nonuniqueness.
        let plane = Plane::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );
        assert_eq!(plane.subs(0.0, 0.5), Point3::new(0.0, 0.5, 0.0));
        assert_eq!(plane.subs(1.0, 0.5), Point3::new(1.0, 0.5, 0.0));
        let options = ValidatedCurveOptions::default();
        let cancel = AtomicBool::new(false);
        assert!(matches!(
            prepared_validated_curve_owned(
                plane_intersection(true),
                [0.0, 1.0],
                &options,
                &mut ValidatedGeometryBudget::new(1_000_000),
                &cancel,
            ),
            Err(ValidatedGeometryError::Singular | ValidatedGeometryError::NonTransverse)
        ));
        assert!(matches!(
            prepared_validated_curve_owned(
                plane_intersection(false),
                [0.0, 1.0],
                &options,
                &mut ValidatedGeometryBudget::new(0),
                &cancel,
            ),
            Err(ValidatedGeometryError::ResourceLimit)
        ));
        assert!(matches!(
            prepared_validated_curve_owned(
                plane_intersection(false),
                [0.0, 1.0],
                &options,
                &mut ValidatedGeometryBudget::new(1_000_000),
                &AtomicBool::new(true),
            ),
            Err(ValidatedGeometryError::Cancelled)
        ));
        let curve = plane_intersection(false);
        let mut budget = ValidatedGeometryBudget::new(1_000_000);
        let source_bytes = curve_owned_bytes(&curve, usize::MAX, &mut budget, &cancel).unwrap();
        let too_small = ValidatedCurveOptions {
            max_retained_bytes: source_bytes,
            ..options
        };
        assert!(matches!(
            prepared_validated_curve_owned(curve, [0.0, 1.0], &too_small, &mut budget, &cancel,),
            Err(ValidatedGeometryError::ResourceLimit)
        ));
        let mut deep = plane_intersection(false);
        for _ in 0..=crate::geometry::MAX_NATIVE_GEOMETRY_NESTING {
            deep = Curve::TrimmedCurve(Processor::new(TrimmedCurve::new(
                Box::new(deep),
                (0.0, 1.0),
            )));
        }
        assert!(matches!(
            prepared_validated_curve_owned(deep, [0.0, 1.0], &options, &mut budget, &cancel,),
            Err(ValidatedGeometryError::ResourceLimit)
        ));
    }

    fn generated_circle_fixture(points: Vec<Point3>) -> (Surface, Plane, Curve) {
        let profile = Curve::Line(Line(
            Point3::new(2.0, 0.0, -1.0),
            Point3::new(2.0, 0.0, 0.0),
        ));
        let surface = Surface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
            profile,
            Point3::new(0.0, 0.0, 0.0),
            crate::Vector3::unit_z(),
        )))
        .inverse();
        let plane = Plane::new(
            Point3::new(0.0, 0.0, 6.0),
            Point3::new(1.0, 0.0, 6.0),
            Point3::new(0.0, 1.0, 6.0),
        );
        let count = points.len() - 1;
        let mut knots = vec![10.0];
        knots.extend((0..=count).map(|index| 10.0 + 4.0 * index as f64 / count as f64));
        knots.push(14.0);
        let guide = Curve::BsplineCurve(crate::BsplineCurve::new(KnotVector::from(knots), points));
        (surface, plane, guide)
    }

    #[test]
    fn generated_dense_circle_coordinates_keep_physical_area_and_original_master_gauge() {
        use std::f64::consts::{FRAC_PI_2, PI};
        let (surface, plane, guide) = generated_circle_fixture(
            (0..=64)
                .map(|index| {
                    let angle = FRAC_PI_2 * f64::from(index) / 64.0;
                    Point3::new(2.0 * angle.cos(), 2.0 * angle.sin(), 6.0)
                })
                .collect(),
        );
        let original = Curve::IntersectionCurve(SurfaceCurve::with_boundaries(
            Box::new(surface.clone()),
            Box::new(Surface::Plane(plane)),
            Box::new(guide.clone()),
            None,
            None,
        ));
        let cancel = AtomicBool::new(false);
        let options = ValidatedCurveOptions::default();
        let mut master_work = ValidatedGeometryBudget::new(2_000_000);
        let master =
            prepared_validated_curve(&original, [10.0, 14.0], &options, &mut master_work, &cancel)
                .unwrap();
        let original_middle = master
            .evaluate(12.0, &mut master_work, &cancel)
            .unwrap()
            .error;
        let mut work = ValidatedGeometryBudget::new(2_000_000);
        let generated = prepared_face_plane_curve(
            &guide,
            ValidatedFacePlaneSource {
                surface: &surface,
                anchor: Some(ValidatedFacePlaneSourceAnchor::Front([0.0, 7.0])),
                endpoint_coordinate_planes: [None; 2],
            },
            &plane,
            [10.0, 14.0],
            &options,
            &mut work,
            &cancel,
        )
        .unwrap();
        assert_eq!(generated.range(), [10.0, 14.0]);
        let mut area = ZERO;
        let mut covered = 10.0;
        for piece in generated.pieces() {
            assert_eq!(piece.parameter[0], covered);
            covered = piece.parameter[1];
            let jet = generated.jet(piece.parameter, &mut work, &cancel).unwrap();
            assert!(jet.position[2].lower <= 6.0 && 6.0 <= jet.position[2].upper);
            let radius_squared = jet.position[0].square() + jet.position[1].square();
            assert!(radius_squared.lower <= 4.0 && 4.0 <= radius_squared.upper);
            let width = ValidatedInterval::point(piece.parameter[1])
                - ValidatedInterval::point(piece.parameter[0]);
            area = area
                + (jet.position[0] * jet.first[1] - jet.position[1] * jet.first[0])
                    * width
                    * ValidatedInterval::point(0.5);
        }
        assert_eq!(covered, 14.0);
        assert!(area.lower > 0.0);
        assert!(
            area.lower <= PI && PI <= area.upper,
            "quarter-circle Green integral"
        );
        for (parameter, expected) in [(10.0, [2.0, 0.0, 6.0]), (14.0, [0.0, 2.0, 6.0])] {
            let point = generated.evaluate(parameter, &mut work, &cancel).unwrap();
            for (axis, expected) in expected.into_iter().enumerate() {
                assert!(point.error[axis].lower <= expected && expected <= point.error[axis].upper);
            }
        }
        // Original-master knot coordinates remain authoritative, not the
        // generated candidate's optional coarser numerical parameterization.
        for parameter in [10.0, 11.0, 12.0, 13.0, 14.0] {
            let angle = FRAC_PI_2 * (parameter - 10.0) / 4.0;
            let point = master
                .evaluate(parameter, &mut master_work, &cancel)
                .unwrap();
            for (axis, expected) in [2.0 * angle.cos(), 2.0 * angle.sin(), 6.0]
                .into_iter()
                .enumerate()
            {
                assert!(point.error[axis].lower <= expected && expected <= point.error[axis].upper);
            }
        }
        assert_eq!(
            master
                .evaluate(12.0, &mut master_work, &cancel)
                .unwrap()
                .error,
            original_middle,
        );
    }

    #[test]
    fn generated_nonuniform_cylinder_generator_keeps_original_knot_coordinates() {
        let surface = Surface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
            Curve::Line(Line(
                Point3::new(24.0, 0.0, 2.0),
                Point3::new(24.0, 1.0, 2.0),
            )),
            Point3::new(22.0, 0.0, 2.0),
            crate::Vector3::new(-0.0, 1.0, 0.0),
        )))
        .inverse();
        let height = 0.0999999285714281;
        let plane = Plane::new(
            Point3::new(-1.0, 19.0, height),
            Point3::new(25.0, 19.0, height),
            Point3::new(-1.0, -1.0, height),
        );
        let guide = Curve::BsplineCurve(crate::BsplineCurve::new(
            KnotVector::from(vec![0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]),
            vec![
                Point3::new(22.62449958252302, 16.0, 0.09999992857142836),
                Point3::new(22.624499582523015, 12.844555966903268, 0.09999992857142811),
                Point3::new(22.624499582523015, 9.0, 0.09999992857142811),
                Point3::new(22.624499582523015, 5.155445987406469, 0.09999992857142811),
                Point3::new(22.62449958252302, 2.0, 0.09999992857142836),
            ],
        ));
        let source = ValidatedFacePlaneSource {
            surface: &surface,
            anchor: Some(ValidatedFacePlaneSourceAnchor::Both {
                front: [1.253236011880648, 16.0],
                back: [1.253236011880648, 2.0],
            }),
            endpoint_coordinate_planes: [None; 2],
        };
        let options = ValidatedCurveOptions::default();
        let cancel = AtomicBool::new(false);
        let mut work = ValidatedGeometryBudget::new(2_000_000);
        let prepared = prepared_face_plane_curve(
            &guide,
            source,
            &plane,
            [0.0, 1.0],
            &options,
            &mut work,
            &cancel,
        );
        let prepared =
            prepared.expect("Original nonuniform knot coordinates retain a complete root.");
        assert_eq!(prepared.range(), [0.0, 1.0]);
        let mut covered = 0.0;
        for piece in prepared.pieces() {
            assert_eq!(piece.parameter[0], covered);
            covered = piece.parameter[1];
            assert!(piece.contraction_bound < 1.0);
        }
        assert_eq!(covered, 1.0);
        let front = prepared.evaluate(0.0, &mut work, &cancel).unwrap();
        let back = prepared.evaluate(1.0, &mut work, &cancel).unwrap();
        assert!(front.error[1].lower <= 16.0 && 16.0 <= front.error[1].upper);
        assert!(back.error[1].lower <= 2.0 && 2.0 <= back.error[1].upper);
        assert!(matches!(
            prepared.evaluate(0.25, &mut ValidatedGeometryBudget::new(0), &cancel,),
            Err(ValidatedGeometryError::ResourceLimit)
        ));
        assert!(matches!(
            prepared.evaluate(0.25, &mut work, &AtomicBool::new(true)),
            Err(ValidatedGeometryError::Cancelled)
        ));
    }

    #[test]
    fn generated_cylinder_generator_keeps_axial_profile_and_endpoint_gauges() {
        let surface = Surface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
            Curve::Line(Line(
                Point3::new(24.0, 0.0, 2.0),
                Point3::new(24.0, 1.0, 2.0),
            )),
            Point3::new(22.0, 0.0, 2.0),
            crate::Vector3::unit_y(),
        )))
        .inverse();
        let angle: f64 = 1.253236011880648;
        let height = 0.0999999285714281;
        let x = 22.0 + 2.0 * angle.cos();
        let plane = Plane::new(
            Point3::new(-1.0000000000000009, 19.000000000000004, height),
            Point3::new(-0.0000000000000009, 19.000000000000004, height),
            Point3::new(-1.0000000000000009, 20.000000000000004, height),
        );
        let cancel = AtomicBool::new(false);
        let mut work = ValidatedGeometryBudget::new(2_000_000);
        let jet = validated_surface_piecewise_jet(
            &surface,
            [[angle, angle], [2.0, 16.0]],
            &mut work,
            &cancel,
        )
        .expect("The bare unit profile retains its actual unbounded affine semantics.");
        for axis in [0, 2] {
            assert!(
                jet.position[axis].width() < 1.0e-11,
                "Axial width must not become a radial-position uncertainty: {jet:?}"
            );
            assert_eq!(jet.dv[axis], ZERO);
            assert_eq!(jet.duv[axis], ZERO);
        }
        assert_eq!(jet.dv[1], ONE);
        assert!(jet.position[1].lower <= 2.0 && jet.position[1].upper >= 16.0);
        for range in [[0.0, 1.0], [10.0, 14.0]] {
            let guide = Curve::BsplineCurve(crate::BsplineCurve::new(
                KnotVector::from(vec![range[0], range[0], range[1], range[1]]),
                vec![Point3::new(x, 16.0, height), Point3::new(x, 2.0, height)],
            ));
            for anchor in [
                ValidatedFacePlaneSourceAnchor::Front([angle, 16.0]),
                ValidatedFacePlaneSourceAnchor::Back([angle, 2.0]),
                ValidatedFacePlaneSourceAnchor::Both {
                    front: [angle, 16.0],
                    back: [angle, 2.0],
                },
            ] {
                let prepared = prepared_face_plane_curve(
                    &guide,
                    ValidatedFacePlaneSource {
                        surface: &surface,
                        anchor: Some(anchor),
                        endpoint_coordinate_planes: [None; 2],
                    },
                    &plane,
                    range,
                    &ValidatedCurveOptions::default(),
                    &mut work,
                    &cancel,
                )
                .unwrap_or_else(|error| {
                    panic!("Generator preparation at {anchor:?}, range {range:?}: {error:?}")
                });
                assert_eq!(prepared.range(), range);
                let mut covered = range[0];
                for piece in prepared.pieces() {
                    assert_eq!(piece.parameter[0], covered);
                    covered = piece.parameter[1];
                    let chart = piece.source_parameters.expect("Original source charts.");
                    assert!(chart[0][0].lower <= angle && angle <= chart[0][0].upper);
                }
                assert_eq!(covered, range[1]);
                for ratio in [0.0, 0.5, 1.0] {
                    let t = range[0] + ratio * (range[1] - range[0]);
                    let point = prepared.evaluate(t, &mut work, &cancel).unwrap();
                    for (axis, expected) in [x, 16.0 - 14.0 * ratio, height].into_iter().enumerate()
                    {
                        assert!(
                            point.error[axis].lower <= expected
                                && expected <= point.error[axis].upper,
                            "Native generator position at {anchor:?}, t={t}: {point:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn generated_axial_root_jet_contracts_nonlinear_chart_over_whole_spans() {
        let surface = Surface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
            Curve::Line(Line(
                Point3::new(24.0, 0.0, 2.0),
                Point3::new(24.0, 1.0, 2.0),
            )),
            Point3::new(22.0, 0.0, 2.0),
            crate::Vector3::unit_y(),
        )))
        .inverse();
        let angle: f64 = 1.253236011880648;
        let height = 0.0999999285714281;
        let x = 22.0 + 2.0 * angle.cos();
        let plane = Plane::new(
            Point3::new(-1.0000000000000009, 19.000000000000004, height),
            Point3::new(-0.0000000000000009, 19.000000000000004, height),
            Point3::new(-1.0000000000000009, 20.000000000000004, height),
        );
        let guide = Curve::BsplineCurve(crate::BsplineCurve::new(
            KnotVector::from(vec![0.0, 0.0, 1.0, 1.0]),
            vec![Point3::new(x, 16.0, height), Point3::new(x, 2.0, height)],
        ));
        let cancel = AtomicBool::new(false);
        let mut work = ValidatedGeometryBudget::new(2_000_000);
        let prepared = prepared_face_plane_curve(
            &guide,
            ValidatedFacePlaneSource {
                surface: &surface,
                anchor: Some(ValidatedFacePlaneSourceAnchor::Both {
                    front: [angle, 16.0],
                    back: [angle, 2.0],
                }),
                endpoint_coordinate_planes: [None; 2],
            },
            &plane,
            [0.0, 1.0],
            &ValidatedCurveOptions::default(),
            &mut work,
            &cancel,
        )
        .unwrap();
        // Independent carrier coefficients prescribe a constant radial root
        // and affine axial coordinate, not a guide-based straightness claim.
        for range in [[0.0, 1.0], [0.0, 0.125], [0.5, 0.75]] {
            let jet = prepared.jet(range, &mut work, &cancel).unwrap();
            for (axis, expected) in [x, 16.0 - 14.0 * range[0], height].into_iter().enumerate() {
                assert!(
                    jet.position[axis].lower <= expected && expected <= jet.position[axis].upper
                );
            }
            for (axis, expected) in [0.0, -14.0, 0.0].into_iter().enumerate() {
                assert!(jet.first[axis].lower <= expected && expected <= jet.first[axis].upper);
                assert!(jet.first[axis].width() < 1.0e-5, "{range:?}: {jet:?}");
                assert!(jet.second[axis].contains_zero());
                assert!(jet.second[axis].magnitude() < 1.0e-4, "{range:?}: {jet:?}");
            }
            assert!(jet.position[0].width() < 1.0e-8, "{range:?}: {jet:?}");
        }
        assert!(matches!(
            prepared.jet([0.0, 0.125], &mut ValidatedGeometryBudget::new(0), &cancel),
            Err(ValidatedGeometryError::ResourceLimit)
        ));
        assert!(matches!(
            prepared.jet([0.0, 0.125], &mut work, &AtomicBool::new(true)),
            Err(ValidatedGeometryError::Cancelled)
        ));
        assert!(matches!(
            prepared.jet([-0.125, 0.125], &mut work, &cancel),
            Err(ValidatedGeometryError::ParameterOutOfRange)
        ));
    }

    #[test]
    fn rejected_coarse_circle_proposal_keeps_original_knot_fallback_and_limits() {
        let (surface, plane, guide) = generated_circle_fixture(vec![
            Point3::new(2.0, 0.0, 6.0),
            Point3::new(0.0, 2.0, 6.0),
            Point3::new(-2.0, 0.0, 6.0),
        ]);
        let options = ValidatedCurveOptions::default();
        let cancel = AtomicBool::new(false);
        let source = ValidatedFacePlaneSource {
            surface: &surface,
            anchor: Some(ValidatedFacePlaneSourceAnchor::Front([0.0, 7.0])),
            endpoint_coordinate_planes: [None; 2],
        };
        let mut work = ValidatedGeometryBudget::new(2_000_000);
        let prepared = prepared_face_plane_curve(
            &guide,
            source,
            &plane,
            [10.0, 14.0],
            &options,
            &mut work,
            &cancel,
        )
        .unwrap();
        let midpoint = prepared.evaluate(12.0, &mut work, &cancel).unwrap();
        for (axis, expected) in [0.0, 2.0, 6.0].into_iter().enumerate() {
            assert!(
                midpoint.error[axis].lower <= expected && expected <= midpoint.error[axis].upper
            );
        }
        for parameter in [11.0, 13.0] {
            let point = prepared.evaluate(parameter, &mut work, &cancel).unwrap();
            let radius_squared = point.error[0].square() + point.error[1].square();
            assert!(radius_squared.lower <= 4.0 && 4.0 <= radius_squared.upper);
            assert!(point.error[2].lower <= 6.0 && 6.0 <= point.error[2].upper);
            assert!(
                point.error[1].lower > 0.0,
                "retain the original upper semicircle"
            );
        }
        assert!(matches!(
            prepared_face_plane_curve(
                &guide,
                source,
                &plane,
                [10.0, 14.0],
                &options,
                &mut ValidatedGeometryBudget::new(0),
                &cancel,
            ),
            Err(ValidatedGeometryError::ResourceLimit)
        ));
        assert!(matches!(
            prepared_face_plane_curve(
                &guide,
                source,
                &plane,
                [10.0, 14.0],
                &options,
                &mut work,
                &AtomicBool::new(true),
            ),
            Err(ValidatedGeometryError::Cancelled)
        ));
        let no_storage = ValidatedCurveOptions {
            max_retained_bytes: 0,
            ..options
        };
        assert!(matches!(
            prepared_face_plane_curve(
                &guide,
                source,
                &plane,
                [10.0, 14.0],
                &no_storage,
                &mut work,
                &cancel,
            ),
            Err(ValidatedGeometryError::ResourceLimit)
        ));
    }
}
