//! Outward-rounded source-geometry arithmetic for validated native queries.
//!
//! Bounds describe the real geometry whose coefficients are the stored binary64
//! values. They do not assert that the legacy sampled or Newton-fallback
//! evaluators are exact. Every operation rejects nonfinite final enclosures.

use std::ops::{Add, Mul, Neg, Sub};
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;

mod explicit;
mod normal;
mod projection;
mod smoothness;
pub(crate) use explicit::{
    processed_parameter, validated_curve_piecewise_jet, validated_curve_position_bounds,
    validated_projection_leader_jet,
};
pub use explicit::{
    validated_boundary_jet, validated_curve_jet, validated_curve2d_jet, validated_surface_jet,
    validated_surface_piecewise_jet,
};
pub use normal::validated_surface_geometric_normal;
pub use projection::validated_curve_projected_jet;
pub(crate) use projection::{
    ProjectionExecution, ScalarProjection, SurfaceFunctional, projected_spline,
    projected_surface_curve_jet, projected_surface_functionals, projected_surface_taylor,
};
pub(crate) use smoothness::{curve_span_smooth, surface_box_smooth};

/// A closed, outward-rounded real interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValidatedInterval {
    /// Inclusive lower bound.
    pub lower: f64,
    /// Inclusive upper bound.
    pub upper: f64,
}

/// The actual failed inequality in a terminal original-plane candidate proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidatedPlaneCoverageRejection {
    /// Query lower bound lies below the original uniqueness tube.
    TubeLower,
    /// Query upper bound lies above the original uniqueness tube.
    TubeUpper,
    /// The prepared coordinate has no consistent strict derivative sign.
    CoordinateMonotonicity,
    /// An increasing coordinate's low-end residual is not nonpositive.
    IncreasingLowBracket,
    /// An increasing coordinate's high-end residual is not nonnegative.
    IncreasingHighBracket,
    /// A decreasing coordinate's low-end residual is not nonnegative.
    DecreasingLowBracket,
    /// A decreasing coordinate's high-end residual is not nonpositive.
    DecreasingHighBracket,
}

/// Fixed-size evidence from an existing terminal-depth candidate attempt.
///
/// This is a refusal diagnostic, never a geometry or continuation receipt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValidatedPlaneCandidateWitness {
    /// Index in the original query's unchanged candidate enumeration.
    pub candidate_index: usize,
    /// First piece in the actual attempted contiguous prepared-piece group.
    pub first_piece_index: usize,
    /// Number of pieces in that group, at most three.
    pub piece_count: u8,
    /// The deciding inequality, with no altered error classification.
    pub rejection: ValidatedPlaneCoverageRejection,
    /// Source-system coordinate 0..3 for tube rejection; 4 for scalar projection.
    pub coordinate: u8,
    /// Tube rejection: offending query bound, then original tube bound.
    /// Other stages: the raw derivative or endpoint residual interval bounds.
    pub bounds: [f64; 2],
}

/// Failed stage of an existing terminal-depth common-junction proof.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidatedPlaneJunctionRejection {
    /// The first end has no retained shared coordinate.
    MissingFirstCoordinate,
    /// The second end has no retained shared coordinate.
    MissingSecondCoordinate,
    /// The two stored endpoint planes are not coefficientwise equal up to sign.
    DifferentCoordinates,
    /// Query lower bound lies below the common original tube.
    TubeLower,
    /// Query upper bound lies above the common original tube.
    TubeUpper,
    /// The first sealed path is not strictly monotone in this world coordinate.
    FirstMonotonicity,
    /// The second sealed path is not strictly monotone in this world coordinate.
    SecondMonotonicity,
    /// Both paths move in the same direction away from their proposed junction.
    SameAwayDirection,
    /// The cell's selected world-coordinate interval is degenerate.
    DegenerateCoordinate,
    /// The actual lower outer endpoint does not bracket the cell.
    OuterLowerBracket,
    /// The actual upper outer endpoint does not bracket the cell.
    OuterUpperBracket,
    /// Common original-root uniqueness refused, including disjoint retained tubes.
    SharedRootUniqueness,
    /// Fresh whole-cell fixed-coordinate uniqueness refused.
    CoordinateUniqueness,
}

/// Typed retryable refusal from a fresh original-junction certificate.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidatedPlaneJunctionCondition {
    /// Inclusion was not established.
    Unresolved,
    /// Interval conditioning failed.
    ConditioningFailure,
    /// Original surfaces were not certified transverse.
    NonTransverse,
    /// A source Jacobian was singular.
    Singular,
    /// A rational denominator met zero.
    DenominatorContainsZero,
    /// An admitted parameter range was exceeded.
    ParameterOutOfRange,
    /// A source knot was nonsmooth.
    NonSmooth,
}

/// Compact evidence from an actual terminal-depth junction attempt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValidatedPlaneJunctionWitness {
    /// Original candidate enumeration; retained query caps bound indices to u32.
    pub candidate_indices: [u32; 2],
    /// Front=false, back=true; None when that end was not reached.
    pub ends: [Option<bool>; 2],
    /// The failed precondition or fresh original certificate.
    pub rejection: ValidatedPlaneJunctionRejection,
    /// Source-system axis for tube failures; world axis for path failures.
    /// Shared-root failure uses a source-system axis when its old tubes separate.
    pub coordinate: Option<u8>,
    /// Raw deciding bounds, or None when the stage has no scalar bounds.
    /// Shared-root failure reports old tube separation, never a period shift.
    pub bounds: Option<[f64; 2]>,
    /// The actual retryable fresh-certificate refusal, if one was attempted.
    pub condition: Option<ValidatedPlaneJunctionCondition>,
}

/// Compact evidence from an actual original-endpoint tube containment refusal.
///
/// Intervals remain in the unchanged original source-system gauge. This is a
/// diagnostic, not an endpoint receipt or permission to rewrap a parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValidatedPlaneEndpointWitness {
    /// Actual matched candidate in the original query.
    pub candidate_index: u32,
    /// Actual first/last prepared piece selected for that bound endpoint.
    pub piece_index: u32,
    /// Actual original coedge occurrence bound to that candidate endpoint.
    pub coedge_index: u32,
    /// Original face carrier index, zero or one, in the candidate source system.
    pub source_index: u8,
    /// Front=false, back=true in the unchanged native candidate parameter.
    pub endpoint_is_back: bool,
    /// First failing source-system coordinate, zero through three.
    /// Coordinates `2*source_index..2*source_index+2` are original face UV;
    /// the other pair is the unchanged original section-plane chart.
    pub coordinate: u8,
    /// Complete retained uniqueness-tube interval on the deciding axis.
    pub tube_bounds: [f64; 2],
    /// Complete query interval on that same original-gauge axis.
    pub query_bounds: [f64; 2],
}

/// Fixed-size evidence from a terminal coverage or mandatory endpoint attempt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ValidatedPlaneCoverageWitness {
    /// An individual candidate or contiguous prepared-piece group refused.
    Candidate(ValidatedPlaneCandidateWitness),
    /// A proposed joined continuation refused.
    Junction(ValidatedPlaneJunctionWitness),
    /// Mandatory original endpoint preseal refused tube containment.
    Endpoint(ValidatedPlaneEndpointWitness),
}

/// Failure to establish a native interval geometry certificate.
#[derive(Clone, Copy, Debug, PartialEq, Error)]
pub enum ValidatedGeometryError {
    /// Cooperative cancellation was requested.
    #[error("validated geometry query cancelled")]
    Cancelled,
    /// The cumulative work allowance was exhausted.
    #[error("validated geometry work allowance exhausted")]
    ResourceLimit,
    /// An input or source coefficient is invalid or nonfinite.
    #[error("invalid validated geometry input")]
    InvalidInput,
    /// The requested parameter lies outside the admitted source domain.
    #[error("parameter outside admitted source domain")]
    ParameterOutOfRange,
    /// This source representation has no implemented interval arithmetic.
    #[error("unsupported source geometry arithmetic")]
    Unsupported,
    /// A rational denominator could not be separated from zero.
    #[error("rational denominator contains zero")]
    DenominatorContainsZero,
    /// The requested interval crosses an insufficiently smooth source knot.
    #[error("source interval crosses a nonsmooth knot")]
    NonSmooth,
    /// Interval conditioning or finite arithmetic could not be established.
    #[error("interval geometry conditioning failed")]
    ConditioningFailure,
    /// Bounded subdivision did not establish the required inclusion.
    #[error("validated geometry inclusion unresolved")]
    Unresolved,
    /// Complete original-plane coverage remained unresolved at a named proof operation.
    #[error(
        "validated plane coverage unresolved during {operation}; original parameter box {parameter_box:?}; terminal coverage witness {witness:?}"
    )]
    PlaneCoverageUnresolved {
        /// The bounded proof operation that failed, without changing its classification.
        operation: &'static str,
        /// Actual unresolved closed box in the unchanged original covering chart.
        /// Absent when the failing proof has no single original source cell.
        parameter_box: Option<[[f64; 2]; 2]>,
        /// Actual endpoint preseal or best-progress terminal-depth coverage refusal.
        witness: Option<ValidatedPlaneCoverageWitness>,
    },
    /// Continuation could not establish a single connected local branch.
    #[error("ambiguous intersection continuation branch")]
    AmbiguousBranch,
    /// The source surfaces are not certified transverse.
    #[error("intersection is not certified transverse")]
    NonTransverse,
    /// The source chart or projected intersection Jacobian is singular.
    #[error("singular source chart or intersection Jacobian")]
    Singular,
}

/// A cumulative allowance shared by all native geometry calls in one query.
#[derive(Debug)]
pub struct ValidatedGeometryBudget {
    limit: usize,
    consumed: usize,
}

impl ValidatedGeometryBudget {
    /// Creates a cumulative allowance; zero admits no work.
    pub const fn new(max_work: usize) -> Self {
        Self {
            limit: max_work,
            consumed: 0,
        }
    }
    /// Returns the unspent allowance.
    pub const fn remaining(&self) -> usize { self.limit - self.consumed }
    /// Returns work charged, including unsuccessful certification attempts.
    pub const fn consumed(&self) -> usize { self.consumed }
    /// Charges work before a node, iteration, coefficient scan, or allocation.
    pub fn charge(
        &mut self,
        work: usize,
        cancel: &AtomicBool,
    ) -> Result<(), ValidatedGeometryError> {
        if cancel.load(Ordering::Relaxed) {
            Err(ValidatedGeometryError::Cancelled)
        } else if work > self.remaining() {
            self.consumed = self.limit;
            Err(ValidatedGeometryError::ResourceLimit)
        } else {
            self.consumed += work;
            Ok(())
        }
    }
}

/// Position and native-parameter derivative enclosures for a spatial curve.
#[derive(Clone, Copy, Debug)]
pub struct ValidatedCurveJet {
    /// Position enclosure.
    pub position: [ValidatedInterval; 3],
    /// First derivative with respect to the native parameter.
    pub first: [ValidatedInterval; 3],
    /// Second derivative with respect to the native parameter.
    pub second: [ValidatedInterval; 3],
    /// Third derivative; implicit curves do not provide this order.
    pub third: [ValidatedInterval; 3],
}

/// Position and native-parameter derivative enclosures for a planar curve.
#[derive(Clone, Copy, Debug)]
pub struct ValidatedCurveJet2D {
    /// Position enclosure.
    pub position: [ValidatedInterval; 2],
    /// First derivative.
    pub first: [ValidatedInterval; 2],
    /// Second derivative.
    pub second: [ValidatedInterval; 2],
    /// Third derivative.
    pub third: [ValidatedInterval; 2],
}

/// Position and first/second native-chart derivatives of a source surface.
#[derive(Clone, Copy, Debug)]
pub struct ValidatedSurfaceJet {
    /// Position enclosure.
    pub position: [ValidatedInterval; 3],
    /// Derivative in the first chart coordinate.
    pub du: [ValidatedInterval; 3],
    /// Derivative in the second chart coordinate.
    pub dv: [ValidatedInterval; 3],
    /// Twice differentiated in the first chart coordinate.
    pub duu: [ValidatedInterval; 3],
    /// Mixed chart derivative.
    pub duv: [ValidatedInterval; 3],
    /// Twice differentiated in the second chart coordinate.
    pub dvv: [ValidatedInterval; 3],
}

/// Correlated scalar bounds for `direction dot (curve(t) - origin)`.
#[derive(Clone, Copy, Debug)]
pub struct ValidatedScalarCurveJet {
    /// Complete positional enclosure on the requested interval.
    pub position: ValidatedInterval,
    /// First native-parameter derivative enclosure.
    pub first: ValidatedInterval,
    /// Second native-parameter derivative enclosure.
    pub second: ValidatedInterval,
}

pub(crate) const ZERO: ValidatedInterval = ValidatedInterval {
    lower: 0.0,
    upper: 0.0,
};
pub(crate) const ONE: ValidatedInterval = ValidatedInterval {
    lower: 1.0,
    upper: 1.0,
};

impl ValidatedInterval {
    /// Constructs a finite ordered interval.
    pub fn new(lower: f64, upper: f64) -> Result<Self, ValidatedGeometryError> {
        if lower.is_finite() && upper.is_finite() && lower <= upper {
            Ok(Self { lower, upper })
        } else {
            Err(ValidatedGeometryError::InvalidInput)
        }
    }
    /// Returns whether this interval contains zero.
    pub fn contains_zero(self) -> bool { self.lower <= 0.0 && self.upper >= 0.0 }
    /// Returns whether both endpoints are finite and ordered.
    pub fn is_finite(self) -> bool {
        self.lower.is_finite() && self.upper.is_finite() && self.lower <= self.upper
    }
    /// Returns an interior floating-point representative, without a proof claim.
    pub fn midpoint(self) -> f64 { self.lower * 0.5 + self.upper * 0.5 }
    /// Returns an outward-rounded width.
    pub fn width(self) -> f64 { (self.upper - self.lower).next_up() }
    /// Returns an upper bound on the absolute value.
    pub fn magnitude(self) -> f64 { self.lower.abs().max(self.upper.abs()) }
    /// Returns the smallest interval containing both arguments.
    pub fn hull(self, other: Self) -> Self {
        Self {
            lower: self.lower.min(other.lower),
            upper: self.upper.max(other.upper),
        }
    }
    /// Returns the intersection, if nonempty.
    pub fn intersection(self, other: Self) -> Option<Self> {
        let result = Self {
            lower: self.lower.max(other.lower),
            upper: self.upper.min(other.upper),
        };
        (result.lower <= result.upper).then_some(result)
    }
    /// Returns the outward-rounded reciprocal, refusing zero denominators.
    pub fn reciprocal(self) -> Result<Self, ValidatedGeometryError> {
        if self.contains_zero() || !self.is_finite() {
            Err(ValidatedGeometryError::DenominatorContainsZero)
        } else if self == ONE || self == -ONE {
            Ok(self)
        } else {
            Ok(Self {
                lower: (1.0 / self.upper).next_down(),
                upper: (1.0 / self.lower).next_up(),
            })
        }
    }
    /// Returns the outward-rounded square, retaining nonnegativity.
    pub fn square(self) -> Self {
        let maximum = self.magnitude();
        let minimum = if self.contains_zero() {
            0.0
        } else {
            self.lower.abs().min(self.upper.abs())
        };
        Self {
            lower: (minimum * minimum).next_down().max(0.0),
            upper: (maximum * maximum).next_up(),
        }
    }
    pub(crate) const fn point(value: f64) -> Self {
        Self {
            lower: value,
            upper: value,
        }
    }
    pub(crate) fn strictly_contains(self, rhs: Self) -> bool {
        self.lower < rhs.lower && rhs.upper < self.upper
    }
}

impl Add for ValidatedInterval {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        if self == ZERO {
            return rhs;
        }
        if rhs == ZERO {
            return self;
        }
        if self.lower == self.upper && rhs.lower == rhs.upper && self.lower == -rhs.lower {
            return ZERO;
        }
        if self.lower == self.upper
            && rhs.lower == rhs.upper
            && self.lower.is_finite()
            && rhs.lower.is_finite()
        {
            // Error-free TwoSum proves when a constant sum is exactly representable.
            let sum = self.lower + rhs.lower;
            let virtual_b = sum - self.lower;
            let virtual_a = sum - virtual_b;
            let roundoff_a = self.lower - virtual_a;
            let roundoff_b = rhs.lower - virtual_b;
            let error = roundoff_a + roundoff_b;
            if error == 0.0
                && [sum, virtual_b, virtual_a, roundoff_a, roundoff_b]
                    .into_iter()
                    .all(f64::is_finite)
            {
                return Self::point(sum);
            }
        }
        Self {
            lower: (self.lower + rhs.lower).next_down(),
            upper: (self.upper + rhs.upper).next_up(),
        }
    }
}
impl Neg for ValidatedInterval {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            lower: -self.upper,
            upper: -self.lower,
        }
    }
}
impl Sub for ValidatedInterval {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        if self.lower == self.upper && rhs.lower == rhs.upper && self.lower == rhs.lower {
            ZERO
        } else {
            self + -rhs
        }
    }
}
impl Mul for ValidatedInterval {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        if self == ZERO || rhs == ZERO {
            return ZERO;
        }
        if self == ONE {
            return rhs;
        }
        if rhs == ONE {
            return self;
        }
        if self == -ONE {
            return -rhs;
        }
        if rhs == -ONE {
            return -self;
        }
        let products = [
            self.lower * rhs.lower,
            self.lower * rhs.upper,
            self.upper * rhs.lower,
            self.upper * rhs.upper,
        ];
        Self {
            lower: products
                .into_iter()
                .fold(f64::INFINITY, f64::min)
                .next_down(),
            upper: products
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max)
                .next_up(),
        }
    }
}

/// A third-order bivariate Taylor jet; entries contain divided derivatives.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Taylor(pub(crate) [[ValidatedInterval; 4]; 4]);

impl Taylor {
    pub(crate) const ZERO: Self = Self([[ZERO; 4]; 4]);
    pub(crate) fn constant(value: f64) -> Self { Self::interval(ValidatedInterval::point(value)) }
    pub(crate) fn interval(value: ValidatedInterval) -> Self {
        let mut result = Self::ZERO;
        result.0[0][0] = value;
        result
    }
    pub(crate) fn variable(value: ValidatedInterval, axis: usize) -> Self {
        let mut result = Self::interval(value);
        result.0[usize::from(axis == 0)][usize::from(axis == 1)] = ONE;
        result
    }
    pub(crate) fn finite(self) -> Result<Self, ValidatedGeometryError> {
        if (0..4).all(|i| (0..4 - i).all(|j| self.0[i][j].is_finite())) {
            Ok(self)
        } else {
            Err(ValidatedGeometryError::ConditioningFailure)
        }
    }
    pub(crate) fn hull(self, other: Self) -> Self {
        Self(std::array::from_fn(|i| {
            std::array::from_fn(|j| self.0[i][j].hull(other.0[i][j]))
        }))
    }
    pub(crate) fn reciprocal(self) -> Result<Self, ValidatedGeometryError> {
        let inverse = self.0[0][0].reciprocal()?;
        let mut delta = self;
        delta.0[0][0] = ZERO;
        let delta = delta * Self::interval(-inverse);
        let square = delta * delta;
        Ok((Self::constant(1.0) + delta + square + square * delta) * Self::interval(inverse))
    }
    pub(crate) fn sin_cos(self) -> (Self, Self) {
        let (sine, cosine) = sin_cos_interval(self.0[0][0]);
        let mut delta = self;
        delta.0[0][0] = ZERO;
        let square = delta * delta * Self::constant(0.5);
        let third = ValidatedInterval {
            lower: (1.0_f64 / 3.0).next_down(),
            upper: (1.0_f64 / 3.0).next_up(),
        };
        let cube = square * delta * Self::interval(third);
        (
            Self::interval(sine) + Self::interval(cosine) * delta
                - Self::interval(sine) * square
                - Self::interval(cosine) * cube,
            Self::interval(cosine) - Self::interval(sine) * delta - Self::interval(cosine) * square
                + Self::interval(sine) * cube,
        )
    }
    pub(crate) fn exp(self) -> Result<Self, ValidatedGeometryError> {
        let value = exp_interval(self.0[0][0])?;
        let mut delta = self;
        delta.0[0][0] = ZERO;
        let square = delta * delta * Self::constant(0.5);
        let cube = square * delta * Self::interval(ValidatedInterval::point(3.0).reciprocal()?);
        (Self::constant(1.0) + delta + square + cube)
            .mul(Self::interval(value))
            .finite()
    }
}
impl Add for Taylor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(std::array::from_fn(|i| {
            std::array::from_fn(|j| self.0[i][j] + rhs.0[i][j])
        }))
    }
}
impl Neg for Taylor {
    type Output = Self;
    fn neg(self) -> Self { Self(self.0.map(|row| row.map(Neg::neg))) }
}
impl Sub for Taylor {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { self + -rhs }
}
impl Mul for Taylor {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self(std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if i + j > 3 {
                    ZERO
                } else {
                    (0..=i)
                        .flat_map(|a| (0..=j).map(move |b| self.0[a][b] * rhs.0[i - a][j - b]))
                        .fold(ZERO, Add::add)
                }
            })
        }))
    }
}

/// Encloses mathematical sine and cosine without assuming libm error bounds.
///
/// Reduction uses an outward enclosure of pi and an explicit Taylor remainder.
/// Wide or extremely large arguments conservatively return the unit interval.
pub fn validated_sin_cos(
    value: ValidatedInterval,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<(ValidatedInterval, ValidatedInterval), ValidatedGeometryError> {
    budget.charge(41, cancel)?;
    if !value.is_finite() {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    Ok(sin_cos_interval(value))
}

fn sin_cos_interval(value: ValidatedInterval) -> (ValidatedInterval, ValidatedInterval) {
    if value == ZERO {
        return (ZERO, ONE);
    }
    let unit = ValidatedInterval {
        lower: -1.0,
        upper: 1.0,
    };
    if !value.is_finite() || value.width() > 1.0 || value.magnitude() > 1_099_511_627_776.0 {
        return (unit, unit);
    }
    // The adjacent binary64 values strictly enclose mathematical pi/2.
    let half_pi = ValidatedInterval {
        lower: std::f64::consts::FRAC_PI_2.next_down(),
        upper: std::f64::consts::FRAC_PI_2.next_up(),
    };
    let quadrant = (value.midpoint() / std::f64::consts::FRAC_PI_2).round();
    let reduced = value - half_pi * ValidatedInterval::point(quadrant);
    if reduced.magnitude() > 2.0 {
        return (unit, unit);
    }
    let mut power = ONE;
    let mut factorial = ONE;
    let mut sine = ZERO;
    let mut cosine = ZERO;
    for order in 0..=40 {
        if order > 0 {
            power = power * reduced;
            factorial = factorial * ValidatedInterval::point(order as f64);
        }
        let inverse_factorial = ValidatedInterval {
            lower: (1.0 / factorial.upper).next_down(),
            upper: (1.0 / factorial.lower).next_up(),
        };
        let term = power * inverse_factorial;
        match order % 4 {
            0 => cosine = cosine + term,
            1 => sine = sine + term,
            2 => cosine = cosine - term,
            _ => sine = sine - term,
        }
    }
    let factorial = factorial * ValidatedInterval::point(41.0);
    let error = (power * reduced).magnitude() * (1.0 / factorial.lower).next_up();
    let remainder = ValidatedInterval {
        lower: -error.next_up(),
        upper: error.next_up(),
    };
    let sine = (sine + remainder).intersection(unit).unwrap_or(unit);
    let cosine = (cosine + remainder).intersection(unit).unwrap_or(unit);
    match (quadrant as i64).rem_euclid(4) {
        0 => (sine, cosine),
        1 => (cosine, -sine),
        2 => (-sine, -cosine),
        _ => (-cosine, sine),
    }
}

fn exp_interval(value: ValidatedInterval) -> Result<ValidatedInterval, ValidatedGeometryError> {
    if !value.is_finite() || value.magnitude() > 700.0 {
        return Err(ValidatedGeometryError::ConditioningFailure);
    }
    let mut reduced = value;
    let mut squarings = 0;
    while reduced.magnitude() > 0.5 {
        reduced = reduced * ValidatedInterval::point(0.5);
        squarings += 1;
    }
    let mut result = ONE;
    let mut term = ONE;
    for order in 1..=30 {
        term = term * reduced * ValidatedInterval::point(order as f64).reciprocal()?;
        result = result + term;
    }
    let error = (term * reduced * ValidatedInterval::point(31.0).reciprocal()?).magnitude();
    let remainder = ValidatedInterval {
        lower: -(2.0 * error).next_up(),
        upper: (2.0 * error).next_up(),
    };
    result = result + remainder;
    for _ in 0..squarings {
        result = result.square();
    }
    if result.is_finite() {
        Ok(result)
    } else {
        Err(ValidatedGeometryError::ConditioningFailure)
    }
}

pub(crate) fn interval_range(range: [f64; 2]) -> Result<ValidatedInterval, ValidatedGeometryError> {
    ValidatedInterval::new(range[0], range[1])
}

pub(crate) fn curve_jet_from_taylor<const N: usize>(
    value: [Taylor; N],
) -> [[ValidatedInterval; N]; 4] {
    [
        value.map(|v| v.0[0][0]),
        value.map(|v| v.0[1][0]),
        value.map(|v| v.0[2][0] * ValidatedInterval::point(2.0)),
        value.map(|v| v.0[3][0] * ValidatedInterval::point(6.0)),
    ]
}
