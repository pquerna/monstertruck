use super::explicit::{
    basis, clipped_parameter, curve_taylor_with_extension, processed_parameter,
    processed_parameter_with_extension, validate_knots,
};
use super::{
    Taylor, ValidatedGeometryBudget, ValidatedGeometryError as E, ValidatedInterval as I,
    ValidatedScalarCurveJet, ZERO, curve_jet_from_taylor, interval_range,
};
use crate::{
    Conic2D, Curve, Curve2D, KnotVector, ParametricCurve, Plane, Surface,
    geometry::MAX_NATIVE_GEOMETRY_NESTING,
};
use std::sync::atomic::AtomicBool;

/// Scalar functional applied to source coefficients before evaluation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScalarProjection<const N: usize> {
    pub(crate) direction: [I; N],
    pub(crate) offset: I,
    pub(crate) center: Option<[f64; N]>,
}

/// Shared work budget and cancellation signal for projection evaluation.
#[derive(Debug)]
pub(crate) struct ProjectionExecution<'a> {
    pub(crate) budget: &'a mut ValidatedGeometryBudget,
    pub(crate) cancel: &'a AtomicBool,
}

/// Encloses `direction dot (curve(t) - origin)` before coordinate separation.
///
/// Line, polynomial/rational spline and plane-carried parameter curves fold the
/// scalar functional into source coefficients before basis evaluation. Other
/// explicit carriers use a conservative source-jet dot product. Implicit curves
/// require their prepared binding's corresponding method instead.
pub fn validated_curve_projected_jet(
    curve: &Curve,
    range: [f64; 2],
    origin: [f64; 3],
    direction: [f64; 3],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<ValidatedScalarCurveJet, E> {
    budget.charge(1, cancel)?;
    if !origin.into_iter().chain(direction).all(f64::is_finite) {
        return Err(E::InvalidInput);
    }
    let t = Taylor::variable(interval_range(range)?, 0);
    let value = projected_curve3d(
        curve,
        t,
        ScalarProjection {
            direction: direction.map(I::point),
            offset: ZERO,
            center: Some(origin),
        },
        0,
        false,
        budget,
        cancel,
    )?
    .finite()?;
    let [[position], [first], [second], _] = curve_jet_from_taylor([value]);
    if ![position, first, second].into_iter().all(I::is_finite) {
        return Err(E::ConditioningFailure);
    }
    Ok(ValidatedScalarCurveJet {
        position,
        first,
        second,
    })
}

fn projected_curve3d(
    curve: &Curve,
    t: Taylor,
    projection: ScalarProjection<3>,
    nesting: usize,
    conic_extension: bool,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Taylor, E> {
    let ScalarProjection {
        direction,
        offset,
        center,
    } = projection;
    budget.charge(1, cancel)?;
    if nesting > MAX_NATIVE_GEOMETRY_NESTING {
        return Err(E::ResourceLimit);
    }
    if !direction.into_iter().chain([offset]).all(I::is_finite) {
        return Err(E::ConditioningFailure);
    }
    Ok(match curve {
        Curve::Line(segment) => {
            let a = coefficient(
                [segment.0.x, segment.0.y, segment.0.z],
                direction,
                offset,
                center,
            )?;
            let b = coefficient(
                [segment.1.x, segment.1.y, segment.1.z],
                direction,
                offset,
                center,
            )?;
            scalar_line(a, b, t)?
        }
        Curve::BsplineCurve(curve) => projected_spline(
            curve.knot_vector(),
            curve.control_points().len(),
            t,
            |i| {
                let p = curve.control_points()[i];
                [p.x, p.y, p.z]
            },
            projection,
            false,
            &mut ProjectionExecution { budget, cancel },
        )?,
        Curve::NurbsCurve(curve) => {
            let constant = match center {
                Some(origin) => offset - dot(origin, direction, ZERO)?,
                None => offset,
            };
            projected_spline(
                curve.knot_vector(),
                curve.control_points().len(),
                t,
                |i| {
                    let p = curve.control_points()[i];
                    [p.x, p.y, p.z, p.w]
                },
                ScalarProjection {
                    direction: [direction[0], direction[1], direction[2], constant],
                    offset: ZERO,
                    center: None,
                },
                true,
                &mut ProjectionExecution { budget, cancel },
            )?
        }
        Curve::ParameterCurve(parameter_curve) => match parameter_curve.surface().as_ref() {
            Surface::Plane(plane) => {
                let [constant, u, v] = plane_coefficients(plane, direction, offset, center)?;
                projected_curve2d(
                    parameter_curve.curve(),
                    t,
                    ScalarProjection {
                        direction: [u, v],
                        offset: constant,
                        center: None,
                    },
                    nesting + 1,
                    conic_extension,
                    budget,
                    cancel,
                )?
            }
            _ => {
                let constant = match center {
                    Some(origin) => offset - dot(origin, direction, ZERO)?,
                    None => offset,
                };
                project_taylor(
                    curve_taylor_with_extension(
                        curve,
                        t,
                        nesting,
                        conic_extension,
                        budget,
                        cancel,
                    )?,
                    direction,
                    constant,
                )
            }
        },
        Curve::TrimmedCurve(wrapper) => {
            let parameter =
                processed_parameter(wrapper.orientation(), Some(wrapper.entity().range()), t)?;
            let matrix = *wrapper.transform();
            if !(0..4).all(|i| (0..4).all(|j| matrix[i][j].is_finite())) {
                return Err(E::InvalidInput);
            }
            if (0..4).all(|i| (0..4).all(|j| matrix[i][j] == if i == j { 1.0 } else { 0.0 })) {
                return projected_curve3d(
                    wrapper.entity().curve(),
                    parameter,
                    projection,
                    nesting + 1,
                    true,
                    budget,
                    cancel,
                );
            }
            let constant = match center {
                Some(origin) => offset - dot(origin, direction, ZERO)?,
                None => offset,
            };
            let mut functional = [ZERO; 4];
            for j in 0..4 {
                functional[j] = dot(
                    [matrix[j][0], matrix[j][1], matrix[j][2]],
                    direction,
                    constant * I::point(matrix[j][3]),
                )?;
            }
            let numerator = projected_curve3d(
                wrapper.entity().curve(),
                parameter,
                ScalarProjection {
                    direction: [functional[0], functional[1], functional[2]],
                    offset: functional[3],
                    center: None,
                },
                nesting + 1,
                true,
                budget,
                cancel,
            )?;
            if (0..3).all(|j| matrix[j][3] == 0.0) && matrix[3][3] == 1.0 {
                return Ok(numerator);
            }
            let denominator = if (0..3).all(|j| matrix[j][3] == 0.0) {
                Taylor::constant(matrix[3][3])
            } else {
                projected_curve3d(
                    wrapper.entity().curve(),
                    parameter,
                    ScalarProjection {
                        direction: [matrix[0][3], matrix[1][3], matrix[2][3]].map(I::point),
                        offset: I::point(matrix[3][3]),
                        center: None,
                    },
                    nesting + 1,
                    true,
                    budget,
                    cancel,
                )?
            };
            numerator * denominator.reciprocal()?
        }
        Curve::IntersectionCurve(_) => return Err(E::Unsupported),
    })
}

fn coefficient<const N: usize>(
    point: [f64; N],
    direction: [I; N],
    offset: I,
    center: Option<[f64; N]>,
) -> Result<I, E> {
    match center {
        Some(center) => Ok(dot_difference(point, center, direction)? + offset),
        None => dot(point, direction, offset),
    }
}

fn dot<const N: usize>(point: [f64; N], direction: [I; N], offset: I) -> Result<I, E> {
    if !point.into_iter().all(f64::is_finite) {
        return Err(E::InvalidInput);
    }
    Ok((0..N)
        .map(|i| I::point(point[i]) * direction[i])
        .fold(offset, |a, b| a + b))
}

fn dot_difference<const N: usize>(
    point: [f64; N],
    origin: [f64; N],
    direction: [I; N],
) -> Result<I, E> {
    if !point.into_iter().chain(origin).all(f64::is_finite) {
        return Err(E::InvalidInput);
    }
    Ok((0..N)
        .map(|i| (I::point(point[i]) - I::point(origin[i])) * direction[i])
        .fold(ZERO, |a, b| a + b))
}

fn project_taylor<const N: usize>(point: [Taylor; N], direction: [I; N], offset: I) -> Taylor {
    (0..N)
        .map(|i| point[i] * Taylor::interval(direction[i]))
        .fold(Taylor::interval(offset), |a, b| a + b)
}

fn line<const N: usize>(
    a: [f64; N],
    b: [f64; N],
    t: Taylor,
    direction: [I; N],
    offset: I,
) -> Result<Taylor, E> {
    let first = dot(a, direction, offset)?;
    let second = dot(b, direction, offset)?;
    scalar_line(first, second, t)
}

fn scalar_line(first: I, second: I, t: Taylor) -> Result<Taylor, E> {
    if !first.is_finite() || !second.is_finite() {
        return Err(E::ConditioningFailure);
    }
    let mut result = Taylor::interval(first) + t * Taylor::interval(second - first);
    if t.0[0][0].lower >= 0.0 && t.0[0][0].upper <= 1.0 {
        result.0[0][0] = result.0[0][0]
            .intersection(first.hull(second))
            .ok_or(E::ConditioningFailure)?;
    }
    Ok(result)
}

fn plane_coefficients(
    plane: &Plane,
    direction: [I; 3],
    offset: I,
    center: Option<[f64; 3]>,
) -> Result<[I; 3], E> {
    let [o, p, q] = plane.defining_points();
    let o = [o.x, o.y, o.z];
    let constant = coefficient(o, direction, offset, center)?;
    let [u, v] = match plane.stored_axes() {
        Some([a, b]) => [
            dot([a.x, a.y, a.z], direction, ZERO)?,
            dot([b.x, b.y, b.z], direction, ZERO)?,
        ],
        None => {
            let a = [p.x, p.y, p.z];
            let b = [q.x, q.y, q.z];
            if !a.into_iter().chain(b).all(f64::is_finite) {
                return Err(E::InvalidInput);
            }
            let project_difference = |point: [f64; 3]| {
                (0..3)
                    .map(|i| (I::point(point[i]) - I::point(o[i])) * direction[i])
                    .fold(ZERO, |a, b| a + b)
            };
            [project_difference(a), project_difference(b)]
        }
    };
    Ok([constant, u, v])
}

fn projected_curve2d(
    curve: &Curve2D,
    t: Taylor,
    projection: ScalarProjection<2>,
    nesting: usize,
    conic_extension: bool,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Taylor, E> {
    let ScalarProjection {
        direction, offset, ..
    } = projection;
    budget.charge(1, cancel)?;
    if nesting > MAX_NATIVE_GEOMETRY_NESTING {
        return Err(E::ResourceLimit);
    }
    if !direction.into_iter().chain([offset]).all(I::is_finite) {
        return Err(E::ConditioningFailure);
    }
    Ok(match curve {
        Curve2D::Line(curve) => line(
            [curve.0.x, curve.0.y],
            [curve.1.x, curve.1.y],
            t,
            direction,
            offset,
        )?,
        Curve2D::BsplineCurve(curve) => projected_spline(
            curve.knot_vector(),
            curve.control_points().len(),
            t,
            |i| {
                let p = curve.control_points()[i];
                [p.x, p.y]
            },
            projection,
            false,
            &mut ProjectionExecution { budget, cancel },
        )?,
        Curve2D::NurbsCurve(curve) => projected_spline(
            curve.knot_vector(),
            curve.control_points().len(),
            t,
            |i| {
                let p = curve.control_points()[i];
                [p.x, p.y, p.z]
            },
            ScalarProjection {
                direction: [direction[0], direction[1], offset],
                offset: ZERO,
                center: None,
            },
            true,
            &mut ProjectionExecution { budget, cancel },
        )?,
        Curve2D::Conic(conic) => {
            let (parameter, matrix, kind) = match conic {
                Conic2D::Ellipse(curve) => (
                    processed_parameter_with_extension(
                        curve.orientation(),
                        curve.try_range_tuple(),
                        t,
                        conic_extension,
                    )?,
                    *curve.transform(),
                    0,
                ),
                Conic2D::Hyperbola(curve) => (
                    processed_parameter_with_extension(
                        curve.orientation(),
                        curve.try_range_tuple(),
                        t,
                        conic_extension,
                    )?,
                    *curve.transform(),
                    1,
                ),
                Conic2D::Parabola(curve) => (
                    processed_parameter_with_extension(
                        curve.orientation(),
                        curve.try_range_tuple(),
                        t,
                        conic_extension,
                    )?,
                    *curve.transform(),
                    2,
                ),
            };
            if !(0..3).all(|i| (0..3).all(|j| matrix[i][j].is_finite())) {
                return Err(E::InvalidInput);
            }
            let a = dot([matrix[0][0], matrix[0][1]], direction, ZERO)?;
            let b = dot([matrix[1][0], matrix[1][1]], direction, ZERO)?;
            let c = dot([matrix[2][0], matrix[2][1]], direction, offset)?;
            let [x, y] = match kind {
                0 => {
                    let (sine, cosine) = parameter.sin_cos();
                    [cosine, sine]
                }
                1 => {
                    let positive = parameter.exp()?;
                    let negative = (-parameter).exp()?;
                    [
                        (positive + negative) * Taylor::constant(0.5),
                        (positive - negative) * Taylor::constant(0.5),
                    ]
                }
                _ => [parameter * parameter, parameter * Taylor::constant(2.0)],
            };
            Taylor::interval(c) + Taylor::interval(a) * x + Taylor::interval(b) * y
        }
        Curve2D::Polyline(curve) => {
            if curve.len() < 2 {
                return Err(E::InvalidInput);
            }
            let range = t.0[0][0];
            if range.lower < 0.0 || range.upper > (curve.len() - 1) as f64 {
                return Err(E::ParameterOutOfRange);
            }
            if range.lower.floor() < range.upper.ceil() - 1.0 {
                return Err(E::NonSmooth);
            }
            let index = (range.midpoint().floor() as usize).min(curve.len() - 2);
            let (p, q) = (curve[index], curve[index + 1]);
            line(
                [p.x, p.y],
                [q.x, q.y],
                t - Taylor::constant(index as f64),
                direction,
                offset,
            )?
        }
        Curve2D::TrimmedCurve(wrapper) => {
            let parameter =
                processed_parameter(wrapper.orientation(), Some(wrapper.entity().range()), t)?;
            let matrix = *wrapper.transform();
            if !(0..3).all(|i| (0..3).all(|j| matrix[i][j].is_finite())) {
                return Err(E::InvalidInput);
            }
            if (0..3).all(|i| (0..3).all(|j| matrix[i][j] == if i == j { 1.0 } else { 0.0 })) {
                return projected_curve2d(
                    wrapper.entity().curve(),
                    parameter,
                    projection,
                    nesting + 1,
                    true,
                    budget,
                    cancel,
                );
            }
            let mut functional = [ZERO; 3];
            for (j, value) in functional.iter_mut().enumerate() {
                *value = dot(
                    [matrix[j][0], matrix[j][1]],
                    direction,
                    if j == 2 { offset } else { ZERO },
                )?;
            }
            projected_curve2d(
                wrapper.entity().curve(),
                parameter,
                ScalarProjection {
                    direction: [functional[0], functional[1]],
                    offset: functional[2],
                    center: None,
                },
                nesting + 1,
                true,
                budget,
                cancel,
            )?
        }
    })
}

/// Evaluates a readonly coefficient-first scalar spline or homogeneous quotient.
///
/// For a rational spline, the final control coordinate is the denominator
/// weight. The scalar numerator includes every supplied direction coefficient;
/// `[axis_unit, -vertex_axis]` on homogeneous controls therefore forms
/// `(N_axis - vertex_axis * W) / W` before basis evaluation. Callers establish
/// that their readonly control view represents the claimed source boundary.
///
/// The scalar functional retains its direction, offset and optional center as
/// one coefficient-space projection. Execution shares the caller's work budget
/// and cancellation signal across every subdivision.
pub(crate) fn projected_spline<const N: usize>(
    knots: &KnotVector,
    count: usize,
    parameter: Taylor,
    control: impl Fn(usize) -> [f64; N],
    projection: ScalarProjection<N>,
    rational: bool,
    execution: &mut ProjectionExecution<'_>,
) -> Result<Taylor, E> {
    let mut stack = [(parameter, 0usize); 49];
    let mut length = 1;
    let mut result: Option<Taylor> = None;
    while length > 0 {
        execution.budget.charge(1, execution.cancel)?;
        length -= 1;
        let (parameter, depth) = stack[length];
        let attempt = projected_spline_cell(
            knots, count, parameter, &control, projection, rational, execution,
        );
        match attempt {
            Ok(value) => {
                result = Some(match result {
                    Some(previous) => previous.hull(value),
                    None => value,
                })
            }
            Err(E::DenominatorContainsZero) => {
                if depth >= 48 {
                    return Err(E::DenominatorContainsZero);
                }
                let interval = parameter.0[0][0];
                let middle = interval.midpoint();
                if middle <= interval.lower || middle >= interval.upper {
                    return Err(E::DenominatorContainsZero);
                }
                execution.budget.charge(2, execution.cancel)?;
                let mut right = parameter;
                right.0[0][0].lower = middle;
                let mut left = parameter;
                left.0[0][0].upper = middle;
                stack[length] = (right, depth + 1);
                stack[length + 1] = (left, depth + 1);
                length += 2;
            }
            Err(error) => return Err(error),
        }
    }
    result.ok_or(E::ParameterOutOfRange)
}

fn projected_spline_cell<const N: usize>(
    knots: &KnotVector,
    count: usize,
    parameter: Taylor,
    control: &impl Fn(usize) -> [f64; N],
    projection: ScalarProjection<N>,
    rational: bool,
    execution: &mut ProjectionExecution<'_>,
) -> Result<Taylor, E> {
    let ScalarProjection {
        direction,
        offset,
        center,
    } = projection;
    let degree = validate_knots(
        knots,
        count,
        parameter.0[0][0],
        false,
        execution.budget,
        execution.cancel,
    )?;
    let mut result: Option<Taylor> = None;
    for span in 0..knots.len() - 1 {
        execution.budget.charge(1, execution.cancel)?;
        if knots[span] == knots[span + 1] {
            continue;
        }
        let Some(parameter) = clipped_parameter(parameter, knots[span], knots[span + 1]) else {
            continue;
        };
        let (start, length, basis) = basis(
            knots,
            count,
            degree,
            span,
            parameter,
            execution.budget,
            execution.cancel,
        )?;
        let mut numerator = Taylor::ZERO;
        let mut denominator = Taylor::ZERO;
        let mut numerator_hull: Option<I> = None;
        let mut weight_hull: Option<I> = None;
        for (index, basis) in basis[..length].iter().enumerate() {
            execution.budget.charge(1, execution.cancel)?;
            let point = control(start + index);
            let coefficient = match center {
                Some(center) => dot_difference(point, center, direction)? + offset,
                None => dot(point, direction, offset)?,
            };
            numerator = numerator + Taylor::interval(coefficient) * *basis;
            numerator_hull =
                Some(numerator_hull.map_or(coefficient, |previous| previous.hull(coefficient)));
            if rational {
                let weight = I::point(point[N - 1]);
                denominator = denominator + Taylor::interval(weight) * *basis;
                weight_hull = Some(weight_hull.map_or(weight, |previous| previous.hull(weight)));
            }
        }
        if span >= degree && span < count {
            if let Some(hull) = numerator_hull {
                if hull.lower == hull.upper {
                    numerator = Taylor::interval(hull);
                } else {
                    numerator.0[0][0] = numerator.0[0][0]
                        .intersection(hull)
                        .ok_or(E::ConditioningFailure)?;
                }
            }
            if let Some(hull) = weight_hull {
                if hull.lower == hull.upper {
                    denominator = Taylor::interval(hull);
                } else {
                    denominator.0[0][0] = denominator.0[0][0]
                        .intersection(hull)
                        .ok_or(E::ConditioningFailure)?;
                }
            }
        }
        let value = if rational {
            numerator * denominator.reciprocal()?
        } else {
            numerator
        };
        result = Some(match result {
            Some(previous) => previous.hull(value),
            None => value,
        });
    }
    result.ok_or(E::ParameterOutOfRange)
}

#[cfg(test)]
mod scalar_projection_regressions {
    use super::*;
    use crate::{BsplineCurve, Line, NurbsCurve, Point3, Tolerance, Vector4};

    #[test]
    fn polynomial_line_preserves_small_plane_residual() {
        let curve = Curve::Line(Line(
            Point3::new(3.0, 5.0, 7.0),
            Point3::new(11.0, 13.0, 15.0),
        ));
        let jet = validated_curve_projected_jet(
            &curve,
            [0.0, 1.0],
            [3.0, 5.0, 7.0],
            [1.0, -1.0, 0.0],
            &mut ValidatedGeometryBudget::new(100_000),
            &AtomicBool::new(false),
        )
        .expect("A finite polynomial line has a bounded scalar projection.");
        assert!(jet.position.contains_zero() && jet.position.magnitude() < 1.0e-12);
        assert!(jet.first.contains_zero() && jet.first.magnitude() < 1.0e-12);
        assert_eq!(jet.second, ZERO);
    }

    #[test]
    fn rational_planarity_survives_whole_span_basis_evaluation() {
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let curve = Curve::NurbsCurve(NurbsCurve::new(BsplineCurve::new(
            KnotVector::from(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            vec![
                Vector4::new(1.0, 0.0, 0.0, 1.0),
                Vector4::new(h, h, 0.0, h),
                Vector4::new(0.0, 1.0, 0.0, 1.0),
            ],
        )));
        let jet = validated_curve_projected_jet(
            &curve,
            [0.0, 1.0],
            [13.0, 17.0, 0.0],
            [0.0, 0.0, 1.0],
            &mut ValidatedGeometryBudget::new(100_000),
            &AtomicBool::new(false),
        )
        .expect("A positive-denominator rational span retains its coefficient plane.");
        assert_eq!([jet.position, jet.first, jet.second], [ZERO; 3]);
    }

    #[test]
    fn native_affine_uv_scalar_projection_retains_constant_value_and_offset_once() {
        use crate::{
            Conic2D, Matrix3, ParameterCurve, Plane, Point2, Processor, TrimmedCurve, UnitCircle,
        };
        let matrix = Matrix3::new(0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
        let line = Curve2D::TrimmedCurve(Processor::with_transform(
            TrimmedCurve::new(
                Box::new(Curve2D::Line(Line(
                    Point2::new(0.0, 0.0),
                    Point2::new(1.0, 0.0),
                ))),
                (0.5, 0.75),
            ),
            matrix,
        ));
        let circle = Curve2D::Conic(Conic2D::Ellipse(Processor::with_transform(
            TrimmedCurve::new(UnitCircle::<Point2>::new(), (0.5, 0.75)),
            matrix,
        )));
        for (uv, circular) in [(line, false), (circle, true)] {
            let curve = Curve::ParameterCurve(ParameterCurve::new(
                uv,
                Box::new(Surface::Plane(Plane::xy())),
            ));
            for t in [0.5_f64, 0.625, 0.75] {
                let expected = Point3::new(1.0, if circular { t.sin() } else { 0.0 }, 0.0);
                assert!(curve.subs(t).near(&expected));
                assert_eq!(curve.der(t).x, 0.0);
            }
            let jet = validated_curve_projected_jet(
                &curve,
                [0.5, 0.75],
                [3.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                &mut ValidatedGeometryBudget::new(100_000),
                &AtomicBool::new(false),
            )
            .unwrap();
            assert!(jet.position.lower <= -4.0 && -4.0 <= jet.position.upper);
            assert!(jet.position.upper - jet.position.lower < 1.0e-10);
        }
    }
}
