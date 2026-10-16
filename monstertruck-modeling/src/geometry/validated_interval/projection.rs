use super::explicit::{
    basis, clipped_parameter, curve_taylor_with_extension, curve2d_taylor, processed_parameter,
    processed_parameter_with_extension, surface_taylor_with_policy, validate_knots,
};
use super::{
    Taylor, ValidatedGeometryBudget, ValidatedGeometryError as E, ValidatedInterval as I,
    ValidatedScalarCurveJet, ZERO, curve_jet_from_taylor, interval_range,
};
use crate::{
    Conic2D, Curve, Curve2D, KnotVector, Matrix4, ParametricCurve, Plane, Surface,
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

/// Projects the original native UV continuation, including its actual derivatives.
pub(crate) fn projected_surface_curve_jet(
    surface: &Surface,
    curve: &Curve2D,
    range: [f64; 2],
    projection: ScalarProjection<3>,
    execution: &mut ProjectionExecution<'_>,
) -> Result<ValidatedScalarCurveJet, E> {
    let uv = curve2d_taylor(
        curve,
        Taylor::variable(interval_range(range)?, 0),
        0,
        false,
        execution.budget,
        execution.cancel,
    )?;
    let value = projected_surface_taylor(surface, uv, projection, 0, execution)?;
    let [[position], [first], [second], _] = curve_jet_from_taylor([value]);
    Ok(ValidatedScalarCurveJet {
        position,
        first,
        second,
    })
}

/// Pulls one interval functional through native surface coefficients.
///
/// The UV Taylor pair belongs to the original chart, not a fitted spatial
/// surrogate. Polynomial tensors use partition unity before basis expansion;
/// rational tensors retain their original denominator even for a zero scalar.
/// Unsupported scalar specializations retain the full certified evaluation.
pub(crate) fn projected_surface_taylor(
    surface: &Surface,
    [u, v]: [Taylor; 2],
    projection: ScalarProjection<3>,
    nesting: usize,
    execution: &mut ProjectionExecution<'_>,
) -> Result<Taylor, E> {
    execution.budget.charge(1, execution.cancel)?;
    if nesting > MAX_NATIVE_GEOMETRY_NESTING {
        return Err(E::ResourceLimit);
    }
    if !projection
        .direction
        .into_iter()
        .chain([projection.offset])
        .all(I::is_finite)
    {
        return Err(E::ConditioningFailure);
    }
    u.finite()?;
    v.finite()?;
    let value = match surface {
        Surface::Plane(plane) => {
            let [constant, a, b] = plane_coefficients(
                plane,
                projection.direction,
                projection.offset,
                projection.center,
            )?;
            Taylor::interval(constant) + u * Taylor::interval(a) + v * Taylor::interval(b)
        }
        Surface::ExtrusionSurface(surface) => {
            let vector = surface.extruding_vector();
            let slope = dot([vector.x, vector.y, vector.z], projection.direction, ZERO)?;
            projected_curve3d(
                surface.entity_curve(),
                u,
                projection,
                nesting + 1,
                false,
                execution.budget,
                execution.cancel,
            )? + v * Taylor::interval(slope)
        }
        Surface::BsplineSurface(surface) => projected_tensor(
            (surface.knot_vectors(), surface.control_points()),
            [u, v],
            |point| [point.x, point.y, point.z],
            projection,
            false,
            execution,
        )?,
        Surface::NurbsSurface(surface) => {
            let constant = projection_constant(projection)?;
            projected_tensor(
                (surface.knot_vectors(), surface.control_points()),
                [u, v],
                |point| [point.x, point.y, point.z, point.w],
                ScalarProjection {
                    direction: [
                        projection.direction[0],
                        projection.direction[1],
                        projection.direction[2],
                        constant,
                    ],
                    offset: ZERO,
                    center: None,
                },
                true,
                execution,
            )?
        }
        Surface::RevolutionSurface(processor) => {
            let (u, v) = if processor.orientation() {
                (u, v)
            } else {
                (v, u)
            };
            let surface = processor.entity();
            let origin = surface.origin();
            let axis = surface.axis();
            let origin = [origin.x, origin.y, origin.z];
            let axis = [axis.x, axis.y, axis.z];
            if !axis.into_iter().chain(origin).all(f64::is_finite) {
                return Err(E::InvalidInput);
            }
            placed_projection(*processor.transform(), projection, |projection| {
                let normal = projection.direction;
                // Project the native rotation coefficients before separating the profile.
                let axial = std::array::from_fn(|j| {
                    (0..3)
                        .map(|i| normal[i] * (I::point(axis[i]) * I::point(axis[j])))
                        .fold(ZERO, |a, b| a + b)
                });
                let radial = std::array::from_fn(|j| {
                    (0..3)
                        .map(|i| {
                            normal[i]
                                * (I::point(if i == j { 1.0 } else { 0.0 })
                                    - I::point(axis[i]) * I::point(axis[j]))
                        })
                        .fold(ZERO, |a, b| a + b)
                });
                let cross = [
                    normal[1] * I::point(axis[2]) - normal[2] * I::point(axis[1]),
                    normal[2] * I::point(axis[0]) - normal[0] * I::point(axis[2]),
                    normal[0] * I::point(axis[1]) - normal[1] * I::point(axis[0]),
                ];
                let constant = coefficient(origin, normal, projection.offset, projection.center)?;
                let mut profile = |direction: [I; 3], required: bool| {
                    if !required && direction == [ZERO; 3] {
                        Ok(Taylor::ZERO)
                    } else {
                        projected_curve3d(
                            surface.entity_curve(),
                            u,
                            ScalarProjection {
                                direction,
                                offset: ZERO,
                                center: Some(origin),
                            },
                            nesting + 1,
                            false,
                            execution.budget,
                            execution.cancel,
                        )
                    }
                };
                // Even a constant functional must admit the original profile.
                let axial_value = profile(axial, radial == [ZERO; 3] && cross == [ZERO; 3])?;
                let radial_value = profile(radial, false)?;
                let cross_value = profile(cross, false)?;
                if radial == [ZERO; 3] && cross == [ZERO; 3] {
                    Ok(Taylor::interval(constant) + axial_value)
                } else {
                    let (sine, cosine) = v.sin_cos();
                    Ok(Taylor::interval(constant)
                        + axial_value
                        + radial_value * cosine
                        + cross_value * sine)
                }
            })?
        }
        Surface::SphericalSurface(processor) => {
            let (u, v) = if processor.orientation() {
                (u, v)
            } else {
                (v, u)
            };
            let sphere = processor.entity();
            let center = sphere.center();
            let radius = sphere.radius();
            if !radius.is_finite() {
                return Err(E::InvalidInput);
            }
            let (su, cu) = u.sin_cos();
            let (sv, cv) = v.sin_cos();
            placed_projection(*processor.transform(), projection, |projection| {
                Ok(Taylor::interval(coefficient(
                    [center.x, center.y, center.z],
                    projection.direction,
                    projection.offset,
                    projection.center,
                )?) + Taylor::constant(radius)
                    * project_taylor([su * cv, su * sv, cu], projection.direction, ZERO))
            })?
        }
        Surface::ToroidalSurface(processor) => {
            let (u, v) = if processor.orientation() {
                (u, v)
            } else {
                (v, u)
            };
            let torus = processor.entity();
            let center = torus.center();
            if ![torus.small_radius(), torus.large_radius()]
                .into_iter()
                .all(f64::is_finite)
            {
                return Err(E::InvalidInput);
            }
            let (su, cu) = u.sin_cos();
            let (sv, cv) = v.sin_cos();
            let minor = Taylor::constant(torus.small_radius());
            let radial = Taylor::constant(torus.large_radius()) + minor * cv;
            placed_projection(*processor.transform(), projection, |projection| {
                Ok(Taylor::interval(coefficient(
                    [center.x, center.y, center.z],
                    projection.direction,
                    projection.offset,
                    projection.center,
                )?) + project_taylor(
                    [radial * cu, radial * su, minor * sv],
                    projection.direction,
                    ZERO,
                ))
            })?
        }
        _ => project_taylor(
            surface_taylor_with_policy(
                surface,
                u,
                v,
                nesting,
                false,
                execution.budget,
                execution.cancel,
            )?,
            projection.direction,
            projection_constant(projection)?,
        ),
    };
    value.finite()
}

fn projection_constant<const N: usize>(projection: ScalarProjection<N>) -> Result<I, E> {
    match projection.center {
        Some(center) => Ok(projection.offset - dot(center, projection.direction, ZERO)?),
        None => Ok(projection.offset),
    }
}

fn placed_projection(
    matrix: Matrix4,
    projection: ScalarProjection<3>,
    mut evaluate: impl FnMut(ScalarProjection<3>) -> Result<Taylor, E>,
) -> Result<Taylor, E> {
    if !(0..4).all(|i| (0..4).all(|j| matrix[i][j].is_finite())) {
        return Err(E::InvalidInput);
    }
    if (0..4).all(|i| (0..4).all(|j| matrix[i][j] == if i == j { 1.0 } else { 0.0 })) {
        return evaluate(projection);
    }
    let affine = (0..3).all(|i| matrix[i][3] == 0.0) && matrix[3][3] == 1.0;
    let constant = projection_constant(projection)?;
    let mut functional = [ZERO; 4];
    for (j, value) in functional.iter_mut().enumerate() {
        let column = [matrix[j][0], matrix[j][1], matrix[j][2]];
        *value = if affine && j == 3 {
            coefficient(
                column,
                projection.direction,
                projection.offset,
                projection.center,
            )?
        } else {
            dot(
                column,
                projection.direction,
                constant * I::point(matrix[j][3]),
            )?
        };
    }
    let numerator = evaluate(ScalarProjection {
        direction: [functional[0], functional[1], functional[2]],
        offset: functional[3],
        center: None,
    })?;
    // `Matrix4`/`Point3` uses homogeneous division; `Matrix3`/`Point2` does not.
    if affine {
        Ok(numerator)
    } else {
        let denominator = evaluate(ScalarProjection {
            direction: [matrix[0][3], matrix[1][3], matrix[2][3]].map(I::point),
            offset: I::point(matrix[3][3]),
            center: None,
        })?;
        Ok(numerator * denominator.reciprocal()?)
    }
}

fn projected_tensor<P, const N: usize>(
    source: (&(KnotVector, KnotVector), &[Vec<P>]),
    uv: [Taylor; 2],
    control: impl Fn(&P) -> [f64; N],
    projection: ScalarProjection<N>,
    rational: bool,
    execution: &mut ProjectionExecution<'_>,
) -> Result<Taylor, E> {
    let mut pending = [(uv, 0usize); 49];
    let mut length = 1;
    let mut result: Option<Taylor> = None;
    while length != 0 {
        execution.budget.charge(1, execution.cancel)?;
        length -= 1;
        let (uv, depth) = pending[length];
        match projected_tensor_cell(source, uv, &control, projection, rational, execution) {
            Ok(value) => result = Some(result.map_or(value, |previous| previous.hull(value))),
            Err(E::DenominatorContainsZero) => {
                let axis = usize::from(uv[1].0[0][0].width() > uv[0].0[0][0].width());
                let middle = uv[axis].0[0][0].midpoint();
                if depth == 48
                    || middle <= uv[axis].0[0][0].lower
                    || middle >= uv[axis].0[0][0].upper
                {
                    return Err(E::DenominatorContainsZero);
                }
                let mut lower = uv;
                let mut upper = uv;
                lower[axis].0[0][0].upper = middle;
                upper[axis].0[0][0].lower = middle;
                execution.budget.charge(2, execution.cancel)?;
                pending[length] = (upper, depth + 1);
                pending[length + 1] = (lower, depth + 1);
                length += 2;
            }
            Err(error) => return Err(error),
        }
    }
    result.ok_or(E::ParameterOutOfRange)
}

fn projected_tensor_cell<P, const N: usize>(
    (knots, controls): (&(KnotVector, KnotVector), &[Vec<P>]),
    [u, v]: [Taylor; 2],
    control: &impl Fn(&P) -> [f64; N],
    projection: ScalarProjection<N>,
    rational: bool,
    execution: &mut ProjectionExecution<'_>,
) -> Result<Taylor, E> {
    if controls.is_empty() || controls[0].is_empty() {
        return Err(E::InvalidInput);
    }
    let nu = controls.len();
    let nv = controls[0].len();
    for row in controls {
        execution.budget.charge(1, execution.cancel)?;
        if row.len() != nv {
            return Err(E::InvalidInput);
        }
    }
    let pu = validate_knots(
        &knots.0,
        nu,
        u.0[0][0],
        true,
        execution.budget,
        execution.cancel,
    )?;
    let pv = validate_knots(
        &knots.1,
        nv,
        v.0[0][0],
        true,
        execution.budget,
        execution.cancel,
    )?;
    let mut result: Option<Taylor> = None;
    for su in 0..knots.0.len() - 1 {
        execution.budget.charge(1, execution.cancel)?;
        if knots.0[su] == knots.0[su + 1] {
            continue;
        }
        let lower = if knots.0[su] == knots.0[0] {
            u.0[0][0].lower.min(knots.0[su])
        } else {
            knots.0[su]
        };
        let upper = if knots.0[su + 1] == knots.0[knots.0.len() - 1] {
            u.0[0][0].upper.max(knots.0[su + 1])
        } else {
            knots.0[su + 1]
        };
        let Some(u) = clipped_parameter(u, lower, upper) else {
            continue;
        };
        let (iu, lu, bu) = basis(&knots.0, nu, pu, su, u, execution.budget, execution.cancel)?;
        for sv in 0..knots.1.len() - 1 {
            execution.budget.charge(1, execution.cancel)?;
            if knots.1[sv] == knots.1[sv + 1] {
                continue;
            }
            let lower = if knots.1[sv] == knots.1[0] {
                v.0[0][0].lower.min(knots.1[sv])
            } else {
                knots.1[sv]
            };
            let upper = if knots.1[sv + 1] == knots.1[knots.1.len() - 1] {
                v.0[0][0].upper.max(knots.1[sv + 1])
            } else {
                knots.1[sv + 1]
            };
            let Some(v) = clipped_parameter(v, lower, upper) else {
                continue;
            };
            let (iv, lv, bv) = basis(&knots.1, nv, pv, sv, v, execution.budget, execution.cancel)?;
            let partition = su >= pu && su < nu && sv >= pv && sv < nv;
            let reference = if partition {
                control(&controls[iu][iv])
            } else {
                [0.0; N]
            };
            let mut numerator = if partition {
                Taylor::interval(coefficient(
                    reference,
                    projection.direction,
                    projection.offset,
                    projection.center,
                )?)
            } else {
                Taylor::ZERO
            };
            let mut denominator = Taylor::constant(if partition { reference[N - 1] } else { 0.0 });
            let mut numerator_hull: Option<I> = None;
            let mut denominator_hull: Option<I> = None;
            for (i, a) in bu[..lu].iter().enumerate() {
                for (j, b) in bv[..lv].iter().enumerate() {
                    execution.budget.charge(1, execution.cancel)?;
                    let point = control(&controls[iu + i][iv + j]);
                    let value = coefficient(
                        point,
                        projection.direction,
                        projection.offset,
                        projection.center,
                    )?;
                    let residual = if partition {
                        dot_difference(point, reference, projection.direction)?
                    } else {
                        value
                    };
                    let basis = *a * *b;
                    numerator = numerator + Taylor::interval(residual) * basis;
                    numerator_hull = Some(numerator_hull.map_or(value, |hull| hull.hull(value)));
                    if rational {
                        let weight = I::point(point[N - 1]);
                        let residual =
                            weight - I::point(if partition { reference[N - 1] } else { 0.0 });
                        denominator = denominator + Taylor::interval(residual) * basis;
                        denominator_hull =
                            Some(denominator_hull.map_or(weight, |hull| hull.hull(weight)));
                    }
                }
            }
            let nonnegative = knots.0[su] <= u.0[0][0].lower
                && u.0[0][0].upper <= knots.0[su + 1]
                && knots.1[sv] <= v.0[0][0].lower
                && v.0[0][0].upper <= knots.1[sv + 1];
            if partition && nonnegative {
                if let Some(hull) = numerator_hull {
                    numerator.0[0][0] = numerator.0[0][0]
                        .intersection(hull)
                        .ok_or(E::ConditioningFailure)?;
                }
                if let Some(hull) = denominator_hull {
                    denominator.0[0][0] = denominator.0[0][0]
                        .intersection(hull)
                        .ok_or(E::ConditioningFailure)?;
                }
            }
            let value = if rational {
                numerator * denominator.reciprocal()?
            } else {
                numerator
            };
            result = Some(result.map_or(value, |previous| previous.hull(value)));
        }
    }
    result.ok_or(E::ParameterOutOfRange)
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

#[cfg(test)]
mod scalar_surface_laws {
    use super::*;
    use crate::{
        BsplineCurve, BsplineSurface, Invertible, Line, Matrix3, NurbsCurve, NurbsSurface, Point2,
        Point3, Processor, RevolutionSurface, TrimmedCurve, Vector3, Vector4,
        validated_surface_jet,
    };

    fn knots() -> KnotVector { KnotVector::from(vec![0.0, 0.0, 1.0, 1.0]) }

    fn projection(direction: [f64; 3], center: [f64; 3]) -> ScalarProjection<3> {
        ScalarProjection {
            direction: direction.map(I::point),
            offset: ZERO,
            center: Some(center),
        }
    }

    fn enclose(surface: &Surface, uv: &Curve2D, range: [f64; 2]) -> ValidatedScalarCurveJet {
        projected_surface_curve_jet(
            surface,
            uv,
            range,
            projection([0.0, 0.0, 1.0], [0.0; 3]),
            &mut ProjectionExecution {
                budget: &mut ValidatedGeometryBudget::new(1_000_000),
                cancel: &AtomicBool::new(false),
            },
        )
        .unwrap()
    }

    fn contains(interval: I, value: f64) {
        assert!(
            interval.lower <= value && value <= interval.upper,
            "{interval:?} misses {value}"
        );
    }

    #[test]
    fn posed_reversed_uv_composes_the_original_polynomial_world_law() {
        let surface = Surface::BsplineSurface(BsplineSurface::new(
            (knots(), knots()),
            vec![
                vec![Point3::new(0.0, 0.0, 5.0), Point3::new(0.0, 1.0, 8.0)],
                vec![Point3::new(1.0, 0.0, 7.0), Point3::new(1.0, 1.0, 10.0)],
            ],
        ));
        let basis = Curve2D::BsplineCurve(BsplineCurve::new(
            KnotVector::from(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            vec![
                Point2::new(0.0, 0.0),
                Point2::new(0.5, 0.0),
                Point2::new(1.0, 1.0),
            ],
        ));
        for reversed in [false, true] {
            let mut wrapper = Processor::with_transform(
                TrimmedCurve::new(Box::new(basis.clone()), (0.0, 1.0)),
                Matrix3::new(2.0, 0.0, 0.5, 0.0, 3.0, 0.25, 1.0, -2.0, -4.0),
            );
            if reversed {
                wrapper.invert();
            }
            let uv = Curve2D::TrimmedCurve(wrapper);
            let jet = projected_surface_curve_jet(
                &surface,
                &uv,
                [0.125, 0.375],
                projection([2.0, -1.0, 4.0], [7.0, 11.0, 13.0]),
                &mut ProjectionExecution {
                    budget: &mut ValidatedGeometryBudget::new(1_000_000),
                    cancel: &AtomicBool::new(false),
                },
            )
            .unwrap();
            for t in [0.125, 0.25, 0.375] {
                let s = if reversed { 1.0 - t } else { t };
                contains(jet.position, 33.0 * s * s + 20.0 * s - 47.0);
                contains(
                    jet.first,
                    (66.0 * s + 20.0) * if reversed { -1.0 } else { 1.0 },
                );
                contains(jet.second, 66.0);
            }
        }
    }

    #[test]
    fn rational_tensor_composes_an_independent_quotient_law_with_fewer_visits() {
        let surface = Surface::NurbsSurface(NurbsSurface::new(BsplineSurface::new(
            (knots(), knots()),
            vec![
                vec![
                    Vector4::new(0.0, 0.0, 3.0, 1.0),
                    Vector4::new(0.0, 1.0, 10.0, 1.0),
                ],
                vec![
                    Vector4::new(2.0, 0.0, 8.0, 2.0),
                    Vector4::new(2.0, 2.0, 26.0, 2.0),
                ],
            ],
        )));
        let uv = Curve2D::BsplineCurve(BsplineCurve::new(
            KnotVector::from(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            vec![
                Point2::new(0.0, 0.0),
                Point2::new(0.5, 0.0),
                Point2::new(1.0, 1.0),
            ],
        ));
        let jet = enclose(&surface, &uv, [0.125, 0.375]);
        for t in [0.125_f64, 0.25, 0.375] {
            contains(jet.position, 11.0 * t * t - 4.0 * t + 9.0 - 6.0 / (1.0 + t));
            contains(jet.first, 22.0 * t - 4.0 + 6.0 / (1.0 + t).powi(2));
            contains(jet.second, 22.0 - 12.0 / (1.0 + t).powi(3));
        }
        let cancel = AtomicBool::new(false);
        let mut scalar = ValidatedGeometryBudget::new(1_000_000);
        let mut full = ValidatedGeometryBudget::new(1_000_000);
        projected_surface_taylor(
            &surface,
            [
                Taylor::variable(I::new(0.125, 0.375).unwrap(), 0),
                Taylor::variable(I::new(0.125, 0.375).unwrap(), 1),
            ],
            projection([0.0, 0.0, 1.0], [0.0; 3]),
            0,
            &mut ProjectionExecution {
                budget: &mut scalar,
                cancel: &cancel,
            },
        )
        .unwrap();
        validated_surface_jet(&surface, [[0.125, 0.375]; 2], &mut full, &cancel).unwrap();
        assert!(scalar.consumed() < full.consumed());
    }

    #[test]
    fn generic_posed_inverted_revolution_retains_scalar_rotation_derivatives() {
        for inverted in [false, true] {
            let mut processor = Processor::with_transform(
                RevolutionSurface::by_revolution(
                    Curve::Line(Line(
                        Point3::new(2.0, 0.0, 5.0),
                        Point3::new(3.0, 0.0, 12.0),
                    )),
                    Point3::new(0.0, 0.0, 0.0),
                    Vector3::new(0.0, 0.0, 1.0),
                ),
                Matrix4::new(
                    2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 3.0, -1.0, 7.0, 1.0,
                ),
            );
            if inverted {
                processor.invert();
            }
            let surface = Surface::RevolutionSurface(processor);
            let end = if inverted {
                Point2::new(2.0, 1.0)
            } else {
                Point2::new(1.0, 2.0)
            };
            let uv = Curve2D::Line(Line(Point2::new(0.0, 0.0), end));
            let jet = projected_surface_curve_jet(
                &surface,
                &uv,
                [0.125, 0.375],
                projection([1.0, 2.0, 3.0], [11.0, 13.0, 17.0]),
                &mut ProjectionExecution {
                    budget: &mut ValidatedGeometryBudget::new(1_000_000),
                    cancel: &AtomicBool::new(false),
                },
            )
            .unwrap();
            for t in [0.125_f64, 0.25, 0.375] {
                let a = 2.0 * (2.0 * t).cos() + 8.0 * (2.0 * t).sin();
                let b = -2.0 * (2.0 * t).sin() + 8.0 * (2.0 * t).cos();
                contains(jet.position, (2.0 + t) * a + 9.0 + 105.0 * t);
                contains(jet.first, a + 2.0 * (2.0 + t) * b + 105.0);
                contains(jet.second, 4.0 * b - 4.0 * (2.0 + t) * a);
            }
        }
    }

    #[test]
    fn interval_functional_retains_all_plane_coefficients() {
        let surface = Surface::Plane(Plane::xy());
        let direction = [I::new(1.0, 2.0).unwrap(), I::new(-3.0, -2.0).unwrap(), ZERO];
        let value = projected_surface_taylor(
            &surface,
            [
                Taylor::variable(I::new(2.0, 3.0).unwrap(), 0),
                Taylor::variable(I::new(4.0, 5.0).unwrap(), 1),
            ],
            ScalarProjection {
                direction,
                offset: I::new(-7.0, -5.0).unwrap(),
                center: Some([1.0, 2.0, 3.0]),
            },
            0,
            &mut ProjectionExecution {
                budget: &mut ValidatedGeometryBudget::new(1_000_000),
                cancel: &AtomicBool::new(false),
            },
        )
        .unwrap();
        for a in [1.0, 2.0] {
            for b in [-3.0, -2.0] {
                for offset in [-7.0, -5.0] {
                    for u in [2.0, 3.0] {
                        for v in [4.0, 5.0] {
                            contains(value.0[0][0], a * (u - 1.0) + b * (v - 2.0) + offset);
                            contains(value.0[1][0], a);
                            contains(value.0[0][1], b);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn zero_functionals_do_not_erase_surface_or_uv_poles_and_source_knots() {
        let surface = Surface::NurbsSurface(NurbsSurface::new(BsplineSurface::new(
            (knots(), knots()),
            vec![
                vec![Vector4::new(0.0, 0.0, 0.0, -1.0); 2],
                vec![Vector4::new(1.0, 0.0, 0.0, 1.0); 2],
            ],
        )));
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(1_000_000);
        let zero = projection([0.0; 3], [0.0; 3]);
        assert!(matches!(
            projected_surface_taylor(
                &surface,
                [
                    Taylor::variable(I::point(0.5), 0),
                    Taylor::variable(I::point(0.5), 1)
                ],
                zero,
                0,
                &mut ProjectionExecution {
                    budget: &mut budget,
                    cancel: &cancel
                },
            ),
            Err(E::DenominatorContainsZero)
        ));
        assert!(budget.consumed() > 0);
        let pole_profile = Curve::NurbsCurve(NurbsCurve::new(BsplineCurve::new(
            knots(),
            vec![
                Vector4::new(2.0, 0.0, 0.0, -1.0),
                Vector4::new(3.0, 0.0, 1.0, 1.0),
            ],
        )));
        let revolution =
            Surface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
                pole_profile,
                Point3::new(0.0, 0.0, 0.0),
                Vector3::new(0.0, 0.0, 1.0),
            )));
        assert!(matches!(
            projected_surface_taylor(
                &revolution,
                [
                    Taylor::variable(I::point(0.5), 0),
                    Taylor::variable(I::point(0.25), 1)
                ],
                zero,
                0,
                &mut ProjectionExecution {
                    budget: &mut budget,
                    cancel: &cancel
                },
            ),
            Err(E::DenominatorContainsZero)
        ));
        let uv = Curve2D::NurbsCurve(NurbsCurve::new(BsplineCurve::new(
            knots(),
            vec![Vector3::new(0.0, 0.0, -1.0), Vector3::new(1.0, 0.0, 1.0)],
        )));
        assert!(matches!(
            projected_surface_curve_jet(
                &Surface::Plane(Plane::xy()),
                &uv,
                [0.5, 0.5],
                zero,
                &mut ProjectionExecution {
                    budget: &mut budget,
                    cancel: &cancel
                },
            ),
            Err(E::DenominatorContainsZero)
        ));
        let crease = Surface::BsplineSurface(BsplineSurface::new(
            (KnotVector::from(vec![0.0, 0.0, 1.0, 2.0, 2.0]), knots()),
            (0..3)
                .map(|u| {
                    vec![
                        Point3::new(u as f64, 0.0, 0.0),
                        Point3::new(u as f64, 1.0, 0.0),
                    ]
                })
                .collect(),
        ));
        assert!(matches!(
            projected_surface_taylor(
                &crease,
                [
                    Taylor::variable(I::new(0.5, 1.5).unwrap(), 0),
                    Taylor::variable(I::new(0.0, 1.0).unwrap(), 1)
                ],
                zero,
                0,
                &mut ProjectionExecution {
                    budget: &mut budget,
                    cancel: &cancel
                },
            ),
            Err(E::NonSmooth)
        ));
        for (limit, cancelled, expected) in
            [(0, false, E::ResourceLimit), (100, true, E::Cancelled)]
        {
            let mut budget = ValidatedGeometryBudget::new(limit);
            assert_eq!(
                projected_surface_taylor(
                    &Surface::Plane(Plane::xy()),
                    [Taylor::constant(0.0); 2],
                    zero,
                    0,
                    &mut ProjectionExecution {
                        budget: &mut budget,
                        cancel: &AtomicBool::new(cancelled)
                    },
                )
                .unwrap_err(),
                expected,
            );
            assert_eq!(budget.consumed(), if cancelled { 0 } else { limit });
        }
    }

    #[test]
    fn mixed_weights_certify_the_actual_nonzero_denominator_not_the_weight_hull() {
        let surface = Surface::NurbsSurface(NurbsSurface::new(BsplineSurface::new(
            (
                KnotVector::from(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
                knots(),
            ),
            [1.0, -0.125, 1.0]
                .into_iter()
                .enumerate()
                .map(|(i, w)| {
                    vec![
                        Vector4::new(i as f64, 0.0, 0.0, w),
                        Vector4::new(i as f64, 1.0, 0.0, w),
                    ]
                })
                .collect(),
        )));
        let value = projected_surface_taylor(
            &surface,
            [
                Taylor::variable(I::new(0.0, 1.0).unwrap(), 0),
                Taylor::variable(I::new(0.0, 1.0).unwrap(), 1),
            ],
            projection([0.0, 0.0, 1.0], [0.0; 3]),
            0,
            &mut ProjectionExecution {
                budget: &mut ValidatedGeometryBudget::new(1_000_000),
                cancel: &AtomicBool::new(false),
            },
        )
        .unwrap();
        assert_eq!(value.0[0][0], ZERO);
        assert_eq!(value.0[1][0], ZERO);
        assert_eq!(value.0[0][1], ZERO);
    }

    #[test]
    fn interval_axis_functional_retains_exact_angular_independence_and_origin_tube() {
        let surface = Surface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
            Curve::Line(Line(
                Point3::new(2.0, 0.0, 5.0),
                Point3::new(3.0, 0.0, 12.0),
            )),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        )));
        let normal = [ZERO, ZERO, I::new(1.0, 2.0).unwrap()];
        let origin_z = I::new(3.0, 4.0).unwrap();
        let center_z = origin_z.midpoint();
        let value = projected_surface_taylor(
            &surface,
            [
                Taylor::variable(I::new(0.0, 1.0).unwrap(), 0),
                Taylor::variable(I::new(-10.0, 10.0).unwrap(), 1),
            ],
            ScalarProjection {
                direction: normal,
                offset: normal[2] * (I::point(center_z) - origin_z),
                center: Some([0.0, 0.0, center_z]),
            },
            0,
            &mut ProjectionExecution {
                budget: &mut ValidatedGeometryBudget::new(1_000_000),
                cancel: &AtomicBool::new(false),
            },
        )
        .unwrap();
        for n in [1.0, 2.0] {
            for origin in [3.0, 4.0] {
                for u in [0.0, 0.5, 1.0] {
                    contains(value.0[0][0], n * (5.0 + 7.0 * u - origin));
                    contains(value.0[1][0], n * 7.0);
                }
            }
        }
        assert_eq!(value.0[0][1], ZERO);
        assert_eq!(value.0[0][2], ZERO);
        assert_eq!(value.0[1][1], ZERO);
    }
}
