use super::*;
use super::{isocurve::ExplicitCurve, planar::Dyadic};
use crate::geometry::MAX_NATIVE_GEOMETRY_NESTING;
use crate::{BsplineCurve, KnotVector, Line, Matrix3, Matrix4, Plane, Point2, Vector3};

/// An original source-use choice on an exact native extrusion chart.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExtrusionChartCurve {
    /// A complete translated profile at this actual finite extrusion coordinate.
    ProfileAt {
        /// Native extrusion coordinate, not a guessed finite carrier domain.
        offset: f64,
    },
    /// A straight generator at this original profile parameter.
    GeneratorAt {
        /// Native profile parameter, independently bound to the original use.
        profile_parameter: f64,
    },
}

fn refused() -> TrimDomainError { ValidatedGeometryError::Unresolved.into() }
fn number(value: f64) -> Result<Dyadic, TrimDomainError> {
    Dyadic::scalar(value).ok_or_else(refused)
}
fn sum(a: Dyadic, b: Dyadic) -> Result<Dyadic, TrimDomainError> { a.add(b).ok_or_else(refused) }
fn product(a: Dyadic, b: Dyadic) -> Result<Dyadic, TrimDomainError> {
    a.multiply(b).ok_or_else(refused)
}
fn difference(a: f64, b: f64) -> Result<Dyadic, TrimDomainError> { sum(number(a)?, number(-b)?) }

fn explicit(curve: &Curve) -> Result<ExplicitCurve<'_>, TrimDomainError> {
    match curve {
        Curve::Line(curve) => Ok(ExplicitCurve::Line(curve)),
        Curve::BsplineCurve(curve) => Ok(ExplicitCurve::Polynomial(curve)),
        Curve::NurbsCurve(curve) => Ok(ExplicitCurve::Rational(curve)),
        _ => Err(ValidatedGeometryError::Unsupported.into()),
    }
}

fn domain(
    curve: &ExplicitCurve<'_>,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<[f64; 2], TrimDomainError> {
    budget.charge(curve.knots_len(), cancel)?;
    let count = curve.count();
    let degree = curve
        .knots_len()
        .checked_sub(count.saturating_add(1))
        .ok_or(TrimDomainError::InvalidChart)?;
    if count == 0 || degree == 0 || degree >= count {
        return Err(TrimDomainError::InvalidChart);
    }
    let bounds = [curve.knot(degree), curve.knot(count)];
    if !bounds.into_iter().all(f64::is_finite) || bounds[0] >= bounds[1] {
        return Err(TrimDomainError::InvalidChart);
    }
    for index in 0..curve.knots_len() {
        let value = curve.knot(index);
        if !value.is_finite()
            || (index > 0 && value < curve.knot(index - 1))
            || (index <= degree && value != bounds[0])
            || (index >= count && value != bounds[1])
        {
            return Err(ValidatedGeometryError::Unsupported.into());
        }
    }
    Ok(bounds)
}

fn uv_line(range: [f64; 2], front: Point2, back: Point2) -> Curve2D {
    Curve2D::BsplineCurve(BsplineCurve::new(
        KnotVector::from(vec![range[0], range[0], range[1], range[1]]),
        vec![front, back],
    ))
}

fn translated_profile(
    edge: &Curve,
    profile: &Curve,
    direction: Vector3,
    offset: f64,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Curve2D, TrimDomainError> {
    let edge = explicit(edge)?;
    let profile = explicit(profile)?;
    let e = domain(&edge, budget, cancel)?;
    let p = domain(&profile, budget, cancel)?;
    if edge.count() != profile.count() || edge.knots_len() != profile.knots_len() {
        return Err(refused());
    }
    for reverse in [false, true] {
        let mut matches = true;
        for index in 0..edge.knots_len() {
            budget.charge(1, cancel)?;
            let offset_p = if reverse {
                difference(p[1], profile.knot(profile.knots_len() - 1 - index))?
            } else {
                difference(profile.knot(index), p[0])?
            };
            if product(difference(edge.knot(index), e[0])?, difference(p[1], p[0])?)?
                != product(offset_p, difference(e[1], e[0])?)?
            {
                matches = false;
                break;
            }
        }
        if !matches {
            continue;
        }
        let mut pivot = None;
        for index in 0..edge.count() {
            budget.charge(1, cancel)?;
            let a = edge.control(index);
            let b = profile.control(if reverse {
                profile.count() - 1 - index
            } else {
                index
            });
            if a[3] != 0.0 && b[3] != 0.0 {
                pivot = Some((number(a[3])?, number(b[3])?));
                break;
            }
        }
        let Some((edge_weight, profile_weight)) = pivot else {
            return Err(ValidatedGeometryError::DenominatorContainsZero.into());
        };
        for index in 0..edge.count() {
            budget.charge(1, cancel)?;
            let a = edge.control(index);
            let b = profile.control(if reverse {
                profile.count() - 1 - index
            } else {
                index
            });
            if product(number(a[3])?, profile_weight)? != product(number(b[3])?, edge_weight)? {
                matches = false;
                break;
            }
            for axis in 0..3 {
                let translated = sum(
                    number(b[axis])?,
                    product(
                        product(number(offset)?, number(direction[axis])?)?,
                        number(b[3])?,
                    )?,
                )?;
                if product(number(a[axis])?, profile_weight)? != product(translated, edge_weight)? {
                    matches = false;
                    break;
                }
            }
            if !matches {
                break;
            }
        }
        if matches {
            let (front, back) = if reverse { (p[1], p[0]) } else { (p[0], p[1]) };
            return Ok(uv_line(
                e,
                Point2::new(front, offset),
                Point2::new(back, offset),
            ));
        }
    }
    Err(refused())
}

fn same_uv(
    a: &Curve2D,
    b: &Curve2D,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<bool, TrimDomainError> {
    same_uv_at_depth(a, b, 1, budget, cancel)
}

fn same_uv_at_depth(
    a: &Curve2D,
    b: &Curve2D,
    depth: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<bool, TrimDomainError> {
    budget.charge(1, cancel)?;
    if depth > MAX_NATIVE_GEOMETRY_NESTING {
        return Err(ValidatedGeometryError::ResourceLimit.into());
    }
    Ok(match (a, b) {
        (Curve2D::Line(a), Curve2D::Line(b)) => {
            budget.charge(2, cancel)?;
            a.0 == b.0 && a.1 == b.1
        }
        (Curve2D::Polyline(a), Curve2D::Polyline(b)) => {
            budget.charge(a.as_slice().len(), cancel)?;
            a.as_slice() == b.as_slice()
        }
        (Curve2D::BsplineCurve(a), Curve2D::BsplineCurve(b)) => {
            budget.charge(
                a.knot_vector()
                    .len()
                    .saturating_add(a.control_points().len()),
                cancel,
            )?;
            a.knot_vector() == b.knot_vector() && a.control_points() == b.control_points()
        }
        (Curve2D::NurbsCurve(a), Curve2D::NurbsCurve(b)) => {
            budget.charge(
                a.knot_vector()
                    .len()
                    .saturating_add(a.control_points().len()),
                cancel,
            )?;
            a.knot_vector() == b.knot_vector() && a.control_points() == b.control_points()
        }
        (Curve2D::Conic(a), Curve2D::Conic(b)) => {
            budget.charge(12, cancel)?;
            match (a, b) {
                (Conic2D::Ellipse(a), Conic2D::Ellipse(b)) => {
                    a.transform() == b.transform()
                        && a.orientation() == b.orientation()
                        && a.entity().range() == b.entity().range()
                }
                (Conic2D::Hyperbola(a), Conic2D::Hyperbola(b)) => {
                    a.transform() == b.transform()
                        && a.orientation() == b.orientation()
                        && a.entity().range() == b.entity().range()
                }
                (Conic2D::Parabola(a), Conic2D::Parabola(b)) => {
                    a.transform() == b.transform()
                        && a.orientation() == b.orientation()
                        && a.entity().range() == b.entity().range()
                }
                _ => false,
            }
        }
        (Curve2D::TrimmedCurve(a), Curve2D::TrimmedCurve(b)) => {
            budget.charge(12, cancel)?;
            a.transform() == b.transform()
                && a.orientation() == b.orientation()
                && a.entity().range() == b.entity().range()
                && same_uv_at_depth(
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

fn translated_parameter_profile(
    edge: &ParameterCurve<Curve2D, Box<Surface>>,
    profile: &ParameterCurve<Curve2D, Box<Surface>>,
    direction: Vector3,
    offset: f64,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Curve2D, TrimDomainError> {
    if !same_uv(edge.curve(), profile.curve(), budget, cancel)? {
        return Err(refused());
    }
    let (Surface::Plane(a), Surface::Plane(b)) =
        (edge.surface().as_ref(), profile.surface().as_ref())
    else {
        return Err(ValidatedGeometryError::Unsupported.into());
    };
    let a = planar::plane_coefficients(a).ok_or_else(refused)?;
    let b = planar::plane_coefficients(b).ok_or_else(refused)?;
    if a[1] != b[1] || a[2] != b[2] {
        return Err(refused());
    }
    for axis in 0..3 {
        budget.charge(1, cancel)?;
        if a[0][axis]
            != sum(
                b[0][axis],
                product(number(offset)?, number(direction[axis])?)?,
            )?
        {
            return Err(refused());
        }
    }
    let range = edge.range_tuple();
    if ![range.0, range.1].into_iter().all(f64::is_finite) || range.0 >= range.1 {
        return Err(TrimDomainError::InvalidChart);
    }
    Ok(uv_line(
        [range.0, range.1],
        Point2::new(range.0, offset),
        Point2::new(range.1, offset),
    ))
}

fn plane_point(plane: &Plane, uv: [Dyadic; 3]) -> Result<[Dyadic; 4], TrimDomainError> {
    let mut point = [number(0.0)?; 4];
    let coefficients = planar::plane_coefficients(plane).ok_or_else(refused)?;
    for axis in 0..3 {
        point[axis] = sum(
            sum(
                product(coefficients[0][axis], uv[2])?,
                product(coefficients[1][axis], uv[0])?,
            )?,
            product(coefficients[2][axis], uv[1])?,
        )?;
    }
    point[3] = uv[2];
    Ok(point)
}

fn profile_endpoint(
    profile: &Curve,
    parameter: f64,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<[Dyadic; 4], TrimDomainError> {
    let view = native_profile_view(profile, budget, cancel)?;
    let parameter = sum(product(number(view.sign)?, number(parameter)?)?, view.shift)?;
    let point = match view.basis {
        Curve::Line(line) => exact_line_point(line, parameter)?,
        Curve::ParameterCurve(profile) => {
            let Surface::Plane(plane) = profile.surface().as_ref() else {
                return Err(ValidatedGeometryError::Unsupported.into());
            };
            plane_point(
                plane,
                exact_uv_point(profile.curve(), parameter, view.depth + 1, budget, cancel)?,
            )?
        }
        basis => {
            let profile = explicit(basis)?;
            let range = domain(&profile, budget, cancel)?;
            let index = if parameter == number(range[0])? {
                0
            } else if parameter == number(range[1])? {
                profile.count() - 1
            } else {
                return Err(ValidatedGeometryError::Unsupported.into());
            };
            budget.charge(1, cancel)?;
            let point = profile.control(index);
            [
                number(point[0])?,
                number(point[1])?,
                number(point[2])?,
                number(point[3])?,
            ]
        }
    };
    transformed_point(&view.pose, point)
}

/// Proves a complete spatial coedge has its selected exact native extrusion chart.
///
/// Translated profiles are matched by whole homogeneous coefficient identities
/// with one global projective scale and exact affine native knot correspondence,
/// including reversed curves. Analytic plane-embedded parameter profiles require
/// identical whole UV coefficients and exact raw plane-axis/translation identity.
/// Nested native trims retain every original reflection anchor; exact dyadic
/// composition compares their complete basis coefficients and affine poses.
/// Straight generators are proved from both line coefficients and exact clamped
/// spline endpoints or algebraic points of native embedded lines and conics.
/// No endpoint sampling is promoted to a whole-curve proof.
///
/// The branch and finite offset are original oriented-use input, not reconstructed
/// by nearest points. A caller can compute candidate offsets but they become
/// authoritative only after these full identities succeed. Extrusion's legacy
/// reported `v=0..1` cell is never used as the trimmed face's V domain.
/// Transcendental conic endpoints, nonclamped spline endpoints and unavailable explicit
/// parameter correspondence return typed refusals. Regularity/denominator checks
/// remain separate native interval obligations.
pub fn certified_extrusion_parameter_curve(
    curve: &Curve,
    surface: &Surface,
    branch: ExtrusionChartCurve,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Curve2D, TrimDomainError> {
    budget.charge(1, cancel)?;
    let Surface::ExtrusionSurface(surface) = surface else {
        return Err(ValidatedGeometryError::Unsupported.into());
    };
    let profile = surface.entity_curve();
    let direction = surface.extruding_vector();
    if ![direction.x, direction.y, direction.z]
        .into_iter()
        .all(f64::is_finite)
    {
        return Err(TrimDomainError::InvalidChart);
    }
    match branch {
        ExtrusionChartCurve::ProfileAt { offset } => {
            if !offset.is_finite() {
                return Err(TrimDomainError::InvalidChart);
            }
            if matches!(curve, Curve::TrimmedCurve(_)) || matches!(profile, Curve::TrimmedCurve(_))
            {
                return translated_wrapped_profile(
                    curve, profile, direction, offset, budget, cancel,
                );
            }
            if let (Curve::ParameterCurve(edge), Curve::ParameterCurve(profile)) = (curve, profile)
            {
                translated_parameter_profile(edge, profile, direction, offset, budget, cancel)
            } else {
                translated_profile(curve, profile, direction, offset, budget, cancel)
            }
        }
        ExtrusionChartCurve::GeneratorAt { profile_parameter } => {
            if !profile_parameter.is_finite() {
                return Err(TrimDomainError::InvalidChart);
            }
            let (range, endpoints) = exact_generator_endpoints(curve, budget, cancel)?;
            let point = profile_endpoint(profile, profile_parameter, budget, cancel)?;
            if point[3] == number(0.0)? {
                return Err(ValidatedGeometryError::DenominatorContainsZero.into());
            }
            let axis = (0..3)
                .find(|axis| direction[*axis] != 0.0)
                .ok_or(TrimDomainError::InvalidChart)?;
            let approximate = point[axis].approximate_quotient(point[3]);
            let mut offsets = [0.0; 2];
            for (index, endpoint) in endpoints.into_iter().enumerate() {
                budget.charge(1, cancel)?;
                let endpoint_axis = endpoint[axis].approximate_quotient(endpoint[3]);
                let offset = (endpoint_axis - approximate) / direction[axis];
                if !offset.is_finite() {
                    return Err(refused());
                }
                for axis in 0..3 {
                    let rhs = sum(
                        point[axis],
                        product(
                            product(number(offset)?, number(direction[axis])?)?,
                            point[3],
                        )?,
                    )?;
                    if product(endpoint[axis], point[3])? != product(rhs, endpoint[3])? {
                        return Err(refused());
                    }
                }
                offsets[index] = offset;
            }
            Ok(uv_line(
                range,
                Point2::new(profile_parameter, offsets[0]),
                Point2::new(profile_parameter, offsets[1]),
            ))
        }
    }
}

struct NativeProfileView<'a> {
    basis: &'a Curve,
    domain: [f64; 2],
    sign: f64,
    shift: Dyadic,
    pose: [[Dyadic; 4]; 4],
    depth: usize,
}

fn identity_pose() -> Result<[[Dyadic; 4]; 4], TrimDomainError> {
    let mut matrix = [[number(0.0)?; 4]; 4];
    for (axis, row) in matrix.iter_mut().enumerate() {
        row[axis] = number(1.0)?;
    }
    Ok(matrix)
}

fn transformed_point(
    matrix: &[[Dyadic; 4]; 4],
    point: [Dyadic; 4],
) -> Result<[Dyadic; 4], TrimDomainError> {
    let mut result = [number(0.0)?; 4];
    for axis in 0..4 {
        for term in 0..4 {
            result[axis] = sum(result[axis], product(matrix[axis][term], point[term])?)?;
        }
    }
    Ok(result)
}

fn compose_pose(
    left: [[Dyadic; 4]; 4],
    right: &Matrix4,
) -> Result<[[Dyadic; 4]; 4], TrimDomainError> {
    if right[0][3] != 0.0 || right[1][3] != 0.0 || right[2][3] != 0.0 || right[3][3] != 1.0 {
        return Err(ValidatedGeometryError::Unsupported.into());
    }
    let mut result = [[number(0.0)?; 4]; 4];
    for axis in 0..4 {
        for column in 0..4 {
            for term in 0..4 {
                result[axis][column] = sum(
                    result[axis][column],
                    product(left[axis][term], number(right[column][term])?)?,
                )?;
            }
        }
    }
    Ok(result)
}

fn native_profile_view<'a>(
    curve: &'a Curve,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<NativeProfileView<'a>, TrimDomainError> {
    let range = curve.range_tuple();
    if ![range.0, range.1].into_iter().all(f64::is_finite) || range.0 >= range.1 {
        return Err(TrimDomainError::InvalidChart);
    }
    let mut view = NativeProfileView {
        basis: curve,
        domain: [range.0, range.1],
        sign: 1.0,
        shift: number(0.0)?,
        pose: identity_pose()?,
        depth: 0,
    };
    let mut depth = 0;
    while let Curve::TrimmedCurve(trimmed) = view.basis {
        budget.charge(66, cancel)?;
        depth += 1;
        if depth > MAX_NATIVE_GEOMETRY_NESTING {
            return Err(ValidatedGeometryError::ResourceLimit.into());
        }
        view.pose = compose_pose(view.pose, trimmed.transform())?;
        if !trimmed.orientation() {
            let range = trimmed.entity().range();
            view.sign = -view.sign;
            view.shift = sum(
                sum(number(range.0)?, number(range.1)?)?,
                view.shift.negated().ok_or_else(refused)?,
            )?;
        }
        view.basis = trimmed.entity().curve();
        view.depth = depth;
    }
    Ok(view)
}

fn translated_homogeneous(
    edge: [Dyadic; 4],
    profile: [Dyadic; 4],
    direction: Vector3,
    offset: f64,
    edge_weight: Dyadic,
    profile_weight: Dyadic,
) -> Result<bool, TrimDomainError> {
    if product(edge[3], profile_weight)? != product(profile[3], edge_weight)? {
        return Ok(false);
    }
    for axis in 0..3 {
        let translated = sum(
            profile[axis],
            product(
                product(number(offset)?, number(direction[axis])?)?,
                profile[3],
            )?,
        )?;
        if product(edge[axis], profile_weight)? != product(translated, edge_weight)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn transformed_control(
    view: &NativeProfileView<'_>,
    curve: &ExplicitCurve<'_>,
    index: usize,
) -> Result<[Dyadic; 4], TrimDomainError> {
    let point = curve.control(index);
    transformed_point(
        &view.pose,
        [
            number(point[0])?,
            number(point[1])?,
            number(point[2])?,
            number(point[3])?,
        ],
    )
}

fn translated_wrapped_profile(
    edge: &Curve,
    profile: &Curve,
    direction: Vector3,
    offset: f64,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Curve2D, TrimDomainError> {
    let edge = native_profile_view(edge, budget, cancel)?;
    let profile = native_profile_view(profile, budget, cancel)?;
    if let (Curve::ParameterCurve(a), Curve::ParameterCurve(b)) = (edge.basis, profile.basis) {
        if !same_uv_at_depth(
            a.curve(),
            b.curve(),
            edge.depth.max(profile.depth) + 1,
            budget,
            cancel,
        )? {
            return Err(refused());
        }
        let (Surface::Plane(a), Surface::Plane(b)) = (a.surface().as_ref(), b.surface().as_ref())
        else {
            return Err(ValidatedGeometryError::Unsupported.into());
        };
        let a = planar::plane_coefficients(a).ok_or_else(refused)?;
        let b = planar::plane_coefficients(b).ok_or_else(refused)?;
        for coefficient in 0..3 {
            budget.charge(32, cancel)?;
            let weight = number(if coefficient == 0 { 1.0 } else { 0.0 })?;
            let a = transformed_point(
                &edge.pose,
                [
                    a[coefficient][0],
                    a[coefficient][1],
                    a[coefficient][2],
                    weight,
                ],
            )?;
            let b = transformed_point(
                &profile.pose,
                [
                    b[coefficient][0],
                    b[coefficient][1],
                    b[coefficient][2],
                    weight,
                ],
            )?;
            if !translated_homogeneous(a, b, direction, offset, number(1.0)?, number(1.0)?)? {
                return Err(refused());
            }
        }
    } else {
        let a = explicit(edge.basis)?;
        let b = explicit(profile.basis)?;
        if a.count() != b.count() || a.knots_len() != b.knots_len() {
            return Err(refused());
        }
        for index in 0..a.knots_len() {
            budget.charge(1, cancel)?;
            if a.knot(index) != b.knot(index) {
                return Err(refused());
            }
        }
        let mut weights = None;
        for index in 0..a.count() {
            budget.charge(32, cancel)?;
            let a = transformed_control(&edge, &a, index)?;
            let b = transformed_control(&profile, &b, index)?;
            if a[3] != number(0.0)? && b[3] != number(0.0)? {
                weights = Some((a[3], b[3]));
                break;
            }
        }
        let (edge_weight, profile_weight) =
            weights.ok_or(ValidatedGeometryError::DenominatorContainsZero)?;
        for index in 0..a.count() {
            budget.charge(32, cancel)?;
            if !translated_homogeneous(
                transformed_control(&edge, &a, index)?,
                transformed_control(&profile, &b, index)?,
                direction,
                offset,
                edge_weight,
                profile_weight,
            )? {
                return Err(refused());
            }
        }
    }
    let mut parameters = [0.0; 2];
    for (index, t) in edge.domain.into_iter().enumerate() {
        let base = sum(product(number(edge.sign)?, number(t)?)?, edge.shift)?;
        parameters[index] = product(
            number(profile.sign)?,
            sum(base, profile.shift.negated().ok_or_else(refused)?)?,
        )?
        .exact_float()
        .ok_or_else(refused)?;
    }
    Ok(uv_line(
        edge.domain,
        Point2::new(parameters[0], offset),
        Point2::new(parameters[1], offset),
    ))
}

fn exact_line_point(
    line: &Line<Point3>,
    parameter: Dyadic,
) -> Result<[Dyadic; 4], TrimDomainError> {
    let mut point = [number(1.0)?; 4];
    for (axis, coordinate) in point.iter_mut().enumerate().take(3) {
        *coordinate = sum(
            number(line.0[axis])?,
            product(parameter, difference(line.1[axis], line.0[axis])?)?,
        )?;
    }
    Ok(point)
}

fn exact_generator_endpoints(
    curve: &Curve,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<([f64; 2], [[Dyadic; 4]; 2]), TrimDomainError> {
    let view = native_profile_view(curve, budget, cancel)?;
    let Curve::Line(line) = view.basis else {
        return Err(ValidatedGeometryError::Unsupported.into());
    };
    let mut endpoints = [[number(0.0)?; 4]; 2];
    for (index, t) in view.domain.into_iter().enumerate() {
        budget.charge(32, cancel)?;
        let parameter = sum(product(number(view.sign)?, number(t)?)?, view.shift)?;
        endpoints[index] = transformed_point(&view.pose, exact_line_point(line, parameter)?)?;
    }
    Ok((view.domain, endpoints))
}

pub(super) fn exact_uv_point(
    curve: &Curve2D,
    parameter: Dyadic,
    depth: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<[Dyadic; 3], TrimDomainError> {
    budget.charge(12, cancel)?;
    if depth > MAX_NATIVE_GEOMETRY_NESTING {
        return Err(ValidatedGeometryError::ResourceLimit.into());
    }
    match curve {
        Curve2D::TrimmedCurve(trimmed) => {
            let parameter = if trimmed.orientation() {
                parameter
            } else {
                let range = trimmed.entity().range();
                sum(
                    sum(number(range.0)?, number(range.1)?)?,
                    parameter.negated().ok_or_else(refused)?,
                )?
            };
            let point = exact_uv_point(
                trimmed.entity().curve(),
                parameter,
                depth + 1,
                budget,
                cancel,
            )?;
            exact_uv_transform(trimmed.transform(), point)
        }
        Curve2D::Line(line) => {
            if parameter == number(0.0)? {
                return Ok([number(line.0.x)?, number(line.0.y)?, number(1.0)?]);
            }
            if parameter == number(1.0)? {
                return Ok([number(line.1.x)?, number(line.1.y)?, number(1.0)?]);
            }
            Ok([
                sum(
                    number(line.0.x)?,
                    product(parameter, difference(line.1.x, line.0.x)?)?,
                )?,
                sum(
                    number(line.0.y)?,
                    product(parameter, difference(line.1.y, line.0.y)?)?,
                )?,
                number(1.0)?,
            ])
        }
        Curve2D::Conic(conic) => {
            let (matrix, orientation, range, kind) = match conic {
                Conic2D::Ellipse(curve) => (
                    curve.transform(),
                    curve.orientation(),
                    curve.entity().range(),
                    0,
                ),
                Conic2D::Hyperbola(curve) => (
                    curve.transform(),
                    curve.orientation(),
                    curve.entity().range(),
                    1,
                ),
                Conic2D::Parabola(curve) => (
                    curve.transform(),
                    curve.orientation(),
                    curve.entity().range(),
                    2,
                ),
            };
            let parameter = if orientation {
                parameter
            } else {
                sum(
                    sum(number(range.0)?, number(range.1)?)?,
                    parameter.negated().ok_or_else(refused)?,
                )?
            };
            let point = if kind == 2 {
                [
                    product(parameter, parameter)?,
                    product(number(2.0)?, parameter)?,
                    number(1.0)?,
                ]
            } else if parameter == number(0.0)? {
                [number(1.0)?, number(0.0)?, number(1.0)?]
            } else {
                return Err(ValidatedGeometryError::Unsupported.into());
            };
            exact_uv_transform(matrix, point)
        }
        Curve2D::Polyline(curve) => {
            let points = curve.as_slice();
            let last = points
                .len()
                .checked_sub(1)
                .ok_or(TrimDomainError::InvalidChart)?;
            let point = if parameter == number(0.0)? {
                points[0]
            } else if parameter == number(last as f64)? {
                points[last]
            } else {
                return Err(ValidatedGeometryError::Unsupported.into());
            };
            Ok([number(point.x)?, number(point.y)?, number(1.0)?])
        }
        Curve2D::BsplineCurve(curve) => {
            let controls = curve.control_points();
            budget.charge(curve.knot_vector().len(), cancel)?;
            let (degree, range) = isocurve::knot_domain(curve.knot_vector(), controls.len())?;
            if degree == 1 && controls.len() == 2 {
                return exact_uv_affine_point(
                    parameter,
                    range,
                    [
                        [controls[0].x, controls[0].y, 1.0],
                        [controls[1].x, controls[1].y, 1.0],
                    ],
                    budget,
                    cancel,
                );
            }
            let index = if parameter == number(range[0])? {
                0
            } else if parameter == number(range[1])? {
                controls.len() - 1
            } else {
                return Err(ValidatedGeometryError::Unsupported.into());
            };
            let point = controls[index];
            Ok([number(point.x)?, number(point.y)?, number(1.0)?])
        }
        Curve2D::NurbsCurve(curve) => {
            let controls = curve.control_points();
            budget.charge(curve.knot_vector().len(), cancel)?;
            let (degree, range) = isocurve::knot_domain(curve.knot_vector(), controls.len())?;
            if degree == 1 && controls.len() == 2 {
                return exact_uv_affine_point(
                    parameter,
                    range,
                    [
                        [controls[0].x, controls[0].y, controls[0].z],
                        [controls[1].x, controls[1].y, controls[1].z],
                    ],
                    budget,
                    cancel,
                );
            }
            let index = if parameter == number(range[0])? {
                0
            } else if parameter == number(range[1])? {
                controls.len() - 1
            } else {
                return Err(ValidatedGeometryError::Unsupported.into());
            };
            let point = controls[index];
            Ok([number(point.x)?, number(point.y)?, number(point.z)?])
        }
    }
}

/// Evaluates an admitted degree-one span without approximating its reciprocal.
/// The positive original span length remains a common homogeneous factor.
fn exact_uv_affine_point(
    parameter: Dyadic,
    range: [f64; 2],
    controls: [[f64; 3]; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<[Dyadic; 3], TrimDomainError> {
    let start = number(range[0])?;
    let end = number(range[1])?;
    if parameter.compare(start).is_lt() || parameter.compare(end).is_gt() {
        return Err(ValidatedGeometryError::ParameterOutOfRange.into());
    }
    let front = sum(end, parameter.negated().ok_or_else(refused)?)?;
    let back = sum(parameter, start.negated().ok_or_else(refused)?)?;
    let mut point = [number(0.0)?; 3];
    for (axis, coordinate) in point.iter_mut().enumerate() {
        budget.charge(4, cancel)?;
        *coordinate = sum(
            product(front, number(controls[0][axis])?)?,
            product(back, number(controls[1][axis])?)?,
        )?;
    }
    Ok(point)
}

fn exact_uv_transform(
    matrix: &Matrix3,
    point: [Dyadic; 3],
) -> Result<[Dyadic; 3], TrimDomainError> {
    let mut result = [number(0.0)?; 3];
    for axis in 0..3 {
        for term in 0..3 {
            result[axis] = sum(
                result[axis],
                product(number(matrix[term][axis])?, point[term])?,
            )?;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod wrapped_extrusion_chart_tests {
    use super::*;
    use crate::{
        EuclideanSpace, ExtrusionSurface, Invertible, Processor, TrimmedCurve, UnitCircle,
        assert_near,
    };

    #[test]
    fn translated_trimmed_profiles_preserve_original_inverse_anchor() {
        let basis = Curve::Line(Line(Point3::origin(), Point3::new(1.0, 0.0, 0.0)));
        let mut profile = Processor::new(TrimmedCurve::new(Box::new(basis.clone()), (-2.0, 7.0)));
        profile.invert();
        let direction = Vector3::new(0.0, 0.0, 2.0);
        let surface = Surface::ExtrusionSurface(ExtrusionSurface::by_extrusion(
            Curve::TrimmedCurve(profile),
            direction,
        ));
        let edge = Curve::TrimmedCurve(Processor::with_transform(
            TrimmedCurve::new(Box::new(basis), (1.0, 4.0)),
            Matrix4::from_translation(direction * 3.0),
        ));
        let uv = certified_extrusion_parameter_curve(
            &edge,
            &surface,
            ExtrusionChartCurve::ProfileAt { offset: 3.0 },
            &mut ValidatedGeometryBudget::new(100_000),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(uv.range_tuple(), (1.0, 4.0));
        for t in [1.0, 2.0, 4.0] {
            let parameter = uv.evaluate(t);
            assert_near!(surface.evaluate(parameter.x, parameter.y), edge.evaluate(t));
        }
    }

    #[test]
    fn trimmed_inverse_generator_uses_its_actual_nondefault_domain() {
        let profile = Curve::Line(Line(Point3::origin(), Point3::new(1.0, 0.0, 0.0)));
        let surface = Surface::ExtrusionSurface(ExtrusionSurface::by_extrusion(
            profile,
            Vector3::new(0.0, 0.0, 2.0),
        ));
        let mut edge = Processor::new(TrimmedCurve::new(
            Box::new(Curve::Line(Line(
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 2.0),
            ))),
            (-3.0, 5.0),
        ));
        edge.invert();
        let edge = Curve::TrimmedCurve(edge);
        let uv = certified_extrusion_parameter_curve(
            &edge,
            &surface,
            ExtrusionChartCurve::GeneratorAt {
                profile_parameter: 1.0,
            },
            &mut ValidatedGeometryBudget::new(100_000),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(uv.range_tuple(), (-3.0, 5.0));
        for t in [-3.0, 0.0, 5.0] {
            let parameter = uv.evaluate(t);
            assert_near!(surface.evaluate(parameter.x, parameter.y), edge.evaluate(t));
        }
    }

    #[test]
    fn analytic_profile_identity_includes_original_conic_domain() {
        let conic = |range| {
            Curve2D::Conic(Conic2D::Ellipse(Processor::new(TrimmedCurve::new(
                UnitCircle::<Point2>::new(),
                range,
            ))))
        };
        let a = conic((0.2, 5.7));
        let b = conic((0.3, 5.6));
        assert!(
            !same_uv(
                &a,
                &b,
                &mut ValidatedGeometryBudget::new(100_000),
                &AtomicBool::new(false)
            )
            .unwrap()
        );
        assert!(
            same_uv(
                &a,
                &a,
                &mut ValidatedGeometryBudget::new(100_000),
                &AtomicBool::new(false)
            )
            .unwrap()
        );
    }
}
