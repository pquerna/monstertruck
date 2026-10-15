use super::isocurve::ExplicitCurve;
use super::planar::Dyadic;
use super::*;
use crate::{KnotVector, Plane, Point3};

fn unresolved() -> TrimDomainError { ValidatedGeometryError::Unresolved.into() }

fn scalar(value: f64) -> Result<Dyadic, TrimDomainError> {
    Dyadic::scalar(value).ok_or_else(unresolved)
}

fn add(a: Dyadic, b: Dyadic) -> Result<Dyadic, TrimDomainError> { a.add(b).ok_or_else(unresolved) }

fn multiply(a: Dyadic, b: Dyadic) -> Result<Dyadic, TrimDomainError> {
    a.multiply(b).ok_or_else(unresolved)
}

fn axis_domain(knots: &KnotVector, count: usize) -> Result<(usize, [f64; 2]), TrimDomainError> {
    let degree = knots
        .len()
        .checked_sub(count.saturating_add(1))
        .ok_or(TrimDomainError::InvalidChart)?;
    if degree == 0 || degree > 32 || count != degree + 1 {
        return Err(ValidatedGeometryError::Unsupported.into());
    }
    let range = [knots[degree], knots[count]];
    if !range.into_iter().all(f64::is_finite)
        || range[0] >= range[1]
        || !knots.iter().take(degree + 1).all(|knot| *knot == range[0])
        || !knots.iter().skip(degree + 1).all(|knot| *knot == range[1])
    {
        return Err(ValidatedGeometryError::Unsupported.into());
    }
    Ok((degree, range))
}

fn frame(
    knots: &(KnotVector, KnotVector),
    counts: [usize; 2],
    control: impl Fn(usize, usize) -> [f64; 4],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<(Plane, [[f64; 2]; 2]), TrimDomainError> {
    budget.charge(knots.0.len().saturating_add(knots.1.len()), cancel)?;
    let (p, urange) = axis_domain(&knots.0, counts[0])?;
    let (q, vrange) = axis_domain(&knots.1, counts[1])?;
    let point = |i, j| {
        let point = control(i, j);
        if point[3] == 0.0 || !point.into_iter().all(f64::is_finite) {
            return Err(TrimDomainError::InvalidChart);
        }
        let point = Point3::new(
            point[0] / point[3],
            point[1] / point[3],
            point[2] / point[3],
        );
        if ![point.x, point.y, point.z].into_iter().all(f64::is_finite) {
            Err(TrimDomainError::InvalidChart)
        } else {
            Ok(point)
        }
    };
    let origin = point(0, 0)?;
    let u = point(p, 0)? - origin;
    let v = point(0, q)? - origin;
    let plane = Plane::from_origin_vectors(origin, u, v);
    let zero = scalar(0.0)?;
    let value = |i: usize, j: usize, axis: usize| -> Result<Dyadic, TrimDomainError> {
        if i <= p && j <= q {
            scalar(control(i, j)[axis])
        } else {
            Ok(zero)
        }
    };
    for i in 0..=p + 1 {
        for j in 0..=q + 1 {
            budget.charge(1, cancel)?;
            let factors = [
                (i, j, (p + 1 - i) * (q + 1 - j)),
                (i.wrapping_sub(1), j, i * (q + 1 - j)),
                (i, j.wrapping_sub(1), (p + 1 - i) * j),
                (i.wrapping_sub(1), j.wrapping_sub(1), i * j),
            ];
            let elevated = |axis| -> Result<Dyadic, TrimDomainError> {
                factors.into_iter().try_fold(zero, |sum, (a, b, factor)| {
                    add(sum, multiply(value(a, b, axis)?, scalar(factor as f64)?)?)
                })
            };
            let weight = elevated(3)?;
            let u_weight = add(
                multiply(
                    value(i.wrapping_sub(1), j, 3)?,
                    scalar((i * (q + 1 - j)) as f64)?,
                )?,
                multiply(
                    value(i.wrapping_sub(1), j.wrapping_sub(1), 3)?,
                    scalar((i * j) as f64)?,
                )?,
            )?;
            let v_weight = add(
                multiply(
                    value(i, j.wrapping_sub(1), 3)?,
                    scalar(((p + 1 - i) * j) as f64)?,
                )?,
                multiply(
                    value(i.wrapping_sub(1), j.wrapping_sub(1), 3)?,
                    scalar((i * j) as f64)?,
                )?,
            )?;
            for axis in 0..3 {
                budget.charge(1, cancel)?;
                let rhs = add(
                    add(
                        multiply(scalar(origin[axis])?, weight)?,
                        multiply(scalar(u[axis])?, u_weight)?,
                    )?,
                    multiply(scalar(v[axis])?, v_weight)?,
                )?;
                if elevated(axis)? != rhs {
                    return Err(unresolved());
                }
            }
        }
    }
    Ok((plane, [urange, vrange]))
}

fn map_uv(
    curve: Curve2D,
    domain: [[f64; 2]; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Curve2D, TrimDomainError> {
    let scale = [domain[0][1] - domain[0][0], domain[1][1] - domain[1][0]];
    for axis in 0..2 {
        if add(scalar(domain[axis][1])?, scalar(-domain[axis][0])?)? != scalar(scale[axis])? {
            return Err(unresolved());
        }
    }
    budget.charge(1, cancel)?;
    if domain == [[0.0, 1.0], [0.0, 1.0]] {
        return Ok(curve);
    }
    let range = curve.range_tuple();
    if ![range.0, range.1].into_iter().all(f64::is_finite) || range.0 >= range.1 {
        return Err(TrimDomainError::InvalidChart);
    }
    Ok(Curve2D::TrimmedCurve(crate::Processor::with_transform(
        crate::TrimmedCurve::new(Box::new(curve), range),
        crate::Matrix3::from_cols(
            crate::Vector3::new(scale[0], 0.0, 0.0),
            crate::Vector3::new(0.0, scale[1], 0.0),
            crate::Vector3::new(domain[0][0], domain[1][0], 1.0),
        ),
    )))
}

fn extrusion_frame(
    surface: &crate::ExtrusionSurface<Curve, crate::Vector3>,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<(Plane, [[f64; 2]; 2]), TrimDomainError> {
    let profile = match surface.entity_curve() {
        Curve::Line(curve) => ExplicitCurve::Line(curve),
        Curve::BsplineCurve(curve) => ExplicitCurve::Polynomial(curve),
        Curve::NurbsCurve(curve) => ExplicitCurve::Rational(curve),
        _ => return Err(ValidatedGeometryError::Unsupported.into()),
    };
    budget.charge(profile.knots_len(), cancel)?;
    let count = profile.count();
    let degree = profile
        .knots_len()
        .checked_sub(count.saturating_add(1))
        .ok_or(TrimDomainError::InvalidChart)?;
    if degree == 0 || degree > 32 || count != degree + 1 {
        return Err(ValidatedGeometryError::Unsupported.into());
    }
    let range = [profile.knot(degree), profile.knot(count)];
    if !range.into_iter().all(f64::is_finite)
        || range[0] >= range[1]
        || !(0..degree + 1).all(|index| profile.knot(index) == range[0])
        || !(degree + 1..profile.knots_len()).all(|index| profile.knot(index) == range[1])
    {
        return Err(ValidatedGeometryError::Unsupported.into());
    }
    let point = |index| -> Result<Point3, TrimDomainError> {
        let point = profile.control(index);
        if point[3] == 0.0 {
            return Err(ValidatedGeometryError::DenominatorContainsZero.into());
        }
        Ok(Point3::new(
            point[0] / point[3],
            point[1] / point[3],
            point[2] / point[3],
        ))
    };
    let origin = point(0)?;
    let u = point(count - 1)? - origin;
    let zero = scalar(0.0)?;
    let coefficient = |index: usize, axis| -> Result<Dyadic, TrimDomainError> {
        if index < count {
            scalar(profile.control(index)[axis])
        } else {
            Ok(zero)
        }
    };
    for index in 0..=degree + 1 {
        budget.charge(1, cancel)?;
        let elevated = |axis| -> Result<Dyadic, TrimDomainError> {
            add(
                multiply(
                    coefficient(index, axis)?,
                    scalar((degree + 1 - index) as f64)?,
                )?,
                multiply(
                    coefficient(index.wrapping_sub(1), axis)?,
                    scalar(index as f64)?,
                )?,
            )
        };
        let weight = elevated(3)?;
        let u_weight = multiply(
            coefficient(index.wrapping_sub(1), 3)?,
            scalar(index as f64)?,
        )?;
        for axis in 0..3 {
            let rhs = add(
                multiply(scalar(origin[axis])?, weight)?,
                multiply(scalar(u[axis])?, u_weight)?,
            )?;
            if elevated(axis)? != rhs {
                return Err(unresolved());
            }
        }
    }
    let direction = surface.extruding_vector();
    if ![direction.x, direction.y, direction.z]
        .into_iter()
        .all(f64::is_finite)
    {
        return Err(TrimDomainError::InvalidChart);
    }
    // The second interval expresses an identity coordinate map, not a V support cell.
    Ok((
        Plane::from_origin_vectors(origin, u, direction),
        [range, [0.0, 1.0]],
    ))
}

/// Extracts exact UV correspondence on an affine native plane or tensor Bézier carrier.
///
/// Supports bounded single-span polynomial and rational surfaces through degree
/// 32 in each axis. Every elevated homogeneous coefficient is compared using
/// exact checked dyadic arithmetic after clearing the Bernstein factors
/// `(p + 1) * (q + 1)`. The complete identity `X = (O + U*s + V*t) * W` is proved,
/// not fitted or sampled. Mixed weight signs are allowed; actual denominator
/// separation remains the responsibility of native interval surface admission.
/// Curve projection and the final native-knot-domain remap also require exact
/// whole-control identities. Rounded candidates, excessive dyadic exponent spread,
/// nonlinear parameterizations and multi-span carriers return typed refusals.
/// No surface, source curve or BREP is copied.
///
/// Exact affine line/Bézier-profile extrusions are also supported without building
/// a rounded tensor net. Their V coordinate is the actual extrusion coordinate;
/// the legacy reported `0..1` V cell is never treated as face support.
pub fn certified_affine_surface_parameter_curve(
    curve: &Curve,
    surface: &Surface,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Curve2D, TrimDomainError> {
    budget.charge(1, cancel)?;
    if let Surface::Plane(plane) = surface {
        return certified_planar_parameter_curve(curve, plane, budget, cancel);
    }
    let (plane, domain) = match surface {
        Surface::BsplineSurface(surface) => {
            let controls = surface.control_points();
            let columns = controls.first().ok_or(TrimDomainError::InvalidChart)?.len();
            budget.charge(controls.len(), cancel)?;
            if columns == 0 || !controls.iter().all(|row| row.len() == columns) {
                return Err(TrimDomainError::InvalidChart);
            }
            frame(
                surface.knot_vectors(),
                [controls.len(), columns],
                |i, j| {
                    let point = controls[i][j];
                    [point.x, point.y, point.z, 1.0]
                },
                budget,
                cancel,
            )?
        }
        Surface::NurbsSurface(surface) => {
            let controls = surface.control_points();
            let columns = controls.first().ok_or(TrimDomainError::InvalidChart)?.len();
            budget.charge(controls.len(), cancel)?;
            if columns == 0 || !controls.iter().all(|row| row.len() == columns) {
                return Err(TrimDomainError::InvalidChart);
            }
            frame(
                surface.knot_vectors(),
                [controls.len(), columns],
                |i, j| {
                    let point = controls[i][j];
                    [point.x, point.y, point.z, point.w]
                },
                budget,
                cancel,
            )?
        }
        Surface::ExtrusionSurface(surface) => extrusion_frame(surface, budget, cancel)?,
        _ => return Err(ValidatedGeometryError::Unsupported.into()),
    };
    map_uv(
        certified_planar_parameter_curve(curve, &plane, budget, cancel)?,
        domain,
        budget,
        cancel,
    )
}
