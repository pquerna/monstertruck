//! Conservative native carrier-component certificates for interval union.
//! A transverse plane cuts a positive-weight degree-one homogeneous fibre in
//! exactly one point for every profile parameter. Uniform coefficient signs
//! certify that fact over the entire profile, rather than at sampled points.

use super::*;
use std::mem::size_of;

pub(super) struct ComponentPoint {
    pub coordinate: f64,
    pub tangent: Vector3,
}

pub(super) enum NativeComponent {
    Line {
        direction: Vector3,
        planes: [Plane; 2],
    },
    Ruled {
        surface: NurbsSurface<Vector4>,
        plane: Plane,
        profile_u: bool,
        positive_coefficient: bool,
        ranges: [(f64, f64); 2],
        period: Option<f64>,
    },
}

fn finite4(point: Vector4) -> bool {
    [point.x, point.y, point.z, point.w]
        .into_iter()
        .all(f64::is_finite)
}

fn plane_form(plane: Plane) -> Option<Vector4> {
    let normal = plane.normal();
    let form = Vector4::new(
        normal.x,
        normal.y,
        normal.z,
        -normal.dot(plane.origin().to_vec()),
    );
    finite4(form).then_some(form)
}

fn dot_error(form: Vector4, point: Vector4) -> f64 {
    128.0
        * f64::EPSILON
        * ((form.x * point.x).abs()
            + (form.y * point.y).abs()
            + (form.z * point.z).abs()
            + (form.w * point.w).abs())
        .max(f64::MIN_POSITIVE)
}

fn admit(surface: &BsplineSurface<Vector4>, work: Option<&BooleanWork<'_>>) -> ClassicResult<()> {
    if let Some(work) = work {
        let rows = surface.control_points();
        let controls = rows.iter().map(Vec::len).sum();
        let bytes = rows
            .capacity()
            .saturating_mul(size_of::<Vec<Vector4>>())
            .saturating_add(
                rows.iter()
                    .map(|row| row.capacity().saturating_mul(size_of::<Vector4>()))
                    .sum::<usize>(),
            )
            .saturating_add(
                surface
                    .knot_vector_u()
                    .capacity()
                    .saturating_add(surface.knot_vector_v().capacity())
                    .saturating_mul(size_of::<f64>()),
            );
        work.limit(
            "native_component_controls",
            controls,
            work.limits.max_mesh_vertices,
        )?;
        work.limit(
            "native_component_bytes",
            bytes,
            work.limits.max_intermediate_prepared_bytes,
        )?;
    }
    Ok(())
}

fn exact_plane<S>(source: &S, work: Option<&BooleanWork<'_>>) -> ClassicResult<Option<Plane>>
where S: TryIntoAnalyticSurfaceKind + TryIntoHomogeneousBsplineSurface {
    let Some(AnalyticSurfaceKind::Plane(plane)) = source.try_into_analytic_surface_kind() else {
        return Ok(None);
    };
    let Some(form) = plane_form(plane) else {
        return Ok(None);
    };
    let Some(net) = source.try_into_homogeneous_bspline_surface() else {
        return Ok(None);
    };
    admit(&net, work)?;
    let rows = net.control_points();
    if rows.is_empty() || rows.iter().any(Vec::is_empty) {
        return Ok(None);
    }
    for &point in rows.iter().flatten() {
        if let Some(work) = work {
            work.check()?;
        }
        let error = dot_error(form, point);
        if !finite4(point)
            || point.w <= f64::MIN_POSITIVE * 1024.0
            || !error.is_finite()
            || error / point.w > TOLERANCE
            || form.dot(point).abs() > error
        {
            return Ok(None);
        }
    }
    Ok(Some(plane))
}

fn plane_fibre(
    surface: &NurbsSurface<Vector4>,
    plane: Plane,
    profile_u: bool,
    fibre: (f64, f64),
    profile: f64,
    positive_coefficient: bool,
) -> Option<(Point3, Vector3)> {
    let net = surface.non_rationalized();
    let (a, b) = if profile_u {
        ((profile, fibre.0), (profile, fibre.1))
    } else {
        ((fibre.0, profile), (fibre.1, profile))
    };
    let h0 = net.evaluate(a.0, a.1);
    let h1 = net.evaluate(b.0, b.1);
    if !finite4(h0) || !finite4(h1) || h0.w != h1.w || h0.w <= f64::MIN_POSITIVE * 1024.0 {
        return None;
    }
    let form = plane_form(plane)?;
    let across = h1 - h0;
    let denominator = form.dot(across);
    let error = dot_error(form, h0) + dot_error(form, h1);
    if !denominator.is_finite()
        || !error.is_finite()
        || denominator.abs() <= error
        || (denominator > 0.0) != positive_coefficient
    {
        return None;
    }
    let fraction = -form.dot(h0) / denominator;
    if !fraction.is_finite() {
        return None;
    }
    // Equal fibre weights make this an affine extension, with no rational
    // pole beyond either original patch endpoint. Never clamp the fibre:
    // original trim provenance determines the retained material interval.
    let value = h0 + fraction * across;
    let d0 = if profile_u {
        net.derivative_u(a.0, a.1)
    } else {
        net.derivative_v(a.0, a.1)
    };
    let d1 = if profile_u {
        net.derivative_u(b.0, b.1)
    } else {
        net.derivative_v(b.0, b.1)
    };
    if !finite4(value) || !finite4(d0) || !finite4(d1) {
        return None;
    }
    let along = d0 + fraction * (d1 - d0);
    let derivative = along - form.dot(along) / denominator * across;
    if !finite4(derivative) {
        return None;
    }
    let point = Point3::from_vec(value.truncate() / value.w);
    let tangent = (derivative.truncate() - point.to_vec() * derivative.w) / value.w;
    ([point.x, point.y, point.z].into_iter().all(f64::is_finite)
        && tangent.magnitude2().is_finite()
        && tangent.magnitude2() > 0.0)
        .then_some((point, tangent))
}

impl NativeComponent {
    pub(super) fn new<S>(
        carriers: [&S; 2],
        work: Option<&BooleanWork<'_>>,
    ) -> ClassicResult<Option<Self>>
    where
        S: TryIntoAnalyticSurfaceKind + TryIntoHomogeneousBsplineSurface,
    {
        if let Some(work) = work {
            work.check()?;
        }
        let planes = [
            exact_plane(carriers[0], work)?,
            exact_plane(carriers[1], work)?,
        ];
        if let [Some(first), Some(second)] = planes {
            let normal0 = first.normal();
            let normal1 = second.normal();
            let cross = normal0.cross(normal1);
            let scale = normal0.magnitude() * normal1.magnitude();
            if cross.magnitude().is_finite() && cross.magnitude() > 128.0 * f64::EPSILON * scale {
                return Ok(Some(Self::Line {
                    direction: cross.normalize(),
                    planes: [first, second],
                }));
            }
            return Ok(None);
        }
        let (plane, ruled) = match planes {
            [Some(plane), None] => (plane, carriers[1]),
            [None, Some(plane)] => (plane, carriers[0]),
            _ => return Ok(None),
        };
        let Some(form) = plane_form(plane) else {
            return Ok(None);
        };
        let Some(net) = ruled.try_into_homogeneous_bspline_surface() else {
            return Ok(None);
        };
        admit(&net, work)?;
        let rows = net.control_points();
        let Some(first) = rows.first() else {
            return Ok(None);
        };
        if rows.len() < 2 || first.len() < 2 || rows.iter().any(|row| row.len() != first.len()) {
            return Ok(None);
        }
        let profile_u = if net.vdegree() == 1 && first.len() == 2 {
            true
        } else if net.udegree() == 1 && rows.len() == 2 {
            false
        } else {
            return Ok(None);
        };
        let fibre_knots = if profile_u {
            net.knot_vector_v()
        } else {
            net.knot_vector_u()
        };
        if fibre_knots.len() != 4
            || fibre_knots[0] != fibre_knots[1]
            || fibre_knots[2] != fibre_knots[3]
            || fibre_knots[1] >= fibre_knots[2]
        {
            return Ok(None);
        }
        let count = if profile_u { rows.len() } else { first.len() };
        let mut sign = None;
        let fibres = (0..count).map(|index| {
            if profile_u {
                (rows[index][0], rows[index][1])
            } else {
                (rows[0][index], rows[1][index])
            }
        });
        for (start, end) in fibres {
            if let Some(work) = work {
                work.check()?;
            }
            // Equal positive fibre weights exclude rational poles even outside
            // the finite patch while a guide is recovered to its original trim.
            if !finite4(start)
                || !finite4(end)
                || start.w != end.w
                || start.w <= f64::MIN_POSITIVE * 1024.0
            {
                return Ok(None);
            }
            let coefficient = form.dot(end - start);
            let error = dot_error(form, start) + dot_error(form, end);
            if !coefficient.is_finite() || !error.is_finite() || coefficient.abs() <= error {
                return Ok(None);
            }
            let positive = coefficient > 0.0;
            if sign.is_some_and(|previous| previous != positive) {
                return Ok(None);
            }
            sign = Some(positive);
        }
        let ku = net.knot_vector_u();
        let kv = net.knot_vector_v();
        if ku.iter().chain(kv.iter()).any(|value| !value.is_finite())
            || ku
                .windows(2)
                .chain(kv.windows(2))
                .any(|pair| pair[0] > pair[1])
        {
            return Ok(None);
        }
        let ranges = [
            (ku[net.udegree()], ku[rows.len()]),
            (kv[net.vdegree()], kv[first.len()]),
        ];
        if ranges.iter().any(|&(front, back)| front >= back) {
            return Ok(None);
        }
        let surface = NurbsSurface::new(net);
        let profile = ranges[usize::from(!profile_u)];
        let fibre = ranges[usize::from(profile_u)];
        let seam = |value| {
            let (a, b) = if profile_u {
                ((profile.0, value), (profile.1, value))
            } else {
                ((value, profile.0), (value, profile.1))
            };
            let p = surface.evaluate(a.0, a.1);
            let q = surface.evaluate(b.0, b.1);
            let dp = if profile_u {
                surface.derivative_u(a.0, a.1)
            } else {
                surface.derivative_v(a.0, a.1)
            };
            let dq = if profile_u {
                surface.derivative_u(b.0, b.1)
            } else {
                surface.derivative_v(b.0, b.1)
            };
            let scale = p.to_vec().magnitude().max(q.to_vec().magnitude()).max(1.0);
            p.distance2(q) <= (128.0 * f64::EPSILON * scale).powi(2)
                && dp.distance2(dq)
                    <= (128.0 * f64::EPSILON * dp.magnitude().max(dq.magnitude()).max(1.0)).powi(2)
        };
        let period = (seam(fibre.0) && seam(fibre.1)).then_some(profile.1 - profile.0);
        let Some(positive_coefficient) = sign else {
            return Ok(None);
        };
        Ok(Some(Self::Ruled {
            surface,
            plane,
            profile_u,
            positive_coefficient,
            ranges,
            period,
        }))
    }

    pub(super) fn period(&self) -> Option<f64> {
        match self {
            Self::Line { .. } => None,
            Self::Ruled { period, .. } => *period,
        }
    }

    pub(super) fn project(
        &self,
        point: Point3,
        previous: Option<f64>,
        work: Option<&BooleanWork<'_>>,
    ) -> Option<ComponentPoint> {
        match self {
            Self::Line { direction, planes } => {
                if planes
                    .iter()
                    .any(|plane| plane.search_parameter(point, None, 100).is_none())
                {
                    return None;
                }
                let coordinate = point.to_vec().dot(*direction);
                coordinate.is_finite().then_some(ComponentPoint {
                    coordinate,
                    tangent: *direction,
                })
            }
            Self::Ruled {
                surface,
                plane,
                profile_u,
                positive_coefficient,
                ranges,
                period,
            } => {
                let profile = ranges[usize::from(!*profile_u)];
                let fibre = ranges[usize::from(*profile_u)];
                let mut parameter = if let Some(previous) = previous {
                    if period.is_some() {
                        profile.0 + (previous - profile.0).rem_euclid(profile.1 - profile.0)
                    } else {
                        previous.clamp(profile.0, profile.1)
                    }
                } else {
                    // Preserve the existing surface initializer. If Newton
                    // steps outside its bounded spline fibre, use that same
                    // fixed presearch seed directly for the certified root.
                    let seed = surface
                        .search_nearest_parameter(point, None, 100)
                        .unwrap_or_else(|| {
                            algo::surface::presearch(surface, point, (ranges[0], ranges[1]), 50)
                        });
                    if *profile_u { seed.0 } else { seed.1 }
                };
                parameter = parameter.clamp(profile.0, profile.1);
                let mut settled = false;
                for _ in 0..100 {
                    if work.is_some_and(|work| work.check().is_err()) {
                        return None;
                    }
                    let (value, tangent) = plane_fibre(
                        surface,
                        *plane,
                        *profile_u,
                        fibre,
                        parameter,
                        *positive_coefficient,
                    )?;
                    let speed2 = tangent.magnitude2();
                    let step = tangent.dot(point - value) / speed2;
                    let step2 = step * step * speed2;
                    if !step.is_finite() || !step2.is_finite() {
                        return None;
                    }
                    let small = value.near(&point) && step2 <= TOLERANCE2;
                    if small && (settled || step2 == 0.0) {
                        let mut coordinate = parameter;
                        if let (Some(period), Some(previous)) = (period, previous) {
                            coordinate += ((previous - coordinate) / period).round() * period;
                        }
                        return coordinate.is_finite().then_some(ComponentPoint {
                            coordinate,
                            tangent,
                        });
                    }
                    settled = small;
                    parameter = (parameter + step).clamp(profile.0, profile.1);
                }
                None
            }
        }
    }
}
