//! Whole original rational arc admission on a literal, independently posed sphere.
//! Directed rounding bounds the complete homogeneous Bernstein identity in world
//! distance. The original coefficients, native domain and owned edge stay intact.

use super::chart_boundary::{clamped, curve_control, curve_data, plane_value};
use super::original_face::{Enclosure, sum};
use super::*;

struct SpherePose {
    rows: [usize; 3],
    axes: [f64; 3],
    translation: [f64; 3],
    scale: f64,
}

impl SpherePose {
    fn new(pose: Option<&Matrix4>) -> Option<Self> {
        let Some(pose) = pose else {
            return Some(Self {
                rows: [0, 1, 2],
                axes: [1.0; 3],
                translation: [0.0; 3],
                scale: 1.0,
            });
        };
        if pose.w.w != 1.0
            || (0..3).any(|axis| pose[axis][3] != 0.0)
            || (0..4).any(|column| (0..4).any(|row| !pose[column][row].is_finite()))
        {
            return None;
        }
        let mut rows = [0; 3];
        let mut axes = [0.0; 3];
        let mut used = [false; 3];
        for axis in 0..3 {
            let mut nonzero = (0..3).filter(|&row| pose[axis][row] != 0.0);
            let row = nonzero.next()?;
            if nonzero.next().is_some() || used[row] {
                return None;
            }
            rows[axis] = row;
            axes[axis] = pose[axis][row];
            used[row] = true;
        }
        let scale = axes[0].abs();
        (scale > 0.0 && axes.iter().all(|axis| axis.abs() == scale)).then_some(Self {
            rows,
            axes,
            translation: [pose.w.x, pose.w.y, pose.w.z],
            scale,
        })
    }

    fn delta(&self, control: Vector4, sphere: &Sphere) -> [Enclosure; 3] {
        std::array::from_fn(|axis| {
            Enclosure::point(control[self.rows[axis]])
                .sub(
                    Enclosure::point(self.translation[self.rows[axis]])
                        .mul(Enclosure::point(control.w)),
                )
                .div(self.axes[axis])
                .sub(Enclosure::point(sphere.center()[axis]).mul(Enclosure::point(control.w)))
        })
    }

    fn transverse(&self, sphere: &Sphere, plane: &OriginalPlane) -> bool {
        let center = std::array::from_fn::<_, 3, _>(|row| {
            let Some(axis) = self.rows.iter().position(|&candidate| candidate == row) else {
                return Enclosure {
                    lo: f64::NAN,
                    hi: f64::NAN,
                };
            };
            Enclosure::point(self.translation[row])
                .add(Enclosure::point(self.axes[axis]).mul(Enclosure::point(sphere.center()[axis])))
                .sub(Enclosure::point(plane.origin[row]))
        });
        let distance =
            sum((0..3).map(|axis| center[axis].mul(Enclosure::point(plane.normal[axis]))));
        let normal2 = sum((0..3).map(|axis| {
            Enclosure::point(plane.normal[axis]).mul(Enclosure::point(plane.normal[axis]))
        }));
        let radius = Enclosure::point(sphere.radius()).mul(Enclosure::point(self.scale));
        let margin = normal2.mul(radius.mul(radius)).sub(distance.mul(distance));
        margin.finite() && margin.lo > 0.0
    }
}

fn plane_support(control: Vector4, plane: &OriginalPlane) -> bool {
    if plane_value(plane, control) != Some(0.0) {
        return false;
    }
    let Some(support) = &plane.support else {
        return true;
    };
    [support.u, support.v].into_iter().flatten().all(|form| {
        let coordinate = sum((0..3).map(|axis| {
            Enclosure::point(control[axis])
                .div(control.w)
                .sub(Enclosure::point(plane.origin[axis]))
                .mul(Enclosure::point(form[axis]))
        }));
        coordinate.finite() && coordinate.lo >= 0.0 && coordinate.hi <= support.denominator
    })
}

fn quadratic_span(
    source: OriginalCurveCoefficients<'_>,
    knots: &KnotVector,
    span: usize,
    sphere: &Sphere,
    pose: &SpherePose,
) -> Option<[[Enclosure; 4]; 3]> {
    let [Some(first), Some(middle), Some(last)] =
        [span - 2, span - 1, span].map(|index| curve_control(source, index))
    else {
        return None;
    };
    let original = [first, middle, last].map(|control| {
        let delta = pose.delta(control, sphere);
        [delta[0], delta[1], delta[2], Enclosure::point(control.w)]
    });
    // The exact de Boor recurrence evaluates the original homogeneous span.
    // These are coefficient enclosures for a proof, never replacement geometry.
    let evaluate = |parameter: Enclosure| -> Option<[Enclosure; 4]> {
        let mut values = original;
        for level in 1..=2 {
            for local in (level..=2).rev() {
                let index = span - 2 + local;
                let alpha = parameter.sub(Enclosure::point(knots[index])).quotient(
                    Enclosure::point(knots[index + 3 - level]).sub(Enclosure::point(knots[index])),
                )?;
                values[local] = std::array::from_fn(|axis| {
                    Enclosure::point(1.0)
                        .sub(alpha)
                        .mul(values[local - 1][axis])
                        .add(alpha.mul(values[local][axis]))
                });
            }
        }
        Some(values[2])
    };
    let start = Enclosure::point(knots[span]);
    let end = Enclosure::point(knots[span + 1]);
    let first = evaluate(start)?;
    let last = evaluate(end)?;
    let midpoint = evaluate(start.add(end).mul(Enclosure::point(0.5)))?;
    let middle = std::array::from_fn(|axis| {
        midpoint[axis]
            .mul(Enclosure::point(2.0))
            .sub(first[axis].add(last[axis]).mul(Enclosure::point(0.5)))
    });
    Some([first, middle, last])
}

fn quadratic_joins(
    source: OriginalCurveCoefficients<'_>,
    knots: &KnotVector,
    count: usize,
    work: Option<&BooleanWork<'_>>,
) -> std::result::Result<bool, AttachmentFailure> {
    let mut index = 3;
    while index < count {
        if let Some(work) = work {
            work.check()?;
        }
        let first = index;
        while index < count && knots[index] == knots[first] {
            if let Some(work) = work {
                work.check()?;
            }
            index += 1;
        }
        match index - first {
            // The original quadratic homogeneous basis is C1 here. Nonzero
            // same-sign weights preserve C1 after dehomogenization.
            1 => {}
            2 => {
                let [Some(before), Some(point), Some(after)] =
                    [first - 2, first - 1, first].map(|index| curve_control(source, index))
                else {
                    return Ok(false);
                };
                let weighted_difference = |first: Vector4, second: Vector4| {
                    Some(Vector3::new(
                        exact_add(
                            exact_mul(first.x, second.w)?,
                            -exact_mul(second.x, first.w)?,
                        )?,
                        exact_add(
                            exact_mul(first.y, second.w)?,
                            -exact_mul(second.y, first.w)?,
                        )?,
                        exact_add(
                            exact_mul(first.z, second.w)?,
                            -exact_mul(second.z, first.w)?,
                        )?,
                    ))
                };
                let Some(incoming) = weighted_difference(point, before) else {
                    return Ok(false);
                };
                let Some(outgoing) = weighted_difference(after, point) else {
                    return Ok(false);
                };
                // A C0 source knot needs its own G1 identity, not an assumed
                // C1 theorem or a fitted smooth replacement.
                if cross(incoming, outgoing) != Some(Vector3::zero())
                    || !exact_dot(incoming, outgoing).is_some_and(|value| value > 0.0)
                {
                    return Ok(false);
                }
            }
            _ => return Ok(false),
        }
    }
    Ok(true)
}
fn proves_arc<C>(
    original: &C,
    absolute: [Point3; 2],
    sphere: &Sphere,
    pose: &SpherePose,
    plane: &OriginalPlane,
    work: Option<&BooleanWork<'_>>,
) -> std::result::Result<bool, AttachmentFailure>
where
    C: ParametricCurve3D + BoundedCurve + TryIntoHomogeneousBsplineCurve,
{
    let Some(source) = original.original_curve_coefficients() else {
        return Ok(false);
    };
    let Some((knots, count)) = curve_data(source) else {
        return Ok(false);
    };
    if clamped(knots, count) != Some(2)
        || original.range_tuple() != (knots[2], knots[count])
        || !sphere.radius().is_finite()
        || sphere.radius() <= 0.0
        || !pose.transverse(sphere, plane)
    {
        return Ok(false);
    }
    let Some(first) = curve_control(source, 0) else {
        return Ok(false);
    };
    let direction = absolute[1] - absolute[0];
    let mut previous: Option<(Enclosure, f64)> = None;
    let mut minimum_weight = f64::INFINITY;
    for index in 0..count {
        let Some(control) = curve_control(source, index) else {
            return Ok(false);
        };
        if let Some(work) = work {
            work.check()?;
        }
        if ![control.x, control.y, control.z, control.w]
            .into_iter()
            .all(f64::is_finite)
            || control.w == 0.0
            || control.w.is_sign_positive() != first.w.is_sign_positive()
        {
            return Ok(false);
        }
        if !plane_support(control, plane) {
            return Ok(false);
        }
        minimum_weight = minimum_weight.min(control.w.abs());
        if index == 0 || index + 1 == count {
            let vertex = absolute[usize::from(index != 0)];
            if (0..3).any(|axis| {
                let residual = Enclosure::point(control[axis])
                    .div(control.w)
                    .sub(Enclosure::point(vertex[axis]));
                !residual.finite() || residual.magnitude() > TOLERANCE
            }) {
                return Ok(false);
            }
        }
        let projected = sum((0..3)
            .map(|axis| Enclosure::point(control[axis]).mul(Enclosure::point(direction[axis]))));
        if let Some((old, weight)) = previous {
            let progression = projected
                .mul(Enclosure::point(weight))
                .sub(old.mul(Enclosure::point(control.w)));
            if !progression.finite() || progression.lo <= 0.0 {
                return Ok(false);
            }
        }
        previous = Some((projected, control.w));
    }
    if !quadratic_joins(source, knots, count, work)? {
        return Ok(false);
    }
    let radius2 = Enclosure::point(sphere.radius()).mul(Enclosure::point(sphere.radius()));
    let binomial = [1.0, 2.0, 1.0];
    let degree_four = [1.0, 4.0, 6.0, 4.0, 1.0];
    let mut residual = 0.0_f64;
    for span in 2..count {
        if knots[span] == knots[span + 1] {
            continue;
        }
        if let Some(work) = work {
            work.charge(3)?;
        }
        let Some(controls) = quadratic_span(source, knots, span, sphere, pose) else {
            return Ok(false);
        };
        for (degree, divisor) in degree_four.into_iter().enumerate() {
            let mut coefficient = Enclosure::point(0.0);
            for first in 0..3 {
                if degree < first || degree - first >= 3 {
                    continue;
                }
                if let Some(work) = work {
                    work.check()?;
                }
                let second = degree - first;
                let product =
                    sum((0..3).map(|axis| controls[first][axis].mul(controls[second][axis])))
                        .sub(radius2.mul(controls[first][3]).mul(controls[second][3]));
                coefficient = coefficient.add(
                    product
                        .mul(Enclosure::point(binomial[first] * binomial[second]))
                        .div(divisor),
                );
            }
            if !coefficient.finite() {
                return Ok(false);
            }
            residual = residual.max(coefficient.magnitude());
        }
    }
    // |rho-r| <= |rho²-r²|/r. Same-sign weights bound the complete native
    // denominator, and the literal similarity converts this to world distance.
    let world_residual = Enclosure::point(residual).mul(Enclosure::point(pose.scale));
    let allowed = Enclosure::point(minimum_weight)
        .mul(Enclosure::point(minimum_weight))
        .mul(Enclosure::point(sphere.radius()))
        .mul(Enclosure::point(TOLERANCE));
    Ok(world_residual.finite() && allowed.finite() && world_residual.hi <= allowed.lo)
}

pub(super) fn proves_overlap<C, S>(
    original: &C,
    absolute: [Point3; 2],
    directed: [Point3; 2],
    intersection: &IntersectionCurve<PolylineCurve, S, S>,
    (side, original_boundary): (usize, &[Wire<Point3, C>]),
    work: Option<&BooleanWork<'_>>,
) -> std::result::Result<bool, AttachmentFailure>
where
    C: ParametricCurve3D
        + BoundedCurve
        + SearchParameter<CurveParameter, Point = Point3>
        + TryIntoHomogeneousBsplineCurve,
    S: ParametricSurface3D
        + SearchNearestParameter<SurfaceParameter, Point = Point3>
        + TryIntoAnalyticSurfaceKind
        + TryIntoHomogeneousBsplineSurface,
{
    let operands = [intersection.surface0(), intersection.surface1()];
    let Some((sphere, stored_pose)) = operands[side].original_sphere() else {
        return Ok(false);
    };
    let Some(plane) = OriginalPlane::from_surface(operands[1 - side]) else {
        return Ok(false);
    };
    let Some(source) = original.original_curve_coefficients() else {
        return Ok(false);
    };
    let Some(pose) = SpherePose::new(stored_pose) else {
        return Ok(false);
    };
    if !proves_arc(original, absolute, sphere, &pose, &plane, work)? {
        return Ok(false);
    }
    if !original_face::contains_curve(source, original_boundary, &plane, work)? {
        return Ok(false);
    }
    if (directed != absolute && directed != [absolute[1], absolute[0]])
        || !owned_endpoint::bounded_endpoints(intersection, directed, work)?
    {
        return Ok(false);
    }
    let (front, back) = original.range_tuple();
    let forward = directed == absolute;
    let mut previous: Option<(f64, Point3)> = None;
    let (native_front, _) = intersection.range_tuple();
    let count = intersection.leader().len();
    for index in 0..count {
        let Some(point) =
            owned_endpoint::projected_point(intersection, native_front + index as f64, work)?
        else {
            return Ok(false);
        };
        // Endpoint identity comes from the complete original branch and its
        // actual bounded endpoint consistency, never nearest-point ranking.
        let parameter = if index == 0 {
            if forward { front } else { back }
        } else if index + 1 == count {
            if forward { back } else { front }
        } else {
            let Some(parameter) = original.search_parameter(point, None, 100) else {
                return Ok(false);
            };
            parameter
        };
        if parameter < front || parameter > back || !original.evaluate(parameter).near(&point) {
            return Ok(false);
        }
        if let Some((old, old_point)) = previous {
            let progression = if forward {
                parameter - old
            } else {
                old - parameter
            };
            if progression < 0.0 || (progression == 0.0 && point != old_point) {
                return Ok(false);
            }
        }
        previous = Some((parameter, point));
    }
    Ok(previous.is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use monstertruck_modeling as modeling;

    fn original_quarter() -> NurbsCurve<Vector4> {
        let weight = 0.5_f64.sqrt();
        NurbsCurve::new(BsplineCurve::new(
            KnotVector::bezier_knot(2),
            vec![
                Vector4::new(5.0, 0.0, 0.0, 1.0),
                Vector4::new(5.0 * weight, 5.0 * weight, 0.0, weight),
                Vector4::new(0.0, 5.0, 0.0, 1.0),
            ],
        ))
    }

    #[test]
    fn original_sphere_arc_uses_whole_bernstein_residual_and_strict_transversality() {
        let curve = original_quarter();
        let sphere = Sphere::new(Point3::origin(), 5.0);
        let pose = SpherePose::new(None).unwrap();
        let plane = OriginalPlane::new(&Plane::xy()).unwrap();
        let endpoints = [curve.front(), curve.back()];
        assert!(proves_arc(&curve, endpoints, &sphere, &pose, &plane, None).unwrap());
        let unrelated = Sphere::new(Point3::origin(), 5.001);
        assert!(!proves_arc(&curve, endpoints, &unrelated, &pose, &plane, None).unwrap());
        let mut off_plane = curve.clone();
        off_plane.control_points_mut().nth(1).unwrap().z = 1.0e-8;
        assert!(!proves_arc(&off_plane, endpoints, &sphere, &pose, &plane, None).unwrap());
        let tangent = OriginalPlane::new(&Plane::new(
            Point3::new(0.0, 0.0, 5.0),
            Point3::new(1.0, 0.0, 5.0),
            Point3::new(0.0, 1.0, 5.0),
        ))
        .unwrap();
        assert!(!pose.transverse(&sphere, &tangent));
    }

    #[test]
    fn literal_sphere_pose_and_inversion_are_not_fitted_or_recomposed_operands() {
        let matrix =
            Matrix4::from_translation(Vector3::new(16.0, -8.0, 4.0)) * Matrix4::from_scale(2.0);
        let mut surface = Processor::with_transform(Sphere::new(Point3::origin(), 5.0), matrix);
        surface.invert();
        let (sphere, stored) = surface.original_sphere().unwrap();
        assert_eq!(stored, Some(&matrix));
        let pose = SpherePose::new(stored).unwrap();
        let mut curve = original_quarter();
        curve
            .control_points_mut()
            .for_each(|point| *point = matrix * *point);
        let plane = OriginalPlane::new(&Plane::new(
            Point3::new(0.0, 0.0, 4.0),
            Point3::new(1.0, 0.0, 4.0),
            Point3::new(0.0, 1.0, 4.0),
        ))
        .unwrap();
        assert!(
            proves_arc(
                &curve,
                [curve.front(), curve.back()],
                sphere,
                &pose,
                &plane,
                None
            )
            .unwrap()
        );
        let nested = Processor::with_transform(surface, matrix);
        assert!(nested.original_sphere().is_none());
        let fitted =
            modeling::Surface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
                modeling::Curve::NurbsCurve(original_quarter()),
                Point3::origin(),
                Vector3::unit_z(),
            )));
        assert!(fitted.original_sphere().is_none());
        assert!(SpherePose::new(Some(&Matrix4::from_nonuniform_scale(1.0, 2.0, 1.0))).is_none());
    }

    #[test]
    fn original_knot_inserted_sphere_arc_keeps_every_quadratic_span_and_raw_gauge() {
        let mut curve = original_quarter();
        for knot in [0.25, 0.5, 0.75] {
            curve.add_knot(knot);
        }
        curve.parameter_transform(2.0, 10.0);
        let before = curve.non_rationalized().clone();
        assert_eq!(curve.non_rationalized().control_points().len(), 6);
        let sphere = Sphere::new(Point3::origin(), 5.0);
        let pose = SpherePose::new(None).unwrap();
        let plane = OriginalPlane::new(&Plane::xy()).unwrap();
        assert!(
            proves_arc(
                &curve,
                [curve.front(), curve.back()],
                &sphere,
                &pose,
                &plane,
                None,
            )
            .unwrap()
        );
        assert_eq!(curve.range_tuple(), (10.0, 12.0));
        assert_eq!(curve.non_rationalized(), &before);
    }

    #[test]
    fn original_quadratic_c0_joins_require_g1_without_claiming_c1_or_smoothing_corners() {
        let knots = KnotVector::from(vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        let curve = |corner| {
            BsplineCurve::new(
                knots.clone(),
                vec![
                    Point3::new(-2.0, 0.0, 0.0),
                    Point3::new(-1.0, 1.0, 0.0),
                    Point3::new(0.0, 1.0, 0.0),
                    Point3::new(2.0, if corner { 2.0 } else { 1.0 }, 0.0),
                    Point3::new(3.0, 0.0, 0.0),
                ],
            )
        };
        let smooth_image = curve(false);
        let corner = curve(true);
        assert!(
            quadratic_joins(
                smooth_image.original_curve_coefficients().unwrap(),
                &knots,
                5,
                None,
            )
            .unwrap()
        );
        assert!(
            !quadratic_joins(
                corner.original_curve_coefficients().unwrap(),
                &knots,
                5,
                None,
            )
            .unwrap()
        );
    }

    #[test]
    fn original_sphere_arc_plane_equation_ignores_inactive_origin_axes_without_reframing() {
        let mut curve = original_quarter();
        for knot in [0.25, 0.5, 0.75] {
            curve.add_knot(knot);
        }
        let sphere = Sphere::new(Point3::origin(), 5.0);
        let pose = SpherePose::new(None).unwrap();
        let plane = OriginalPlane::new(&Plane::new(
            Point3::new(-6.0, -6.0, 0.0),
            Point3::new(-6.0, 6.0, 0.0),
            Point3::new(6.0, -6.0, 0.0),
        ))
        .unwrap();
        assert_eq!(plane.origin, Vector3::new(-6.0, -6.0, 0.0));
        assert_eq!(plane.normal, Vector3::new(0.0, 0.0, -144.0));
        assert!(
            proves_arc(
                &curve,
                [curve.front(), curve.back()],
                &sphere,
                &pose,
                &plane,
                None,
            )
            .unwrap()
        );
        assert_eq!(
            plane_value(&plane, Vector4::new(f64::NAN, 0.0, 0.0, 1.0)),
            None,
        );
        let mut pole = curve.clone();
        pole.control_points_mut().last().unwrap().w = -1.0;
        assert!(
            !proves_arc(
                &pole,
                [pole.front(), pole.back()],
                &sphere,
                &pose,
                &plane,
                None,
            )
            .unwrap()
        );
    }
}
