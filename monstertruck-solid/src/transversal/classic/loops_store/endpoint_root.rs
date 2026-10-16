//! Identify a declared native endpoint by its original plane-root law.
//! Numerical proposals do not acquire identity from proximity or parameter tolerances.

use super::chart_boundary::{clamped, curve_control, curve_data, plane_value};
use super::*;

pub(in crate::transversal::classic::loops_store) fn canonical_plane_endpoint<C, S>(
    edge: &Edge<Point3, C>,
    opposite: &S,
    parameter: f64,
    point: Point3,
    work: Option<&BooleanWork<'_>>,
) -> std::result::Result<Option<f64>, AttachmentFailure>
where
    C: ParametricCurve3D + BoundedCurve + TryIntoHomogeneousBsplineCurve,
    S: ParametricSurface3D + TryIntoAnalyticSurfaceKind + TryIntoHomogeneousBsplineSurface,
{
    if !parameter.is_finite() {
        return Ok(None);
    }
    let Some(plane) = OriginalPlane::from_surface(opposite) else {
        return Ok(None);
    };
    edge.with_curve(|curve| {
        let Some(source) = curve.original_curve_coefficients() else {
            return Ok(None);
        };
        let (front, back) = curve.range_tuple();
        let (count, degree, first_end, last_start) = match source {
            OriginalCurveCoefficients::Line(line) => {
                if curve.parameter_range() != line.parameter_range() {
                    return Ok(None);
                }
                (2, 1, back, front)
            }
            _ => {
                let Some((knots, count)) = curve_data(source) else {
                    return Ok(None);
                };
                if let Some(work) = work {
                    work.charge(knots.len())?;
                }
                let Some(degree) = clamped(knots, count) else {
                    return Ok(None);
                };
                if (front, back) != (knots[degree], knots[count]) {
                    return Ok(None);
                }
                (count, degree, knots[degree + 1], knots[count - 1])
            }
        };
        let control = |index| match source {
            OriginalCurveCoefficients::Line(line) => match index {
                0 => Some(line.0.to_homogeneous()),
                1 => Some(line.1.to_homogeneous()),
                _ => None,
            },
            _ => curve_control(source, index),
        };
        let mut selected = None;
        for at_front in [true, false] {
            if (at_front && parameter >= first_end) || (!at_front && parameter <= last_start) {
                continue;
            }
            let vertex = if at_front {
                edge.absolute_front()
            } else {
                edge.absolute_back()
            };
            if !point.near(&vertex.point()) {
                continue;
            }
            let endpoint_index = if at_front { 0 } else { count - 1 };
            let Some(endpoint) = control(endpoint_index) else {
                continue;
            };
            if coefficient_endpoint(endpoint, vertex.point()) != Some(true)
                || plane.contains(vertex.point()) != Some(true)
                || plane_value(&plane, endpoint) != Some(0.0)
            {
                continue;
            }
            let start = if at_front { 0 } else { count - degree - 1 };
            let mut positive = None;
            let mut valid = true;
            for index in start..=start + degree {
                if let Some(work) = work {
                    work.check()?;
                }
                let Some(coefficient) = control(index) else {
                    valid = false;
                    break;
                };
                if ![coefficient.x, coefficient.y, coefficient.z, coefficient.w]
                    .into_iter()
                    .all(f64::is_finite)
                    || coefficient.w == 0.0
                    || coefficient.w.is_sign_positive() != endpoint.w.is_sign_positive()
                {
                    valid = false;
                    break;
                }
                if index == endpoint_index {
                    continue;
                }
                let Some(value) = plane_value(&plane, coefficient) else {
                    valid = false;
                    break;
                };
                if value == 0.0 || positive.is_some_and(|positive| positive != (value > 0.0)) {
                    valid = false;
                    break;
                }
                positive = Some(value > 0.0);
            }
            if !valid || positive.is_none() {
                continue;
            }
            // Nonnegative original B-spline basis functions and same-sign weights
            // exclude every other root on this active endpoint span. The strictly
            // nonzero adjacent coefficient also proves a transverse endpoint jet.
            let endpoint_parameter = if at_front { front } else { back };
            if !curve.evaluate(endpoint_parameter).near(&vertex.point()) {
                continue;
            }
            if selected.is_some() {
                return Err("intersection_chart_ambiguous_endpoint_root".into());
            }
            selected = Some(endpoint_parameter);
        }
        Ok(selected)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use monstertruck_modeling as modeling;

    fn edge(controls: Vec<Vector4>, knots: KnotVector) -> modeling::Edge {
        let curve = NurbsCurve::new(BsplineCurve::new(knots, controls));
        let front = Vertex::new(curve.front());
        let back = Vertex::new(curve.back());
        Edge::new(&front, &back, modeling::Curve::NurbsCurve(curve))
    }

    #[test]
    fn original_endpoint_law_binds_shared_identity_without_parameter_tolerance() {
        let mut curve = NurbsCurve::new(BsplineCurve::new(
            KnotVector::from(vec![2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0]),
            vec![
                Vector4::new(16.0, 0.0, 0.0, 4.0),
                Vector4::new(16.0, 0.0, -8.0, 4.0),
                Vector4::new(48.0, 0.0, -16.0, 8.0),
                Vector4::new(64.0, 0.0, -16.0, 8.0),
                Vector4::new(128.0, 0.0, 0.0, 16.0),
            ],
        ));
        curve.parameter_transform(2.0, 10.0);
        let front = Vertex::new(curve.front());
        let back = Vertex::new(curve.back());
        let edge: modeling::Edge = Edge::new(&front, &back, modeling::Curve::NurbsCurve(curve));
        let plane = modeling::Surface::Plane(Plane::xy());
        assert_eq!(
            canonical_plane_endpoint(
                &edge,
                &plane,
                14.0 - 1.0e-13,
                front.point() + Vector3::new(0.0, 0.0, 1.4e-14),
                None
            )
            .unwrap(),
            Some(14.0)
        );
        assert_eq!(edge.absolute_front(), &front);
        assert_eq!(
            canonical_plane_endpoint(&edge, &plane, 18.0 + 1.0e-13, back.point(), None).unwrap(),
            Some(18.0)
        );
    }

    #[test]
    fn tangent_overlap_poles_and_another_scalar_root_do_not_supply_endpoint_identity() {
        let plane = modeling::Surface::Plane(Plane::xy());
        for (heights, weights) in [
            ([0.0, 0.0, 1.0], [1.0; 3]),
            ([0.0; 3], [1.0; 3]),
            ([0.0, 1.0, -1.0], [1.0; 3]),
            ([0.0, 1.0, 1.0], [1.0, -1.0, 1.0]),
        ] {
            let edge = edge(
                (0..3)
                    .map(|index| {
                        Vector4::new(4.0 * weights[index], 0.0, heights[index], weights[index])
                    })
                    .collect(),
                KnotVector::bezier_knot(2),
            );
            assert_eq!(
                canonical_plane_endpoint(
                    &edge,
                    &plane,
                    -1.0e-14,
                    edge.absolute_front().point(),
                    None
                )
                .unwrap(),
                None
            );
        }
        let near = edge(
            vec![
                Vector4::new(4.0, 0.0, 1.0e-12, 1.0),
                Vector4::new(4.0, 0.0, 1.0, 1.0),
                Vector4::new(4.0, 0.0, 2.0, 1.0),
            ],
            KnotVector::bezier_knot(2),
        );
        assert_eq!(
            canonical_plane_endpoint(&near, &plane, -1.0e-14, near.absolute_front().point(), None)
                .unwrap(),
            None
        );
    }
}
