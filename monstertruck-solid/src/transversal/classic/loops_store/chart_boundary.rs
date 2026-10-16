//! Whole literal tensor-boundary overlap, retaining the owned native edges.
//! A mesh leader selects an already proved regular branch; it does not prove
//! carrier membership, plane containment or source-edge identity.

use super::*;

type OriginalBoundaryLoop<C> = (Wire<Point3, C>, Wire<Point3, PolylineCurve>);

pub(super) fn curve_data(source: OriginalCurveCoefficients<'_>) -> Option<(&KnotVector, usize)> {
    match source {
        OriginalCurveCoefficients::Polynomial(curve) => {
            Some((curve.knot_vector(), curve.control_points().len()))
        }
        OriginalCurveCoefficients::Rational(curve) => {
            let curve = curve.non_rationalized();
            Some((curve.knot_vector(), curve.control_points().len()))
        }
        OriginalCurveCoefficients::Line(_) => None,
    }
}

pub(super) fn curve_control(
    source: OriginalCurveCoefficients<'_>,
    index: usize,
) -> Option<Vector4> {
    match source {
        OriginalCurveCoefficients::Polynomial(curve) => {
            Some(curve.control_points().get(index)?.to_homogeneous())
        }
        OriginalCurveCoefficients::Rational(curve) => {
            curve.non_rationalized().control_points().get(index)
        }
        OriginalCurveCoefficients::Line(_) => None,
    }
}

pub(super) fn clamped(knots: &KnotVector, count: usize) -> Option<usize> {
    let degree = knots.len().checked_sub(count + 1)?;
    if degree == 0 || degree >= count || !knots.iter().all(|knot| knot.is_finite()) {
        return None;
    }
    let front = knots[degree];
    let back = knots[count];
    (front < back
        && knots[..=degree].iter().all(|&knot| knot == front)
        && knots[count..].iter().all(|&knot| knot == back)
        && knots[degree + 1] > front
        && knots[count - 1] < back
        && knots.windows(2).all(|pair| pair[0] <= pair[1]))
    .then_some(degree)
}

fn tensor_data(
    source: OriginalSurfaceCoefficients<'_>,
) -> Option<(&KnotVector, &KnotVector, usize, usize)> {
    match source {
        OriginalSurfaceCoefficients::Polynomial(surface) => {
            let controls = surface.control_points();
            let (u, v) = surface.knot_vectors();
            let nv = controls.first()?.len();
            controls
                .iter()
                .all(|row| row.len() == nv)
                .then_some((u, v, controls.len(), nv))
        }
        OriginalSurfaceCoefficients::Rational(surface) => {
            let surface = surface.non_rationalized();
            let controls = surface.control_points();
            let (u, v) = surface.knot_vectors();
            let nv = controls.first()?.len();
            controls
                .iter()
                .all(|row| row.len() == nv)
                .then_some((u, v, controls.len(), nv))
        }
        OriginalSurfaceCoefficients::Extrusion { .. } => None,
    }
}

fn tensor_control(source: OriginalSurfaceCoefficients<'_>, u: usize, v: usize) -> Option<Vector4> {
    match source {
        OriginalSurfaceCoefficients::Polynomial(surface) => {
            Some(surface.control_points().get(u)?.get(v)?.to_homogeneous())
        }
        OriginalSurfaceCoefficients::Rational(surface) => {
            surface.non_rationalized().control_points().get(u)?.get(v)
        }
        OriginalSurfaceCoefficients::Extrusion { .. } => None,
    }
}

pub(super) fn plane_value(plane: &OriginalPlane, control: Vector4) -> Option<f64> {
    if ![control.x, control.y, control.z, control.w]
        .into_iter()
        .all(f64::is_finite)
    {
        return None;
    }
    let mut value = 0.0;
    for axis in 0..3 {
        if plane.normal[axis] != 0.0 {
            let origin = exact_mul(plane.origin[axis], control.w)?;
            let displacement = exact_add(control[axis], -origin)?;
            value = exact_add(value, exact_mul(plane.normal[axis], displacement)?)?;
        }
    }
    Some(value)
}

fn proves_boundary<C, S>(
    edge: &Edge<Point3, C>,
    surface: &S,
    plane: &OriginalPlane,
    work: Option<&BooleanWork<'_>>,
) -> std::result::Result<bool, AttachmentFailure>
where
    C: ParametricCurve3D + BoundedCurve + TryIntoHomogeneousBsplineCurve,
    S: ParametricSurface3D + TryIntoHomogeneousBsplineSurface,
{
    let Some(tensor) = surface.original_surface_coefficients() else {
        return Ok(false);
    };
    let Some((uk, vk, nu, nv)) = tensor_data(tensor) else {
        return Ok(false);
    };
    let (Some(up), Some(vp)) = (clamped(uk, nu), clamped(vk, nv)) else {
        return Ok(false);
    };
    let expected_domain = (
        (
            std::ops::Bound::Included(uk[up]),
            std::ops::Bound::Included(uk[nu]),
        ),
        (
            std::ops::Bound::Included(vk[vp]),
            std::ops::Bound::Included(vk[nv]),
        ),
    );
    if surface.parameter_range() != expected_domain {
        return Ok(false);
    }
    edge.with_curve(|curve| {
        let Some(source) = curve.original_curve_coefficients() else { return Ok(false) };
        let Some((knots, count)) = curve_data(source) else { return Ok(false) };
        let Some(degree) = clamped(knots, count) else { return Ok(false) };
        if curve.range_tuple() != (knots[degree], knots[count]) {
            return Ok(false);
        }
        let mut failure = None;
        let mut checkpoint = || {
            if let Some(work) = work
                && let Err(error) = work.check()
            {
                failure = Some(error);
                return None;
            }
            Some(())
        };
        let mut proof = || -> Option<bool> {
            let first = curve_control(source, 0)?;
            let last = curve_control(source, count - 1)?;
            if !coefficient_endpoint(first, edge.absolute_front().point())?
                || !coefficient_endpoint(last, edge.absolute_back().point())?
            {
                return Some(false);
            }
            let direction = difference(exact_point(last)?.to_vec(), exact_point(first)?.to_vec())?;
            if direction.magnitude2() == 0.0 {
                return Some(false);
            }
            // Each original span is Bernstein, and adjacent one-sided
            // tangents agree in direction. Strict scalar variation inside
            // each span and nonzero endpoint tangents prove regularity also
            // at the original knots, without fitting a smoother curve.
            if !(count - 1).is_multiple_of(degree) {
                return Some(false);
            }
            for repeated in knots[degree + 1..count].chunks(degree) {
                checkpoint()?;
                if repeated.len() != degree || repeated.iter().any(|&knot| knot != repeated[0]) {
                    return Some(false);
                }
            }
            for join in (degree..count - 1).step_by(degree) {
                checkpoint()?;
                let before = exact_point(curve_control(source, join - 1)?)?;
                let point = exact_point(curve_control(source, join)?)?;
                let after = exact_point(curve_control(source, join + 1)?)?;
                let incoming = difference(point.to_vec(), before.to_vec())?;
                let outgoing = difference(after.to_vec(), point.to_vec())?;
                if cross(incoming, outgoing)? != Vector3::zero()
                    || exact_dot(incoming, outgoing)? <= 0.0
                {
                    return Some(false);
                }
            }
            // Positive rational B-spline weights preserve monotonic control
            // order. Every active span has strict scalar variation; distinct
            // endpoint controls also exclude a stationary bounded endpoint.
            for index in 0..count {
                checkpoint()?;
                let control = curve_control(source, index)?;
                if control.w == 0.0 || control.w.is_sign_positive() != first.w.is_sign_positive()
                    || !plane.contains(exact_point(control)?)?
                {
                    return Some(false);
                }
                if index != 0 {
                    let previous = exact_point(curve_control(source, index - 1)?)?;
                    let actual = exact_point(control)?;
                    if actual == previous
                        || projected_difference(direction, actual.to_vec(), previous.to_vec())? < 0.0
                    {
                        return Some(false);
                    }
                }
            }
            for span in degree..count {
                checkpoint()?;
                if knots[span] < knots[span + 1]
                    && projected_difference(
                        direction,
                        exact_point(curve_control(source, span)?)?.to_vec(),
                        exact_point(curve_control(source, span - degree)?)?.to_vec(),
                    )? <= 0.0
                {
                    return Some(false);
                }
            }
            for axis in 0..2 {
                let (along, along_count, transverse_count, along_degree) =
                    if axis == 0 { (uk, nu, nv, up) } else { (vk, nv, nu, vp) };
                if degree != along_degree || count > along_count {
                    continue;
                }
                for offset in 0..=along_count - count {
                    checkpoint()?;
                    let low = along[offset + degree];
                    let high = along[offset + count];
                    // A copied row slice is the restricted original basis only
                    // at a global clamped endpoint or an exact multiplicity-p
                    // knot. Lower multiplicity needs actual knot insertion;
                    // a discontinuous multiplicity-(p+1) split is not this row.
                    if (offset != 0
                        && (along[offset] >= low || along[offset + degree + 1] <= low))
                        || (offset + count != along_count
                            && (along[offset + count - 1] >= high
                                || along[offset + count + degree] <= high))
                    {
                        continue;
                    }
                    let mut restricted_endpoints = true;
                    for index in 0..degree {
                        checkpoint()?;
                        restricted_endpoints &= along[offset + 1 + index] == low
                            && along[offset + count + index] == high;
                    }
                    if !restricted_endpoints {
                        continue;
                    }
                    let native_span = exact_add(knots[count], -knots[degree])?;
                    let source_span = exact_add(high, -low)?;
                    if source_span <= 0.0 {
                        continue;
                    }
                    let same_knots = (degree + 1..count).all(|index| {
                        let native = exact_add(knots[index], -knots[degree])
                            .and_then(|delta| exact_mul(delta, source_span));
                        let original = exact_add(along[offset + index], -low)
                            .and_then(|delta| exact_mul(delta, native_span));
                        matches!((native, original), (Some(native), Some(original)) if native == original)
                    });
                    if !same_knots {
                        continue;
                    }
                    for upper in [false, true] {
                        let boundary = if upper { transverse_count - 1 } else { 0 };
                        let neighbor = if upper { boundary - 1 } else { 1 };
                        let control = |index, row| {
                            if axis == 0 { tensor_control(tensor, index, row) }
                            else { tensor_control(tensor, row, index) }
                        };
                        let mut sign = None;
                        let mut matches = true;
                        for index in 0..count {
                            checkpoint()?;
                            let original = curve_control(source, index)?;
                            let boundary = control(offset + index, boundary)?;
                            let neighbor = control(offset + index, neighbor)?;
                            let value = plane_value(plane, neighbor)?;
                            matches &= original == boundary
                                && neighbor.w != 0.0
                                && neighbor.w.is_sign_positive() == first.w.is_sign_positive()
                                && value != 0.0;
                            let positive = value.is_sign_positive();
                            matches &= sign.is_none_or(|previous| previous == positive);
                            sign = Some(positive);
                        }
                        // The exact boundary row is zero in the plane functional.
                        // Its adjacent row has one strict sign, proving the full
                        // original tensor's transverse boundary derivative.
                        if matches {
                            return Some(true);
                        }
                    }
                }
            }
            Some(false)
        };
        let proved = proof() == Some(true);
        if let Some(error) = failure {
            return Err(error.into());
        }
        Ok(proved)
    })
}

fn selects_branch<C, S>(
    wire: &Wire<Point3, C>,
    intersection: &IntersectionCurve<PolylineCurve, S, S>,
    work: Option<&BooleanWork<'_>>,
) -> std::result::Result<bool, AttachmentFailure>
where
    C: ParametricCurve3D + BoundedCurve + SearchParameter<CurveParameter, Point = Point3>,
    S: ParametricSurface3D + SearchNearestParameter<SurfaceParameter, Point = Point3>,
{
    let mut previous = None;
    let mut first = None;
    let mut winding = 0;
    for point in intersection.leader().iter().copied() {
        let mut selected = None;
        for (index, edge) in wire.iter().enumerate() {
            if let Some(work) = work {
                work.check()?;
            }
            let parameter = edge.with_curve(|curve| {
                let (front, back) = curve.range_tuple();
                let t = if curve.front().near(&point) {
                    front
                } else if curve.back().near(&point) {
                    back
                } else {
                    curve.search_parameter(point, None, 100)?
                };
                (front <= t && t <= back && curve.evaluate(t).near(&point)).then_some(
                    index as f64
                        + if edge.orientation() {
                            (t - front) / (back - front)
                        } else {
                            (back - t) / (back - front)
                        },
                )
            });
            if let Some(parameter) = parameter {
                let parameter = if parameter == wire.len() as f64 {
                    0.0
                } else {
                    parameter
                };
                if selected.is_some_and(|previous: f64| !previous.near(&parameter)) {
                    return Ok(false);
                }
                selected = Some(parameter);
            }
        }
        let Some(mut parameter) = selected else {
            return Ok(false);
        };
        if let Some(previous) = previous {
            parameter += winding as f64 * wire.len() as f64;
            if parameter < previous {
                winding += 1;
                parameter += wire.len() as f64;
            }
            if winding > 1 || parameter < previous {
                return Ok(false);
            }
        } else {
            first = Some(parameter);
        }
        previous = Some(parameter);
    }
    let complete = first
        .zip(previous)
        .is_some_and(|(first, last)| (last - first).near(&(wire.len() as f64)));
    Ok(complete)
}

pub(in crate::transversal::classic::loops_store) fn original_loop<C, S>(
    native: &Loops<Point3, C>,
    polygon: &Loops<Point3, PolylineCurve>,
    surface: &S,
    (opposite, original_boundary): (&S, &[Wire<Point3, C>]),
    intersection: &IntersectionCurve<PolylineCurve, S, S>,
    chart_edges: &[(usize, usize)],
    work: Option<&BooleanWork<'_>>,
) -> std::result::Result<Option<OriginalBoundaryLoop<C>>, AttachmentFailure>
where
    C: Clone
        + ParametricCurve3D
        + BoundedCurve
        + SearchParameter<CurveParameter, Point = Point3>
        + TryIntoHomogeneousBsplineCurve,
    S: ParametricSurface3D
        + SearchNearestParameter<SurfaceParameter, Point = Point3>
        + TryIntoAnalyticSurfaceKind
        + TryIntoHomogeneousBsplineSurface,
{
    let Some(plane) = OriginalPlane::from_surface(opposite) else {
        return Ok(None);
    };
    if let Some(work) = work {
        work.limit(
            "intermediate_prepared_bytes",
            chart_edges
                .len()
                .saturating_mul(size_of::<(Edge<Point3, C>, Edge<Point3, PolylineCurve>)>()),
            work.limits.max_intermediate_prepared_bytes,
        )?;
    }
    let mut candidates = Vec::with_capacity(chart_edges.len());
    if let Some(work) = work {
        work.limit(
            "intermediate_prepared_bytes",
            candidates
                .capacity()
                .saturating_mul(size_of::<(Edge<Point3, C>, Edge<Point3, PolylineCurve>)>()),
            work.limits.max_intermediate_prepared_bytes,
        )?;
    }
    for &(wire, edge) in chart_edges {
        let native_edge = &native[wire][edge];
        if proves_boundary(native_edge, surface, &plane, work)? {
            let supported =
                native_edge.with_curve(|curve| match curve.original_curve_coefficients() {
                    Some(source) => {
                        original_face::contains_curve(source, original_boundary, &plane, work)
                    }
                    None => Ok(false),
                })?;
            if supported {
                candidates.push((native_edge.clone(), polygon[wire][edge].clone()));
            }
        }
    }
    let mut selected = None;
    while let Some((edge, polygon_edge)) = candidates.pop() {
        let mut wire: Wire<_, _> = std::iter::once(edge).collect();
        let mut polygon_wire: Wire<_, _> = std::iter::once(polygon_edge).collect();
        while wire.back_vertex() != wire.front_vertex() {
            let next = {
                let mut matches = candidates
                    .iter()
                    .enumerate()
                    .filter_map(|(index, (edge, _))| {
                        (Some(edge.front()) == wire.back_vertex()).then_some(index)
                    });
                let first = matches.next();
                if matches.next().is_some() {
                    return Err("intersection_original_chart_boundary_incidence".into());
                }
                let Some(first) = first else { return Ok(None) };
                first
            };
            let (edge, polygon_edge) = candidates.swap_remove(next);
            wire.push_back(edge);
            polygon_wire.push_back(polygon_edge);
        }
        if let Some(anchor) = native
            .iter()
            .flat_map(|boundary| boundary.iter())
            .find_map(|edge| wire.iter().position(|candidate| candidate == edge))
        {
            wire.rotate_left(anchor);
            polygon_wire.rotate_left(anchor);
        }
        let forward = selects_branch(&wire, intersection, work)?;
        wire.invert();
        polygon_wire.invert();
        let reverse = selects_branch(&wire, intersection, work)?;
        if forward && reverse {
            return Err("intersection_original_chart_boundary_ambiguous".into());
        }
        if !forward && !reverse {
            continue;
        }
        if forward {
            wire.invert();
            polygon_wire.invert();
        }
        if selected.is_some() {
            return Err("intersection_original_chart_boundary_ambiguous".into());
        }
        selected = Some((wire, polygon_wire));
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use monstertruck_modeling as modeling;

    fn curved_boundary(
        curve_middle: Point3,
        carrier_middle: Point3,
        rise: f64,
    ) -> (modeling::Edge, modeling::Surface) {
        let front = Point3::new(-1.0, 0.0, 0.0);
        let back = Point3::new(1.0, 0.0, 0.0);
        let mut curve =
            BsplineCurve::new(KnotVector::bezier_knot(2), vec![front, curve_middle, back]);
        curve.parameter_transform(2.0, 10.0);
        let vertices = Vertex::from_points([front, back]);
        let edge = Edge::new(
            &vertices[0],
            &vertices[1],
            modeling::Curve::BsplineCurve(curve),
        );
        let surface = modeling::Surface::BsplineSurface(BsplineSurface::new(
            (KnotVector::bezier_knot(2), KnotVector::bezier_knot(1)),
            [front, carrier_middle, back]
                .map(|point| vec![point, point + Vector3::new(0.0, 0.0, rise)])
                .into(),
        ));
        (edge, surface)
    }

    #[test]
    fn whole_curved_chart_boundary_binds_original_row_and_independent_native_gauge() {
        let middle = Point3::new(0.0, 1.0, 0.0);
        let (edge, surface) = curved_boundary(middle, middle, 1.0);
        let plane = OriginalPlane::new(&Plane::xy()).expect("literal original plane");
        assert!(proves_boundary(&edge, &surface, &plane, None).unwrap());
        assert_eq!(edge.with_curve(BoundedCurve::range_tuple), (10.0, 12.0));
        let (off_plane, _) = curved_boundary(middle + Vector3::new(0.0, 0.0, 1.0e-8), middle, 1.0);
        assert!(!proves_boundary(&off_plane, &surface, &plane, None).unwrap());
        let (_, unrelated) = curved_boundary(middle, Point3::new(0.0, 2.0, 0.0), 1.0);
        assert!(!proves_boundary(&edge, &unrelated, &plane, None).unwrap());
        let (_, tangent) = curved_boundary(middle, middle, 0.0);
        assert!(!proves_boundary(&edge, &tangent, &plane, None).unwrap());
    }

    #[test]
    fn an_original_curved_boundary_with_a_knot_corner_is_not_a_regular_overlap() {
        let controls = [
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, -1.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        ];
        let knots = KnotVector::from(vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        let vertices = Vertex::from_points([controls[0], controls[4]]);
        let edge = Edge::new(
            &vertices[0],
            &vertices[1],
            modeling::Curve::BsplineCurve(BsplineCurve::new(knots.clone(), controls.into())),
        );
        let surface = modeling::Surface::BsplineSurface(BsplineSurface::new(
            (knots, KnotVector::bezier_knot(1)),
            controls
                .map(|point| vec![point, point + Vector3::unit_z()])
                .into(),
        ));
        let plane = OriginalPlane::new(&Plane::xy()).expect("literal original plane");
        assert!(!proves_boundary(&edge, &surface, &plane, None).unwrap());
    }

    #[test]
    fn copied_bezier_controls_do_not_replace_an_uninserted_original_tensor_basis() {
        let controls = [
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        ];
        let vertices = Vertex::from_points([controls[0], controls[2]]);
        let edge = Edge::new(
            &vertices[0],
            &vertices[1],
            modeling::Curve::BsplineCurve(BsplineCurve::new(
                KnotVector::bezier_knot(2),
                controls[..3].to_vec(),
            )),
        );
        let source = modeling::Surface::BsplineSurface(BsplineSurface::new(
            (
                KnotVector::from(vec![0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0]),
                KnotVector::bezier_knot(1),
            ),
            controls
                .map(|point| vec![point, point + Vector3::unit_z()])
                .into(),
        ));
        assert_eq!(source.evaluate(1.0, 0.0), Point3::new(0.5, 0.5, 0.0));
        let plane = OriginalPlane::new(&Plane::xy()).unwrap();
        assert!(!proves_boundary(&edge, &source, &plane, None).unwrap());
    }

    #[test]
    fn exact_c0_tensor_span_restrictions_keep_both_original_endpoint_bases() {
        let controls = [
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, -1.0, 0.0),
            Point3::new(3.0, 0.0, 0.0),
        ];
        let source = modeling::Surface::BsplineSurface(BsplineSurface::new(
            (
                KnotVector::from(vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
                KnotVector::bezier_knot(1),
            ),
            controls
                .map(|point| vec![point, point + Vector3::unit_z()])
                .into(),
        ));
        let plane = OriginalPlane::new(&Plane::xy()).unwrap();
        for offset in [0, 2] {
            let vertices = Vertex::from_points([controls[offset], controls[offset + 2]]);
            let mut curve = BsplineCurve::new(
                KnotVector::bezier_knot(2),
                controls[offset..offset + 3].to_vec(),
            );
            curve.parameter_transform(2.0, 10.0);
            let edge = Edge::new(
                &vertices[0],
                &vertices[1],
                modeling::Curve::BsplineCurve(curve),
            );
            assert!(proves_boundary(&edge, &source, &plane, None).unwrap());
        }
    }
}
