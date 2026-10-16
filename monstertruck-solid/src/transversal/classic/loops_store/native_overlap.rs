//! Reuse an owned original edge only for a whole, exactly proved plane overlap.
//! Literal source coefficients and native domains remain authoritative; neither
//! converted nets nor geometric proximity establish this construction identity.

use super::periodic_loops::AttachmentFailure;
use super::*;

fn exact_add(a: f64, b: f64) -> Option<f64> {
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    let sum = a + b;
    let virtual_b = sum - a;
    let recovered_a = sum - virtual_b;
    let error_a = a - recovered_a;
    let error_b = b - virtual_b;
    let error = error_a + error_b;
    [sum, virtual_b, recovered_a, error_a, error_b, error]
        .into_iter()
        .all(f64::is_finite)
        .then_some(())?;
    (error == 0.0).then_some(sum)
}

fn dyadic(value: f64) -> (u128, i32) {
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    if exponent == 0 {
        (u128::from(fraction), -1074)
    } else {
        (u128::from(fraction | (1_u64 << 52)), exponent - 1075)
    }
}

fn exact_mul(a: f64, b: f64) -> Option<f64> {
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    let product = a * b;
    if !product.is_finite() {
        return None;
    }
    if a == 0.0 || b == 0.0 {
        return Some(product);
    }
    if product == 0.0 {
        return None;
    }
    let (first, first_exponent) = dyadic(a);
    let (second, second_exponent) = dyadic(b);
    let (rounded, rounded_exponent) = dyadic(product);
    let exact = first * second;
    let exact_shift = exact.trailing_zeros();
    let rounded_shift = rounded.trailing_zeros();
    ((exact >> exact_shift) == (rounded >> rounded_shift)
        && first_exponent + second_exponent + exact_shift as i32
            == rounded_exponent + rounded_shift as i32)
        .then_some(product)
}

fn difference(first: Vector3, second: Vector3) -> Option<Vector3> {
    Some(Vector3::new(
        exact_add(first.x, -second.x)?,
        exact_add(first.y, -second.y)?,
        exact_add(first.z, -second.z)?,
    ))
}

fn cross(first: Vector3, second: Vector3) -> Option<Vector3> {
    Some(Vector3::new(
        exact_add(
            exact_mul(first.y, second.z)?,
            -exact_mul(first.z, second.y)?,
        )?,
        exact_add(
            exact_mul(first.z, second.x)?,
            -exact_mul(first.x, second.z)?,
        )?,
        exact_add(
            exact_mul(first.x, second.y)?,
            -exact_mul(first.y, second.x)?,
        )?,
    ))
}

fn projected_difference(direction: Vector3, first: Vector3, second: Vector3) -> Option<f64> {
    let mut value = 0.0;
    for axis in 0..3 {
        if direction[axis] != 0.0 {
            let delta = exact_add(first[axis], -second[axis])?;
            value = exact_add(value, exact_mul(direction[axis], delta)?)?;
        }
    }
    Some(value)
}

struct AffineSupport {
    u: Option<Vector3>,
    v: Option<Vector3>,
    denominator: f64,
}

struct OriginalPlane {
    origin: Vector3,
    normal: Vector3,
    support: Option<AffineSupport>,
}

fn exact_dot(first: Vector3, second: Vector3) -> Option<f64> {
    let mut value = 0.0;
    for axis in 0..3 {
        if first[axis] != 0.0 {
            value = exact_add(value, exact_mul(first[axis], second[axis])?)?;
        }
    }
    Some(value)
}

fn exact_point(control: Vector4) -> Option<Point3> {
    if ![control.x, control.y, control.z, control.w]
        .into_iter()
        .all(f64::is_finite)
        || control.w == 0.0
    {
        return None;
    }
    let quotient = |value: f64| {
        let result = value / control.w;
        (result.is_finite() && exact_mul(result, control.w) == Some(value)).then_some(result)
    };
    Some(Point3::new(
        quotient(control.x)?,
        quotient(control.y)?,
        quotient(control.z)?,
    ))
}

impl OriginalPlane {
    fn from_axes(origin: Point3, u: Vector3, v: Vector3, bounded: [bool; 2]) -> Option<Self> {
        if ![origin.x, origin.y, origin.z]
            .into_iter()
            .all(f64::is_finite)
        {
            return None;
        }
        let normal = cross(u, v)?;
        if normal == Vector3::new(0.0, 0.0, 0.0) {
            return None;
        }
        let support = if bounded.into_iter().any(|bound| bound) {
            let denominator = exact_dot(normal, normal)?;
            if denominator <= 0.0 {
                return None;
            }
            Some(AffineSupport {
                u: if bounded[0] {
                    Some(cross(v, normal)?)
                } else {
                    None
                },
                v: if bounded[1] {
                    Some(cross(normal, u)?)
                } else {
                    None
                },
                denominator,
            })
        } else {
            None
        };
        Some(Self {
            origin: origin.to_vec(),
            normal,
            support,
        })
    }

    fn new(plane: &Plane) -> Option<Self> {
        let [origin, first, second] = plane.defining_points();
        let [u, v] = match plane.stored_axes() {
            Some(axes) => axes,
            None => [
                difference(first.to_vec(), origin.to_vec())?,
                difference(second.to_vec(), origin.to_vec())?,
            ],
        };
        Self::from_axes(origin, u, v, [false; 2])
    }

    fn tensor(controls: [[Vector4; 2]; 2]) -> Option<Self> {
        let weight = controls[0][0].w;
        if controls.iter().flatten().any(|control| control.w != weight) {
            return None;
        }
        let origin = exact_point(controls[0][0])?;
        let along_u = exact_point(controls[1][0])?;
        let along_v = exact_point(controls[0][1])?;
        let opposite = exact_point(controls[1][1])?;
        let u = difference(along_u.to_vec(), origin.to_vec())?;
        let v = difference(along_v.to_vec(), origin.to_vec())?;
        if difference(opposite.to_vec(), along_u.to_vec())? != v
            || difference(opposite.to_vec(), along_v.to_vec())? != u
        {
            return None;
        }
        // These dual forms only prove support membership. The original tensor
        // knots and native parameter frame are neither normalized nor replaced.
        Self::from_axes(origin, u, v, [true; 2])
    }

    fn from_surface<S>(surface: &S) -> Option<Self>
    where S: ParametricSurface3D + TryIntoAnalyticSurfaceKind + TryIntoHomogeneousBsplineSurface
    {
        if let Some(plane) = surface.original_plane() {
            if surface.parameter_range() != plane.parameter_range() {
                return None;
            }
            return Self::new(plane);
        }
        match surface.original_surface_coefficients()? {
            OriginalSurfaceCoefficients::Polynomial(source) => {
                let controls = source.control_points();
                let knots = source.knot_vectors();
                if controls.len() != 2
                    || controls.iter().any(|row| row.len() != 2)
                    || !clamped_linear(&knots.0, 2)
                    || !clamped_linear(&knots.1, 2)
                    || surface.parameter_range() != source.parameter_range()
                {
                    return None;
                }
                Self::tensor([
                    [
                        controls[0][0].to_homogeneous(),
                        controls[0][1].to_homogeneous(),
                    ],
                    [
                        controls[1][0].to_homogeneous(),
                        controls[1][1].to_homogeneous(),
                    ],
                ])
            }
            OriginalSurfaceCoefficients::Rational(source) => {
                let tensor = source.non_rationalized();
                let controls = tensor.control_points();
                let knots = tensor.knot_vectors();
                if controls.len() != 2
                    || controls.iter().any(|row| row.len() != 2)
                    || !clamped_linear(&knots.0, 2)
                    || !clamped_linear(&knots.1, 2)
                    || surface.parameter_range() != source.parameter_range()
                {
                    return None;
                }
                Self::tensor([
                    [controls[0][0], controls[0][1]],
                    [controls[1][0], controls[1][1]],
                ])
            }
            OriginalSurfaceCoefficients::Extrusion { profile, vector } => {
                let unbounded_profile = matches!(profile, OriginalCurveCoefficients::Line(_));
                let (controls, domain) = segment_coefficients(profile)?;
                let unit = (
                    std::ops::Bound::Included(0.0),
                    std::ops::Bound::Included(1.0),
                );
                if surface.parameter_range() != (domain, unit) || controls[0].w != controls[1].w {
                    return None;
                }
                let origin = exact_point(controls[0])?;
                let end = exact_point(controls[1])?;
                Self::from_axes(
                    origin,
                    difference(end.to_vec(), origin.to_vec())?,
                    vector,
                    // Extrusion is affine for every finite v. Literal Line also
                    // continues in u; its native unit interval is a display gauge.
                    [!unbounded_profile, false],
                )
            }
        }
    }

    fn contains(&self, point: Point3) -> Option<bool> {
        if projected_difference(self.normal, point.to_vec(), self.origin)? != 0.0 {
            return Some(false);
        }
        if let Some(support) = &self.support {
            for form in [support.u, support.v].into_iter().flatten() {
                let numerator = projected_difference(form, point.to_vec(), self.origin)?;
                if numerator < 0.0 || numerator > support.denominator {
                    return Some(false);
                }
            }
        }
        Some(true)
    }
}

fn clamped_linear(knots: &KnotVector, controls: usize) -> bool {
    controls == 2
        && knots.len() == 4
        && knots.iter().copied().all(f64::is_finite)
        && knots[0] == knots[1]
        && knots[2] == knots[3]
        && knots[1] < knots[2]
        && exact_add(knots[2], -knots[1]).is_some_and(|span| span > 0.0)
}

fn segment_coefficients(
    coefficients: OriginalCurveCoefficients<'_>,
) -> Option<([Vector4; 2], ParameterRange)> {
    let (controls, domain, range) = match coefficients {
        OriginalCurveCoefficients::Line(line) => (
            [line.0.to_homogeneous(), line.1.to_homogeneous()],
            line.parameter_range(),
            line.range_tuple(),
        ),
        OriginalCurveCoefficients::Polynomial(spline) => {
            let controls = spline.control_points();
            if !clamped_linear(spline.knot_vector(), controls.len()) {
                return None;
            }
            (
                [controls[0].to_homogeneous(), controls[1].to_homogeneous()],
                spline.parameter_range(),
                spline.range_tuple(),
            )
        }
        OriginalCurveCoefficients::Rational(spline) => {
            let source = spline.non_rationalized();
            let controls = source.control_points();
            if !clamped_linear(source.knot_vector(), controls.len()) {
                return None;
            }
            (
                [controls[0], controls[1]],
                spline.parameter_range(),
                spline.range_tuple(),
            )
        }
    };
    let (start, end) = range;
    (start.is_finite()
        && end.is_finite()
        && start < end
        && exact_add(end, -start).is_some_and(|span| span > 0.0)
        && controls.iter().all(|control| {
            [control.x, control.y, control.z, control.w]
                .into_iter()
                .all(f64::is_finite)
                && control.w != 0.0
                && control.w.is_sign_positive() == controls[0].w.is_sign_positive()
        }))
    .then_some((controls, domain))
}

fn original_segment<C>(curve: &C) -> Option<[Vector4; 2]>
where C: BoundedCurve<Point = Point3> + TryIntoHomogeneousBsplineCurve {
    let (controls, domain) = segment_coefficients(curve.original_curve_coefficients()?)?;
    (curve.parameter_range() == domain).then_some(controls)
}

fn coefficient_endpoint(control: Vector4, point: Point3) -> Option<bool> {
    Some(
        exact_mul(point.x, control.w)? == control.x
            && exact_mul(point.y, control.w)? == control.y
            && exact_mul(point.z, control.w)? == control.z,
    )
}

fn proves_overlap<C, S>(
    original: &C,
    absolute_endpoints: [Point3; 2],
    directed_endpoints: [Point3; 2],
    intersection: &IntersectionCurve<PolylineCurve, S, S>,
    planes: &[OriginalPlane; 2],
    work: Option<&BooleanWork<'_>>,
) -> std::result::Result<bool, AttachmentFailure>
where
    C: BoundedCurve<Point = Point3> + TryIntoHomogeneousBsplineCurve,
{
    if let Some(work) = work {
        work.check()?;
    }
    let Some(controls) = original_segment(original) else {
        return Ok(false);
    };
    let first = &planes[0];
    let second = &planes[1];
    let proof = || -> Option<(Vector3, bool)> {
        if !coefficient_endpoint(controls[0], absolute_endpoints[0])?
            || !coefficient_endpoint(controls[1], absolute_endpoints[1])?
        {
            return None;
        }
        let direction = cross(first.normal, second.normal)?;
        if direction == Vector3::new(0.0, 0.0, 0.0) {
            return None;
        }
        for point in absolute_endpoints {
            if !first.contains(point)? || !second.contains(point)? {
                return None;
            }
        }
        let span = projected_difference(
            direction,
            directed_endpoints[1].to_vec(),
            directed_endpoints[0].to_vec(),
        )?;
        (span != 0.0).then_some((direction, span > 0.0))
    };
    let Some((direction, positive)) = proof() else {
        return Ok(false);
    };
    let leader = intersection.leader();
    if leader.len() < 2
        || leader.first() != Some(&directed_endpoints[0])
        || leader.last() != Some(&directed_endpoints[1])
    {
        return Ok(false);
    }
    for point in leader.iter().copied() {
        if let Some(work) = work {
            work.check()?;
        }
        if ![point.x, point.y, point.z].into_iter().all(f64::is_finite)
            || first.contains(point) != Some(true)
            || second.contains(point) != Some(true)
        {
            return Ok(false);
        }
    }
    for pair in leader.windows(2) {
        if let Some(work) = work {
            work.check()?;
        }
        let step = projected_difference(direction, pair[1].to_vec(), pair[0].to_vec());
        if !step.is_some_and(|step| step != 0.0 && (step > 0.0) == positive) {
            return Ok(false);
        }
    }
    // Same-sign nonzero two-control weights prove one traversal of the complete source
    // segment. Whole original affine operands and their convex support prove a
    // unique line branch, not a plane fitted to samples or converted controls.
    Ok(true)
}

type OriginalOverlap<C> = (Edge<Point3, C>, Edge<Point3, PolylineCurve>, [bool; 2]);
type OriginalLoopPair<'a, C> = (&'a Loops<Point3, C>, &'a Loops<Point3, PolylineCurve>);

pub(super) fn original_overlap<C, S>(
    intersection: &IntersectionCurve<PolylineCurve, S, S>,
    stores: [OriginalLoopPair<'_, C>; 2],
    vertices: [&Vertex<Point3>; 2],
    polygon_vertices: [&Vertex<Point3>; 2],
    work: Option<&BooleanWork<'_>>,
) -> std::result::Result<Option<OriginalOverlap<C>>, AttachmentFailure>
where
    C: Clone + BoundedCurve<Point = Point3> + TryIntoHomogeneousBsplineCurve,
    S: ParametricSurface3D + TryIntoAnalyticSurfaceKind + TryIntoHomogeneousBsplineSurface,
{
    if vertices[0] == vertices[1] {
        return Ok(None);
    }
    let Some(first) = OriginalPlane::from_surface(intersection.surface0()) else {
        return Ok(None);
    };
    let Some(second) = OriginalPlane::from_surface(intersection.surface1()) else {
        return Ok(None);
    };
    let planes = [first, second];
    let mut selected: Option<OriginalOverlap<C>> = None;
    for (side, (native, polygon)) in stores.into_iter().enumerate() {
        for (wire_index, wire) in native.iter().enumerate() {
            for (edge_index, edge) in wire.iter().enumerate() {
                if let Some(work) = work {
                    work.check()?;
                }
                let forward = edge.front() == vertices[0] && edge.back() == vertices[1];
                let reverse = edge.back() == vertices[0] && edge.front() == vertices[1];
                if !forward && !reverse {
                    continue;
                }
                let polygon = &polygon[wire_index][edge_index];
                let orientation = if forward {
                    edge.orientation()
                } else {
                    !edge.orientation()
                };
                let polygon_orientation = if forward {
                    polygon.orientation()
                } else {
                    !polygon.orientation()
                };
                let polygon_endpoints = if forward {
                    [polygon.front(), polygon.back()]
                } else {
                    [polygon.back(), polygon.front()]
                };
                if let Some((old, old_polygon, present)) = selected.as_mut()
                    && old.id() == edge.id()
                {
                    if old.orientation() != orientation
                        || old_polygon.id() != polygon.id()
                        || old_polygon.orientation() != polygon_orientation
                        || polygon_endpoints != polygon_vertices
                    {
                        return Err("intersection_original_overlap_ambiguous".into());
                    }
                    present[side] = true;
                    continue;
                }
                if !edge.with_curve(|curve| {
                    proves_overlap(
                        curve,
                        [edge.absolute_front().point(), edge.absolute_back().point()],
                        vertices.map(Vertex::point),
                        intersection,
                        &planes,
                        work,
                    )
                })? {
                    continue;
                }
                if polygon_endpoints != polygon_vertices {
                    return Err("intersection_original_overlap_polygon_incidence".into());
                }
                if selected.is_some() {
                    return Err("intersection_original_overlap_ambiguous".into());
                }
                let native = if forward {
                    edge.clone()
                } else {
                    edge.inverse()
                };
                let polygon = if forward {
                    polygon.clone()
                } else {
                    polygon.inverse()
                };
                let mut present = [false; 2];
                present[side] = true;
                selected = Some((native, polygon, present));
            }
        }
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use monstertruck_modeling as modeling;

    fn intersection(
        points: Vec<Point3>,
    ) -> IntersectionCurve<PolylineCurve, modeling::Surface, modeling::Surface> {
        IntersectionCurve::new(
            modeling::Surface::Plane(Plane::xy()),
            modeling::Surface::Plane(Plane::new(
                Point3::origin(),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 0.0, 1.0),
            )),
            PolylineCurve(points),
        )
    }

    fn prove_curve(
        original: &modeling::Curve,
        absolute: [Point3; 2],
        directed: [Point3; 2],
        branch: &IntersectionCurve<PolylineCurve, modeling::Surface, modeling::Surface>,
        work: Option<&BooleanWork<'_>>,
    ) -> std::result::Result<bool, AttachmentFailure> {
        let Some(first) = OriginalPlane::from_surface(branch.surface0()) else {
            return Ok(false);
        };
        let Some(second) = OriginalPlane::from_surface(branch.surface1()) else {
            return Ok(false);
        };
        proves_overlap(original, absolute, directed, branch, &[first, second], work)
    }

    #[test]
    fn original_affine_tensors_and_extrusions_keep_their_full_native_support() {
        let knots = (
            KnotVector::from(vec![-3.0, -3.0, 5.0, 5.0]),
            KnotVector::from(vec![2.0, 2.0, 9.0, 9.0]),
        );
        let controls = vec![
            vec![Point3::origin(), Point3::new(0.0, 1.0, 0.0)],
            vec![Point3::new(2.0, 0.0, 0.0), Point3::new(2.0, 1.0, 0.0)],
        ];
        let polynomial = BsplineSurface::new(knots.clone(), controls.clone());
        let rational = NurbsSurface::new(BsplineSurface::new(
            knots,
            controls
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|point| point.to_homogeneous() * 2.0)
                        .collect()
                })
                .collect(),
        ));
        let negative_rational = NurbsSurface::new(BsplineSurface::new(
            polynomial.knot_vectors().clone(),
            controls
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|point| point.to_homogeneous() * -2.0)
                        .collect()
                })
                .collect(),
        ));
        let profile = BsplineCurve::new(
            KnotVector::from(vec![-3.0, -3.0, 5.0, 5.0]),
            vec![Point3::origin(), Point3::new(2.0, 0.0, 0.0)],
        );
        let rational_profile = NurbsCurve::new(BsplineCurve::new(
            profile.knot_vector().clone(),
            profile
                .control_points()
                .iter()
                .map(|point| point.to_homogeneous() * 2.0)
                .collect(),
        ));
        let negative_profile = NurbsCurve::new(BsplineCurve::new(
            profile.knot_vector().clone(),
            profile
                .control_points()
                .iter()
                .map(|point| point.to_homogeneous() * -2.0)
                .collect(),
        ));
        let profiles = [
            modeling::Curve::Line(Line(Point3::origin(), Point3::new(2.0, 0.0, 0.0))),
            modeling::Curve::BsplineCurve(profile),
            modeling::Curve::NurbsCurve(rational_profile),
            modeling::Curve::NurbsCurve(negative_profile),
        ];
        let surfaces = [
            modeling::Surface::BsplineSurface(polynomial),
            modeling::Surface::NurbsSurface(rational),
            modeling::Surface::NurbsSurface(negative_rational),
        ]
        .into_iter()
        .chain(profiles.into_iter().map(|profile| {
            modeling::Surface::ExtrusionSurface(ExtrusionSurface::by_extrusion(
                profile,
                Vector3::unit_y(),
            ))
        }));
        let endpoints = [Point3::origin(), Point3::new(1.0, 0.0, 0.0)];
        let line = modeling::Curve::Line(Line(endpoints[0], endpoints[1]));
        for surface in surfaces {
            let bounded = match surface.original_surface_coefficients().unwrap() {
                OriginalSurfaceCoefficients::Extrusion {
                    profile: OriginalCurveCoefficients::Line(_),
                    ..
                } => [false; 2],
                OriginalSurfaceCoefficients::Extrusion { .. } => [true, false],
                _ => [true; 2],
            };
            let equation = OriginalPlane::from_surface(&surface).expect("original affine operands");
            assert_eq!(equation.contains(Point3::new(2.0, 1.0, 0.0)), Some(true));
            assert_eq!(
                equation.contains(Point3::new(2.5, 0.0, 0.0)),
                Some(!bounded[0])
            );
            assert_eq!(
                equation.contains(Point3::new(0.0, 1.5, 0.0)),
                Some(!bounded[1])
            );
            let branch = IntersectionCurve::new(
                surface,
                intersection(endpoints.to_vec()).surface1().clone(),
                PolylineCurve(endpoints.to_vec()),
            );
            assert!(matches!(
                prove_curve(&line, endpoints, endpoints, &branch, None),
                Ok(true)
            ));
        }
    }

    #[test]
    fn original_affine_surface_proof_refuses_nonaffine_and_rounded_operands() {
        let knots = (
            KnotVector::from(vec![0.0, 0.0, 1.0, 1.0]),
            KnotVector::from(vec![0.0, 0.0, 1.0, 1.0]),
        );
        let controls = vec![
            vec![Point3::origin(), Point3::new(0.0, 1.0, 0.0)],
            vec![
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, f64::from_bits(1)),
            ],
        ];
        let nonaffine =
            modeling::Surface::BsplineSurface(BsplineSurface::new(knots.clone(), controls));
        assert!(OriginalPlane::from_surface(&nonaffine).is_none());
        let rounded = modeling::Surface::NurbsSurface(NurbsSurface::new(BsplineSurface::new(
            knots,
            vec![
                vec![
                    Vector4::new(1.0, 0.0, 0.0, 3.0),
                    Vector4::new(1.0, 3.0, 0.0, 3.0),
                ],
                vec![
                    Vector4::new(4.0, 0.0, 0.0, 3.0),
                    Vector4::new(4.0, 3.0, 0.0, 3.0),
                ],
            ],
        )));
        assert!(OriginalPlane::from_surface(&rounded).is_none());
        let profile = modeling::Curve::Line(Line(Point3::origin(), Point3::new(1.0, 0.0, 0.0)));
        let tangent = modeling::Surface::ExtrusionSurface(ExtrusionSurface::by_extrusion(
            profile,
            Vector3::unit_x(),
        ));
        assert!(OriginalPlane::from_surface(&tangent).is_none());
    }

    fn owned_face(curve: modeling::Curve) -> modeling::Face {
        let vertices = Vertex::from_points([
            Point3::origin(),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ]);
        let wire: modeling::Wire =
            std::iter::once(Edge::new(&vertices[0], &vertices[1], curve))
                .chain((1..4).map(|index| {
                    modeling::builder::line(&vertices[index], &vertices[(index + 1) % 4])
                }))
                .collect();
        modeling::Face::new(vec![wire], modeling::Surface::Plane(Plane::xy()))
    }

    #[test]
    fn exact_arithmetic_never_turns_underflow_overflow_or_rounded_products_into_identities() {
        assert_eq!(exact_mul(f64::from_bits(1), 0.5), None);
        assert_eq!(exact_mul(f64::MIN_POSITIVE, f64::MIN_POSITIVE), None);
        assert_eq!(exact_mul(f64::MAX, 2.0), None);
        assert_eq!(exact_mul(1.0 + f64::EPSILON, 1.0 + f64::EPSILON), None);
        assert_eq!(exact_add(f64::MAX, f64::MAX), None);
        assert_eq!(exact_add(1.0, f64::from_bits(1)), None);
        assert_eq!(exact_mul(f64::from_bits(1), 1.0), Some(f64::from_bits(1)));
        assert_eq!(exact_add(0.5, -0.25), Some(0.25));
        let origin = Point3::new(1.0e16, 0.0, 0.0);
        let plane = Plane::from_origin_vectors(origin, Vector3::unit_x(), Vector3::unit_z());
        assert_eq!(plane.defining_points()[0], plane.defining_points()[1]);
        let Some(original) = OriginalPlane::new(&plane) else {
            panic!("the original vector-defined axes remain independent");
        };
        assert_eq!(original.contains(origin + Vector3::unit_z()), Some(true));
    }

    #[test]
    fn literal_polynomial_and_rational_overlaps_reuse_original_edges_and_nonunit_gauges() {
        let first = Point3::origin();
        let second = Point3::new(1.0, 0.0, 0.0);
        let curves = [
            modeling::Curve::Line(Line(first, second)),
            modeling::Curve::BsplineCurve(BsplineCurve::new(
                KnotVector::from(vec![-2.0, -2.0, 5.0, 5.0]),
                vec![first, second],
            )),
            modeling::Curve::NurbsCurve(NurbsCurve::new(BsplineCurve::new(
                KnotVector::from(vec![3.0, 3.0, 7.0, 7.0]),
                vec![
                    Vector4::new(0.0, 0.0, 0.0, 2.0),
                    Vector4::new(4.0, 0.0, 0.0, 4.0),
                ],
            ))),
            modeling::Curve::NurbsCurve(NurbsCurve::new(BsplineCurve::new(
                KnotVector::from(vec![3.0, 3.0, 7.0, 7.0]),
                vec![
                    Vector4::new(0.0, 0.0, 0.0, -2.0),
                    Vector4::new(-4.0, 0.0, 0.0, -4.0),
                ],
            ))),
        ];
        for curve in curves {
            let face = owned_face(curve);
            let polygon = face.mapped(
                |point| *point,
                |curve| PolylineCurve::from_curve(curve, curve.range_tuple(), 1.0e-5),
                |_| (),
            );
            let native: Loops<_, _> = Loops::from(&face);
            let polygon: Loops<_, _> = Loops::from(&polygon);
            let empty_native: Loops<Point3, modeling::Curve> = std::iter::empty().collect();
            let empty_polygon: Loops<Point3, PolylineCurve> = std::iter::empty().collect();
            let edge = &native[0][0];
            let polygon_edge = &polygon[0][0];
            let branch = intersection(vec![first, Point3::new(0.5, 0.0, 0.0), second]);
            let selected = original_overlap(
                &branch,
                [(&native, &polygon), (&empty_native, &empty_polygon)],
                [edge.front(), edge.back()],
                [polygon_edge.front(), polygon_edge.back()],
                None,
            );
            let Ok(Some((selected, selected_polygon, present))) = selected else {
                panic!("literal original whole overlap must retain its native source");
            };
            assert_eq!(present, [true, false]);
            assert_eq!(selected.id(), edge.id());
            assert_eq!(selected_polygon.id(), polygon_edge.id());
            assert_eq!(
                selected.with_curve(ParametricCurve::parameter_range),
                edge.with_curve(ParametricCurve::parameter_range),
            );
            assert!(selected.is_geometric_consistent());
            assert_eq!(face.absolute_boundaries()[0][0].id(), edge.id());
        }
    }

    #[test]
    fn rounded_wrapper_extraction_and_near_plane_literals_are_not_original_overlap_proofs() {
        let tiny = 2.0_f64.powi(-54);
        let wrapper = modeling::Curve::TrimmedCurve(Processor::with_transform(
            TrimmedCurve::new(
                Box::new(modeling::Curve::Line(Line(
                    Point3::new(1.0, tiny, 0.0),
                    Point3::new(1.0, tiny, 1.0),
                ))),
                (0.0, 1.0),
            ),
            Matrix4::from_cols(
                Vector4::new(0.0, 0.0, 1.0, 0.0),
                Vector4::new(0.0, 1.0, 1.0, 0.0),
                Vector4::new(1.0, 0.0, 0.0, 0.0),
                Vector4::new(0.0, 0.0, -1.0, 1.0),
            ),
        ));
        let rounded = wrapper.try_into_homogeneous_bspline_curve().unwrap();
        assert!(rounded.control_points().iter().all(|point| point.z == 0.0));
        assert!(wrapper.original_curve_coefficients().is_none());
        let endpoints = [Point3::origin(), Point3::new(1.0, 0.0, 0.0)];
        let branch = intersection(endpoints.to_vec());
        assert!(matches!(
            prove_curve(&wrapper, endpoints, endpoints, &branch, None),
            Ok(false)
        ));
        let off_plane = endpoints.map(|point| point + Vector3::new(0.0, 0.0, f64::from_bits(1)));
        let near = modeling::Curve::Line(Line(off_plane[0], off_plane[1]));
        assert!(off_plane[0].near(&endpoints[0]));
        assert!(matches!(
            prove_curve(&near, off_plane, off_plane, &branch, None),
            Ok(false)
        ));
        let reversed = intersection(vec![endpoints[0], endpoints[1], endpoints[0], endpoints[1]]);
        let line = modeling::Curve::Line(Line(endpoints[0], endpoints[1]));
        assert!(matches!(
            prove_curve(&line, endpoints, endpoints, &reversed, None),
            Ok(false)
        ));
        let parallel = IntersectionCurve::new(
            modeling::Surface::Plane(Plane::xy()),
            modeling::Surface::Plane(Plane::xy()),
            PolylineCurve(endpoints.to_vec()),
        );
        assert!(matches!(
            prove_curve(&line, endpoints, endpoints, &parallel, None),
            Ok(false)
        ));
        let mixed_sign = modeling::Curve::NurbsCurve(NurbsCurve::new(BsplineCurve::new(
            KnotVector::bezier_knot(1),
            vec![
                Vector4::new(0.0, 0.0, 0.0, 1.0),
                Vector4::new(-1.0, 0.0, 0.0, -1.0),
            ],
        )));
        assert!(matches!(
            prove_curve(&mixed_sign, endpoints, endpoints, &branch, None),
            Ok(false)
        ));
    }

    #[test]
    fn multiple_owned_original_overlap_edges_are_not_arbitrarily_welded() {
        let a = Vertex::new(Point3::origin());
        let b = Vertex::new(Point3::new(1.0, 0.0, 0.0));
        let face = |ends: [Point3; 2], plane: Plane| {
            let c = Vertex::new(ends[0]);
            let d = Vertex::new(ends[1]);
            let wire: modeling::Wire = [
                modeling::builder::line(&a, &b),
                modeling::builder::line(&b, &c),
                modeling::builder::line(&c, &d),
                modeling::builder::line(&d, &a),
            ]
            .into_iter()
            .collect();
            modeling::Face::new(vec![wire], modeling::Surface::Plane(plane))
        };
        let first = face(
            [Point3::new(1.0, 1.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
            Plane::xy(),
        );
        let second = face(
            [Point3::new(1.0, 0.0, 1.0), Point3::new(0.0, 0.0, 1.0)],
            Plane::new(
                Point3::origin(),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 0.0, 1.0),
            ),
        );
        let shell: modeling::Shell = vec![first, second].into();
        let polygon = shell.mapped(
            |point| *point,
            |curve| PolylineCurve::from_curve(curve, curve.range_tuple(), 1.0e-5),
            |_| (),
        );
        let native = [Loops::from(&shell[0]), Loops::from(&shell[1])];
        let polygon = [Loops::from(&polygon[0]), Loops::from(&polygon[1])];
        let polygon_edge = &polygon[0][0][0];
        let result = original_overlap(
            &intersection(vec![a.point(), b.point()]),
            [(&native[0], &polygon[0]), (&native[1], &polygon[1])],
            [&a, &b],
            [polygon_edge.front(), polygon_edge.back()],
            None,
        );
        assert!(matches!(
            result,
            Err(AttachmentFailure::Geometry(
                "intersection_original_overlap_ambiguous"
            ))
        ));
    }
}
