//! Bounded outward native broad-phase boxes without similarity guesses.

use crate::geometry::validated_curve::{ValidatedCurveOptions, prepared_validated_curve};
use crate::geometry::validated_interval::{
    ValidatedGeometryBudget, ValidatedGeometryError, ValidatedInterval,
    validated_curve_position_bounds, validated_surface_jet,
};
use crate::*;
use std::result::Result;
use std::sync::atomic::AtomicBool;

type Interval = ValidatedInterval;

fn point(value: f64) -> Result<Interval, ValidatedGeometryError> { Interval::new(value, value) }

fn dot(a: [Interval; 3], b: [Interval; 3]) -> Interval { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }

fn finite(values: [Interval; 3]) -> Result<[Interval; 3], ValidatedGeometryError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(values)
    } else {
        Err(ValidatedGeometryError::ConditioningFailure)
    }
}

fn coefficient_point(value: Point3) -> Result<[Interval; 3], ValidatedGeometryError> {
    Ok([point(value.x)?, point(value.y)?, point(value.z)?])
}

fn boundary_box(
    face: &Face,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<[Interval; 3], ValidatedGeometryError> {
    let mut result: Option<[Interval; 3]> = None;
    for edge in face.edge_iter() {
        budget.charge(1, cancel)?;
        let bounds = edge.with_curve(|curve| -> Result<[Interval; 3], ValidatedGeometryError> {
            curve_owned_bytes(curve, usize::MAX, budget, cancel)?;
            // Exact polynomial convex hulls are independent of knot smoothness.
            if let Curve::Line(line) = curve {
                let a = coefficient_point(line.0)?;
                let b = coefficient_point(line.1)?;
                return Ok(std::array::from_fn(|axis| a[axis].hull(b[axis])));
            }
            if let Curve::BsplineCurve(spline) = curve {
                budget.charge(spline.control_points().len(), cancel)?;
                return spline
                    .control_points()
                    .iter()
                    .try_fold(None, |bounds: Option<[Interval; 3]>, coefficient| {
                        let value = coefficient_point(*coefficient)?;
                        Ok::<_, ValidatedGeometryError>(Some(match bounds {
                            None => value,
                            Some(previous) => {
                                std::array::from_fn(|axis| previous[axis].hull(value[axis]))
                            }
                        }))
                    })?
                    .ok_or(ValidatedGeometryError::InvalidInput);
            }
            let range = curve.range_tuple();
            match validated_curve_position_bounds(curve, [range.0, range.1], budget, cancel) {
                Err(ValidatedGeometryError::Unsupported) => {
                    // Native implicit curves need their original prepared root
                    // tubes, never the guide or a sampled control polygon.
                    let prepared = prepared_validated_curve(
                        curve,
                        [range.0, range.1],
                        &ValidatedCurveOptions::default(),
                        budget,
                        cancel,
                    )?;
                    prepared.position_bounds([range.0, range.1], budget, cancel)
                }
                result => result,
            }
        })?;
        result = Some(match result {
            None => bounds,
            Some(previous) => std::array::from_fn(|i| previous[i].hull(bounds[i])),
        });
    }
    finite(result.ok_or(ValidatedGeometryError::InvalidInput)?)
}

fn inverse_affine(matrix: Matrix4) -> Result<[[Interval; 3]; 3], ValidatedGeometryError> {
    if matrix.x.w != 0.0 || matrix.y.w != 0.0 || matrix.z.w != 0.0 || matrix.w.w != 1.0 {
        return Err(ValidatedGeometryError::Unsupported);
    }
    let a = [
        [point(matrix.x.x)?, point(matrix.y.x)?, point(matrix.z.x)?],
        [point(matrix.x.y)?, point(matrix.y.y)?, point(matrix.z.y)?],
        [point(matrix.x.z)?, point(matrix.y.z)?, point(matrix.z.z)?],
    ];
    let cofactors: [[Interval; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let rows = [(i + 1) % 3, (i + 2) % 3];
            let cols = [(j + 1) % 3, (j + 2) % 3];
            a[rows[0]][cols[0]] * a[rows[1]][cols[1]] - a[rows[0]][cols[1]] * a[rows[1]][cols[0]]
        })
    });
    let determinant =
        a[0][0] * cofactors[0][0] + a[0][1] * cofactors[0][1] + a[0][2] * cofactors[0][2];
    let reciprocal = determinant.reciprocal()?;
    let inverse = std::array::from_fn(|i| std::array::from_fn(|j| cofactors[j][i] * reciprocal));
    if inverse.iter().flatten().all(|value| value.is_finite()) {
        Ok(inverse)
    } else {
        Err(ValidatedGeometryError::ConditioningFailure)
    }
}

/// Applies the inverse of a stored affine frame using outward cofactor arithmetic.
///
/// A determinant interval containing zero refuses; a rounded ordinary matrix
/// inverse is never treated as the original frame's exact inverse.
pub fn validated_inverse_affine_point(
    matrix: Matrix4,
    position: [Interval; 3],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<[Interval; 3], ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    finite(position)?;
    let inverse = inverse_affine(matrix)?;
    let translation = coefficient_point(Point3::from_vec(matrix.w.truncate()))?;
    let shifted = std::array::from_fn(|i| position[i] - translation[i]);
    finite(std::array::from_fn(|i| dot(inverse[i], shifted)))
}

/// Encloses an admitted compact native plane or straight-revolution face.
///
/// Polynomial hull coefficients are exact stored binary64 values. Homogeneous
/// quotients, affine inversion, axial projection and chart evaluation use
/// outward arithmetic at every operation. Both positive and all-negative
/// homogeneous weights obey the same convex-hull argument; mixed weights
/// refuse this fast path. No approximate similarity test authorizes a bound.
///
/// For a straight revolution, regular profile-coordinate extrema of a compact
/// face occur on its chart boundary. Possible profile-axis intersections are
/// included separately because a cone apex can be an interior singular stratum.
/// The actual stored-axis norm defect is retained in the Rodrigues projection
/// formula, rather than assuming its rounded normalization is mathematically
/// unit length. The resulting box is a conservative broad-phase enclosure,
/// never a tight support witness or a trim-coverage certificate.
///
/// The caller must already have admitted the face topology and boundary/carrier
/// correspondence. A disconnected or geometrically inconsistent input does not
/// acquire validity from its box. Cancellation and work charges are cumulative.
pub fn validated_face_bounding_box(
    face: &Face,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<BoundingBox<Point3>, ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    let boundary = boundary_box(face, budget, cancel)?;
    validated_face_bounding_box_with_boundary_enclosure(face, boundary, budget, cancel)
}

/// Applies the native compact-face ruling theorem to a caller-certified boundary.
///
/// # Required geometric preconditions
///
/// The caller must already have admitted the compact face topology and proved
/// that `boundary` encloses the WHOLE original actual oriented face boundary,
/// including every wire and edge, in this face's native spatial units. For
/// implicit edges this must be the whole-image enclosure of the same retained
/// original prepared branches, not fresh independently selected roots, sampled
/// points, polygon guides, mesh vertices or an unproved UV candidate box.
///
/// This function validates finite ordered intervals and applies the same plane,
/// extrusion or straight-revolution theorem as [`validated_face_bounding_box`].
/// It does not validate arbitrary supplied intervals' boundary incidence,
/// completeness, orientation or correspondence. It never prepares edge roots.
pub fn validated_face_bounding_box_with_boundary_enclosure(
    face: &Face,
    boundary: [Interval; 3],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<BoundingBox<Point3>, ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    let boundary = finite(boundary)?;
    let bounds = face.with_surface(|surface| -> Result<[Interval; 3], ValidatedGeometryError> {
        match surface {
            Surface::Plane(_) => Ok(boundary),
            // Each compact ruling has its coordinate extrema on the boundary.
            Surface::ExtrusionSurface(_) => Ok(boundary),
            Surface::RevolutionSurface(processor) => {
                let revolution = processor.entity();
                let Curve::Line(profile) = revolution.entity_curve() else {
                    return Err(ValidatedGeometryError::Unsupported);
                };
                let inverse = inverse_affine(*processor.transform())?;
                let translation =
                    coefficient_point(Point3::from_vec(processor.transform().w.truncate()))?;
                let shifted = std::array::from_fn(|i| boundary[i] - translation[i]);
                let local: [Interval; 3] = std::array::from_fn(|i| dot(inverse[i], shifted));
                let origin = coefficient_point(revolution.origin())?;
                let axis = coefficient_point(Point3::from_vec(revolution.axis()))?;
                let p0 = coefficient_point(profile.0)?;
                let p1 = coefficient_point(profile.1)?;
                let offset = std::array::from_fn(|i| p0[i] - origin[i]);
                let direction = std::array::from_fn(|i| p1[i] - p0[i]);
                let axis_squared = dot(axis, axis);
                let factor = point(1.0)? + Interval::new(0.0, 2.0)? * (axis_squared - point(1.0)?);
                let along = dot(axis, std::array::from_fn(|i| local[i] - origin[i]))
                    * factor.reciprocal()?;
                let slope = dot(axis, direction);
                let intercept = dot(axis, offset);
                let mut profile_range = (along - intercept) * slope.reciprocal()?;
                let inverse_axis_squared = axis_squared.reciprocal()?;
                let radial_offset: [Interval; 3] =
                    std::array::from_fn(|i| offset[i] - axis[i] * intercept * inverse_axis_squared);
                let radial_direction: [Interval; 3] =
                    std::array::from_fn(|i| direction[i] - axis[i] * slope * inverse_axis_squared);
                if let Some(index) = (0..3)
                    .filter(|i| !radial_direction[*i].contains_zero())
                    .max_by(|a, b| {
                        radial_direction[*a]
                            .magnitude()
                            .total_cmp(&radial_direction[*b].magnitude())
                    })
                {
                    let apex = -radial_offset[index] * radial_direction[index].reciprocal()?;
                    if (0..3)
                        .all(|i| (radial_offset[i] + radial_direction[i] * apex).contains_zero())
                    {
                        profile_range = profile_range.hull(apex);
                    }
                } else {
                    let native_axis = revolution.axis();
                    let exact_coordinate_parallel = (0..3).any(|coordinate| {
                        native_axis[coordinate].abs() == 1.0
                            && (0..3)
                                .filter(|i| *i != coordinate)
                                .all(|i| native_axis[i] == 0.0 && profile.0[i] == profile.1[i])
                    });
                    if !exact_coordinate_parallel {
                        return Err(ValidatedGeometryError::ConditioningFailure);
                    }
                }
                if !profile_range.is_finite() {
                    return Err(ValidatedGeometryError::ConditioningFailure);
                }
                let profile_interval = [profile_range.lower, profile_range.upper];
                let angles = [0.0, std::f64::consts::TAU.next_up()];
                let domain = if processor.orientation() {
                    [profile_interval, angles]
                } else {
                    [angles, profile_interval]
                };
                Ok(validated_surface_jet(surface, domain, budget, cancel)?.position)
            }
            _ => Err(ValidatedGeometryError::Unsupported),
        }
    })?;
    let bounds = finite(bounds)?;
    Ok(BoundingBox::from_iter([
        Point3::new(bounds[0].lower, bounds[1].lower, bounds[2].lower),
        Point3::new(bounds[0].upper, bounds[1].upper, bounds[2].upper),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_affine_uses_actual_nonsimilarity_coefficients() {
        let frame = Matrix4::from_cols(
            Vector4::new(2.0, 0.0, 0.0, 0.0),
            Vector4::new(0.5, 3.0, 0.0, 0.0),
            Vector4::new(0.0, 0.25, 4.0, 0.0),
            Vector4::new(5.0, -6.0, 7.0, 1.0),
        );
        let source = Point3::new(1.0, 2.0, -1.0);
        let world = frame.transform_point(source);
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(100);
        let enclosure = validated_inverse_affine_point(
            frame,
            coefficient_point(world).expect("finite world point"),
            &mut budget,
            &cancel,
        )
        .expect("invertible actual affine coefficients");
        (0..3).for_each(|i| {
            assert!(enclosure[i].lower <= source[i] && source[i] <= enclosure[i].upper);
        });
    }

    #[test]
    fn exact_planar_boundary_hull_retains_constant_height() {
        let vertices = [
            Vertex::new(Point3::new(0.0, 0.0, 8.0)),
            Vertex::new(Point3::new(2.0, 0.0, 8.0)),
            Vertex::new(Point3::new(2.0, 3.0, 8.0)),
            Vertex::new(Point3::new(0.0, 3.0, 8.0)),
        ];
        let wire = Wire::from(
            (0..4)
                .map(|i| builder::line(&vertices[i], &vertices[(i + 1) % 4]))
                .collect::<Vec<_>>(),
        );
        let face = Face::new(
            vec![wire],
            Surface::Plane(Plane::new(
                Point3::new(0.0, 0.0, 8.0),
                Point3::new(1.0, 0.0, 8.0),
                Point3::new(0.0, 1.0, 8.0),
            )),
        );
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(100);
        let bounds = validated_face_bounding_box(&face, &mut budget, &cancel).expect("planar hull");
        assert_eq!(bounds.min(), Point3::new(0.0, 0.0, 8.0));
        assert_eq!(bounds.max(), Point3::new(2.0, 3.0, 8.0));
    }

    #[test]
    fn retained_original_implicit_boundary_encloses_the_face_without_repreparing_roots() {
        let points = [
            Point3::new(0.0, 0.0, 8.0),
            Point3::new(2.0, 0.0, 8.0),
            Point3::new(2.0, 3.0, 8.0),
            Point3::new(0.0, 3.0, 8.0),
        ];
        let vertices = points.map(Vertex::new);
        let carrier = Plane::new(
            Point3::new(0.0, 0.0, 8.0),
            Point3::new(1.0, 0.0, 8.0),
            Point3::new(0.0, 1.0, 8.0),
        );
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(4_000_000);
        let mut prepared = Vec::new();
        let mut edges = Vec::new();
        for i in 0..4 {
            let next = (i + 1) % 4;
            let side = Plane::new(
                points[i],
                points[next],
                Point3::new(points[i].x, points[i].y, points[i].z + 1.0),
            );
            let curve = Curve::IntersectionCurve(SurfaceCurve::with_boundaries(
                Box::new(Surface::Plane(carrier)),
                Box::new(Surface::Plane(side)),
                Box::new(Curve::Line(Line(points[i], points[next]))),
                None,
                None,
            ));
            prepared.push(
                prepared_validated_curve_owned(
                    curve.clone(),
                    [0.0, 1.0],
                    &ValidatedCurveOptions::default(),
                    &mut budget,
                    &cancel,
                )
                .unwrap(),
            );
            edges.push(Edge::new(&vertices[i], &vertices[next], curve));
        }
        let face = Face::new(vec![Wire::from(edges)], Surface::Plane(carrier));
        let mut boundary: Option<[Interval; 3]> = None;
        for binding in &prepared {
            let image = binding
                .position_bounds([0.0, 1.0], &mut budget, &cancel)
                .unwrap();
            boundary = Some(match boundary {
                None => image,
                Some(previous) => std::array::from_fn(|axis| previous[axis].hull(image[axis])),
            });
        }
        // One theorem charge cannot prepare even one implicit edge: the bound
        // must consume the already certified complete boundary, not reselect it.
        let bounds = validated_face_bounding_box_with_boundary_enclosure(
            &face,
            boundary.unwrap(),
            &mut ValidatedGeometryBudget::new(1),
            &cancel,
        )
        .unwrap();
        for x in [0.0, 0.5, 2.0] {
            for y in [0.0, 1.5, 3.0] {
                let actual = carrier.subs(x, y);
                assert!(bounds.min().x <= actual.x && actual.x <= bounds.max().x);
                assert!(bounds.min().y <= actual.y && actual.y <= bounds.max().y);
                assert_eq!(bounds.min().z, actual.z);
                assert_eq!(bounds.max().z, actual.z);
            }
        }
        assert!(bounds.min().x >= -1.0e-8 && bounds.max().x <= 2.0 + 1.0e-8);
        assert!(bounds.min().y >= -1.0e-8 && bounds.max().y <= 3.0 + 1.0e-8);
        for bad in [
            Interval {
                lower: 1.0,
                upper: -1.0,
            },
            Interval {
                lower: f64::NAN,
                upper: 1.0,
            },
            Interval {
                lower: 0.0,
                upper: f64::INFINITY,
            },
        ] {
            assert!(matches!(
                validated_face_bounding_box_with_boundary_enclosure(
                    &face,
                    [bad; 3],
                    &mut ValidatedGeometryBudget::new(1),
                    &cancel,
                ),
                Err(ValidatedGeometryError::ConditioningFailure)
            ));
        }
        assert!(matches!(
            validated_face_bounding_box_with_boundary_enclosure(
                &face,
                boundary.unwrap(),
                &mut ValidatedGeometryBudget::new(0),
                &cancel,
            ),
            Err(ValidatedGeometryError::ResourceLimit)
        ));
        assert!(matches!(
            validated_face_bounding_box_with_boundary_enclosure(
                &face,
                boundary.unwrap(),
                &mut ValidatedGeometryBudget::new(1),
                &AtomicBool::new(true),
            ),
            Err(ValidatedGeometryError::Cancelled)
        ));
    }

    #[test]
    fn planar_boundary_bounds_accept_actual_mixed_weights_and_continuous_c0_edges() {
        let vertices = [
            Vertex::new(Point3::new(0.0, 0.0, 8.0)),
            Vertex::new(Point3::new(2.0, 0.0, 8.0)),
            Vertex::new(Point3::new(2.0, 3.0, 8.0)),
            Vertex::new(Point3::new(0.0, 3.0, 8.0)),
        ];
        // X=2tW and Z=8W exactly; W has mixed Bernstein signs but is
        // 1 - (27/8)t + (27/8)t² >= 5/32 over the entire active domain.
        let weights = [1.0, 0.15625, -0.125, 0.15625, 1.0];
        let xs = [0.0, 0.5, -0.125, -0.1875, 2.0];
        let mixed = NurbsCurve::new(BsplineCurve::new(
            KnotVector::from(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            (0..5)
                .map(|i| Vector4::new(xs[i], 0.0, 8.0 * weights[i], weights[i]))
                .collect(),
        ));
        let c0 = BsplineCurve::new(
            KnotVector::from(vec![0.0, 0.0, 0.5, 1.0, 1.0]),
            vec![
                vertices[1].point(),
                Point3::new(2.0, 1.0, 8.0),
                vertices[2].point(),
            ],
        );
        let wire = Wire::from(vec![
            Edge::new(&vertices[0], &vertices[1], Curve::NurbsCurve(mixed)),
            Edge::new(&vertices[1], &vertices[2], Curve::BsplineCurve(c0)),
            builder::line(&vertices[2], &vertices[3]),
            builder::line(&vertices[3], &vertices[0]),
        ]);
        let face = Face::new(
            vec![wire],
            Surface::Plane(Plane::from_origin_vectors(
                Point3::new(0.0, 0.0, 8.0),
                Vector3::unit_x(),
                Vector3::unit_y(),
            )),
        );
        let mut budget = ValidatedGeometryBudget::new(500_000);
        let cancel = AtomicBool::new(false);
        let bounds = validated_face_bounding_box(&face, &mut budget, &cancel)
            .expect("same native position pipeline, not coefficient-sign admission");
        assert_eq!(bounds.min().z, 8.0);
        assert_eq!(bounds.max().z, 8.0);
        for point in [
            Point3::new(0.0, 0.0, 8.0),
            Point3::new(0.74, 0.0, 8.0),
            Point3::new(2.0, 0.74, 8.0),
            Point3::new(2.0, 2.26, 8.0),
            Point3::new(0.0, 3.0, 8.0),
        ] {
            for axis in 0..3 {
                assert!(bounds.min()[axis] <= point[axis] && point[axis] <= bounds.max()[axis]);
            }
        }
    }
}
