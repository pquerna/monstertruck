use super::*;

#[derive(Clone, ParametricSurface)]
enum Carrier {
    Plane(Plane),
    Spline(BsplineSurface<Point3>),
    Rational(NurbsSurface<Vector4>),
    Posed(Processor<RevolutionSurface<Line<Point3>>, Matrix4>),
    Offset(OffsetSurface<Plane, BsplineSurface<Vector3>>),
}

fn carriers() -> Vec<Carrier> {
    let plane = Plane::new(
        Point3::new(10.0, -20.0, 30.0),
        Point3::new(12.0, -20.0, 30.0),
        Point3::new(10.0, -17.0, 30.0),
    );
    let spline = BsplineSurface::new(
        (KnotVector::bezier_knot(1), KnotVector::bezier_knot(1)),
        vec![
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 2.0, 0.0)],
            vec![Point3::new(3.0, 0.0, 0.0), Point3::new(3.0, 2.0, 1.0)],
        ],
    );
    let weight = 0.5_f64.sqrt();
    let rational = NurbsSurface::new(BsplineSurface::new(
        (KnotVector::bezier_knot(2), KnotVector::bezier_knot(1)),
        vec![
            vec![
                Vector4::new(5.0, 0.0, 0.0, 1.0),
                Vector4::new(5.0, 0.0, 8.0, 1.0),
            ],
            vec![
                Vector4::new(5.0 * weight, 5.0 * weight, 0.0, weight),
                Vector4::new(5.0 * weight, 5.0 * weight, 8.0 * weight, weight),
            ],
            vec![
                Vector4::new(0.0, 5.0, 0.0, 1.0),
                Vector4::new(0.0, 5.0, 8.0, 1.0),
            ],
        ],
    ));
    let revolution = RevolutionSurface::by_revolution(
        Line(Point3::new(5.0, 0.0, 0.0), Point3::new(6.0, 0.0, 8.0)),
        Point3::origin(),
        Vector3::unit_z(),
    );
    let pose = Matrix4::from_translation(Vector3::new(40.0, -30.0, 0.08726203218641757))
        * Matrix4::from_angle_x(Deg(13.0))
        * Matrix4::from_nonuniform_scale(1.2, 0.8, 1.1);
    let mut posed = Processor::with_transform(revolution, pose);
    posed.invert();
    // OffsetSurface has a genuine custom derivative-batch override. Its
    // nonconstant offset affects both position and the coupled Jacobian.
    let offset = BsplineSurface::new(
        (KnotVector::bezier_knot(1), KnotVector::bezier_knot(1)),
        vec![
            vec![Vector3::new(1.0, -2.0, 0.5), Vector3::new(1.0, 3.0, 1.5)],
            vec![Vector3::new(4.0, -2.0, 2.5), Vector3::new(4.0, 3.0, 4.5)],
        ],
    );
    vec![
        Carrier::Plane(plane),
        Carrier::Spline(spline),
        Carrier::Rational(rational),
        Carrier::Posed(posed),
        Carrier::Offset(OffsetSurface::new(plane, offset)),
    ]
}

fn vector_bits(vector: Vector3) -> [u64; 3] {
    [vector.x.to_bits(), vector.y.to_bits(), vector.z.to_bits()]
}

fn solve<S: ParametricSurface<Point = Point3, Vector = Vector3>>(
    surface: &S,
    plane: &Plane,
    point: Point3,
    normal: Vector3,
    compact: bool,
) -> Vector4 {
    let function = |Vector4 { x, y, z, w }| {
        let entries = |surface: &S| {
            if compact {
                surface.first_derivatives(x, y)
            } else {
                let batch = surface.derivatives(1, x, y);
                [batch[0][0], batch[1][0], batch[0][1]]
            }
        };
        let [pt0, uder0, vder0] = entries(surface);
        let [pt1, uder1, vder1] = if compact {
            plane.first_derivatives(z, w)
        } else {
            let batch = plane.derivatives(1, z, w);
            [batch[0][0], batch[1][0], batch[0][1]]
        };
        CalcOutput {
            value: (pt0 - pt1).extend(normal.dot((pt0 + pt1) / 2.0 - point.to_vec())),
            derivation: Matrix4::from_cols(
                uder0.extend(normal.dot(uder0) / 2.0),
                vder0.extend(normal.dot(vder0) / 2.0),
                (-uder1).extend(normal.dot(uder1) / 2.0),
                (-vder1).extend(normal.dot(vder1) / 2.0),
            ),
        }
    };
    newton::solve(function, Vector4::new(0.37, 0.41, 0.0, 0.0), 100)
        .expect("transverse native coupled root")
}

#[test]
fn native_compact_jets_preserve_carriers_and_bitwise_coupled_roots_including_custom_batches() {
    for surface in carriers() {
        for (u, v) in [(0.0, 0.0), (0.37, 0.41), (1.0, 1.0)] {
            let batch = surface.derivatives(1, u, v);
            let expected = [batch[0][0], batch[1][0], batch[0][1]];
            let compact = surface.first_derivatives(u, v);
            assert_eq!(compact.map(vector_bits), expected.map(vector_bits));
            assert!(Point3::from_vec(compact[0]).near(&surface.evaluate(u, v)));
            assert!(compact[1].cross(compact[2]).magnitude2() > 0.0);
        }
        let target = surface.evaluate(0.37, 0.41);
        let clip_normal = Vector3::new(0.13, -0.09, 0.98).normalize();
        let x = clip_normal.cross(Vector3::unit_y()).normalize();
        let plane = Plane::new(target, target + x, target + clip_normal.cross(x));
        let tangent = surface
            .derivative_u(0.37, 0.41)
            .cross(surface.derivative_v(0.37, 0.41))
            .cross(clip_normal)
            .normalize();
        let point = target + tangent * 0.01 + clip_normal * 0.005;
        let expected = solve(&surface, &plane, point, tangent, false);
        // References and boxes must retain the same native batch authority;
        // their compact jets are used to construct a real coupled root.
        let borrowed = &surface;
        let actual = solve(&borrowed, &plane, point, tangent, true);
        let boxed = Box::new(surface.clone());
        let boxed_root = solve(&boxed, &plane, point, tangent, true);
        assert_eq!(
            [
                actual.x.to_bits(),
                actual.y.to_bits(),
                actual.z.to_bits(),
                actual.w.to_bits()
            ],
            [
                expected.x.to_bits(),
                expected.y.to_bits(),
                expected.z.to_bits(),
                expected.w.to_bits()
            ]
        );
        assert_eq!(actual, boxed_root);
        let carrier_point = surface.evaluate(actual.x, actual.y);
        assert!(carrier_point.near(&plane.evaluate(actual.z, actual.w)));
        assert!((carrier_point - point).dot(tangent).abs() < TOLERANCE);
    }
}

#[test]
fn public_native_intersection_projection_retains_both_actual_carrier_residuals() {
    let rational = match carriers().remove(2) {
        Carrier::Rational(surface) => surface,
        _ => unreachable!(),
    };
    let plane = Plane::new(
        Point3::new(0.0, 0.0, 4.0),
        Point3::new(1.0, 0.0, 4.0),
        Point3::new(0.0, 1.0, 4.0),
    );
    for u in [0.1, 0.37, 0.8] {
        let point = rational.evaluate(u, 0.5) + Vector3::new(0.001, -0.001, 0.002);
        let tangent = rational.normal(u, 0.5).cross(plane.normal());
        let (point, uv0, uv1) =
            double_projection(&rational, None, &plane, None, point, tangent, 100)
                .expect("original native coupled projection");
        assert!(point.near(&rational.evaluate(uv0.x, uv0.y)));
        assert!(point.near(&plane.evaluate(uv1.x, uv1.y)));
        assert!((point.z - 4.0).abs() < TOLERANCE);
    }
}
