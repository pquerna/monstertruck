//! A native intersection curve is not its approximate marching leader.

use monstertruck_modeling::*;

fn cylinder(radius: f64) -> Surface {
    Surface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
        Curve::Line(Line(
            Point3::new(radius, 0.0, 2.0),
            Point3::new(radius, 0.0, 0.0),
        )),
        Point3::origin(),
        Vector3::unit_z(),
    )))
}

fn plane() -> Surface {
    Surface::Plane(Plane::new(
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(1.0, 0.0, 1.0),
        Point3::new(0.0, 1.0, 1.0),
    ))
}

fn leader(front_radius: f64) -> BsplineCurve<Point3> {
    BsplineCurve::new(
        KnotVector::from(vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0]),
        vec![
            Point3::new(front_radius, 0.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
            Point3::new(-1.0, 0.0, 1.0),
            Point3::new(0.0, -1.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
        ],
    )
}

#[test]
fn projected_circle_is_included_without_accepting_its_inscribed_guide() {
    for transform in [
        Matrix4::identity(),
        Matrix4::from_translation(Vector3::new(3.0, -2.0, 4.0))
            * Matrix4::from_nonuniform_scale(2.0, 1.0, 1.0),
    ] {
        let wall = cylinder(1.0).transformed(transform);
        let cap = plane().transformed(transform);
        let guide = leader(1.0).transformed(transform);
        assert!(!wall.include(&Curve::BsplineCurve(guide.clone())));
        let curve = Curve::from(IntersectionCurve::new(wall.clone(), cap.clone(), guide));
        let inverse = transform.invert().unwrap();
        for index in 0..=64 {
            let point = inverse.transform_point(curve.evaluate(index as f64 / 16.0));
            assert!((point.x * point.x + point.y * point.y - 1.0).abs() < 1.0e-6);
            assert!((point.z - 1.0).abs() < 1.0e-6);
        }
        assert!(wall.include(&curve));
        assert!(cap.include(&curve));
        assert!(!cylinder(2.0).transformed(transform).include(&curve));
    }
}

#[test]
fn malformed_unprojected_endpoint_is_not_granted_carrier_membership() {
    let wall = cylinder(1.0);
    let cap = plane();
    let curve = Curve::from(IntersectionCurve::new(wall.clone(), cap, leader(0.75)));
    assert!(!wall.include(&curve));
    assert!((curve.front().x - 0.75).abs() < 1.0e-12);
}

#[test]
fn ordinary_curves_use_original_posed_inverted_affine_revolution_carrier() {
    let line = Curve::Line(Line(Point3::new(1.0, 0.0, 0.1), Point3::new(1.0, 0.0, 1.9)));
    let spline = Curve::BsplineCurve(BsplineCurve::from(Line(
        Point3::new(1.0, 0.0, 0.1),
        Point3::new(1.0, 0.0, 1.9),
    )));
    let rational = Curve::NurbsCurve(NurbsCurve::new(spline.lift_up()));
    let circle: Wire = primitive::circle(
        Point3::new(1.0, 0.0, 1.0),
        Point3::new(0.0, 0.0, 1.0),
        Vector3::unit_z(),
        2,
    );
    let arcs = [circle[0].curve(), circle[1].curve()];
    let bowed = Curve::BsplineCurve(BsplineCurve::new(
        KnotVector::bezier_knot(2),
        vec![
            Point3::new(1.0, 0.0, 0.1),
            Point3::new(1.2, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.9),
        ],
    ));
    let bowed_rational = Curve::NurbsCurve(NurbsCurve::new(bowed.lift_up()));
    let offset = Curve::Line(Line(
        Point3::new(1.05, 0.0, 0.1),
        Point3::new(1.05, 0.0, 1.9),
    ));
    for transform in [
        Matrix4::identity(),
        Matrix4::from_translation(Vector3::new(40.0, 20.0, 0.0)),
        Matrix4::from_translation(Vector3::new(80.0, -7.0, 3.0))
            * Matrix4::from_cols(
                Vector4::new(2.0, 0.0, 0.0, 0.0),
                Vector4::new(0.3, 1.0, 0.0, 0.0),
                Vector4::new(0.0, 0.2, 0.5, 0.0),
                Vector4::new(0.0, 0.0, 0.0, 1.0),
            ),
    ] {
        let inverse = transform.invert().unwrap();
        let wall = cylinder(1.0).transformed(transform);
        for inverted in [false, true] {
            let wall = if inverted {
                wall.inverse()
            } else {
                wall.clone()
            };
            for curve in [&line, &spline, &rational, &arcs[0], &arcs[1]] {
                let curve = curve.transformed(transform);
                let (start, end) = curve.range_tuple();
                for index in 0..=8 {
                    let point = curve.evaluate(start + (end - start) * index as f64 / 8.0);
                    let local = inverse.transform_point(point);
                    assert!((local.x * local.x + local.y * local.y - 1.0).abs() < 1.0e-10);
                    let (u, v) = wall.search_parameter(point, None, 100).unwrap();
                    assert!(wall.evaluate(u, v).near(&point));
                }
                assert!(wall.include(&curve));
            }
            for curve in [&bowed, &bowed_rational, &offset] {
                let curve = curve.transformed(transform);
                let (start, end) = curve.range_tuple();
                let local = inverse.transform_point(curve.evaluate((start + end) / 2.0));
                assert!(local.x.hypot(local.y) > 1.01);
                assert!(!wall.include(&curve));
            }
        }
    }
}
