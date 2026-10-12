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
