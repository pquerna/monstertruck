// Licensed under the Apache License, Version 2.0.
use monstertruck_geometry::prelude::*;
use monstertruck_topology::{Edge, Vertex};

fn line(length: f64) -> Edge<Point3, Line<Point3>> {
    let first = Vertex::new(Point3::origin());
    let last = Vertex::new(Point3::new(length, 0.0, 0.0));
    Edge::new(&first, &last, Line(first.point(), last.point()))
}

#[test]
fn fast_native_curve_interior_cut_preserves_carriers_and_shared_inverse_topology() {
    let original = line(100.0);
    let original_curve = original.curve();
    let parameter = 1.0 - 3.0e-8;
    assert!(parameter.near(&1.0));
    let point = original.with_curve(|curve| curve.evaluate(parameter));
    assert!(!point.near(&original.absolute_back().point()));
    for source in [original.clone(), original.inverse()] {
        let contact = Vertex::new(point);
        let (first, second) = source.cut_with_parameter(&contact, parameter)
            .expect("world-distinct native interior contact must cut even when its parameter is endpoint-near");
        assert_eq!(first.front(), source.front());
        assert_eq!(first.back(), &contact);
        assert_eq!(second.front(), &contact);
        assert_eq!(second.back(), source.back());
        assert!(first.is_geometric_consistent() && second.is_geometric_consistent());
        assert_eq!(first.orientation(), source.orientation());
        assert_eq!(second.orientation(), source.orientation());
        for piece in [&first, &second] {
            let curve = piece.curve();
            assert!(curve.0.y.abs() < 1.0e-12 && curve.1.y.abs() < 1.0e-12);
            assert!(curve.0.z.abs() < 1.0e-12 && curve.1.z.abs() < 1.0e-12);
            assert!((curve.1.x - curve.0.x).abs() > TOLERANCE);
        }
    }
    assert_eq!(original.curve(), original_curve);
    assert_eq!(original.absolute_front().point(), original_curve.0);
    assert_eq!(original.absolute_back().point(), original_curve.1);
}

#[test]
fn native_cut_refuses_world_endpoints_invalid_domain_and_off_carrier_contacts() {
    let short = line(5.0e-4);
    for parameter in [1.0e-4, 1.0 - 1.0e-4] {
        let contact = Vertex::new(short.with_curve(|curve| curve.evaluate(parameter)));
        assert!(
            short.cut_with_parameter(&contact, parameter).is_none(),
            "parameter-distinct but world-near endpoint must not create a degenerate cut"
        );
    }
    let source = line(100.0);
    let original = source.curve();
    for parameter in [
        -1.0e-8,
        0.0,
        1.0,
        1.0 + 1.0e-8,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        let contact = Vertex::new(Point3::new(50.0, 0.0, 0.0));
        assert!(source.cut_with_parameter(&contact, parameter).is_none());
    }
    let off_carrier = Vertex::new(Point3::new(50.0, 3.0e-6, 0.0));
    assert!(source.cut_with_parameter(&off_carrier, 0.5).is_none());
    assert!(
        source
            .cut_with_parameter(source.absolute_front(), 0.0)
            .is_none()
    );
    assert!(
        source
            .cut_with_parameter(source.absolute_back(), 1.0)
            .is_none()
    );
    assert_eq!(source.curve(), original);
    assert!(source.is_geometric_consistent());
}
