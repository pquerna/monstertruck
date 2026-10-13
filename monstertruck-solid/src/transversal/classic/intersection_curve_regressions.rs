use super::*;

fn xy(z: f64) -> Plane {
    Plane::new(
        Point3::new(0.0, 0.0, z),
        Point3::new(1.0, 0.0, z),
        Point3::new(0.0, 1.0, z),
    )
}

#[test]
fn separate_parallel_carriers_cannot_certify_an_independent_projection_midpoint() {
    let leader = PolylineCurve(vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)]);
    assert!(build_intersection_curve(xy(0.0), xy(4.0 * TOLERANCE), leader, None).is_none());
}

#[test]
fn coplanar_carriers_do_not_certify_a_transverse_intersection() {
    let leader = PolylineCurve(vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)]);
    assert!(build_intersection_curve(xy(0.0), xy(0.0), leader, None).is_none());
}

#[test]
fn native_transverse_leader_is_certified_on_both_original_carriers() {
    let first = xy(0.0);
    let second = Plane::new(
        Point3::origin(),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
    );
    let leader = PolylineCurve(vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)]);
    let curve =
        build_intersection_curve(first, second, leader, None).expect("native transverse curve");
    for carrier in [curve.surface0(), curve.surface1()] {
        for point in curve.leader().iter() {
            let uv = carrier
                .search_parameter(*point, None, 100)
                .expect("carrier certificate");
            assert!(point.near(&carrier.evaluate(uv.0, uv.1)));
        }
    }
}
