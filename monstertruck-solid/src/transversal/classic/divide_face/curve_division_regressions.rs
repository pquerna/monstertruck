use super::*;
use crate::transversal::prepared::BooleanWorkLimits;
use monstertruck_modeling::{self as modeling, Plane};
use std::sync::atomic::{AtomicBool, Ordering};

fn policy() -> BooleanWorkLimits {
    BooleanWorkLimits {
        max_faces: 64,
        max_edges: 512,
        max_mesh_vertices: 50_000,
        max_mesh_triangles: 50_000,
        max_triangle_pairs: 100_000,
        max_interference_segments: 10_000,
        max_output_faces: 64,
        max_output_edges: 512,
        max_prepared_bytes: 4 * 1024 * 1024,
        max_source_prepared_bytes: 4 * 1024 * 1024,
        max_other_prepared_bytes: 4 * 1024 * 1024,
        max_intermediate_prepared_bytes: 4 * 1024 * 1024,
    }
}

#[test]
fn opposite_native_coedges_share_subdivision_but_preserve_both_face_parameter_contours() {
    let a = Vertex::new(Point3::new(0.0, 0.0, 0.0));
    let b = Vertex::new(Point3::new(2.0, 0.0, 0.0));
    let c = Vertex::new(Point3::new(2.0, 1.0, 0.0));
    let d = Vertex::new(Point3::new(0.0, 1.0, 0.0));
    let e = Vertex::new(Point3::new(0.0, 0.0, 3.0));
    let f = Vertex::new(Point3::new(2.0, 0.0, 3.0));
    let shared: modeling::Edge = modeling::builder::line(&a, &b);
    let xy: Wire<_, _> = [
        shared.clone(),
        modeling::builder::line(&b, &c),
        modeling::builder::line(&c, &d),
        modeling::builder::line(&d, &a),
    ]
    .into_iter()
    .collect();
    let xz: Wire<_, _> = [
        shared.inverse(),
        modeling::builder::line(&a, &e),
        modeling::builder::line(&e, &f),
        modeling::builder::line(&f, &b),
    ]
    .into_iter()
    .collect();
    let plane0 = Plane::new(
        a.point(),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    );
    let plane1 = Plane::new(
        a.point(),
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(1.0, 0.0, 0.0),
    );
    let face0 = Face::new(vec![xy], modeling::Surface::Plane(plane0));
    let face1 = Face::new(vec![xz], modeling::Surface::Plane(plane1));
    let cancel = AtomicBool::new(false);
    let work = BooleanWork::new(policy(), &cancel);
    let mut cache = CurveDivisionCache::new(1.0e-5);
    let roots = HashMap::default();
    let first = create_parameter_boundary(
        &face0,
        &face0.absolute_boundaries()[0],
        &mut cache,
        None,
        &roots,
        Some(&work),
    )
    .unwrap_or_else(|_| panic!("XY native parameter boundary"));
    let second = create_parameter_boundary(
        &face1,
        &face1.absolute_boundaries()[0],
        &mut cache,
        None,
        &roots,
        Some(&work),
    )
    .unwrap_or_else(|_| panic!("XZ native parameter boundary"));
    assert!((first.area() - 2.0).abs() < 1.0e-12);
    assert!((second.area() - 6.0).abs() < 1.0e-12);
    for (points, face) in [(&first, &face0), (&second, &face1)] {
        assert_eq!(points.first(), points.last());
        assert!(points.iter().all(|point| {
            let value = face.surface().evaluate(point.x, point.y);
            [value.x, value.y, value.z].into_iter().all(f64::is_finite)
        }));
    }
    assert!(face0.is_geometric_consistent() && face1.is_geometric_consistent());
    assert_eq!(shared.front().point(), a.point());
    assert_eq!(shared.back().point(), b.point());
}

#[test]
fn sampling_byte_refusal_and_cached_hit_cancellation_remain_typed() {
    let first = Vertex::new(Point3::origin());
    let last = Vertex::new(Point3::new(2.0, 0.0, 0.0));
    let edge: modeling::Edge = modeling::builder::line(&first, &last);
    let cancel = AtomicBool::new(false);
    let mut tiny = policy();
    tiny.max_intermediate_prepared_bytes = 0;
    let limited = BooleanWork::new(tiny, &cancel);
    let mut cache = CurveDivisionCache::new(1.0e-5);
    assert!(matches!(
        cache.get(&edge, Some(&limited)),
        Err(ShapeOpsError::WorkLimitExceeded {
            resource: "curve_division_admission_bytes",
            ..
        })
    ));
    let work = BooleanWork::new(policy(), &cancel);
    cache.get(&edge, Some(&work)).expect("admitted native line");
    cancel.store(true, Ordering::Relaxed);
    assert!(matches!(
        cache.get(&edge, Some(&work)),
        Err(ShapeOpsError::Cancelled)
    ));
}
