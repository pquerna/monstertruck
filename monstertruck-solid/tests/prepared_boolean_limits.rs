use monstertruck_meshing::prelude::*;
use monstertruck_modeling::*;
use monstertruck_solid::{
    BooleanWorkLimits, ShapeOpsError, and, and_prepared, prepare_boolean_source,
};
use std::sync::atomic::AtomicBool;

fn limits() -> BooleanWorkLimits {
    BooleanWorkLimits {
        max_faces: 64,
        max_edges: 512,
        max_mesh_vertices: 20_000,
        max_mesh_triangles: 20_000,
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

fn source() -> Solid {
    primitive::cuboid(BoundingBox::from_iter([
        Point3::origin(),
        Point3::new(1.0, 1.0, 1.0),
    ]))
}

#[test]
fn prepared_queries_share_native_and_carriers_and_leave_the_source_unchanged() {
    let source = source();
    let before = serde_json::to_vec(&source.boundaries()[0].compress()).expect("source snapshot");
    let clip = builder::translated(&source, Vector3::new(0.5, 0.5, 0.5));
    let input_carriers = source
        .face_iter()
        .chain(clip.face_iter())
        .map(|face| {
            face.with_surface(|surface| serde_json::to_value(surface))
                .expect("original native carrier")
        })
        .collect::<Vec<_>>();
    let cancel = AtomicBool::new(false);
    let prepared =
        prepare_boolean_source(&source, 1.0e-5, limits(), &cancel).expect("native preparation");
    let ordinary = and(&source, &clip, 1.0e-5).expect("ordinary native AND");
    let repeated = and_prepared(&prepared, &clip, &cancel).expect("prepared native AND");
    for result in [&ordinary, &repeated] {
        result.boundaries()[0]
            .check_solid_boundary()
            .expect("native manifold");
        assert!(result.is_geometric_consistent());
        assert!(
            (result.boundaries()[0]
                .triangulation(0.005)
                .to_polygon()
                .volume()
                - 0.125)
                .abs()
                < 1.0e-6
        );
        assert!(result.face_iter().all(|face| {
            let carrier = face
                .with_surface(|surface| serde_json::to_value(surface))
                .expect("retained native carrier");
            input_carriers.contains(&carrier)
        }));
    }
    assert_eq!(
        before,
        serde_json::to_vec(&source.boundaries()[0].compress()).expect("immutable source")
    );
}

#[test]
fn cancellation_is_typed_before_preparation_or_any_query_work() {
    let source = source();
    let cancelled = AtomicBool::new(true);
    assert!(matches!(
        prepare_boolean_source(&source, 1.0e-5, limits(), &cancelled),
        Err(ShapeOpsError::Cancelled)
    ));
    let prepared = prepare_boolean_source(&source, 1.0e-5, limits(), &AtomicBool::new(false))
        .expect("preparation");
    assert!(matches!(
        and_prepared(&prepared, &source, &cancelled),
        Err(ShapeOpsError::Cancelled)
    ));
}

#[test]
fn source_byte_admission_fails_before_meshing_and_query_byte_admission_is_aggregate() {
    let source = source();
    let cancel = AtomicBool::new(false);
    let mut policy = limits();
    policy.max_prepared_bytes = 0;
    assert!(matches!(
        prepare_boolean_source(&source, 1.0e-5, policy, &cancel),
        Err(ShapeOpsError::WorkLimitExceeded {
            resource: "prepared_bytes",
            ..
        })
    ));
    let first = prepare_boolean_source(&source, 1.0e-5, limits(), &cancel).expect("preparation");
    policy.max_prepared_bytes = first.retained_bytes();
    let prepared = prepare_boolean_source(&source, 1.0e-5, policy, &cancel)
        .expect("exact retained buffer admission");
    let clip = builder::translated(&source, Vector3::new(0.5, 0.5, 0.5));
    assert!(matches!(
        and_prepared(&prepared, &clip, &cancel),
        Err(ShapeOpsError::WorkLimitExceeded {
            resource: "prepared_bytes",
            ..
        })
    ));
}

#[test]
fn bounded_triangle_work_segments_and_selected_output_refuse_excess_without_fallback() {
    let source = source();
    let clip = builder::translated(&source, Vector3::new(0.5, 0.5, 0.5));
    let cancel = AtomicBool::new(false);
    for resource in [
        "triangle_pairs",
        "interference_segments",
        "output_faces",
        "output_edges",
    ] {
        let mut policy = limits();
        match resource {
            "triangle_pairs" => policy.max_triangle_pairs = 0,
            "interference_segments" => policy.max_interference_segments = 0,
            "output_faces" => policy.max_output_faces = 0,
            "output_edges" => policy.max_output_edges = 0,
            _ => unreachable!("fixed resource list"),
        }
        let prepared = prepare_boolean_source(&source, 1.0e-5, policy, &cancel)
            .expect("source within admission");
        let error = and_prepared(&prepared, &clip, &cancel).expect_err("excess must refuse");
        assert!(
            matches!(error, ShapeOpsError::WorkLimitExceeded { resource: failed, .. } if failed == resource)
        );
    }
}

#[test]
fn exact_source_cache_admission_keeps_a_distinct_query_seed_allowance() {
    let source = source();
    let cancel = AtomicBool::new(false);
    let initial =
        prepare_boolean_source(&source, 1.0e-5, limits(), &cancel).expect("initial preparation");
    let mut policy = limits();
    policy.max_source_prepared_bytes = initial.retained_bytes();
    let prepared = prepare_boolean_source(&source, 1.0e-5, policy, &cancel)
        .expect("exact source cache admission");
    let clip = builder::translated(&source, Vector3::new(0.5, 0.5, 0.5));
    and_prepared(&prepared, &clip, &cancel).expect("independent query seed allowance");
}
