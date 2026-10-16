use super::*;
use crate::alternative::Alternative;
use crate::transversal::prepared::{BooleanWorkBudget, BooleanWorkLimits};
use std::sync::atomic::{AtomicBool, Ordering};

type TestCurve = Alternative<Line<Point3>, IntersectionCurve<PolylineCurve, Plane, Plane>>;

fn limits() -> BooleanWorkLimits {
    BooleanWorkLimits {
        max_faces: 64,
        max_edges: 512,
        max_mesh_vertices: 50_000,
        max_mesh_triangles: 50_000,
        max_triangle_pairs: 0,
        max_interference_segments: 10_000,
        max_output_faces: 64,
        max_output_edges: 512,
        max_prepared_bytes: 4 * 1024 * 1024,
        max_source_prepared_bytes: 4 * 1024 * 1024,
        max_other_prepared_bytes: 4 * 1024 * 1024,
        max_intermediate_prepared_bytes: 4 * 1024 * 1024,
    }
}

fn mesh(origin: Point3, triangles: usize) -> PolygonMesh {
    PolygonMesh::new(
        StandardAttributes {
            positions: vec![
                origin,
                origin + Vector3::unit_x(),
                origin + Vector3::unit_y(),
            ],
            ..Default::default()
        },
        Faces::from_iter(std::iter::repeat_n([0, 1, 2], triangles)),
    )
}

fn shell(mesh: PolygonMesh) -> MeshShell {
    [Face::new(Vec::new(), Some(mesh))].into_iter().collect()
}

fn run(
    first: &MeshShell,
    second: &MeshShell,
    work: &BooleanWork<'_>,
) -> ClassicResult<LoopsStoreQuadruple<TestCurve>> {
    let plane = Plane::new(
        Point3::origin(),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    );
    let native: Shell<Point3, TestCurve, Plane> =
        [Face::new(Vec::new(), plane)].into_iter().collect();
    create_loops_stores(&native, first, &native, second, TOLERANCE, Some(work), None)
}

#[test]
fn strictly_separated_dense_pairs_bypass_triangle_admission_on_each_axis() {
    for axis in 0..3 {
        for direction in [-1.0, 1.0] {
            let mut origin = Point3::origin();
            origin[axis] = direction * 4.0;
            let first = shell(mesh(Point3::origin(), 1000));
            let second = shell(mesh(origin, 1000));
            let cancel = AtomicBool::new(false);
            let budget = BooleanWorkBudget::new(100);
            let work = BooleanWork::with_budget(limits(), &cancel, Some(&budget));
            run(&first, &second, &work).expect("strict seed separation");
            assert!(budget.consumed() < 100);
        }
    }
}

#[test]
fn touching_overlapping_and_unknown_bounds_keep_full_pair_admission() {
    for origin in [
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.5, 0.0, 0.0),
        Point3::origin(),
        Point3::new(f64::NAN, 10.0, 0.0),
        Point3::new(f64::INFINITY, 10.0, 0.0),
    ] {
        let first = shell(mesh(Point3::origin(), 3));
        let second = shell(mesh(origin, 5));
        let cancel = AtomicBool::new(false);
        let work = BooleanWork::new(limits(), &cancel);
        assert!(matches!(
            run(&first, &second, &work),
            Err(ShapeOpsError::WorkLimitExceeded {
                resource: "triangle_pairs",
                actual: 15,
                limit: 0,
            })
        ));
    }
}

#[test]
fn cache_scans_and_pair_rejections_honor_cancellation_and_work_budget() {
    let first = shell(mesh(Point3::origin(), 1));
    let second = shell(mesh(Point3::new(4.0, 0.0, 0.0), 1));
    let cancel = AtomicBool::new(true);
    let work = BooleanWork::new(limits(), &cancel);
    assert!(matches!(
        SeedBounds::new([&first, &second], Some(&work)),
        Err(ShapeOpsError::Cancelled)
    ));
    cancel.store(false, Ordering::Relaxed);
    let bounds = SeedBounds::new([&first, &second], Some(&work)).expect("cache");
    cancel.store(true, Ordering::Relaxed);
    assert!(matches!(
        bounds.disjoint(0, 0, Some(&work)),
        Err(ShapeOpsError::Cancelled)
    ));
    cancel.store(false, Ordering::Relaxed);
    let budget = BooleanWorkBudget::new(5);
    let work = BooleanWork::with_budget(limits(), &cancel, Some(&budget));
    assert!(matches!(
        run(&first, &second, &work),
        Err(ShapeOpsError::WorkLimitExceeded {
            resource: "native_work",
            ..
        })
    ));
    assert_eq!(budget.consumed(), 5);
}

#[test]
fn cache_faces_positions_and_combined_buffer_are_admitted_before_rejection() {
    let first = shell(mesh(Point3::origin(), 1));
    let second = shell(mesh(Point3::new(4.0, 0.0, 0.0), 1));
    let cancel = AtomicBool::new(false);
    for (resource, policy) in [
        (
            "input_faces",
            BooleanWorkLimits {
                max_faces: 1,
                ..limits()
            },
        ),
        (
            "mesh_vertices",
            BooleanWorkLimits {
                max_mesh_vertices: 5,
                ..limits()
            },
        ),
        (
            "intermediate_prepared_bytes",
            BooleanWorkLimits {
                max_intermediate_prepared_bytes: size_of::<Option<Bounds>>(),
                ..limits()
            },
        ),
    ] {
        let budget = BooleanWorkBudget::new(100);
        let work = BooleanWork::with_budget(policy, &cancel, Some(&budget));
        match run(&first, &second, &work) {
            Err(ShapeOpsError::WorkLimitExceeded {
                resource: actual, ..
            }) => assert_eq!(actual, resource),
            _ => panic!("missing cache resource rejection: {resource}"),
        }
        assert!(budget.consumed() > 0);
    }
}

#[test]
fn empty_or_missing_meshes_never_prove_separation_and_missing_keeps_its_error() {
    let first = shell(PolygonMesh::default());
    let second = shell(mesh(Point3::new(4.0, 0.0, 0.0), 1));
    let cancel = AtomicBool::new(false);
    let work = BooleanWork::new(limits(), &cancel);
    let bounds = SeedBounds::new([&first, &second], Some(&work)).expect("empty cache");
    assert!(!bounds.disjoint(0, 0, Some(&work)).expect("unknown"));
    let missing: MeshShell = [Face::new(Vec::new(), None)].into_iter().collect();
    assert!(matches!(
        run(&missing, &second, &work),
        Err(ShapeOpsError::InputTessellationFailed {
            shell_index: 0,
            face_index: 0
        })
    ));
}
