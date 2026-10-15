// SPDX-License-Identifier: Apache-2.0
//! Candidate seed resolution does not replace native intersection accuracy.

use monstertruck_modeling::*;
use monstertruck_solid::{
    BooleanPreparationOptions, BooleanWorkBudget, BooleanWorkLimits, ShapeOpsError,
    and_prepared_with_provenance_counted_with_query_bytes,
    prepare_boolean_source_counted_with_options,
};
use std::sync::atomic::AtomicBool;

const GEOMETRIC_TOLERANCE: f64 = 1.0e-5;
const SEED_TOLERANCE: f64 = 0.005;

fn limits() -> BooleanWorkLimits {
    BooleanWorkLimits {
        max_faces: 128,
        max_edges: 2048,
        max_mesh_vertices: 100_000,
        max_mesh_triangles: 100_000,
        max_triangle_pairs: 1_000_000,
        max_interference_segments: 100_000,
        max_output_faces: 128,
        max_output_edges: 2048,
        max_prepared_bytes: 16_777_216,
        max_source_prepared_bytes: 8_388_608,
        max_other_prepared_bytes: 8_388_608,
        max_intermediate_prepared_bytes: 16_777_216,
    }
}

fn cut(source: &Solid, clip: &Solid) -> Solid {
    let cancel = AtomicBool::new(false);
    let budget = BooleanWorkBudget::new(10_000_000);
    let prepared = prepare_boolean_source_counted_with_options(
        source,
        BooleanPreparationOptions {
            geometric_tolerance: GEOMETRIC_TOLERANCE,
            seed_tolerance: SEED_TOLERANCE,
        },
        limits(),
        &cancel,
        &budget,
    )
    .expect("source-owned candidate seed preparation");
    let result = and_prepared_with_provenance_counted_with_query_bytes(
        &prepared,
        clip,
        &cancel,
        &budget,
        limits().max_prepared_bytes - prepared.retained_bytes(),
    )
    .expect("native reconstruction at independent geometric tolerance")
    .solid;
    assert!(result.is_geometric_consistent());
    for shell in result.boundaries() {
        shell.check_solid_boundary().expect("closed native output");
    }
    result
}

fn through_hole(radius: f64) -> Solid {
    let bottom: Wire = primitive::circle(
        Point3::new(radius, 0.0, 0.0),
        Point3::origin(),
        Vector3::unit_z(),
        2,
    );
    let top = builder::translated(&bottom, Vector3::new(0.0, 0.0, 3.0));
    let seam = builder::line(bottom.front_vertex().unwrap(), top.front_vertex().unwrap());
    let mut side = top.inverse();
    side.push_back(seam.inverse());
    side.extend(bottom.iter().cloned());
    side.push_back(seam);
    let carrier = Surface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
        Curve::Line(Line(
            Point3::new(radius, 0.0, 3.0),
            Point3::new(radius, 0.0, 0.0),
        )),
        Point3::origin(),
        Vector3::unit_z(),
    )));
    let hole = Face::try_new(vec![side], carrier).unwrap();
    let mut shell = primitive::cuboid(BoundingBox::from_iter([
        Point3::new(-2.0, -2.0, 0.0),
        Point3::new(2.0, 2.0, 3.0),
    ]))
    .into_boundaries()
    .pop()
    .unwrap();
    for face in shell.face_iter_mut() {
        if face.edge_iter().all(|edge| edge.front().point().z == 3.0) {
            face.try_add_boundary(top.inverse()).unwrap();
        } else if face.edge_iter().all(|edge| edge.front().point().z == 0.0) {
            face.try_add_boundary(bottom.clone()).unwrap();
        }
    }
    shell.push(hole.inverse());
    Solid::try_new(vec![shell]).unwrap()
}

#[test]
fn coarse_candidates_reconstruct_small_hole_on_original_native_carrier() {
    let radius = 0.05;
    let source = through_hole(radius);
    let clip = primitive::cuboid(BoundingBox::from_iter([
        Point3::new(-3.0, -3.0, 1.5),
        Point3::new(3.0, 3.0, 4.0),
    ]));
    let original_hole = source
        .face_iter()
        .find_map(|face| {
            matches!(face.surface(), Surface::RevolutionSurface(_)).then(|| face.surface())
        })
        .unwrap();
    let result = cut(&source, &clip);
    assert_eq!(result.boundaries().len(), 1);
    let caps: Vec<_> = result
        .face_iter()
        .filter(|face| {
            matches!(face.surface(), Surface::Plane(_))
                && face
                    .edge_iter()
                    .all(|edge| (edge.front().point().z - 1.5).abs() <= GEOMETRIC_TOLERANCE)
        })
        .collect();
    assert_eq!(caps.len(), 1);
    assert_eq!(caps[0].absolute_boundaries().len(), 2);
    let inner: Vec<_> = caps[0]
        .absolute_boundaries()
        .iter()
        .filter(|wire| {
            wire.iter().all(|edge| {
                let point = edge.front().point();
                (point.x.hypot(point.y) - radius).abs() <= GEOMETRIC_TOLERANCE
            })
        })
        .collect();
    assert_eq!(inner.len(), 1);
    for edge in inner[0].iter() {
        edge.with_curve(|curve| {
            assert!(original_hole.include(curve));
            let (start, end) = curve.range_tuple();
            for fraction in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let point = curve.evaluate(start + fraction * (end - start));
                assert!((point.x.hypot(point.y) - radius).abs() <= GEOMETRIC_TOLERANCE);
                assert!((point.z - 1.5).abs() <= GEOMETRIC_TOLERANCE);
            }
        });
    }
    assert!(
        result
            .vertex_iter()
            .all(|vertex| vertex.point().z >= 1.5 - GEOMETRIC_TOLERANCE)
    );
}

#[test]
fn coarse_candidates_keep_both_disconnected_native_cap_patches() {
    let vertices = [
        (0.0, 0.0),
        (6.0, 0.0),
        (6.0, 4.0),
        (4.0, 4.0),
        (4.0, 1.0),
        (2.0, 1.0),
        (2.0, 4.0),
        (0.0, 4.0),
    ]
    .map(|(x, y)| Vertex::new(Point3::new(x, y, 0.0)));
    let wire: Wire = vertices
        .iter()
        .zip(vertices.iter().cycle().skip(1))
        .map(|(front, back)| builder::line(front, back))
        .collect();
    let face = builder::try_attach_plane(vec![wire]).unwrap();
    let source = builder::extrude(&face, Vector3::unit_z());
    let clip = primitive::cuboid(BoundingBox::from_iter([
        Point3::new(-1.0, -1.0, -1.0),
        Point3::new(7.0, 2.0, 2.0),
    ]));
    let result = cut(&source, &clip);
    assert_eq!(result.boundaries().len(), 1);
    let caps: Vec<_> = result
        .face_iter()
        .filter(|face| {
            matches!(face.surface(), Surface::Plane(_))
                && face
                    .edge_iter()
                    .all(|edge| (edge.front().point().y - 2.0).abs() <= GEOMETRIC_TOLERANCE)
        })
        .collect();
    assert_eq!(caps.len(), 2);
    let mut extents = Vec::new();
    for cap in caps {
        assert_eq!(cap.absolute_boundaries().len(), 1);
        let points: Vec<_> = cap.absolute_boundaries()[0]
            .iter()
            .map(|edge| edge.front().point())
            .collect();
        let area = points
            .iter()
            .zip(points.iter().cycle().skip(1))
            .map(|(a, b)| a.to_vec().cross(b.to_vec()).dot(Vector3::unit_y()))
            .sum::<f64>()
            / 2.0;
        assert!((area - 2.0).abs() <= GEOMETRIC_TOLERANCE);
        extents.push((
            points
                .iter()
                .map(|point| point.x)
                .fold(f64::INFINITY, f64::min),
            points
                .iter()
                .map(|point| point.x)
                .fold(f64::NEG_INFINITY, f64::max),
        ));
    }
    extents.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert!((extents[0].0 - 0.0).abs() <= GEOMETRIC_TOLERANCE);
    assert!((extents[0].1 - 2.0).abs() <= GEOMETRIC_TOLERANCE);
    assert!((extents[1].0 - 4.0).abs() <= GEOMETRIC_TOLERANCE);
    assert!((extents[1].1 - 6.0).abs() <= GEOMETRIC_TOLERANCE);
}

#[test]
fn explicit_options_validate_both_tolerances() {
    let source: Solid = primitive::cuboid(BoundingBox::from_iter([
        Point3::origin(),
        Point3::new(1.0, 1.0, 1.0),
    ]));
    let cancel = AtomicBool::new(false);
    let budget = BooleanWorkBudget::new(10_000_000);
    for options in [
        BooleanPreparationOptions {
            geometric_tolerance: f64::NAN,
            seed_tolerance: SEED_TOLERANCE,
        },
        BooleanPreparationOptions {
            geometric_tolerance: GEOMETRIC_TOLERANCE,
            seed_tolerance: 0.0,
        },
    ] {
        assert!(matches!(
            prepare_boolean_source_counted_with_options(
                &source,
                options,
                limits(),
                &cancel,
                &budget,
            ),
            Err(ShapeOpsError::InvalidTolerance)
        ));
    }
}
