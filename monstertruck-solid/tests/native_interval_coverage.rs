// SPDX-License-Identifier: Apache-2.0
//! A connected source can have distinct, non-covered intersections on one face pair.

use monstertruck_meshing::prelude::*;
use monstertruck_modeling::*;
use monstertruck_solid::{BooleanWorkLimits, and, and_prepared, prepare_boolean_source};
use std::sync::atomic::AtomicBool;

fn limits() -> BooleanWorkLimits {
    BooleanWorkLimits {
        max_faces: 128,
        max_edges: 1024,
        max_mesh_vertices: 20_000,
        max_mesh_triangles: 20_000,
        max_triangle_pairs: 100_000,
        max_interference_segments: 20_000,
        max_output_faces: 128,
        max_output_edges: 2048,
        max_prepared_bytes: 4 * 1024 * 1024,
        max_source_prepared_bytes: 3 * 1024 * 1024,
        max_other_prepared_bytes: 1024 * 1024,
        max_intermediate_prepared_bytes: 1024 * 1024,
    }
}

fn u_prism() -> Solid {
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
        .map(|(first, second)| builder::line(first, second))
        .collect();
    let face = builder::try_attach_plane(vec![wire]).expect("native concave profile");
    builder::extrude(&face, Vector3::unit_z())
}

#[test]
fn separate_intervals_on_one_native_face_pair_survive_ordinary_and_prepared_and() {
    let source = u_prism();
    let snapshot = serde_json::to_vec(&source.boundaries()[0].compress()).expect("source snapshot");
    let identities = source
        .face_iter()
        .map(|face| (face.id(), face.stable_id(), face.orientation()))
        .collect::<Vec<_>>();
    let clip = primitive::cuboid(BoundingBox::from_iter([
        Point3::new(-1.0, -1.0, -1.0),
        Point3::new(7.0, 2.0, 2.0),
    ]));
    let cap = clip
        .face_iter()
        .find_map(|face| {
            let Surface::Plane(plane) = face.surface() else {
                return None;
            };
            ((plane.origin().y - 2.0).abs() < 1.0e-12
                && plane
                    .normal()
                    .normalize()
                    .cross(Vector3::unit_y())
                    .magnitude()
                    < 1.0e-12)
                .then_some(plane)
        })
        .expect("unchanged clip cap carrier");
    let input_carriers = source
        .face_iter()
        .chain(clip.face_iter())
        .map(|face| {
            face.with_surface(|surface| serde_json::to_value(surface))
                .expect("original native carrier")
        })
        .collect::<Vec<_>>();
    let cancel = AtomicBool::new(false);
    let prepared = prepare_boolean_source(&source, 1.0e-5, limits(), &cancel).expect("preparation");
    let ordinary = and(&source, &clip, 1.0e-5).expect("ordinary native AND");
    let cached = and_prepared(&prepared, &clip, &cancel).expect("prepared native AND");
    for result in [&ordinary, &cached] {
        assert_eq!(
            result.boundaries().len(),
            1,
            "the retained U remains connected"
        );
        result.boundaries()[0]
            .check_solid_boundary()
            .expect("native manifold");
        assert!(result.is_geometric_consistent());
        assert!(
            (result.boundaries()[0]
                .triangulation(0.005)
                .to_polygon()
                .volume()
                - 10.0)
                .abs()
                < 1.0e-6
        );
        assert!(
            result
                .vertex_iter()
                .all(|vertex| vertex.point().y <= 2.0 + 1.0e-6)
        );
        assert!(result.face_iter().all(|face| {
            let carrier = face
                .with_surface(|surface| serde_json::to_value(surface))
                .expect("retained native carrier");
            input_carriers.contains(&carrier)
        }));
        let caps = result
            .face_iter()
            .filter(|face| matches!(face.surface(), Surface::Plane(plane) if plane == cap))
            .collect::<Vec<_>>();
        assert_eq!(
            caps.len(),
            2,
            "neither distinct transverse interval may be discarded"
        );
        let mut extents = Vec::new();
        for face in caps {
            assert_eq!(
                face.absolute_boundaries().len(),
                1,
                "each arm has one outer contour"
            );
            let points = face.absolute_boundaries()[0]
                .iter()
                .map(|edge| edge.front().point())
                .collect::<Vec<_>>();
            let area = points
                .iter()
                .zip(points.iter().cycle().skip(1))
                .map(|(first, second)| {
                    first
                        .to_vec()
                        .cross(second.to_vec())
                        .dot(cap.normal().normalize())
                })
                .sum::<f64>()
                / 2.0;
            assert!(
                (area - 2.0).abs() < 1.0e-6,
                "positive native outer contour area"
            );
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
            assert!(points.iter().all(|point| (point.y - 2.0).abs() < 1.0e-6));
        }
        extents.sort_by(|first, second| first.0.total_cmp(&second.0));
        for ((low, high), (expected_low, expected_high)) in
            extents.into_iter().zip([(0.0, 2.0), (4.0, 6.0)])
        {
            assert!((low - expected_low).abs() < 1.0e-6 && (high - expected_high).abs() < 1.0e-6);
        }
    }
    assert_eq!(
        snapshot,
        serde_json::to_vec(&source.boundaries()[0].compress()).expect("immutable source")
    );
    assert_eq!(
        identities,
        source
            .face_iter()
            .map(|face| (face.id(), face.stable_id(), face.orientation()))
            .collect::<Vec<_>>()
    );
}
