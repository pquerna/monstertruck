// SPDX-FileCopyrightText: 2026 Spiling contributors
// SPDX-License-Identifier: Apache-2.0
//! Original operator planes on the unchanged analytic STEP source.

use monstertruck_io::step::load::{Table, step_geometry::Curve3D};
use monstertruck_meshing::prelude::*;
use monstertruck_modeling::*;
use monstertruck_solid::{
    BooleanPreparationOptions, BooleanWorkBudget, BooleanWorkLimits, PreparedBooleanSource,
    and_prepared, prepare_boolean_source_counted_with_options,
};
use std::f64::consts::PI;
use std::sync::atomic::AtomicBool;

const TOLERANCE_MM: f64 = 1.0e-5;
const SEED_TOLERANCE_MM: f64 = 0.005;

fn limits() -> BooleanWorkLimits {
    BooleanWorkLimits {
        max_faces: 256,
        max_edges: 2048,
        max_mesh_vertices: 200_000,
        max_mesh_triangles: 200_000,
        max_triangle_pairs: 10_000_000,
        max_interference_segments: 200_000,
        max_output_faces: 256,
        max_output_edges: 4096,
        max_prepared_bytes: 32 * 1024 * 1024,
        max_source_prepared_bytes: 28 * 1024 * 1024,
        max_other_prepared_bytes: 4 * 1024 * 1024,
        max_intermediate_prepared_bytes: 4 * 1024 * 1024,
    }
}

fn prepare_operator_source<'a>(
    source: &'a Solid,
    cancel: &AtomicBool,
) -> PreparedBooleanSource<'a, Curve, Surface> {
    let budget = BooleanWorkBudget::new(usize::MAX);
    prepare_boolean_source_counted_with_options(
        source,
        BooleanPreparationOptions {
            geometric_tolerance: TOLERANCE_MM,
            seed_tolerance: SEED_TOLERANCE_MM,
        },
        limits(),
        cancel,
        &budget,
    )
    .expect("bounded original source with explicit candidate seed policy")
}

fn modeling_curve(curve: &Curve3D) -> Option<Curve> {
    match curve {
        Curve3D::IntersectionCurve(curve) => modeling_curve(curve.leader()),
        Curve3D::SurfaceCurve(curve) => modeling_curve(curve.leader()),
        curve => Curve::try_from(curve).ok(),
    }
}

fn source() -> Solid { source_from_step(include_str!("fixtures/operator-cylinder.step"), 4) }

fn source_from_step(step: &str, expected_faces: usize) -> Solid {
    let table = Table::from_step(step).expect("original source STEP");
    let (compressed, reports) = table
        .to_compressed_solid_reported(&table.manifold_solid_brep[&16])
        .expect("lossless original topology");
    assert!(reports.iter().all(|report| report.is_lossless()));
    assert_eq!(compressed.boundaries[0].faces.len(), expected_faces);
    let slots = compressed.boundaries[0]
        .faces
        .iter()
        .map(|face| (face.surface.clone(), face.orientation))
        .collect::<Vec<_>>();
    let native = monstertruck_topology::Solid::extract(compressed).expect("native source");
    assert!(
        native.boundaries()[0]
            .iter()
            .zip(slots)
            .all(|(face, slot)| { face.surface() == slot.0 && face.orientation() == slot.1 })
    );
    let result = native
        .try_mapped(
            |point| Some(*point),
            modeling_curve,
            |surface| Surface::try_from(surface).ok(),
        )
        .expect("exact analytic mapping");
    assert!(result.is_geometric_consistent());
    result.boundaries()[0]
        .check_solid_boundary()
        .expect("original manifold");
    result
}

fn frame(normal: Vector3) -> (Vector3, Vector3, Vector3) {
    let normal = normal.normalize();
    let least = (1..3).fold(0, |least, axis| {
        if normal[axis].abs() < normal[least].abs() {
            axis
        } else {
            least
        }
    });
    let mut axis = Vector3::new(0.0, 0.0, 0.0);
    axis[least] = 1.0;
    let x = axis.cross(normal).normalize();
    (x, normal.cross(x), normal)
}

fn clip(source: &Solid, normal: Vector3, origin: Point3) -> Solid {
    let (x, y, normal) = frame(normal);
    let bounds = bounding::certified_solid_bounding_box(source).expect("certified native bounds");
    let mut min = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut max = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    (0..8).for_each(|mask| {
        let point = Point3::from(std::array::from_fn::<_, 3, _>(|axis| {
            if mask & (1 << axis) == 0 {
                bounds.min()[axis]
            } else {
                bounds.max()[axis]
            }
        }));
        let delta = point - origin;
        let local = Point3::new(x.dot(delta), y.dot(delta), normal.dot(delta));
        (0..3).for_each(|axis| {
            min[axis] = min[axis].min(local[axis]);
            max[axis] = max[axis].max(local[axis]);
        });
    });
    min = Point3::new(min.x - 1.0, min.y - 1.0, 0.0);
    max += Vector3::new(1.0, 1.0, 1.0);
    let clip = primitive::cuboid(BoundingBox::from_iter([min, max]));
    builder::transformed(
        &clip,
        Matrix4::from_cols(
            x.extend(0.0),
            y.extend(0.0),
            normal.extend(0.0),
            origin.to_vec().extend(1.0),
        ),
    )
}

fn assert_cut(
    source: &Solid,
    normal: Vector3,
    origin: Point3,
    prepared: Option<&PreparedBooleanSource<'_, Curve, Surface>>,
) -> f64 {
    let before = serde_json::to_vec(&source.boundaries()[0].compress()).expect("source snapshot");
    let identities = source.boundaries()[0]
        .iter()
        .map(|face| {
            (
                face.id(),
                face.stable_id(),
                face.orientation(),
                face.with_surface(|surface| serde_json::to_value(surface))
                    .expect("original native carrier"),
            )
        })
        .collect::<Vec<_>>();
    let clip = clip(source, normal, origin);
    let cap_carrier = clip
        .face_iter()
        .find_map(|face| {
            let Surface::Plane(plane) = face.surface() else {
                return None;
            };
            ((plane.origin() - origin).dot(normal).abs() < 1.0e-10
                && plane.normal().normalize().cross(normal).magnitude() < 1.0e-10)
                .then_some(plane)
        })
        .expect("exact clip cap carrier");
    let cut = if let Some(prepared) = prepared {
        and_prepared(prepared, &clip, &AtomicBool::new(false))
    } else {
        let cancel = AtomicBool::new(false);
        let prepared = prepare_operator_source(source, &cancel);
        and_prepared(&prepared, &clip, &cancel)
    }
    .expect("original transverse AND");
    assert_eq!(cut.boundaries().len(), 1);
    cut.boundaries()[0]
        .check_solid_boundary()
        .expect("release manifold checks");
    assert!(cut.is_geometric_consistent());
    assert_eq!(
        before,
        serde_json::to_vec(&source.boundaries()[0].compress()).expect("unchanged source")
    );
    assert_eq!(
        identities,
        source.boundaries()[0]
            .iter()
            .map(|face| {
                (
                    face.id(),
                    face.stable_id(),
                    face.orientation(),
                    face.with_surface(|surface| serde_json::to_value(surface))
                        .expect("unchanged native carrier"),
                )
            })
            .collect::<Vec<_>>()
    );
    cut.face_iter().for_each(|face| {
        let carrier = face.surface();
        face.edge_iter().for_each(|edge| {
            edge.with_curve(|curve| {
                let (start, end) = curve.range_tuple();
                (0..=8).for_each(|index| {
                    let point = curve.evaluate(start + (end - start) * index as f64 / 8.0);
                    assert!((point - origin).dot(normal) >= -TOLERANCE_MM);
                    let uv = carrier
                        .search_parameter(point, None, 100)
                        .expect("original carrier residual");
                    assert!(point.distance(carrier.evaluate(uv.0, uv.1)) <= 1.0e-6);
                });
            })
        });
    });
    let caps = cut
        .face_iter()
        .filter(|face| matches!(face.surface(), Surface::Plane(plane) if plane == cap_carrier))
        .collect::<Vec<_>>();
    assert_eq!(caps.len(), 1);
    let has_hole = source
        .face_iter()
        .any(|face| !face.orientation() && matches!(face.surface(), Surface::RevolutionSurface(_)));
    assert_eq!(
        caps[0].absolute_boundaries().len(),
        if has_hole { 2 } else { 1 }
    );
    for (index, wire) in caps[0].absolute_boundaries().iter().enumerate() {
        let points = wire
            .iter()
            .flat_map(|edge| {
                edge.with_curve(|curve| {
                    let (start, end) = curve.range_tuple();
                    (0..64)
                        .map(|index| {
                            let fraction = index as f64 / 64.0;
                            let fraction = if edge.orientation() {
                                fraction
                            } else {
                                1.0 - fraction
                            };
                            curve.evaluate(start + (end - start) * fraction)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        let area = points
            .iter()
            .zip(points.iter().cycle().skip(1))
            .map(|(a, b)| (*a - origin).cross(*b - origin).dot(cap_carrier.normal()))
            .sum::<f64>();
        assert!(area.is_finite() && if index == 0 { area > 0.0 } else { area < 0.0 });
    }
    let mesh = cut.boundaries()[0].triangulation(0.005).to_polygon();
    let volume = mesh.volume();
    assert!(volume.is_finite() && volume > 0.0);
    volume
}

fn original_queries(world_z: f64) {
    let source = source();
    let angle = 1.0_f64.to_radians();
    let transverse = angle.sin() / 2.0_f64.sqrt();
    let normal = Vector3::new(-transverse, transverse, angle.cos());
    let origin = Point3::from_vec(normal * (world_z - 5.0 * angle.sin()));
    let expected = assert_cut(&source, normal, origin, None);
    let cancel = AtomicBool::new(false);
    let prepared = prepare_operator_source(&source, &cancel);
    let (qw, qx, qy) = (
        0.9999619230641713,
        0.006170592427165337,
        0.006170592427165337,
    );
    let equivalent_normal = Vector3::new(
        -2.0 * qy * qw,
        2.0 * qx * qw,
        1.0 - 2.0 * (qx * qx + qy * qy),
    );
    assert!((normal - equivalent_normal).magnitude() < 1.0e-14);
    let (x, y, _) = frame(normal);
    for (query_normal, query_origin) in [
        (equivalent_normal, origin),
        (normal, origin + 3.25 * x - 2.5 * y),
    ] {
        let volume = assert_cut(&source, query_normal, query_origin, Some(&prepared));
        assert!((volume - expected).abs() < 0.005);
    }
    let lower = assert_cut(&source, -normal, origin, Some(&prepared));
    let mesh_tolerance = 0.005_f64;
    let boundary_area = 2.0 * PI * 5.0 * 8.0 + 4.0 * PI * 25.0;
    let volume_budget = boundary_area * mesh_tolerance + 4.0 / 3.0 * PI * mesh_tolerance.powi(3);
    assert!((lower + expected - PI * 25.0 * 8.0).abs() < volume_budget);
}

#[test]
fn original_partial_bottom_plane_and_equivalent_representations() { original_queries(0.1); }

#[test]
fn original_mid_plane_and_equivalent_representations() { original_queries(4.0); }

#[test]
fn original_partial_top_plane_and_equivalent_representations() { original_queries(8.0); }

#[test]
fn original_quaternion_and_bed_translation_keep_partial_bottom_mid_and_top_material() {
    let source = source();
    let (qw, qx, qy) = (
        0.9999619230641713,
        0.006170592427165337,
        0.006170592427165337,
    );
    let transform = Matrix4::from_cols(
        Vector4::new(1.0 - 2.0 * qy * qy, 2.0 * qx * qy, -2.0 * qw * qy, 0.0),
        Vector4::new(2.0 * qx * qy, 1.0 - 2.0 * qx * qx, 2.0 * qw * qx, 0.0),
        Vector4::new(
            2.0 * qw * qy,
            -2.0 * qw * qx,
            1.0 - 2.0 * (qx * qx + qy * qy),
            0.0,
        ),
        Vector4::new(40.0, 40.0, 0.08726203218641757, 1.0),
    );
    let posed = builder::transformed(&source, transform);
    let cancel = AtomicBool::new(false);
    let prepared = prepare_operator_source(&posed, &cancel);
    for world_z in [0.1, 4.0, 8.0] {
        let retained = assert_cut(
            &posed,
            Vector3::unit_z(),
            Point3::new(0.0, 0.0, world_z),
            Some(&prepared),
        );
        assert!(retained < PI * 25.0 * 8.0);
    }
}

#[test]
fn original_tilted_hole_keeps_inward_walls_and_opposite_native_cap_orientation() {
    let source = source_from_step(include_str!("fixtures/operator-through-hole.step"), 8);
    let angle = 1.0_f64.to_radians();
    let transverse = angle.sin() / 2.0_f64.sqrt();
    let normal = Vector3::new(-transverse, transverse, angle.cos());
    let origin = Point3::from_vec(normal * (4.0 - 5.0 * angle.sin()));
    let cancel = AtomicBool::new(false);
    let prepared = prepare_operator_source(&source, &cancel);
    assert_cut(&source, normal, origin, Some(&prepared));
    let clip = clip(&source, normal, origin);
    let cut = and_prepared(&prepared, &clip, &cancel).expect("hole material AND");
    for face in cut
        .face_iter()
        .filter(|face| matches!(face.surface(), Surface::RevolutionSurface(_)))
    {
        let surface = face.oriented_surface();
        let point = face
            .edge_iter()
            .next()
            .expect("hole wall boundary")
            .front()
            .point();
        let uv = surface
            .search_parameter(point, None, 100)
            .expect("hole carrier");
        let radial = Vector3::new(point.x - 10.0, point.y - 10.0, 0.0);
        assert!(surface.normal(uv.0, uv.1).dot(radial) < 0.0);
    }
}
