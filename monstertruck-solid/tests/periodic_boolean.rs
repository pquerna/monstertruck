//! Native periodic chart regressions, constructed without STEP or an exporter.

use monstertruck_meshing::prelude::*;
use monstertruck_modeling::*;

const HEIGHT: f64 = 3.0;
const BOOLEAN_TOLERANCE: f64 = 1.0e-3;
const MESH_TOLERANCE: f64 = 2.0e-3;

fn cylinder_shell(radius: f64) -> (Shell, Wire, Wire) {
    let bottom: Wire = primitive::circle(
        Point3::new(radius, 0.0, 0.0),
        Point3::origin(),
        Vector3::unit_z(),
        2,
    );
    let top = builder::translated(&bottom, Vector3::new(0.0, 0.0, HEIGHT));
    let seam = builder::line(bottom.front_vertex().unwrap(), top.front_vertex().unwrap());
    let mut side = top.inverse();
    side.push_back(seam.inverse());
    side.extend(bottom.iter().cloned());
    side.push_back(seam);
    let surface = Surface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
        Curve::Line(Line(
            Point3::new(radius, 0.0, HEIGHT),
            Point3::new(radius, 0.0, 0.0),
        )),
        Point3::origin(),
        Vector3::unit_z(),
    )));
    let lateral = Face::try_new(vec![side], surface).unwrap();
    let top_face: Face = builder::try_attach_plane(vec![top.clone()]).unwrap();
    let bottom_face: Face = builder::try_attach_plane(vec![bottom.clone()]).unwrap();
    (
        vec![lateral, top_face, bottom_face.inverse()].into(),
        top,
        bottom,
    )
}

fn through_hole(radius: f64) -> Solid {
    let (hole, top_circle, bottom_circle) = cylinder_shell(radius);
    let mut shell = primitive::cuboid(BoundingBox::from_iter([
        Point3::new(-2.0, -2.0, 0.0),
        Point3::new(2.0, 2.0, HEIGHT),
    ]))
    .into_boundaries()
    .pop()
    .unwrap();
    for face in shell.face_iter_mut() {
        let top = face
            .edge_iter()
            .all(|edge| (edge.front().point().z - HEIGHT).abs() < 1.0e-12);
        let bottom = face
            .edge_iter()
            .all(|edge| edge.front().point().z.abs() < 1.0e-12);
        if top {
            face.try_add_boundary(top_circle.inverse()).unwrap();
        } else if bottom {
            face.try_add_boundary(bottom_circle.clone()).unwrap();
        }
    }
    shell.push(hole[0].inverse());
    Solid::try_new(vec![shell]).unwrap()
}

fn clip(oblique: bool) -> (Solid, Vector3) {
    let box_solid = primitive::cuboid(BoundingBox::from_iter([
        Point3::new(-3.0, -3.0, 0.0),
        Point3::new(3.0, 3.0, HEIGHT),
    ]));
    let (a, b) = if oblique { (0.1, 0.05) } else { (0.0, 0.0) };
    let transform = Matrix4::from_cols(
        Vector4::new(1.0, 0.0, -a, 0.0),
        Vector4::new(0.0, 1.0, -b, 0.0),
        Vector4::new(0.0, 0.0, 1.0, 0.0),
        Vector4::new(0.0, 0.0, HEIGHT / 2.0, 1.0),
    );
    (
        builder::transformed(&box_solid, transform),
        Vector3::new(a, b, 1.0),
    )
}

fn source_snapshot(solid: &Solid) -> impl PartialEq + std::fmt::Debug {
    let geometry = serde_json::to_vec(&solid.boundaries()[0].compress()).unwrap();
    let identities: Vec<_> = solid.boundaries()[0]
        .iter()
        .map(|face| {
            let boundaries: Vec<_> = face
                .absolute_boundaries()
                .iter()
                .map(|wire| {
                    wire.iter()
                        .map(|edge| {
                            (
                                edge.id(),
                                edge.stable_id(),
                                edge.orientation(),
                                edge.front().id(),
                                edge.front().stable_id(),
                                edge.back().id(),
                                edge.back().stable_id(),
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            (face.id(), face.stable_id(), face.orientation(), boundaries)
        })
        .collect();
    (geometry, identities)
}

fn consistency_witness(solid: &Solid) -> String {
    for (face_index, face) in solid.boundaries()[0].iter().enumerate() {
        let surface = face.surface();
        let failure = face.edge_iter().find_map(|edge| {
            let endpoints = edge.is_geometric_consistent();
            edge.with_curve(|curve| {
                let included = surface.include(curve);
                if endpoints && included {
                    return None;
                }
                let (start, end) = curve.range_tuple();
                let actual_samples: Vec<_> = (0..=8).map(|index| {
                    let point = curve.evaluate(start + (end - start) * index as f64 / 8.0);
                    surface.search_parameter(point, None, 100)
                        .map(|(u, v)| point.distance(surface.evaluate(u, v)))
                }).collect();
                Some(format!(
                    "face={face_index}, carrier={:?}, edge={:?}, curve={:?}, endpoints={endpoints}, included={included}, actual_curve_carrier_residuals={actual_samples:?}",
                    std::mem::discriminant(&surface), edge.id(), std::mem::discriminant(curve),
                ))
            })
        });
        if let Some(failure) = failure {
            return failure;
        }
    }
    "no failing edge/carrier incidence found".to_owned()
}

fn assert_upper_material(source: &Solid, radius: f64, hole: bool, oblique: bool) {
    assert!(source.is_geometric_consistent());
    let before = source_snapshot(source);
    let (clip, normal) = clip(oblique);
    let result = monstertruck_solid::and(source, &clip, BOOLEAN_TOLERANCE)
        .expect("closed periodic chart divides into the actual upper material");
    assert!(
        result.is_geometric_consistent(),
        "{}",
        consistency_witness(&result)
    );
    let shell = &result.boundaries()[0];
    shell.check_solid_boundary().unwrap();
    for edge in shell.edge_iter() {
        assert!(edge.is_geometric_consistent());
        edge.with_curve(|curve| {
            let (start, end) = curve.range_tuple();
            for fraction in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let point = curve.evaluate(start + fraction * (end - start));
                assert!(
                    normal.dot(point.to_vec()) >= HEIGHT / 2.0 - BOOLEAN_TOLERANCE,
                    "the old lower circle/rectangle must not survive as an open side"
                );
            }
        });
    }
    let caps: Vec<_> = shell
        .iter()
        .filter(|face| {
            matches!(face.surface(), Surface::Plane(_))
                && face.edge_iter().all(|edge| {
                    (normal.dot(edge.front().point().to_vec()) - HEIGHT / 2.0).abs()
                        < BOOLEAN_TOLERANCE
                })
        })
        .collect();
    assert_eq!(caps.len(), 1);
    assert_eq!(
        caps[0].absolute_boundaries().len(),
        if hole { 2 } else { 1 }
    );
    assert!(
        caps[0]
            .oriented_surface()
            .normal(0.0, 0.0)
            .dot(normal.normalize())
            < -0.999
    );
    let circular_loops: Vec<_> = caps[0]
        .absolute_boundaries()
        .iter()
        .filter(|wire| {
            wire.iter().all(|edge| {
                let point = edge.front().point();
                (point.x.hypot(point.y) - radius).abs() < BOOLEAN_TOLERANCE
            })
        })
        .collect();
    assert_eq!(circular_loops.len(), 1);
    assert!(circular_loops[0].is_closed());
    for edge in circular_loops[0].iter() {
        edge.with_curve(|curve| {
            let (start, end) = curve.range_tuple();
            for fraction in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let point = curve.evaluate(start + fraction * (end - start));
                assert!((point.x.hypot(point.y) - radius).abs() < BOOLEAN_TOLERANCE);
                assert!((normal.dot(point.to_vec()) - HEIGHT / 2.0).abs() < BOOLEAN_TOLERANCE);
            }
        });
    }
    let laterals: Vec<_> = shell
        .iter()
        .filter(|face| matches!(face.surface(), Surface::RevolutionSurface(_)))
        .collect();
    assert_eq!(laterals.len(), 1);
    let surface = laterals[0].oriented_surface();
    let point = Point3::new(radius, 0.0, HEIGHT - 0.5);
    let (u, v) = surface.search_parameter(point, None, 100).unwrap();
    let radial_normal = surface.normal(u, v).dot(Vector3::unit_x());
    assert!(if hole {
        radial_normal < -0.999
    } else {
        radial_normal > 0.999
    });

    let pi = std::f64::consts::PI;
    let section_area = if hole {
        16.0 - pi * radius * radius
    } else {
        pi * radius * radius
    };
    // The centered material has zero first x/y moments, so both clip planes
    // retain exactly half its volume. Tessellation is only an independent audit.
    let expected_volume = section_area * HEIGHT / 2.0;
    let surface_area_bound = if hole {
        96.0 + 2.0 * pi * radius * HEIGHT
    } else {
        2.0 * pi * radius * HEIGHT + 2.0 * pi * radius * radius * normal.magnitude()
    };
    let mesh = result.triangulation(MESH_TOLERANCE).to_polygon();
    assert!(mesh.faces().triangle_iter().count() > 0);
    let budget = surface_area_bound * MESH_TOLERANCE
        + 4.0 * pi * MESH_TOLERANCE.powi(2)
        + 4.0 / 3.0 * pi * MESH_TOLERANCE.powi(3);
    assert!((mesh.volume() - expected_volume).abs() <= budget);
    assert_eq!(
        source_snapshot(source),
        before,
        "AND never mutates source geometry or identity"
    );
}

#[test]
fn periodic_cylinder_and_preserves_closed_material_horizontal_and_oblique() {
    let (shell, _, _) = cylinder_shell(2.0);
    let source = Solid::try_new(vec![shell]).unwrap();
    for oblique in [false, true] {
        assert_upper_material(&source, 2.0, false, oblique);
    }
}

#[test]
fn periodic_hole_and_preserves_inward_wall_horizontal_and_oblique() {
    let source = through_hole(0.6);
    for oblique in [false, true] {
        assert_upper_material(&source, 0.6, true, oblique);
    }
}

#[test]
fn placed_low_layer_periodic_and_retains_transverse_caps_and_inward_hole_wall() {
    for hole in [false, true] {
        let radius = if hole { 0.6 } else { 2.0 };
        let source = if hole {
            through_hole(radius)
        } else {
            let (shell, _, _) = cylinder_shell(radius);
            Solid::try_new(vec![shell]).unwrap()
        };
        let translation = if hole {
            Vector3::new(80.0, 0.0, 0.0)
        } else {
            Vector3::new(40.0, 20.0, 0.0)
        };
        let source = builder::translated(&source, translation);
        let before = source_snapshot(&source);
        for height in [0.1, 0.3] {
            let clip: Solid = primitive::cuboid(BoundingBox::from_iter([
                Point3::new(-3.0, -3.0, height) + translation,
                Point3::new(3.0, 3.0, HEIGHT + 1.0) + translation,
            ]));
            let result = monstertruck_solid::and(&source, &clip, BOOLEAN_TOLERANCE)
                .expect("placed low layer is a transverse native periodic intersection");
            assert!(
                result.is_geometric_consistent(),
                "{}",
                consistency_witness(&result)
            );
            assert_eq!(result.boundaries().len(), 1);
            let shell = &result.boundaries()[0];
            shell.check_solid_boundary().unwrap();
            let caps: Vec<_> = shell
                .iter()
                .filter(|face| {
                    matches!(face.surface(), Surface::Plane(_))
                        && face
                            .edge_iter()
                            .all(|edge| (edge.front().point().z - height).abs() < BOOLEAN_TOLERANCE)
                })
                .collect();
            assert_eq!(caps.len(), 1);
            assert_eq!(
                caps[0].absolute_boundaries().len(),
                if hole { 2 } else { 1 }
            );
            assert!(
                caps[0]
                    .oriented_surface()
                    .normal(0.0, 0.0)
                    .dot(Vector3::unit_z())
                    < -0.999
            );
            let circles: Vec<_> = caps[0]
                .absolute_boundaries()
                .iter()
                .filter(|wire| {
                    wire.iter().all(|edge| {
                        let point = edge.front().point() - translation;
                        (point.x.hypot(point.y) - radius).abs() < BOOLEAN_TOLERANCE
                    })
                })
                .collect();
            assert_eq!(circles.len(), 1);
            assert!(circles[0].is_closed());
            for edge in circles[0].iter() {
                edge.with_curve(|curve| {
                    let (start, end) = curve.range_tuple();
                    for index in 0..=8 {
                        let point = curve.evaluate(start + (end - start) * index as f64 / 8.0)
                            - translation;
                        assert!((point.x.hypot(point.y) - radius).abs() < BOOLEAN_TOLERANCE);
                        assert!((point.z - height).abs() < BOOLEAN_TOLERANCE);
                    }
                });
            }
            let lateral = shell
                .iter()
                .find(|face| matches!(face.surface(), Surface::RevolutionSurface(_)))
                .unwrap()
                .oriented_surface();
            let point = Point3::new(radius, 0.0, HEIGHT - 0.5) + translation;
            let (u, v) = lateral.search_parameter(point, None, 100).unwrap();
            let radial = lateral.normal(u, v).dot(Vector3::unit_x());
            assert!(if hole {
                radial < -0.999
            } else {
                radial > 0.999
            });
            assert_eq!(source_snapshot(&source), before);
        }
    }
}
