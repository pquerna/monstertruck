//! Native periodic chart regressions, constructed without STEP or an exporter.

use monstertruck_meshing::prelude::*;
use monstertruck_modeling::*;
use monstertruck_solid::{
    BooleanCoedgeOrigin, BooleanEdgeDerivation, BooleanOperand, BooleanPreparationOptions,
    BooleanWorkBudget, BooleanWorkLimits, and_prepared_with_provenance,
    prepare_boolean_source_counted_with_options,
};
use std::sync::atomic::AtomicBool;

const HEIGHT: f64 = 3.0;
const BOOLEAN_TOLERANCE: f64 = 1.0e-3;
const MESH_TOLERANCE: f64 = 2.0e-3;

fn cylinder_shell(radius: f64, height: f64, profile_step: f64) -> (Shell, Wire, Wire) {
    let bottom: Wire = primitive::circle(
        Point3::new(radius, 0.0, 0.0),
        Point3::origin(),
        Vector3::unit_z(),
        2,
    );
    let top = builder::translated(&bottom, Vector3::new(0.0, 0.0, height));
    let seam = builder::line(bottom.front_vertex().unwrap(), top.front_vertex().unwrap());
    let mut side = top.inverse();
    side.push_back(seam.inverse());
    side.extend(bottom.iter().cloned());
    side.push_back(seam);
    let surface = Surface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
        Curve::Line(Line(
            Point3::new(radius, 0.0, height),
            Point3::new(radius, 0.0, height - profile_step),
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
    let (hole, top_circle, bottom_circle) = cylinder_shell(radius, HEIGHT, HEIGHT);
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
    let (shell, _, _) = cylinder_shell(2.0, HEIGHT, HEIGHT);
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
            let (shell, _, _) = cylinder_shell(radius, HEIGHT, HEIGHT);
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

#[test]
fn near_terminal_open_cut_attaches_both_rotated_periodic_seam_uses_with_original_genealogy() {
    let angle = 1.0_f64.to_radians();
    let (sine, cosine) = angle.sin_cos();
    let normal = Vector3::new(sine, 0.0, cosine);
    let height = 8.0 * cosine + 5.0 * sine - 0.00099;
    let clip = primitive::cuboid(BoundingBox::from_iter([
        Point3::new(-6.0, -6.0, 0.0),
        Point3::new(6.0, 6.0, 2.0),
    ]));
    let clip = builder::transformed(
        &clip,
        Matrix4::from_cols(
            Vector4::new(cosine, 0.0, -sine, 0.0),
            Vector4::new(0.0, 1.0, 0.0, 0.0),
            normal.extend(0.0),
            (normal * height).extend(1.0),
        ),
    );
    let limits = BooleanWorkLimits {
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
    };
    for (phase, profile_step) in [0.0, 0.07]
        .into_iter()
        .flat_map(|phase| [1.0, 8.0, 64.0].map(move |step| (phase, step)))
    {
        // Only the native axial gauge and periodic chart phase change.
        // Original bounded material and the physical cutting plane do not.
        let (shell, _, _) = cylinder_shell(5.0, 8.0, profile_step);
        let source = Solid::try_new(vec![shell]).expect("independent periodic cylinder");
        let source = builder::transformed(&source, Matrix4::from_angle_z(Rad(phase)));
        let before = source_snapshot(&source);
        let original_side = &source.boundaries()[0][0];
        let seam = original_side.absolute_boundaries()[0][2].id();
        let cancel = AtomicBool::new(false);
        let budget = BooleanWorkBudget::new(10_000_000);
        let prepared = prepare_boolean_source_counted_with_options(
            &source,
            BooleanPreparationOptions {
                geometric_tolerance: 1.0e-5,
                seed_tolerance: 0.005,
            },
            limits,
            &cancel,
            &budget,
        )
        .expect("unchanged original geometry with independent seed policy");
        let result = and_prepared_with_provenance(&prepared, &clip, &cancel)
            .expect("open near-terminal cut crosses the actual periodic seam");
        assert_eq!(result.solid.boundaries().len(), 1);
        let shell = &result.solid.boundaries()[0];
        shell
            .check_solid_boundary()
            .expect("closed oriented native material");
        assert!(
            result.solid.is_geometric_consistent(),
            "{}",
            consistency_witness(&result.solid)
        );
        for face in shell {
            for wire in face.absolute_boundaries() {
                assert!(wire.is_closed());
                assert!(
                    wire.is_simple(),
                    "the selected seam pieces have distinct material corners"
                );
            }
            let carrier = face.surface();
            for edge in face.edge_iter() {
                edge.with_curve(|curve| {
                    let (start, end) = curve.range_tuple();
                    for fraction in [0.0, 0.25, 0.5, 0.75, 1.0] {
                        let point = curve.evaluate(start + fraction * (end - start));
                        assert!(normal.dot(point.to_vec()) >= height - 1.0e-5);
                        assert!(point.z <= 8.0 + 1.0e-6);
                        assert!(point.x.hypot(point.y) <= 5.0 + 1.0e-6);
                        let (u, v) = carrier.search_parameter(point, None, 100).unwrap();
                        assert!(point.near(&carrier.evaluate(u, v)));
                    }
                });
            }
        }
        let caps: Vec<_> = shell
            .iter()
            .filter(|face| {
                face.edge_iter()
                    .all(|edge| (normal.dot(edge.front().point().to_vec()) - height).abs() < 1.0e-6)
            })
            .collect();
        assert_eq!(caps.len(), 1);
        assert!(caps[0].oriented_surface().normal(0.0, 0.0).dot(normal) < -0.999);
        for origin in &result.face_origins {
            let operand = if origin.source.operand == BooleanOperand::Source {
                &source
            } else {
                &clip
            };
            let original =
                &operand.boundaries()[origin.source.shell_index][origin.source.face_index];
            let output = shell
                .iter()
                .find(|face| face.id() == origin.output_face)
                .unwrap();
            assert_eq!(origin.source.face_id, original.id());
            assert_eq!(output.orientation(), original.orientation());
            assert_eq!(
                serde_json::to_value(output.surface()).unwrap(),
                serde_json::to_value(original.surface()).unwrap(),
            );
        }
        assert_eq!(
            result
                .face_origins
                .iter()
                .filter(|origin| {
                    origin.source.operand == BooleanOperand::Source
                        && origin.source.face_id == original_side.id()
                })
                .count(),
            2,
            "both original material sides of the seam survive every native axial gauge",
        );
        let mut seam_uses = [0; 2];
        let mut shared_seam = None;
        for coedge in &result.coedge_origins {
            let BooleanCoedgeOrigin::Original {
                face,
                wire_index,
                coedge_index,
                edge_id,
                orientation,
                derivation,
            } = &coedge.source
            else {
                assert!(!matches!(
                    coedge.source,
                    BooleanCoedgeOrigin::AmbiguousOriginalUse { .. }
                ));
                continue;
            };
            let operand = if face.operand == BooleanOperand::Source {
                &source
            } else {
                &clip
            };
            let original = &operand.boundaries()[face.shell_index][face.face_index];
            assert_eq!(face.face_id, original.id());
            let original_edge = &original.absolute_boundaries()[*wire_index][*coedge_index];
            assert_eq!(*edge_id, original_edge.id());
            assert_eq!(*orientation, original_edge.orientation());
            assert_eq!(coedge.orientation, *orientation);
            let output = shell
                .iter()
                .find(|face| face.id() == coedge.output_face)
                .unwrap();
            let output_edge = &output.absolute_boundaries()[coedge.wire_index][coedge.coedge_index];
            assert_eq!(output_edge.id(), coedge.output_edge);
            let output_domain = output_edge.with_curve(ParametricCurve::parameter_range);
            let recorded_domain = match &result.edge_derivations[*derivation] {
                BooleanEdgeDerivation::Source { domain, .. } => *domain,
                BooleanEdgeDerivation::Cut { child_domain, .. } => *child_domain,
            };
            assert_eq!(output_domain, recorded_domain);
            let mut node = *derivation;
            let mut cut = false;
            loop {
                match &result.edge_derivations[node] {
                    BooleanEdgeDerivation::Source {
                        edge_id: root,
                        domain,
                    } => {
                        assert_eq!(root, edge_id);
                        assert_eq!(
                            *domain,
                            original_edge.with_curve(ParametricCurve::parameter_range)
                        );
                        break;
                    }
                    BooleanEdgeDerivation::Cut {
                        parent,
                        parent_domain,
                        parameter,
                        ..
                    } => {
                        assert!(*parent < node);
                        assert!(parameter.is_finite());
                        let recorded_parent = match &result.edge_derivations[*parent] {
                            BooleanEdgeDerivation::Source { domain, .. } => *domain,
                            BooleanEdgeDerivation::Cut { child_domain, .. } => *child_domain,
                        };
                        assert_eq!(*parent_domain, recorded_parent);
                        cut = true;
                        node = *parent;
                    }
                }
            }
            if face.operand == BooleanOperand::Source && *edge_id == seam {
                assert_eq!(face.face_id, original_side.id());
                assert!(
                    cut,
                    "both directed original seam uses descend from the native cut"
                );
                seam_uses[usize::from(*orientation)] += 1;
                let selected = (coedge.output_edge, *derivation);
                if let Some(previous) = shared_seam {
                    assert_eq!(
                        selected, previous,
                        "opposite seam uses share one actual cut gauge"
                    );
                }
                shared_seam = Some(selected);
            }
        }
        assert_eq!(seam_uses, [1, 1]);
        assert_eq!(source_snapshot(&source), before);
    }
}

fn original_tensor_torus(independent_parameters: bool, pose: Matrix4) -> Solid {
    let circle = [
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 2.0, 2.0],
        [-2.0, 2.0, 2.0],
        [-4.0, 0.0, 4.0],
        [-4.0, -4.0, 4.0],
        [0.0, -8.0, 8.0],
        [8.0, -8.0, 8.0],
        [16.0, 0.0, 16.0],
    ];
    let knots = KnotVector::from(vec![
        0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0,
    ]);
    let tensor = |[x, y, w]: [f64; 3], [a, b, v]: [f64; 3]| {
        Vector4::new(
            (6.0 * v + 2.0 * a) * x,
            (6.0 * v + 2.0 * a) * y,
            2.0 * b * w,
            v * w,
        )
    };
    let surface = Surface::NurbsSurface(NurbsSurface::new(BsplineSurface::new(
        (knots.clone(), knots),
        circle
            .iter()
            .map(|&theta| circle.iter().map(|&phi| tensor(theta, phi)).collect())
            .collect(),
    )));
    let vertices = Vertex::from_points([
        Point3::new(8.0, 0.0, 0.0),
        Point3::new(-8.0, 0.0, 0.0),
        Point3::new(4.0, 0.0, 0.0),
    ]);
    let seam = |half: usize, meridian: bool| {
        let front = if half == 0 {
            0
        } else if meridian {
            2
        } else {
            1
        };
        let back = if half == 1 {
            0
        } else if meridian {
            2
        } else {
            1
        };
        let start = 2.0 * half as f64;
        let mut curve = NurbsCurve::new(BsplineCurve::new(
            KnotVector::from(vec![
                start,
                start,
                start,
                start + 1.0,
                start + 1.0,
                start + 2.0,
                start + 2.0,
                start + 2.0,
            ]),
            circle[4 * half..=4 * half + 4]
                .iter()
                .map(|&point| {
                    if meridian {
                        tensor(circle[0], point)
                    } else {
                        tensor(point, circle[0])
                    }
                })
                .collect(),
        ));
        if independent_parameters {
            curve.parameter_transform(2.0, 10.0);
        }
        Edge::new(&vertices[front], &vertices[back], Curve::NurbsCurve(curve))
    };
    let theta = [seam(0, false), seam(1, false)];
    let phi = [seam(0, true), seam(1, true)];
    let boundary = [
        theta[0].clone(),
        theta[1].clone(),
        phi[0].clone(),
        phi[1].clone(),
        theta[1].inverse(),
        theta[0].inverse(),
        phi[1].inverse(),
        phi[0].inverse(),
    ]
    .into_iter()
    .collect();
    let face =
        Face::try_new(vec![boundary], surface).expect("one original doubly periodic tensor face");
    let solid =
        Solid::try_new(vec![vec![face].into()]).expect("actual shared periodic seam identities");
    builder::transformed(&solid, pose)
}

fn closed_material_limits() -> BooleanWorkLimits {
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

fn assert_original_rational_torus_material(
    independent_parameters: bool,
    pose: Matrix4,
    height: f64,
) {
    let limits = closed_material_limits();
    let source = original_tensor_torus(independent_parameters, pose);
    let before = source_snapshot(&source);
    let original = &source.boundaries()[0][0];
    let equator = original.absolute_boundaries()[0]
        .iter()
        .take(2)
        .map(Edge::id)
        .collect::<Vec<_>>();
    let cancel = AtomicBool::new(false);
    let budget = BooleanWorkBudget::new(100_000_000);
    let prepared = prepare_boolean_source_counted_with_options(
        &source,
        BooleanPreparationOptions {
            geometric_tolerance: 1.0e-5,
            seed_tolerance: 0.005,
        },
        limits,
        &cancel,
        &budget,
    )
    .expect("original rational torus with unchanged independent seed policy");
    let clip: Solid = primitive::cuboid(BoundingBox::from_iter([
        Point3::new(-9.0, -9.0, height),
        Point3::new(9.0, 9.0, 3.0),
    ]));
    let result = and_prepared_with_provenance(&prepared, &clip, &cancel)
                .unwrap_or_else(|error| panic!(
                    "native torus material: height={height:?}, independent_parameters={independent_parameters}, pose={pose:?}, prepared_vertices={}, prepared_triangles={}, prepared_bytes={}: {error:?}",
                    prepared.retained_mesh_vertices(), prepared.retained_mesh_triangles(),
                    prepared.retained_bytes(),
                ));
    assert_eq!(result.solid.boundaries().len(), 1);
    let shell = &result.solid.boundaries()[0];
    shell
        .check_solid_boundary()
        .expect("closed oriented retained torus material");
    assert!(
        result.solid.is_geometric_consistent(),
        "{}",
        consistency_witness(&result.solid)
    );
    let caps = shell
        .iter()
        .filter(|face| {
            matches!(face.surface(), Surface::Plane(_))
                && face
                    .edge_iter()
                    .all(|edge| (edge.front().point().z - height).abs() < 1.0e-6)
        })
        .collect::<Vec<_>>();
    assert_eq!(caps.len(), 1);
    let cap = caps[0];
    assert_eq!(
        cap.absolute_boundaries().len(),
        2,
        "the toroidal void is not a filled disk"
    );
    assert!(cap.oriented_surface().normal(0.0, 0.0).z < -0.999);
    let radial = (4.0_f64 - height * height).sqrt();
    let mut inner = false;
    let mut outer = false;
    for wire in cap.absolute_boundaries() {
        assert!(wire.is_closed() && wire.is_simple());
        let radius = wire
            .front_vertex()
            .unwrap()
            .point()
            .to_vec()
            .truncate()
            .magnitude();
        inner |= (radius - (6.0 - radial)).abs() < 1.0e-6;
        outer |= (radius - (6.0 + radial)).abs() < 1.0e-6;
    }
    assert!(inner && outer);
    for edge in shell.edge_iter() {
        edge.with_curve(|curve| {
            let (front, back) = curve.range_tuple();
            for fraction in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let point = curve.evaluate(front + fraction * (back - front));
                assert!(point.z >= height - 1.0e-5 && point.z <= 2.0 + 1.0e-6);
                let radial = point.x.hypot(point.y) - 6.0;
                assert!((radial * radial + point.z * point.z - 4.0).abs() < 1.0e-5);
            }
        });
    }
    for origin in &result.face_origins {
        if origin.source.operand == BooleanOperand::Source {
            let output = shell
                .iter()
                .find(|face| face.id() == origin.output_face)
                .unwrap();
            assert_eq!(origin.source.face_id, original.id());
            assert_eq!(
                serde_json::to_value(output.surface()).unwrap(),
                serde_json::to_value(original.surface()).unwrap()
            );
        }
    }
    let mut retained_equator = 0;
    let mut generated_equator = 0;
    for coedge in &result.coedge_origins {
        assert!(!matches!(
            coedge.source,
            BooleanCoedgeOrigin::AmbiguousOriginalUse { .. }
        ));
        let output = shell
            .iter()
            .find(|face| face.id() == coedge.output_face)
            .unwrap();
        let edge = &output.absolute_boundaries()[coedge.wire_index][coedge.coedge_index];
        if let BooleanCoedgeOrigin::Original {
            face,
            wire_index,
            coedge_index,
            edge_id,
            orientation,
            derivation,
        } = &coedge.source
        {
            let operand = if face.operand == BooleanOperand::Source {
                &source
            } else {
                &clip
            };
            let source_face = &operand.boundaries()[face.shell_index][face.face_index];
            let source_edge = &source_face.absolute_boundaries()[*wire_index][*coedge_index];
            assert_eq!(face.face_id, source_face.id());
            assert_eq!(*edge_id, source_edge.id());
            assert_eq!(*orientation, source_edge.orientation());
            assert_eq!(coedge.orientation, *orientation);
            let domain = match &result.edge_derivations[*derivation] {
                BooleanEdgeDerivation::Source { domain, .. } => *domain,
                BooleanEdgeDerivation::Cut { child_domain, .. } => *child_domain,
            };
            assert_eq!(edge.with_curve(ParametricCurve::parameter_range), domain);
            let mut node = *derivation;
            loop {
                match &result.edge_derivations[node] {
                    BooleanEdgeDerivation::Source {
                        edge_id: root,
                        domain,
                    } => {
                        assert_eq!(root, edge_id);
                        assert_eq!(
                            *domain,
                            source_edge.with_curve(ParametricCurve::parameter_range)
                        );
                        break;
                    }
                    BooleanEdgeDerivation::Cut {
                        parent,
                        parent_domain,
                        parameter,
                        ..
                    } => {
                        assert!(*parent < node && parameter.is_finite());
                        let recorded = match &result.edge_derivations[*parent] {
                            BooleanEdgeDerivation::Source { domain, .. } => *domain,
                            BooleanEdgeDerivation::Cut { child_domain, .. } => *child_domain,
                        };
                        assert_eq!(*parent_domain, recorded);
                        node = *parent;
                    }
                }
            }
        }
        if height == 0.0 && equator.contains(&edge.id()) {
            if output.id() == cap.id() {
                assert!(
                    matches!(&coedge.source, BooleanCoedgeOrigin::GeneratedIntersection { face }
                            if face.operand == BooleanOperand::Other)
                );
                generated_equator += 1;
            } else {
                assert!(
                    matches!(&coedge.source, BooleanCoedgeOrigin::Original { face, edge_id, .. }
                            if face.operand == BooleanOperand::Source && edge_id == &edge.id())
                );
                retained_equator += 1;
            }
        }
    }
    if height == 0.0 {
        assert_eq!((retained_equator, generated_equator), (2, 2));
    }
    assert_eq!(source_snapshot(&source), before);
}

#[test]
fn original_rational_torus_closed_section_native_gauge_equator() {
    assert_original_rational_torus_material(false, Matrix4::identity(), 0.0);
}

#[test]
fn original_rational_torus_closed_section_native_gauge_terminal() {
    assert_original_rational_torus_material(false, Matrix4::identity(), 1.99901);
}

#[test]
fn original_rational_torus_closed_section_independent_gauge_equator() {
    assert_original_rational_torus_material(true, Matrix4::identity(), 0.0);
}

#[test]
fn original_rational_torus_closed_section_independent_gauge_terminal() {
    assert_original_rational_torus_material(true, Matrix4::identity(), 1.99901);
}

fn original_torus_quarter_turn() -> Matrix4 {
    Matrix4::from_cols(
        Vector4::new(0.0, 1.0, 0.0, 0.0),
        Vector4::new(-1.0, 0.0, 0.0, 0.0),
        Vector4::new(0.0, 0.0, 1.0, 0.0),
        Vector4::new(0.0, 0.0, 0.0, 1.0),
    )
}

#[test]
fn original_rational_torus_closed_section_quarter_turn_independent_gauge_equator() {
    assert_original_rational_torus_material(true, original_torus_quarter_turn(), 0.0);
}

#[test]
fn original_rational_torus_closed_section_quarter_turn_independent_gauge_terminal() {
    assert_original_rational_torus_material(true, original_torus_quarter_turn(), 1.99901);
}

fn original_quadrant_sphere() -> Solid {
    let radius = 5.0;
    let diagonal = radius / 2.0_f64.sqrt();
    let vertices = builder::vertices([(0.0, 0.0, -radius), (radius, 0.0, 0.0), (0.0, 0.0, radius)]);
    let profile: Wire = [
        builder::circle_arc(
            &vertices[0],
            &vertices[1],
            Point3::new(diagonal, 0.0, -diagonal),
        ),
        builder::circle_arc(
            &vertices[1],
            &vertices[2],
            Point3::new(diagonal, 0.0, diagonal),
        ),
    ]
    .into_iter()
    .collect();
    let mut shell = builder::revolve_wire::<Curve, Surface>(
        &profile,
        Point3::origin(),
        Vector3::unit_z(),
        builder::SweepAngle::Closed,
        4,
    );
    shell.face_iter_mut().for_each(|face| {
        face.invert();
    });
    let shell = shell.mapped(
        |point| *point,
        Clone::clone,
        |surface| {
            let mut replacement =
                Surface::SphericalSurface(Processor::new(Sphere::new(Point3::origin(), radius)));
            let point = surface.evaluate(0.31, 0.43);
            let parameter = replacement.search_parameter(point, None, 100).unwrap();
            if surface
                .normal(0.31, 0.43)
                .dot(replacement.normal(parameter.0, parameter.1))
                < 0.0
            {
                replacement.invert();
            }
            replacement
        },
    );
    Solid::try_new(vec![shell])
        .expect("original analytic sphere with shared quadrant arcs and poles")
}

#[test]
fn original_sphere_equatorial_section_reuses_native_arcs_and_shared_endpoint_identities() {
    let source = original_quadrant_sphere();
    let before = source_snapshot(&source);
    let upper_faces = source.boundaries()[0]
        .iter()
        .filter(|face| {
            face.edge_iter()
                .all(|edge| edge.front().point().z >= 0.0 && edge.back().point().z >= 0.0)
        })
        .collect::<Vec<_>>();
    assert_eq!(upper_faces.len(), 4);
    let upper_edges = upper_faces
        .iter()
        .flat_map(|face| face.edge_iter())
        .map(|edge| edge.id())
        .fold(Vec::with_capacity(8), |mut identities, identity| {
            if !identities.contains(&identity) {
                identities.push(identity);
            }
            identities
        });
    assert_eq!(upper_edges.len(), 8, "four original meridians and equators");
    let equator = source.boundaries()[0]
        .edge_iter()
        .filter(|edge| edge.front().point().z == 0.0 && edge.back().point().z == 0.0)
        .map(|edge| edge.id())
        .fold(Vec::with_capacity(4), |mut identities, identity| {
            if !identities.contains(&identity) {
                identities.push(identity);
            }
            identities
        });
    assert_eq!(equator.len(), 4);
    let cancel = AtomicBool::new(false);
    let budget = BooleanWorkBudget::new(100_000_000);
    let prepared = prepare_boolean_source_counted_with_options(
        &source,
        BooleanPreparationOptions {
            geometric_tolerance: 1.0e-5,
            seed_tolerance: 0.005,
        },
        closed_material_limits(),
        &cancel,
        &budget,
    )
    .expect("original sphere preparation without rational replacement surfaces");
    let clip: Solid = primitive::cuboid(BoundingBox::from_iter([
        Point3::new(-6.0, -6.0, 0.0),
        Point3::new(6.0, 6.0, 6.0),
    ]));
    let result = and_prepared_with_provenance(&prepared, &clip, &cancel)
        .expect("the actual cut is already an original curved boundary of both hemispheres");
    assert_eq!(result.solid.boundaries().len(), 1);
    let shell = &result.solid.boundaries()[0];
    shell
        .check_solid_boundary()
        .expect("closed upper spherical material");
    assert!(
        result.solid.is_geometric_consistent(),
        "{}",
        consistency_witness(&result.solid)
    );
    let caps = shell
        .iter()
        .filter(|face| matches!(face.surface(), Surface::Plane(_)))
        .collect::<Vec<_>>();
    assert_eq!(caps.len(), 1);
    let cap = caps[0];
    assert_eq!(shell.len(), upper_faces.len() + 1);
    for original in &upper_faces {
        let retained = shell
            .iter()
            .find(|face| face.id() == original.id())
            .expect("retain every whole original upper spherical patch");
        assert_eq!(retained.orientation(), original.orientation());
        assert!(matches!(retained.surface(), Surface::SphericalSurface(_)));
    }
    assert_eq!(cap.absolute_boundaries().len(), 1);
    assert_eq!(cap.absolute_boundaries()[0].len(), 4);
    assert!(cap.oriented_surface().normal(0.0, 0.0).z < -0.999);
    for edge in cap.edge_iter() {
        assert!(
            equator.contains(&edge.id()),
            "reuse the actual original native arc, not a fitted substitute"
        );
    }
    let mut retained = 0;
    let mut generated = 0;
    let mut retained_meridians = 0;
    for coedge in &result.coedge_origins {
        assert!(!matches!(
            coedge.source,
            BooleanCoedgeOrigin::AmbiguousOriginalUse { .. }
        ));
        assert!(upper_edges.contains(&coedge.output_edge));
        if !equator.contains(&coedge.output_edge) {
            assert!(
                matches!(&coedge.source, BooleanCoedgeOrigin::Original { face, edge_id, .. }
                if face.operand == BooleanOperand::Source && edge_id == &coedge.output_edge)
            );
            retained_meridians += 1;
            continue;
        }
        if coedge.output_face == cap.id() {
            assert!(
                matches!(&coedge.source, BooleanCoedgeOrigin::GeneratedIntersection { face }
                if face.operand == BooleanOperand::Other)
            );
            generated += 1;
        } else {
            assert!(
                matches!(&coedge.source, BooleanCoedgeOrigin::Original { face, edge_id, .. }
                if face.operand == BooleanOperand::Source && edge_id == &coedge.output_edge)
            );
            retained += 1;
        }
    }
    assert_eq!((retained, generated), (4, 4));
    assert_eq!(
        retained_meridians, 8,
        "both original uses of every upper meridian"
    );
    for identity in &upper_edges {
        assert!(
            shell.edge_iter().any(|edge| edge.id() == *identity),
            "retain every actual original upper native edge"
        );
    }
    for edge in shell.edge_iter() {
        assert!(upper_edges.contains(&edge.id()));
        edge.with_curve(|curve| {
            let (front, back) = curve.range_tuple();
            for fraction in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let point = curve.evaluate(front + fraction * (back - front));
                assert!(point.z >= -1.0e-6);
                assert!((point.to_vec().magnitude2() - 25.0).abs() < 1.0e-5);
            }
        });
    }
    assert_eq!(source_snapshot(&source), before);
}
