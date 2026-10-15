//! Unit tests for the parent module.
//!
//! Split out of the module file so the source stays readable; the module
//! path is unchanged.

use super::*;

fn direction(ratios: [f64; 3]) -> Direction {
    Direction {
        label: String::new(),
        direction_ratios: ratios.to_vec(),
    }
}

/// Regression for `axis ∥ ref_direction`: ISO 10303 allows the placement's
/// `axis` and `ref_direction` to be parallel, but the Gram-Schmidt step
/// then normalized a zero vector and produced `NaN`, which later panicked
/// during meshing (`tolerance must be no less than 1e-6`). The conversion
/// must yield a finite, orthonormal basis instead.
#[test]
fn axis2_placement3d_parallel_axis_and_ref_direction_is_finite() {
    let placement = Axis2Placement3d {
        label: String::new(),
        location: CartesianPoint {
            label: String::new(),
            coordinates: vec![1.0, 2.0, 3.0],
        },
        axis: Some(direction([0.0, 0.0, 1.0])),
        // Parallel to `axis` -- the degenerate case.
        ref_direction: Some(direction([0.0, 0.0, 1.0])),
    };

    let matrix = Matrix4::from(&placement);
    let (x, y, z) = (
        matrix.x.truncate(),
        matrix.y.truncate(),
        matrix.z.truncate(),
    );
    for component in [x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z] {
        assert!(component.is_finite(), "placement basis must be finite");
    }
    // The recovered basis is orthonormal.
    assert!((x.magnitude() - 1.0).abs() < 1.0e-9);
    assert!((y.magnitude() - 1.0).abs() < 1.0e-9);
    assert!(x.dot(z).abs() < 1.0e-9);
    assert!(y.dot(z).abs() < 1.0e-9);
}

fn original_line_edge(same_sense: bool) -> EdgeCurve {
    let point = |coordinates: [f64; 3]| CartesianPoint {
        label: String::new(),
        coordinates: coordinates.to_vec(),
    };
    let vertex = |coordinates| VertexPoint {
        label: String::new(),
        vertex_geometry: point(coordinates),
    };
    let (start, end) = if same_sense {
        ([1.0, -11.0, 11.0], [16.0, 9.0, 11.0])
    } else {
        ([16.0, 9.0, 11.0], [1.0, -11.0, 11.0])
    };
    EdgeCurve {
        label: String::new(),
        edge_start: vertex(start),
        edge_end: vertex(end),
        edge_geometry: CurveAny::Line(Box::new(Line {
            label: String::new(),
            pnt: point([7.0, -3.0, 11.0]),
            dir: Vector {
                label: String::new(),
                orientation: direction([3.0, 4.0, 0.0]),
                magnitude: 5.0,
            },
        })),
        same_sense,
    }
}

#[test]
fn spatial_line_edge_preserves_original_nonunit_directed_parameter_gauge() {
    for same_sense in [true, false] {
        let curve = original_line_edge(same_sense)
            .parse_curve3d()
            .expect("ordinary source LINE restriction");
        let (native, map) = curve
            .to_modeling_with_parameter_map()
            .expect("original LINE modeling conversion");
        // The original equation is (7,-3,11) + lambda*(3,4,0),
        // independently of the BREP endpoint chord and its unit interval.
        assert_eq!(curve.range_tuple(), (-2.0, 3.0));
        let chart = curve
            .raw_parameter_chart()
            .expect("raw chart")
            .expect("retained LINE gauge");
        for raw in [-2.0, -0.75, 0.0, 1.25, 3.0] {
            let decoded = chart.nominal(raw).expect("finite original gauge");
            let native_parameter = map.source_to_native(decoded).expect("native parameter");
            let expected = Point3::new(7.0 + 3.0 * raw, -3.0 + 4.0 * raw, 11.0);
            assert!(curve.evaluate(decoded).near(&expected));
            assert!(native.evaluate(native_parameter).near(&expected));
            let expected_derivative =
                Vector3::new(3.0, 4.0, 0.0) * if same_sense { 1.0 } else { -1.0 };
            assert!(
                native
                    .derivative(native_parameter)
                    .near(&expected_derivative)
            );
        }
        assert!(native.front().near(&Point3::from(
            &original_line_edge(same_sense).edge_start.vertex_geometry,
        )));
    }
}

#[test]
fn spatial_line_edge_refuses_displaced_and_wrong_direction_original_carriers() {
    for adversary in 0..3 {
        let mut edge = original_line_edge(true);
        let CurveAny::Line(line) = &mut edge.edge_geometry else {
            unreachable!("fixture is an original LINE");
        };
        match adversary {
            0 => line.pnt.coordinates[2] = 12.0,
            1 => line.dir.orientation = direction([-4.0, 3.0, 0.0]),
            _ => line.dir.orientation = direction([-3.0, -4.0, 0.0]),
        }
        let error = edge
            .parse_curve3d()
            .expect_err("unrelated original LINE must not become a chord");
        let expected = if adversary == 2 {
            ExactConversionError::LineEdgeSense
        } else {
            ExactConversionError::LineEdgeVertexOffCarrier
        };
        assert_eq!(
            error.downcast_ref::<ExactConversionError>(),
            Some(&expected)
        );
    }
}

#[test]
fn spatial_line_edge_keeps_original_carrier_with_tolerated_vertex_residual() {
    let mut edge = original_line_edge(true);
    edge.edge_start.vertex_geometry.coordinates[2] += TOLERANCE * 0.25;
    edge.edge_end.vertex_geometry.coordinates[2] += TOLERANCE * 0.25;
    let curve = edge
        .parse_curve3d()
        .expect("vertices within native residual");
    // Admission tolerance does not authorize fitting the original carrier.
    assert_eq!(curve.evaluate(0.0), Point3::new(7.0, -3.0, 11.0));
    assert_eq!(curve.derivative(0.0), Vector3::new(3.0, 4.0, 0.0));
}

#[test]
fn spatial_line_edge_retains_small_vector_at_large_original_origin() {
    let mut edge = original_line_edge(true);
    let CurveAny::Line(line) = &mut edge.edge_geometry else {
        unreachable!("fixture is an original LINE");
    };
    line.pnt.coordinates = vec![1.0e16, 0.0, 0.0];
    line.dir.orientation = direction([1.0, 0.0, 0.0]);
    line.dir.magnitude = 1.0;
    edge.edge_start.vertex_geometry.coordinates = vec![1.0e16, 0.0, 0.0];
    edge.edge_end.vertex_geometry.coordinates = vec![1.0e16 + 4.0, 0.0, 0.0];
    let curve = edge
        .parse_curve3d()
        .expect("original vector must not vanish");
    let (native, _) = curve
        .to_modeling_with_parameter_map()
        .expect("native original LINE");
    assert_eq!(curve.range_tuple(), (0.0, 4.0));
    assert_eq!(curve.derivative(2.0), Vector3::unit_x());
    assert_eq!(native.derivative(2.0), Vector3::unit_x());
    assert_eq!(native.back(), Point3::new(1.0e16 + 4.0, 0.0, 0.0));
}

#[test]
fn spatial_line_edge_refuses_residual_hidden_by_world_coordinate_rounding() {
    let mut edge = original_line_edge(true);
    let CurveAny::Line(line) = &mut edge.edge_geometry else {
        unreachable!("fixture is an original LINE");
    };
    line.pnt.coordinates = vec![0.0, 1.0e16, 0.0];
    line.dir.orientation = direction([4.0, 1.0, 0.0]);
    line.dir.magnitude = 4.25_f64.sqrt();
    edge.edge_start.vertex_geometry.coordinates = vec![0.0, 1.0e16, 0.0];
    edge.edge_end.vertex_geometry.coordinates = vec![2.0, 1.0e16, 0.0];
    // The original end has a transverse displacement of about 0.5, even
    // though rounding origin + lambda*vector hides it at a 1e16 origin.
    let error = edge
        .parse_curve3d()
        .expect_err("original residual must be bounded");
    assert_eq!(
        error.downcast_ref::<ExactConversionError>(),
        Some(&ExactConversionError::LineEdgeVertexOffCarrier),
    );
}

#[test]
fn explicit_surface_curve_line_keeps_original_geometry_and_refuses_wrong_carrier() {
    for displaced in [false, true] {
        let mut edge = original_line_edge(false);
        if displaced {
            let CurveAny::Line(line) = &mut edge.edge_geometry else {
                unreachable!("fixture is an original LINE");
            };
            line.pnt.coordinates[2] = 12.0;
        }
        edge.edge_geometry = CurveAny::SurfaceCurve(Box::new(SurfaceCurve {
            label: String::new(),
            curve_3d: edge.edge_geometry,
            associated_geometry: Vec::new(),
            master_representation: PreferredSurfaceCurveRepresentation::Curve3D,
            kind: SurfaceCurveEntityKind::Surface,
        }));
        if displaced {
            let error = edge
                .parse_curve3d()
                .expect_err("original nested LINE is displaced");
            assert_eq!(
                error.downcast_ref::<ExactConversionError>(),
                Some(&ExactConversionError::LineEdgeVertexOffCarrier),
            );
        } else {
            let curve = edge.parse_curve3d().expect("original nested LINE");
            let (native, _) = curve
                .to_modeling_with_parameter_map()
                .expect("native leader");
            assert_eq!(native.range_tuple(), (-2.0, 3.0));
            assert!(native.evaluate(0.0).near(&Point3::new(10.0, 1.0, 11.0)));
            assert!(native.derivative(0.0).near(&Vector3::new(-3.0, -4.0, 0.0)));
        }
    }
}
