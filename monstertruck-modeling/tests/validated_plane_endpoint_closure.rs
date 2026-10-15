//! Original-trim endpoint closure must prove physical roots, not guide proximity.

use monstertruck_modeling::*;
use monstertruck_topology::trimmed::TrimmedFace;
use std::sync::atomic::AtomicBool;

const CHART_ERROR: f64 = 1.0e-7;
const SHORTENING: f64 = 1.0e-6;
const EXTENSION_LIMIT: f64 = 1.0e-5;

#[derive(Clone, Copy)]
enum Case {
    Valid,
    RawBoundaryGauge,
    ScaledBoundary,
    ScaledRawBoundaryGauge,
    NearWrongSource,
    Interior,
    WrongCandidate,
    WrongEnd,
    WrongCut,
    WrongEdge,
    WrongBoundary,
    MissingBinding,
    MissingCandidate,
    Disabled,
    NoBytes,
    NoWork,
    Cancelled,
    Hole,
}

fn rectangle(points: [Point2; 4]) -> (Wire, Vec<Option<Curve2D>>, [Curve; 4]) {
    rectangle_with_gauge(points, false, 1.0)
}

fn rectangle_with_gauge(
    points: [Point2; 4],
    raw_boundary_gauge: bool,
    boundary_scale: f64,
) -> (Wire, Vec<Option<Curve2D>>, [Curve; 4]) {
    let vertices = points.map(|p| builder::vertex(Point3::new(p.x, p.y, 0.0) / boundary_scale));
    let curves = std::array::from_fn(|i| {
        let front = Point3::new(points[i].x, points[i].y, 0.0) / boundary_scale;
        let back = Point3::new(points[(i + 1) % 4].x, points[(i + 1) % 4].y, 0.0) / boundary_scale;
        if raw_boundary_gauge && i % 2 == 0 {
            // Horizontal edges retain an affine raw gauge on `[2, 5]`.
            let step = (back - front) / 3.0;
            Curve::TrimmedCurve(Processor::new(TrimmedCurve::new(
                Box::new(Curve::Line(Line(front - 2.0 * step, front - step))),
                (2.0, 5.0),
            )))
        } else {
            Curve::Line(Line(front, back))
        }
    });
    let wire: Wire = (0..4)
        .map(|i| Edge::new(&vertices[i], &vertices[(i + 1) % 4], curves[i].clone()))
        .collect();
    let wire = if boundary_scale == 1.0 {
        wire
    } else {
        builder::transformed(&wire, Matrix4::from_scale(boundary_scale))
    };
    let trims = (0..4)
        .map(|i| Some(Curve2D::Line(Line(points[i], points[(i + 1) % 4]))))
        .collect();
    (wire, trims, curves)
}

fn coverage(
    x: f64,
    case: Case,
) -> std::result::Result<ValidatedPlaneCoverage, ValidatedGeometryError> {
    // The original material rectangle is six by four millimeters in the XY plane.
    let boundary_scale = if matches!(case, Case::ScaledBoundary | Case::ScaledRawBoundaryGauge) {
        1000.0
    } else {
        1.0
    };
    let (wire, trims, boundaries) = rectangle_with_gauge(
        [
            Point2::new(-3.0, -2.0),
            Point2::new(3.0, -2.0),
            Point2::new(3.0, 2.0),
            Point2::new(-3.0, 2.0),
        ],
        matches!(case, Case::RawBoundaryGauge | Case::ScaledRawBoundaryGauge),
        boundary_scale,
    );
    let surface = Surface::Plane(Plane::new(
        Point3::origin(),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ));
    let (wires, trim_wires) = if matches!(case, Case::Hole) {
        // Clockwise inner circuit removes a two-by-two material region.
        let (hole, hole_trims, _) = rectangle([
            Point2::new(-1.0, -1.0),
            Point2::new(-1.0, 1.0),
            Point2::new(1.0, 1.0),
            Point2::new(1.0, -1.0),
        ]);
        (vec![wire, hole], vec![trims, hole_trims])
    } else {
        (vec![wire], vec![trims])
    };
    let trimmed = TrimmedFace::new(Face::new(wires, surface.clone()), trim_wires);
    let cancel = AtomicBool::new(false);
    let mut preparation = ValidatedGeometryBudget::new(2_000_000);
    let domain = prepare_face_parameter_domain_with_uncertainty(
        &trimmed,
        [CHART_ERROR; 2],
        TrimDomainLimits {
            max_coedges: 16,
            max_subdivisions: 65_536,
            max_retained_bytes: 32 * 1024 * 1024,
        },
        &mut preparation,
        &cancel,
    )
    .expect("exact original rectangle chart with explicitly retained correspondence error");
    assert_eq!(domain.uncertainty_uv(), [CHART_ERROR; 2]);
    let plane = Plane::new(
        Point3::new(x, 0.0, 0.0),
        Point3::new(x, 1.0, 0.0),
        Point3::new(x, 0.0, 1.0),
    );
    let guide = Curve::Line(Line(Point3::new(x, -2.0, 0.0), Point3::new(x, 2.0, 0.0)));
    let range = if matches!(case, Case::Interior) {
        [0.25, 0.75]
    } else {
        [SHORTENING / 4.0, 1.0 - SHORTENING / 4.0]
    };
    let curve_options = ValidatedCurveOptions::default();
    let candidate = prepared_face_plane_curve(
        &guide,
        &surface,
        &plane,
        range,
        &curve_options,
        &mut preparation,
        &cancel,
    )
    .expect("regular transverse original face/plane continuation");
    let boundary_range = if matches!(case, Case::RawBoundaryGauge | Case::ScaledRawBoundaryGauge) {
        [2.0, 5.0]
    } else {
        [0.0, 1.0]
    };
    let native_cut =
        |unit_cut: f64| boundary_range[0] + (boundary_range[1] - boundary_range[0]) * unit_cut;
    // This forged receipt is within both the endpoint cap and retained chart error.
    // Its correct edge identity must not authorize a different original 3D carrier.
    let forged_bottom =
        boundaries[0].transformed(Matrix4::from_translation(Vector3::new(0.0, 1.0e-9, 0.0)));
    let bottom = prepared_validated_curve(
        if matches!(case, Case::NearWrongSource) {
            &forged_bottom
        } else {
            &boundaries[0]
        },
        boundary_range,
        &curve_options,
        &mut preparation,
        &cancel,
    )
    .expect("retained original bottom boundary");
    let top = prepared_validated_curve(
        &boundaries[2],
        boundary_range,
        &curve_options,
        &mut preparation,
        &cancel,
    )
    .expect("retained original top boundary");
    let mut bindings = vec![
        ValidatedPlaneTrimEndpointBinding {
            coedge_index: 0,
            original_edge_id: domain.coedges()[0].identity.edge_id,
            boundary: &bottom,
            boundary_range,
            boundary_scale,
            candidate_index: 0,
            endpoint_is_back: false,
            source_cut_parameter: native_cut((x + 3.0) / 6.0),
            source_chart_error_mm: CHART_ERROR,
        },
        ValidatedPlaneTrimEndpointBinding {
            coedge_index: 2,
            original_edge_id: domain.coedges()[2].identity.edge_id,
            boundary: &top,
            boundary_range,
            boundary_scale,
            candidate_index: 0,
            endpoint_is_back: true,
            source_cut_parameter: native_cut((3.0 - x) / 6.0),
            source_chart_error_mm: CHART_ERROR,
        },
    ];
    match case {
        Case::WrongCandidate => bindings[0].candidate_index = 1,
        Case::WrongEnd => bindings[0].endpoint_is_back = true,
        Case::WrongCut => bindings[0].source_cut_parameter = 0.0,
        Case::WrongEdge => bindings[0].original_edge_id = domain.coedges()[1].identity.edge_id,
        Case::WrongBoundary => bindings[0].boundary = &top,
        Case::MissingBinding => {
            bindings.pop();
        }
        _ => {}
    }
    let options = ValidatedPlaneCoverageOptions {
        maximum_endpoint_extension_mm: if matches!(case, Case::Disabled) {
            0.0
        } else {
            EXTENSION_LIMIT
        },
        max_retained_bytes: if matches!(case, Case::NoBytes) {
            0
        } else {
            32 * 1024 * 1024
        },
        ..ValidatedPlaneCoverageOptions::default()
    };
    let mut budget = ValidatedGeometryBudget::new(if matches!(case, Case::NoWork) {
        0
    } else {
        2_000_000
    });
    let query_cancel = AtomicBool::new(matches!(case, Case::Cancelled));
    let candidates = if matches!(case, Case::MissingCandidate) {
        vec![]
    } else {
        vec![&candidate]
    };
    validated_face_plane_coverage(
        &domain,
        &plane,
        &candidates,
        &bindings,
        &options,
        &mut budget,
        &query_cancel,
    )
}

fn assert_physical_receipts(x: f64, case: Case) {
    let proof = coverage(x, case).expect("complete original transverse trim-root census");
    assert!(proof.covered_cells > 0);
    assert_eq!(proof.endpoint_strata.len(), 2);
    for back in [false, true] {
        let receipt = proof
            .endpoint_strata
            .iter()
            .find(|receipt| receipt.candidate_index() == 0 && receipt.endpoint_is_back() == back)
            .expect("one receipt for each finite endpoint of the same retained branch");
        let root = [x, if back { 2.0 } else { -2.0 }, 0.0];
        let bounds = receipt.position_bounds();
        for axis in 0..3 {
            assert!(bounds[axis].lower <= root[axis] && root[axis] <= bounds[axis].upper);
        }
        // The physical missing straight segment is exactly one micrometer long.
        assert!(receipt.maximum_extension_mm() >= SHORTENING * (1.0 - 1.0e-8));
        assert!(receipt.maximum_extension_mm() <= EXTENSION_LIMIT);
        assert!(receipt.maximum_arc_length_mm() >= SHORTENING * (1.0 - 1.0e-8));
        assert!(receipt.maximum_arc_length_mm() <= EXTENSION_LIMIT);
        assert!(receipt.maximum_plane_error_mm() >= 0.0);
        assert!(receipt.maximum_plane_error_mm() <= EXTENSION_LIMIT);
        // Independent Green correction relative to the origin, in square millimeters.
        let max_norm = bounds
            .iter()
            .map(|v| v.lower.abs().max(v.upper.abs()).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(0.5 * max_norm * receipt.maximum_arc_length_mm() <= 0.0001);
    }
}

#[test]
fn original_box_side_endpoint_closure_has_physical_receipts() {
    assert_physical_receipts(0.0, Case::Valid);
}

#[test]
fn nonbinary_original_trim_root_has_bounded_endpoint_receipts() {
    // The source cut parameters are non-binary fractions, not subdivision endpoints.
    assert_physical_receipts(1.0 / 3.0, Case::Valid);
}

#[test]
fn original_boundary_root_gauge_is_independent_of_stored_uv_gauge() {
    // Original horizontal carriers retain [2, 5]; authoritative UV coedges retain [0, 1].
    assert_physical_receipts(1.0 / 3.0, Case::RawBoundaryGauge);
}

#[test]
fn normalized_original_source_matches_native_builder_scaled_edge() {
    assert_physical_receipts(1.0 / 3.0, Case::ScaledBoundary);
}

#[test]
fn normalized_original_source_preserves_active_raw_restriction_when_scaled() {
    assert_physical_receipts(1.0 / 3.0, Case::ScaledRawBoundaryGauge);
}

#[test]
fn correct_edge_id_cannot_bind_nearby_forged_original_source() {
    assert!(coverage(0.0, Case::NearWrongSource).is_err());
}

#[test]
fn endpoint_bindings_cannot_invent_original_trim_incidence() {
    for case in [
        Case::Interior,
        Case::WrongCandidate,
        Case::WrongEnd,
        Case::WrongCut,
        Case::WrongEdge,
        Case::WrongBoundary,
        Case::MissingBinding,
        Case::MissingCandidate,
        Case::Disabled,
    ] {
        assert!(coverage(0.0, case).is_err());
    }
}

#[test]
fn omitted_hole_trim_roots_refuse_endpoint_closure() {
    assert!(coverage(0.0, Case::Hole).is_err());
}

#[test]
fn endpoint_receipts_respect_storage_work_and_cancellation() {
    assert!(matches!(
        coverage(0.0, Case::NoBytes),
        Err(ValidatedGeometryError::ResourceLimit)
    ));
    assert!(matches!(
        coverage(0.0, Case::NoWork),
        Err(ValidatedGeometryError::ResourceLimit)
    ));
    assert!(matches!(
        coverage(0.0, Case::Cancelled),
        Err(ValidatedGeometryError::Cancelled)
    ));
}

#[test]
fn missing_original_boundary_chart_refuses_preparation() {
    let (wire, mut trims, _) = rectangle([
        Point2::new(-3.0, -2.0),
        Point2::new(3.0, -2.0),
        Point2::new(3.0, 2.0),
        Point2::new(-3.0, 2.0),
    ]);
    trims[0] = None;
    let surface = Surface::Plane(Plane::new(
        Point3::origin(),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ));
    let trimmed = TrimmedFace::new(Face::new(vec![wire], surface), vec![trims]);
    let mut budget = ValidatedGeometryBudget::new(2_000_000);
    let cancel = AtomicBool::new(false);
    assert!(matches!(
        prepare_face_parameter_domain_with_uncertainty(
            &trimmed,
            [CHART_ERROR; 2],
            TrimDomainLimits {
                max_coedges: 16,
                max_subdivisions: 65_536,
                max_retained_bytes: 32 * 1024 * 1024,
            },
            &mut budget,
            &cancel,
        ),
        Err(TrimDomainError::MissingExactTrim {
            wire_index: 0,
            coedge_index: 0
        }),
    ));
}
