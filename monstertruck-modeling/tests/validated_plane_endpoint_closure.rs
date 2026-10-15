//! Original-trim endpoint closure must prove physical roots, not guide proximity.

use monstertruck_modeling::*;
use monstertruck_topology::trimmed::TrimmedFace;
use std::f64::consts::{PI, TAU};
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
    InvertedRawBoundaryGauge,
    InvertedRawGaugeWrongEnd,
    InvertedHole,
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
    let raw_boundary_gauge = matches!(
        case,
        Case::RawBoundaryGauge
            | Case::ScaledRawBoundaryGauge
            | Case::InvertedRawBoundaryGauge
            | Case::InvertedRawGaugeWrongEnd
    );
    let (wire, trims, boundaries) = rectangle_with_gauge(
        [
            Point2::new(-3.0, -2.0),
            Point2::new(3.0, -2.0),
            Point2::new(3.0, 2.0),
            Point2::new(-3.0, 2.0),
        ],
        raw_boundary_gauge,
        boundary_scale,
    );
    let surface = Surface::Plane(Plane::new(
        Point3::origin(),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ));
    let (wires, trim_wires) = if matches!(case, Case::Hole | Case::InvertedHole) {
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
    let mut face = Face::new(wires, surface.clone());
    if matches!(
        case,
        Case::InvertedRawBoundaryGauge | Case::InvertedRawGaugeWrongEnd | Case::InvertedHole
    ) {
        face.invert();
    }
    let trimmed = TrimmedFace::new(face, trim_wires);
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
        ValidatedFacePlaneSource {
            surface: &surface,
            anchor: None,
            endpoint_coordinate_planes: [None; 2],
        },
        &plane,
        range,
        &curve_options,
        &mut preparation,
        &cancel,
    )
    .expect("regular transverse original face/plane continuation");
    let boundary_range = if raw_boundary_gauge {
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
        Case::WrongEnd | Case::InvertedRawGaugeWrongEnd => bindings[0].endpoint_is_back = true,
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
fn inverted_face_preserves_original_material_side_with_independent_raw_gauge() {
    assert_physical_receipts(1.0 / 3.0, Case::InvertedRawBoundaryGauge);
}

#[test]
fn inverted_face_does_not_authorize_wrong_end_or_omitted_hole_roots() {
    assert!(coverage(1.0 / 3.0, Case::InvertedRawGaugeWrongEnd).is_err());
    assert!(coverage(0.0, Case::InvertedHole).is_err());
}

#[test]
fn periodic_original_seam_root_seeds_complete_closed_face_coverage() {
    let surface = Surface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
        Curve::Line(Line(
            Point3::new(1.0, 0.0, -2.0),
            Point3::new(1.0, 0.0, 2.0),
        )),
        Point3::origin(),
        Vector3::unit_z(),
    )));
    let bottom: Wire = primitive::circle(
        Point3::new(1.0, 0.0, -2.0),
        Point3::new(0.0, 0.0, -2.0),
        Vector3::unit_z(),
        2,
    );
    let top: Wire = primitive::circle(
        Point3::new(1.0, 0.0, 2.0),
        Point3::new(0.0, 0.0, 2.0),
        Vector3::unit_z(),
        2,
    );
    // The native seam gauge is `[2, 6]`, independent of both stored UV gauges.
    let seam_curve = Curve::TrimmedCurve(Processor::new(TrimmedCurve::new(
        Box::new(Curve::Line(Line(
            Point3::new(1.0, 0.0, -4.0),
            Point3::new(1.0, 0.0, -3.0),
        ))),
        (2.0, 6.0),
    )));
    let seam = Edge::new(bottom[0].front(), top[0].front(), seam_curve.clone());
    let wire: Wire = [
        seam.clone(),
        top[0].clone(),
        top[1].clone(),
        seam.inverse(),
        bottom[1].inverse(),
        bottom[0].inverse(),
    ]
    .into_iter()
    .collect();
    assert_eq!(wire[0].id(), wire[3].id());
    assert_ne!(wire[0].orientation(), wire[3].orientation());
    // Stored curves are per-use and in absolute edge direction, including the
    // two distinct covering-chart copies of the same authoritative seam edge.
    let trims = [
        [Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)],
        [Point2::new(1.0, 0.0), Point2::new(1.0, PI)],
        [Point2::new(1.0, PI), Point2::new(1.0, TAU)],
        [Point2::new(0.0, TAU), Point2::new(1.0, TAU)],
        [Point2::new(0.0, PI), Point2::new(0.0, TAU)],
        [Point2::new(0.0, 0.0), Point2::new(0.0, PI)],
    ]
    .map(|[front, back]| Some(Curve2D::Line(Line(front, back))))
    .to_vec();
    let trimmed = TrimmedFace::new(Face::new(vec![wire], surface.clone()), vec![trims]);
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
    .expect("authoritative full-period revolution chart with opposite shared seam uses");
    assert_eq!(domain.uncertainty_uv(), [CHART_ERROR; 2]);
    assert_eq!(domain.coedges().len(), 6);
    assert_eq!(domain.coedges()[0].identity.edge_id, seam.id());
    assert_eq!(domain.coedges()[3].identity.edge_id, seam.id());
    let plane = Plane::new(
        Point3::origin(),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    );
    let curve_options = ValidatedCurveOptions::default();
    let boundary_range = [2.0, 6.0];
    let source_cut_parameter = 4.0;
    let cut_vertex = builder::vertex(seam_curve.evaluate(source_cut_parameter));
    let (before_cut, after_cut) = seam
        .cut_with_parameter(&cut_vertex, source_cut_parameter)
        .expect("actual shared original edge cuts at the section plane");
    assert_eq!(before_cut.back(), after_cut.front());
    assert_eq!(
        before_cut.curve().range_tuple(),
        (2.0, source_cut_parameter)
    );
    assert_eq!(after_cut.curve().range_tuple(), (source_cut_parameter, 6.0));
    let master = prepared_validated_curve(
        &seam_curve,
        boundary_range,
        &curve_options,
        &mut preparation,
        &cancel,
    )
    .expect("immutable original raw-gauge seam master, prepared without a query seed");
    let original_cut = master
        .evaluate(source_cut_parameter, &mut preparation, &cancel)
        .expect("original source cut before generated-query preparation")
        .error;
    let initial_binding = ValidatedPlaneTrimEndpointBinding {
        coedge_index: 3,
        original_edge_id: seam.id(),
        boundary: &master,
        boundary_range,
        boundary_scale: 1.0,
        candidate_index: 0,
        endpoint_is_back: false,
        source_cut_parameter,
        source_chart_error_mm: CHART_ERROR,
    };
    let bindings = [
        initial_binding,
        ValidatedPlaneTrimEndpointBinding {
            coedge_index: 0,
            endpoint_is_back: true,
            ..initial_binding
        },
    ];
    let initial_chart = validated_plane_trim_root_chart(
        &domain,
        &plane,
        &bindings[0],
        32 * 1024 * 1024,
        &mut preparation,
        &cancel,
    )
    .expect("complete original and nominal root censuses select the actual seam occurrence");
    assert!(initial_chart[0].lower <= 0.5 && 0.5 <= initial_chart[0].upper);
    assert!(initial_chart[1].lower <= TAU && TAU <= initial_chart[1].upper);
    let unrelated_candidate_tags = ValidatedPlaneTrimEndpointBinding {
        candidate_index: usize::MAX,
        endpoint_is_back: true,
        ..bindings[0]
    };
    assert_eq!(
        validated_plane_trim_root_chart(
            &domain,
            &plane,
            &unrelated_candidate_tags,
            32 * 1024 * 1024,
            &mut preparation,
            &cancel,
        )
        .expect("a source-chart proposal does not depend on generated candidate tags"),
        initial_chart
    );
    // This is a clockwise closed physical section, shortened only at its back.
    // A tiny signed start-coordinate roundoff lies on the wrong inverse sheet
    // near zero, instead of the actual original seam use at the end of a period.
    let guide = Curve::BsplineCurve(BsplineCurve::new(
        KnotVector::from(
            (0_usize..=34)
                .map(|index| index.saturating_sub(1).min(32) as f64)
                .collect::<Vec<_>>(),
        ),
        (0..=32)
            .map(|index| {
                let angle = -(TAU - SHORTENING) * index as f64 / 32.0;
                Point3::new(
                    angle.cos(),
                    if index == 0 { 1.0e-14 } else { angle.sin() },
                    0.0,
                )
            })
            .collect(),
    ));
    let range = [0.0, 32.0];
    let mut unseeded_preparation = ValidatedGeometryBudget::new(2_000_000);
    let unseeded = prepared_face_plane_curve(
        &guide,
        ValidatedFacePlaneSource {
            surface: &surface,
            anchor: None,
            endpoint_coordinate_planes: [None; 2],
        },
        &plane,
        range,
        &curve_options,
        &mut unseeded_preparation,
        &cancel,
    )
    .expect("unseeded numerical continuation exists on a different periodic sheet");
    let unseeded_chart = unseeded.pieces()[0]
        .source_parameters
        .expect("original carrier uniqueness tube")[0];
    assert!(unseeded_chart[1].upper < PI);
    let candidate = prepared_face_plane_curve(
        &guide,
        ValidatedFacePlaneSource {
            surface: &surface,
            anchor: Some(ValidatedFacePlaneSourceAnchor::Front(
                initial_chart.map(ValidatedInterval::midpoint),
            )),
            endpoint_coordinate_planes: [None; 2],
        },
        &plane,
        range,
        &curve_options,
        &mut preparation,
        &cancel,
    )
    .expect("actual original coedge root proposes the generated continuation's initial sheet");
    let candidate_chart = candidate.pieces()[0]
        .source_parameters
        .expect("same original carrier, certified with the actual seam-use start")[0];
    assert!(candidate_chart[1].lower > PI);
    let options = ValidatedPlaneCoverageOptions {
        maximum_endpoint_extension_mm: EXTENSION_LIMIT,
        ..ValidatedPlaneCoverageOptions::default()
    };
    let mut wrong_sheet_budget = ValidatedGeometryBudget::new(2_000_000);
    assert!(
        validated_face_plane_coverage(
            &domain,
            &plane,
            &[&unseeded],
            &bindings,
            &options,
            &mut wrong_sheet_budget,
            &cancel,
        )
        .is_err()
    );
    let mut budget = ValidatedGeometryBudget::new(2_000_000);
    let proof = validated_face_plane_coverage(
        &domain,
        &plane,
        &[&candidate],
        &bindings,
        &options,
        &mut budget,
        &cancel,
    )
    .expect("complete physical periodic-face coverage with sealed original seam endpoints");
    assert!(proof.covered_cells > 0);
    assert_eq!(proof.endpoint_strata.len(), 2);
    for binding in bindings {
        let receipt = proof
            .endpoint_strata
            .iter()
            .find(|receipt| receipt.endpoint_is_back() == binding.endpoint_is_back)
            .expect("each actual opposite seam use seals its own endpoint");
        assert_eq!(receipt.candidate_index(), 0);
        assert_eq!(receipt.coedge_index(), binding.coedge_index);
        assert_eq!(receipt.original_edge_id(), seam.id());
        let native_root = receipt.source_root_parameter();
        assert!(native_root[0] <= source_cut_parameter && source_cut_parameter <= native_root[1]);
        for (axis, coordinate) in [1.0, 0.0, 0.0].into_iter().enumerate() {
            let position = receipt.position_bounds()[axis];
            assert!(position.lower <= coordinate && coordinate <= position.upper);
        }
        assert!(receipt.maximum_extension_mm() >= 0.0);
        assert!(receipt.maximum_extension_mm() <= EXTENSION_LIMIT);
        assert!(receipt.maximum_arc_length_mm() >= 0.0);
        assert!(receipt.maximum_arc_length_mm() <= EXTENSION_LIMIT);
        assert!(receipt.maximum_plane_error_mm() >= 0.0);
        assert!(receipt.maximum_plane_error_mm() <= EXTENSION_LIMIT);
        if binding.endpoint_is_back {
            assert!(receipt.maximum_extension_mm() >= SHORTENING * (1.0 - 1.0e-8));
            assert!(receipt.maximum_arc_length_mm() >= SHORTENING * (1.0 - 1.0e-8));
        }
        let max_norm = receipt
            .position_bounds()
            .iter()
            .map(|value| value.lower.abs().max(value.upper.abs()).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(0.5 * max_norm * receipt.maximum_arc_length_mm() <= 0.0001);
    }
    assert_eq!(master.range(), boundary_range);
    assert_eq!(
        master
            .evaluate(source_cut_parameter, &mut preparation, &cancel)
            .expect("generated-query seeding leaves the retained original master unchanged")
            .error,
        original_cut
    );
}

#[derive(Clone, Copy)]
enum BackAnchorCase {
    Complete,
    UnseededBack,
    WrongSeam,
    GeneratedGap,
    BoundedProfile,
}

fn back_anchored_taper_coverage(
    case: BackAnchorCase,
) -> std::result::Result<ValidatedPlaneCoverage, ValidatedGeometryError> {
    // A physical tapered face with its ORIGINAL negative full-period chart.
    // The two half-circle sections meet at a generated interior junction;
    // the second half has no original front root, only its actual raw-u=0 back.
    let height = 0.0999999912280684;
    let radius = 6.0 - height / 6.0;
    // STEP CONICAL_SURFACE retains an untrimmed original Line profile; the face
    // itself is bounded by its original coedges, not by a new carrier trim.
    let profile = Curve::Line(Line(
        Point3::new(6.0, 0.0, 0.0),
        Point3::new(6.0 + 1.0 / 6.0, 0.0, -1.0),
    ));
    let profile = if matches!(case, BackAnchorCase::BoundedProfile) {
        Curve::TrimmedCurve(Processor::new(TrimmedCurve::new(
            Box::new(profile),
            (-12.0, 0.0),
        )))
    } else {
        profile
    };
    let surface = Surface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
        profile,
        Point3::origin(),
        Vector3::unit_z(),
    )))
    .inverse();
    let bottom: Wire = primitive::circle(
        Point3::new(6.0, 0.0, 0.0),
        Point3::origin(),
        Vector3::unit_z(),
        2,
    );
    let top: Wire = primitive::circle(
        Point3::new(4.0, 0.0, 12.0),
        Point3::new(0.0, 0.0, 12.0),
        Vector3::unit_z(),
        2,
    );
    let seam_length = 148.0_f64.sqrt();
    let seam_step = Vector3::new(-2.0 / seam_length, 0.0, 12.0 / seam_length);
    let seam_curve = Curve::TrimmedCurve(Processor::new(TrimmedCurve::new(
        Box::new(Curve::Line(Line(
            Point3::new(6.0, 0.0, 0.0),
            Point3::new(6.0, 0.0, 0.0) + seam_step,
        ))),
        (0.0, seam_length),
    )));
    let seam = Edge::new(bottom[0].front(), top[0].front(), seam_curve.clone());
    let wire: Wire = [
        seam.clone(),
        top[0].clone(),
        top[1].clone(),
        seam.inverse(),
        bottom[1].inverse(),
        bottom[0].inverse(),
    ]
    .into_iter()
    .collect();
    let trims = [
        [Point2::new(-TAU, 0.0), Point2::new(-TAU, -12.0)],
        [Point2::new(-TAU, -12.0), Point2::new(-PI, -12.0)],
        [Point2::new(-PI, -12.0), Point2::new(0.0, -12.0)],
        [Point2::new(0.0, 0.0), Point2::new(0.0, -12.0)],
        [Point2::new(-PI, 0.0), Point2::new(0.0, 0.0)],
        [Point2::new(-TAU, 0.0), Point2::new(-PI, 0.0)],
    ]
    .map(|[front, back]| Some(Curve2D::Line(Line(front, back))))
    .to_vec();
    let trimmed = TrimmedFace::new(Face::new(vec![wire], surface.clone()), vec![trims]);
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
    .expect("original negative-chart taper retains both distinct seam uses");
    let plane = Plane::new(
        Point3::new(0.0, 0.0, height),
        Point3::new(1.0, 0.0, height),
        Point3::new(0.0, 1.0, height),
    );
    let options = ValidatedCurveOptions::default();
    let boundary_range = [0.0, seam_length];
    let source_cut_parameter = height / seam_step.z;
    let master = prepared_validated_curve(
        &seam_curve,
        boundary_range,
        &options,
        &mut preparation,
        &cancel,
    )
    .expect("same original sloping seam, independent arc-length native gauge");
    let original_cut = master
        .evaluate(source_cut_parameter, &mut preparation, &cancel)
        .unwrap()
        .error;
    let guides = [false, true].map(|second| {
        let gap = if second && matches!(case, BackAnchorCase::GeneratedGap) {
            0.001
        } else {
            0.0
        };
        let (front, back) = if second {
            (-PI + gap, 0.0)
        } else {
            (-TAU, -PI)
        };
        let points = (0..=16)
            .map(|index| {
                if (!second && index == 0) || (second && index == 16) {
                    Point3::new(radius, 0.0, height)
                } else if (!second && index == 16) || (second && index == 0 && gap == 0.0) {
                    Point3::new(-radius, 0.0, height)
                } else {
                    let angle = front + (back - front) * index as f64 / 16.0;
                    Point3::new(radius * angle.cos(), radius * angle.sin(), height)
                }
            })
            .collect();
        Curve::BsplineCurve(BsplineCurve::new(
            KnotVector::from(
                std::iter::once(0.0)
                    .chain((0..=16).map(|index| index as f64))
                    .chain(std::iter::once(16.0))
                    .collect::<Vec<_>>(),
            ),
            points,
        ))
    });
    let shared = CoordinatePlane {
        normal: [0.0, 1.0, 0.0],
        offset: 0.0,
    };
    let curves = [0, 1].map(|index| {
        let anchor = if index == 0 {
            Some(ValidatedFacePlaneSourceAnchor::Front([-TAU, -height]))
        } else if matches!(case, BackAnchorCase::UnseededBack) {
            None
        } else {
            Some(ValidatedFacePlaneSourceAnchor::Back([0.0, -height]))
        };
        let coordinates = if index == 0 {
            [None, Some(shared)]
        } else {
            [
                Some(CoordinatePlane {
                    offset: if matches!(case, BackAnchorCase::GeneratedGap) {
                        radius * (-PI + 0.001).sin()
                    } else {
                        0.0
                    },
                    ..shared
                }),
                None,
            ]
        };
        prepared_face_plane_curve(
            &guides[index],
            ValidatedFacePlaneSource {
                surface: &surface,
                anchor,
                endpoint_coordinate_planes: coordinates,
            },
            &plane,
            [0.0, 16.0],
            &options,
            &mut ValidatedGeometryBudget::new(2_000_000),
            &cancel,
        )
        .expect("whole original tapered half-circle continuation")
    });
    let mut covered_until = 0.0;
    for piece in curves[1].pieces() {
        assert_eq!(piece.parameter[0], covered_until);
        assert!(piece.parameter[1] > piece.parameter[0]);
        covered_until = piece.parameter[1];
    }
    assert_eq!(covered_until, 16.0);
    if !matches!(case, BackAnchorCase::UnseededBack) {
        let first_uv = curves[1].pieces()[0].source_parameters.unwrap()[0];
        assert!(
            first_uv[0].upper < 0.0,
            "back anchor retains the original negative sheet"
        );
        let jet = curves[1].jet([16.0; 2], &mut preparation, &cancel).unwrap();
        assert!(
            jet.first[1].lower > 0.0,
            "native ascending derivative is not reversed"
        );
    }
    let wrong_seam = matches!(case, BackAnchorCase::WrongSeam);
    let bindings = [
        ValidatedPlaneTrimEndpointBinding {
            coedge_index: if wrong_seam { 3 } else { 0 },
            original_edge_id: seam.id(),
            boundary: &master,
            boundary_range,
            boundary_scale: 1.0,
            candidate_index: 0,
            endpoint_is_back: false,
            source_cut_parameter,
            source_chart_error_mm: CHART_ERROR,
        },
        ValidatedPlaneTrimEndpointBinding {
            coedge_index: if wrong_seam { 0 } else { 3 },
            original_edge_id: seam.id(),
            boundary: &master,
            boundary_range,
            boundary_scale: 1.0,
            candidate_index: 1,
            endpoint_is_back: true,
            source_cut_parameter,
            source_chart_error_mm: CHART_ERROR,
        },
    ];
    let result = validated_face_plane_coverage(
        &domain,
        &plane,
        &[&curves[0], &curves[1]],
        &bindings,
        &ValidatedPlaneCoverageOptions {
            maximum_endpoint_extension_mm: EXTENSION_LIMIT,
            ..ValidatedPlaneCoverageOptions::default()
        },
        &mut ValidatedGeometryBudget::new(2_000_000),
        &cancel,
    );
    assert_eq!(master.range(), boundary_range);
    assert_eq!(
        master
            .evaluate(source_cut_parameter, &mut preparation, &cancel)
            .unwrap()
            .error,
        original_cut
    );
    result
}

#[test]
fn original_back_root_anchors_complete_negative_chart_taper_coverage() {
    let coverage = back_anchored_taper_coverage(BackAnchorCase::Complete)
        .expect("back-only original anchor must close the entire physical tapered section");
    assert!(coverage.covered_cells > 0);
    assert_eq!(coverage.endpoint_strata.len(), 2);
    for (candidate, coedge, back) in [(0, 0, false), (1, 3, true)] {
        assert!(coverage.endpoint_strata.iter().any(|receipt| {
            receipt.candidate_index() == candidate
                && receipt.coedge_index() == coedge
                && receipt.endpoint_is_back() == back
                && receipt.maximum_extension_mm() <= EXTENSION_LIMIT
        }));
    }
}

#[test]
fn back_anchor_cannot_weld_wrong_original_seam_or_generated_gap() {
    for case in [
        BackAnchorCase::UnseededBack,
        BackAnchorCase::WrongSeam,
        BackAnchorCase::GeneratedGap,
    ] {
        assert!(back_anchored_taper_coverage(case).is_err());
    }
}

#[test]
fn back_anchor_does_not_extend_finite_profile_for_whole_face_census() {
    // The retained trim uncertainty and outward full-span boundary jets reach
    // outside this deliberately finite carrier. An anchor cannot authorize
    // evaluating its original profile beyond its admitted parameter domain.
    assert!(matches!(
        back_anchored_taper_coverage(BackAnchorCase::BoundedProfile),
        Err(ValidatedGeometryError::ParameterOutOfRange)
    ));
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
