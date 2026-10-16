use super::*;
use crate::*;
use monstertruck_topology::trimmed::TrimmedFace;

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
    DisconnectedComponent,
}

fn rectangle_with_gauge(
    points: [Point2; 4],
    raw: bool,
    scale: f64,
) -> (Wire, Vec<Option<Curve2D>>, [Curve; 4]) {
    let vertices = points.map(|p| builder::vertex(Point3::new(p.x, p.y, 0.0) / scale));
    let curves = std::array::from_fn(|i| {
        let front = Point3::new(points[i].x, points[i].y, 0.0) / scale;
        let back = Point3::new(points[(i + 1) % 4].x, points[(i + 1) % 4].y, 0.0) / scale;
        if raw && i % 2 == 0 {
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
    let wire = if scale == 1.0 {
        wire
    } else {
        builder::transformed(&wire, Matrix4::from_scale(scale))
    };
    let trims = (0..4)
        .map(|i| Some(Curve2D::Line(Line(points[i], points[(i + 1) % 4]))))
        .collect();
    (wire, trims, curves)
}

fn coverage_with_inventory(
    x: f64,
    case: Case,
    cached: bool,
) -> std::result::Result<ValidatedPlaneCoverage, ValidatedGeometryError> {
    let scale = if matches!(case, Case::ScaledBoundary | Case::ScaledRawBoundaryGauge) {
        1000.0
    } else {
        1.0
    };
    let raw = matches!(
        case,
        Case::RawBoundaryGauge | Case::ScaledRawBoundaryGauge | Case::InvertedRawBoundaryGauge
    );
    let (wire, trims, originals) = rectangle_with_gauge(
        [
            Point2::new(-3.0, -2.0),
            Point2::new(3.0, -2.0),
            Point2::new(3.0, 2.0),
            Point2::new(-3.0, 2.0),
        ],
        raw,
        scale,
    );
    let surface = Surface::Plane(Plane::new(
        Point3::origin(),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ));
    let (wires, trims) = if matches!(
        case,
        Case::Hole | Case::InvertedHole | Case::DisconnectedComponent
    ) {
        let points = if matches!(case, Case::DisconnectedComponent) {
            [
                Point2::new(-3.0, 4.0),
                Point2::new(3.0, 4.0),
                Point2::new(3.0, 6.0),
                Point2::new(-3.0, 6.0),
            ]
        } else {
            [
                Point2::new(-1.0, -1.0),
                Point2::new(-1.0, 1.0),
                Point2::new(1.0, 1.0),
                Point2::new(1.0, -1.0),
            ]
        };
        let (other, other_trims, _) = rectangle_with_gauge(points, false, 1.0);
        (vec![wire, other], vec![trims, other_trims])
    } else {
        (vec![wire], vec![trims])
    };
    let mut face = Face::new(wires, surface.clone());
    if matches!(case, Case::InvertedRawBoundaryGauge | Case::InvertedHole) {
        face.invert();
    }
    let trimmed = TrimmedFace::new(face, trims);
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
    .map_err(trim_error)?;
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
    let options = ValidatedCurveOptions::default();
    let candidate = prepared_face_plane_curve(
        &guide,
        ValidatedFacePlaneSource {
            surface: &surface,
            anchor: None,
            endpoint_coordinate_planes: [None; 2],
        },
        &plane,
        range,
        &options,
        &mut preparation,
        &cancel,
    )?;
    let boundary_range = if raw { [2.0, 5.0] } else { [0.0, 1.0] };
    let forged =
        originals[0].transformed(Matrix4::from_translation(Vector3::new(0.0, 1.0e-9, 0.0)));
    let bottom = prepared_validated_curve(
        if matches!(case, Case::NearWrongSource) {
            &forged
        } else {
            &originals[0]
        },
        boundary_range,
        &options,
        &mut preparation,
        &cancel,
    )?;
    let top = prepared_validated_curve(
        &originals[2],
        boundary_range,
        &options,
        &mut preparation,
        &cancel,
    )?;
    let cut = |unit: f64| boundary_range[0] + (boundary_range[1] - boundary_range[0]) * unit;
    let mut bindings = vec![
        ValidatedPlaneTrimEndpointBinding {
            coedge_index: 0,
            original_edge_id: domain.coedges()[0].identity.edge_id,
            boundary: &bottom,
            boundary_range,
            boundary_scale: scale,
            candidate_index: 0,
            endpoint_is_back: false,
            source_cut_parameter: cut((x + 3.0) / 6.0),
            source_chart_error_mm: CHART_ERROR,
        },
        ValidatedPlaneTrimEndpointBinding {
            coedge_index: 2,
            original_edge_id: domain.coedges()[2].identity.edge_id,
            boundary: &top,
            boundary_range,
            boundary_scale: scale,
            candidate_index: 0,
            endpoint_is_back: true,
            source_cut_parameter: cut((3.0 - x) / 6.0),
            source_chart_error_mm: CHART_ERROR,
        },
    ];
    let sources = [
        bindings[0].boundary_binding(),
        bindings[1].boundary_binding(),
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
        ..Default::default()
    };
    let mut budget = ValidatedGeometryBudget::new(if matches!(case, Case::NoWork) {
        0
    } else {
        2_000_000
    });
    let cancel = AtomicBool::new(matches!(case, Case::Cancelled));
    let roots = if cached {
        Some(prepared_face_plane_roots(
            &domain,
            &plane,
            &sources,
            options.max_retained_bytes,
            &mut budget,
            &cancel,
        )?)
    } else {
        None
    };
    let candidates = if matches!(case, Case::MissingCandidate) {
        vec![]
    } else {
        vec![&candidate]
    };
    validated_face_plane_coverage(
        &domain,
        &ValidatedPlaneCoverageQuery {
            plane: &plane,
            candidates: &candidates,
            endpoint_bindings: &bindings,
            prepared_roots: roots.as_ref(),
        },
        &options,
        &mut budget,
        &cancel,
    )
}

#[test]
fn cached_and_fresh_roots_preserve_independent_physical_incidence() {
    for x in [0.0, 1.0 / 3.0, -0.7] {
        for case in [
            Case::Valid,
            Case::RawBoundaryGauge,
            Case::ScaledBoundary,
            Case::ScaledRawBoundaryGauge,
            Case::InvertedRawBoundaryGauge,
        ] {
            for cached in [false, true] {
                let proof = coverage_with_inventory(x, case, cached).unwrap();
                assert_eq!(proof.endpoint_strata.len(), 2);
                for back in [false, true] {
                    let receipt = proof
                        .endpoint_strata
                        .iter()
                        .find(|receipt| receipt.endpoint_is_back() == back)
                        .unwrap();
                    let actual_source = [x, if back { 2.0 } else { -2.0 }, 0.0];
                    for (bound, actual) in receipt.position_bounds().into_iter().zip(actual_source)
                    {
                        assert!(bound.lower <= actual && actual <= bound.upper);
                    }
                    assert!(receipt.maximum_extension_mm() >= SHORTENING * (1.0 - 1.0e-8));
                    assert!(receipt.maximum_extension_mm() <= EXTENSION_LIMIT);
                    assert!(receipt.maximum_arc_length_mm() <= EXTENSION_LIMIT);
                }
            }
        }
    }
    for case in [
        Case::Hole,
        Case::InvertedHole,
        Case::DisconnectedComponent,
        Case::NearWrongSource,
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
        for cached in [false, true] {
            assert!(coverage_with_inventory(0.0, case, cached).is_err());
        }
    }
    for cached in [false, true] {
        assert!(matches!(
            coverage_with_inventory(0.0, Case::NoBytes, cached),
            Err(ValidatedGeometryError::ResourceLimit),
        ));
        assert!(matches!(
            coverage_with_inventory(0.0, Case::NoWork, cached),
            Err(ValidatedGeometryError::ResourceLimit),
        ));
        assert!(matches!(
            coverage_with_inventory(0.0, Case::Cancelled, cached),
            Err(ValidatedGeometryError::Cancelled),
        ));
    }
}

#[test]
fn root_inventory_rejects_foreign_queries_bindings_and_mutated_originals() {
    let (wire, trims, originals) = rectangle_with_gauge(
        [
            Point2::new(-3.0, -2.0),
            Point2::new(3.0, -2.0),
            Point2::new(3.0, 2.0),
            Point2::new(-3.0, 2.0),
        ],
        true,
        1.0,
    );
    let surface = Surface::Plane(Plane::new(
        Point3::origin(),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ));
    let trimmed = TrimmedFace::new(Face::new(vec![wire], surface.clone()), vec![trims]);
    let cancel = AtomicBool::new(false);
    let mut budget = ValidatedGeometryBudget::new(4_000_000);
    let limits = TrimDomainLimits {
        max_coedges: 16,
        max_subdivisions: 65_536,
        max_retained_bytes: 32 * 1024 * 1024,
    };
    let domain = prepare_face_parameter_domain_with_uncertainty(
        &trimmed,
        [CHART_ERROR; 2],
        limits,
        &mut budget,
        &cancel,
    )
    .unwrap();
    let foreign_domain = prepare_face_parameter_domain_with_uncertainty(
        &trimmed,
        [CHART_ERROR; 2],
        limits,
        &mut budget,
        &cancel,
    )
    .unwrap();
    let foreign_chart = prepare_face_parameter_domain_with_uncertainty(
        &trimmed,
        [2.0 * CHART_ERROR; 2],
        limits,
        &mut budget,
        &cancel,
    )
    .unwrap();
    let inverted_trimmed = TrimmedFace::new(domain.face().inverse(), trimmed.trims().to_vec());
    let foreign_orientation = prepare_face_parameter_domain_with_uncertainty(
        &inverted_trimmed,
        [CHART_ERROR; 2],
        limits,
        &mut budget,
        &cancel,
    )
    .unwrap();
    let x = 1.0 / 3.0;
    let plane_at = |x| {
        Plane::new(
            Point3::new(x, 0.0, 0.0),
            Point3::new(x, 1.0, 0.0),
            Point3::new(x, 0.0, 1.0),
        )
    };
    let plane = plane_at(x);
    let options = ValidatedCurveOptions::default();
    let prepared: Vec<_> = originals
        .iter()
        .enumerate()
        .map(|(index, curve)| {
            prepared_validated_curve(
                curve,
                if index % 2 == 0 {
                    [2.0, 5.0]
                } else {
                    [0.0, 1.0]
                },
                &options,
                &mut budget,
                &cancel,
            )
            .unwrap()
        })
        .collect();
    let bindings: Vec<_> = prepared
        .iter()
        .enumerate()
        .map(|(index, boundary)| ValidatedPlaneTrimBoundaryBinding {
            coedge_index: index,
            original_edge_id: domain.coedges()[index].identity.edge_id,
            boundary,
            boundary_range: boundary.range(),
            boundary_scale: 1.0,
            source_chart_error_mm: CHART_ERROR,
        })
        .collect();
    let before_preparation = budget.consumed();
    let roots = prepared_face_plane_roots(
        &domain,
        &plane,
        &bindings,
        32 * 1024 * 1024,
        &mut budget,
        &cancel,
    )
    .unwrap();
    assert!(budget.consumed() > before_preparation);
    let cuts = [2.0 + (x + 3.0) * 0.5, 2.0 + (3.0 - x) * 0.5];
    for ((coedge, y), cut) in [(0, -2.0), (2, 2.0)].into_iter().zip(cuts) {
        let before_lookup = budget.consumed();
        let chart = roots
            .root_chart_for_cut(coedge, cut, &mut budget, &cancel)
            .unwrap();
        assert!(budget.consumed() > before_lookup);
        assert!(chart[0].lower <= x && x <= chart[0].upper);
        assert!(chart[1].lower <= y && y <= chart[1].upper);
        let physical = prepared[coedge]
            .evaluate(cut, &mut budget, &cancel)
            .unwrap()
            .error;
        for (bound, actual) in physical.into_iter().zip([x, y, 0.0]) {
            assert!(bound.lower <= actual && actual <= bound.upper);
        }
    }
    let guide = Curve::Line(Line(Point3::new(x, -2.0, 0.0), Point3::new(x, 2.0, 0.0)));
    let candidate = prepared_face_plane_curve(
        &guide,
        ValidatedFacePlaneSource {
            surface: &surface,
            anchor: None,
            endpoint_coordinate_planes: [None; 2],
        },
        &plane,
        [SHORTENING / 4.0, 1.0 - SHORTENING / 4.0],
        &options,
        &mut budget,
        &cancel,
    )
    .unwrap();
    let endpoints = [(0, false, cuts[0]), (2, true, cuts[1])].map(|(index, back, cut)| {
        ValidatedPlaneTrimEndpointBinding {
            coedge_index: index,
            original_edge_id: bindings[index].original_edge_id,
            boundary: bindings[index].boundary,
            boundary_range: bindings[index].boundary_range,
            boundary_scale: 1.0,
            source_chart_error_mm: CHART_ERROR,
            candidate_index: 0,
            endpoint_is_back: back,
            source_cut_parameter: cut,
        }
    });
    let coverage_options = ValidatedPlaneCoverageOptions {
        maximum_endpoint_extension_mm: EXTENSION_LIMIT,
        ..Default::default()
    };
    let cover = |query_domain: &PreparedFaceParameterDomain<'_>,
                 query_plane: &Plane,
                 endpoints: &[ValidatedPlaneTrimEndpointBinding<'_>],
                 budget: &mut ValidatedGeometryBudget| {
        validated_face_plane_coverage(
            query_domain,
            &ValidatedPlaneCoverageQuery {
                plane: query_plane,
                candidates: &[&candidate],
                endpoint_bindings: endpoints,
                prepared_roots: Some(&roots),
            },
            &coverage_options,
            budget,
            &cancel,
        )
    };
    assert!(cover(&domain, &plane, &endpoints, &mut budget).is_ok());
    let mut fresh_work = ValidatedGeometryBudget::new(2_000_000);
    let fresh = validated_face_plane_coverage(
        &domain,
        &ValidatedPlaneCoverageQuery {
            plane: &plane,
            candidates: &[&candidate],
            endpoint_bindings: &endpoints,
            prepared_roots: None,
        },
        &coverage_options,
        &mut fresh_work,
        &cancel,
    )
    .unwrap();
    let mut reused_work = ValidatedGeometryBudget::new(2_000_000);
    let reused = cover(&domain, &plane, &endpoints, &mut reused_work).unwrap();
    for proof in [&fresh, &reused] {
        for receipt in &proof.endpoint_strata {
            let actual = [
                x,
                if receipt.endpoint_is_back() {
                    2.0
                } else {
                    -2.0
                },
                0.0,
            ];
            for (bound, actual) in receipt.position_bounds().into_iter().zip(actual) {
                assert!(bound.lower <= actual && actual <= bound.upper);
            }
        }
    }
    assert!(reused_work.consumed() < fresh_work.consumed());
    let tight_options = ValidatedPlaneCoverageOptions {
        max_retained_bytes: roots.retained_bytes() - 1,
        ..coverage_options
    };
    assert!(matches!(
        validated_face_plane_coverage(
            &domain,
            &ValidatedPlaneCoverageQuery {
                plane: &plane,
                candidates: &[&candidate],
                endpoint_bindings: &endpoints,
                prepared_roots: Some(&roots),
            },
            &tight_options,
            &mut budget,
            &cancel,
        ),
        Err(ValidatedGeometryError::ResourceLimit),
    ));
    for other_plane in [
        plane_at(x + 1.0e-10),
        Plane::new(
            Point3::new(x, 0.0, 1.0),
            Point3::new(x, 1.0, 1.0),
            Point3::new(x, 0.0, 2.0),
        ),
    ] {
        assert!(matches!(
            cover(&domain, &other_plane, &endpoints, &mut budget),
            Err(ValidatedGeometryError::InvalidInput),
        ));
    }
    for foreign in [&foreign_domain, &foreign_chart, &foreign_orientation] {
        assert!(matches!(
            cover(foreign, &plane, &endpoints, &mut budget),
            Err(ValidatedGeometryError::InvalidInput),
        ));
    }
    let distinct_binding =
        prepared_validated_curve(&originals[0], [2.0, 5.0], &options, &mut budget, &cancel)
            .unwrap();
    for change in 0..5 {
        let mut foreign = endpoints;
        match change {
            0 => foreign[0].boundary = &distinct_binding,
            1 => foreign[0].boundary_range = [2.25, 4.75],
            2 => foreign[0].boundary_scale = 2.0,
            3 => foreign[0].source_chart_error_mm += 1.0e-10,
            _ => foreign[0].coedge_index = 2,
        }
        assert!(matches!(
            cover(&domain, &plane, &foreign, &mut budget),
            Err(ValidatedGeometryError::InvalidInput),
        ));
    }
    domain
        .face()
        .set_surface(surface.transformed(Matrix4::from_translation(Vector3::unit_z())));
    assert!(matches!(
        roots.root_chart_for_cut(0, cuts[0], &mut budget, &cancel),
        Err(ValidatedGeometryError::InvalidInput),
    ));
    assert!(matches!(
        cover(&domain, &plane, &endpoints, &mut budget),
        Err(ValidatedGeometryError::InvalidInput),
    ));
    domain.face().set_surface(surface.clone());
    let edge = &domain.face().absolute_boundaries()[0][0];
    edge.set_curve(
        originals[0].transformed(Matrix4::from_translation(Vector3::new(0.0, 1.0e-10, 0.0))),
    );
    assert!(
        roots
            .root_chart_for_cut(0, cuts[0], &mut budget, &cancel)
            .is_err()
    );
    assert!(matches!(
        cover(&domain, &plane, &endpoints, &mut budget),
        Err(ValidatedGeometryError::InvalidInput),
    ));
    edge.set_curve(originals[0].clone());
    assert!(
        roots
            .root_chart_for_cut(0, cuts[0], &mut budget, &cancel)
            .is_ok()
    );
    let before_failure = budget.consumed();
    assert!(
        roots
            .root_chart_for_cut(0, 100.0, &mut budget, &cancel)
            .is_err()
    );
    assert!(budget.consumed() > before_failure);
    assert!(matches!(
        prepared_face_plane_roots(
            &domain,
            &plane,
            &bindings,
            roots.retained_bytes() - 1,
            &mut budget,
            &cancel,
        ),
        Err(ValidatedGeometryError::ResourceLimit),
    ));
    assert!(matches!(
        roots.root_chart_for_cut(0, cuts[0], &mut ValidatedGeometryBudget::new(0), &cancel),
        Err(ValidatedGeometryError::ResourceLimit),
    ));
    assert!(matches!(
        roots.root_chart_for_cut(0, cuts[0], &mut budget, &AtomicBool::new(true)),
        Err(ValidatedGeometryError::Cancelled),
    ));
}

#[test]
fn root_inventory_compares_actual_processor_transform_and_orientation() {
    let surface = Surface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
        Curve::Line(Line(
            Point3::new(1.0, 0.0, -2.0),
            Point3::new(1.0, 0.0, 2.0),
        )),
        Point3::origin(),
        Vector3::unit_z(),
    )));
    let chart = [
        Point2::new(0.1, 0.2),
        Point2::new(0.9, 0.2),
        Point2::new(0.9, 2.0),
        Point2::new(0.1, 2.0),
    ];
    let uv = std::array::from_fn::<_, 4, _>(|index| {
        Curve2D::Line(Line(chart[index], chart[(index + 1) % 4]))
    });
    let originals = uv.each_ref().map(|curve| {
        Curve::ParameterCurve(ParameterCurve::new(
            curve.clone(),
            Box::new(surface.clone()),
        ))
    });
    let vertices = chart.map(|point| builder::vertex(surface.subs(point.x, point.y)));
    let wire: Wire = (0..4)
        .map(|index| {
            Edge::new(
                &vertices[index],
                &vertices[(index + 1) % 4],
                originals[index].clone(),
            )
        })
        .collect();
    let trimmed = TrimmedFace::new(
        Face::new(vec![wire], surface.clone()),
        vec![uv.into_iter().map(Some).collect()],
    );
    let cancel = AtomicBool::new(false);
    let mut budget = ValidatedGeometryBudget::new(2_000_000);
    let domain = prepare_face_parameter_domain_with_uncertainty(
        &trimmed,
        [0.0; 2],
        TrimDomainLimits {
            max_coedges: 16,
            max_subdivisions: 65_536,
            max_retained_bytes: 32 * 1024 * 1024,
        },
        &mut budget,
        &cancel,
    )
    .unwrap();
    let prepared: Vec<_> = originals
        .iter()
        .map(|curve| {
            prepared_validated_curve(
                curve,
                [0.0, 1.0],
                &ValidatedCurveOptions::default(),
                &mut budget,
                &cancel,
            )
            .unwrap()
        })
        .collect();
    let bindings: Vec<_> = prepared
        .iter()
        .enumerate()
        .map(|(index, boundary)| ValidatedPlaneTrimBoundaryBinding {
            coedge_index: index,
            original_edge_id: domain.coedges()[index].identity.edge_id,
            boundary,
            boundary_range: [0.0, 1.0],
            boundary_scale: 1.0,
            source_chart_error_mm: 0.0,
        })
        .collect();
    let plane = Plane::new(
        Point3::origin(),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    );
    let roots = prepared_face_plane_roots(
        &domain,
        &plane,
        &bindings,
        32 * 1024 * 1024,
        &mut budget,
        &cancel,
    )
    .unwrap();
    assert!(
        roots
            .root_chart_for_cut(0, 0.5, &mut budget, &cancel)
            .is_ok()
    );
    for changed in [
        surface.transformed(Matrix4::from_nonuniform_scale(1.0, 1.0, 1.001)),
        surface.clone().inverse(),
    ] {
        domain.face().set_surface(changed);
        assert!(matches!(
            roots.root_chart_for_cut(0, 0.5, &mut budget, &cancel),
            Err(ValidatedGeometryError::InvalidInput),
        ));
    }
    domain.face().set_surface(surface);
    assert!(
        roots
            .root_chart_for_cut(0, 0.5, &mut budget, &cancel)
            .is_ok()
    );
}
