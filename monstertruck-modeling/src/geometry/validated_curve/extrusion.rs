//! Constructive generated sections of an original extrusion by a transverse plane.
//!
//! The original profile is borrowed from the existing retained carrier. Its
//! parameter is prescribed, not enclosed in an impossible open endpoint tube.
//! A certified nonsingular linear system determines the other three coordinates.
use super::*;
use crate::BoundedCurve;

/// Coefficients owned by the same immutable binding as the original carriers.
/// No profile, surface or alternative geometry is retained here.
#[derive(Debug)]
pub(super) struct Graph {
    parameter: [f64; 2],
    profile_parameter: [f64; 2],
    profile_domain: [f64; 2],
    scale: [ValidatedInterval; 3],
    inverse: [[ValidatedInterval; 3]; 3],
    origin: [ValidatedInterval; 3],
    linear: [[ValidatedInterval; 3]; 3],
    offset: [ValidatedInterval; 3],
}

fn profile(curve: &Curve) -> Result<&Curve, ValidatedGeometryError> {
    let (Surface::ExtrusionSurface(source), Surface::Plane(_), _) = carriers(curve)? else {
        return Err(ValidatedGeometryError::InvalidInput);
    };
    Ok(source.entity_curve())
}

pub(super) fn bounded_profile_domain(curve: &Curve) -> Result<[f64; 2], ValidatedGeometryError> {
    // Bare native LINE has unbounded affine semantics, despite its nominal
    // drawing range. A genuine outer trim, spline support or bounded parameter
    // expression supplies the closed source domain used by this operator.
    match curve {
        Curve::BsplineCurve(_) | Curve::NurbsCurve(_) | Curve::TrimmedCurve(_) => {}
        Curve::ParameterCurve(parameter) => match parameter.curve() {
            Curve2D::Line(_) => return Err(ValidatedGeometryError::Unsupported),
            Curve2D::Polyline(_)
            | Curve2D::BsplineCurve(_)
            | Curve2D::NurbsCurve(_)
            | Curve2D::Conic(_)
            | Curve2D::TrimmedCurve(_) => {}
        },
        Curve::Line(_) | Curve::IntersectionCurve(_) => {
            return Err(ValidatedGeometryError::Unsupported);
        }
    }
    let range = curve.range_tuple();
    if !range.0.is_finite() || !range.1.is_finite() || range.0 >= range.1 {
        return Err(ValidatedGeometryError::ParameterOutOfRange);
    }
    Ok([range.0, range.1])
}

fn magnitude(vector: [ValidatedInterval; 3]) -> Result<f64, ValidatedGeometryError> {
    let square = vector
        .into_iter()
        .fold(ZERO, |sum, value| sum + value.square());
    if !square.is_finite() || square.upper < 0.0 {
        return Err(ValidatedGeometryError::ConditioningFailure);
    }
    Ok(square.upper.sqrt().next_up())
}

impl Graph {
    fn new(
        curve: &Curve,
        parameter: [f64; 2],
        anchors: [[f64; 2]; 2],
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> Result<Self, ValidatedGeometryError> {
        budget.charge(256, cancel)?;
        let (Surface::ExtrusionSurface(source), Surface::Plane(plane), _) = carriers(curve)? else {
            return Err(ValidatedGeometryError::Unsupported);
        };
        let profile_domain = bounded_profile_domain(source.entity_curve())?;
        let profile_parameter = anchors.map(|anchor| anchor[0]);
        if !parameter
            .into_iter()
            .chain(profile_domain)
            .chain(anchors.into_iter().flatten())
            .all(f64::is_finite)
            || parameter[0] >= parameter[1]
            || profile_domain[0] >= profile_domain[1]
            || profile_parameter[0] == profile_parameter[1]
            || profile_parameter
                .into_iter()
                .any(|u| u < profile_domain[0] || u > profile_domain[1])
        {
            return Err(ValidatedGeometryError::ParameterOutOfRange);
        }
        let slope = (ValidatedInterval::point(profile_parameter[1])
            - ValidatedInterval::point(profile_parameter[0]))
            * (ValidatedInterval::point(parameter[1]) - ValidatedInterval::point(parameter[0]))
                .reciprocal()?;
        if slope.contains_zero() {
            return Err(ValidatedGeometryError::NonTransverse);
        }
        let square = slope.square();
        let scale = [slope, square, square * slope];
        if !scale.into_iter().all(ValidatedInterval::is_finite) {
            return Err(ValidatedGeometryError::ConditioningFailure);
        }
        let direction = source.extruding_vector();
        let direction = [direction.x, direction.y, direction.z].map(ValidatedInterval::point);
        let (a, b) = coverage::plane_axes(plane);
        let columns = [direction, negate3(a), negate3(b)];
        let cofactors = [
            cross3(columns[1], columns[2]),
            cross3(columns[2], columns[0]),
            cross3(columns[0], columns[1]),
        ];
        let determinant = dot3(columns[0], cofactors[0]);
        if determinant.contains_zero() {
            return Err(ValidatedGeometryError::NonTransverse);
        }
        let inverse_determinant = determinant.reciprocal()?;
        let inverse = cofactors.map(|row| row.map(|value| value * inverse_determinant));
        let origin =
            [plane.origin().x, plane.origin().y, plane.origin().z].map(ValidatedInterval::point);
        let linear = std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                (if row == column { ONE } else { ZERO }) - direction[row] * inverse[0][column]
            })
        });
        let displacement = dot3(inverse[0], origin);
        let offset = direction.map(|value| value * displacement);
        if !inverse
            .into_iter()
            .flatten()
            .chain(linear.into_iter().flatten())
            .chain(offset)
            .all(ValidatedInterval::is_finite)
        {
            return Err(ValidatedGeometryError::ConditioningFailure);
        }
        Ok(Self {
            parameter,
            profile_parameter,
            profile_domain,
            scale,
            inverse,
            origin,
            linear,
            offset,
        })
    }

    /// Exact endpoint identities of the prescribed affine map avoid outward
    /// arithmetic crossing a genuine finite profile endpoint. No UV is clamped.
    pub(super) fn profile_span(
        &self,
        parameter: ValidatedInterval,
    ) -> Result<ValidatedInterval, ValidatedGeometryError> {
        if parameter.lower < self.parameter[0] || parameter.upper > self.parameter[1] {
            return Err(ValidatedGeometryError::ParameterOutOfRange);
        }
        let endpoint = |t| {
            if t == self.parameter[0] {
                ValidatedInterval::point(self.profile_parameter[0])
            } else if t == self.parameter[1] {
                ValidatedInterval::point(self.profile_parameter[1])
            } else {
                ValidatedInterval::point(self.profile_parameter[0])
                    + (ValidatedInterval::point(t) - ValidatedInterval::point(self.parameter[0]))
                        * self.scale[0]
            }
        };
        let span = endpoint(parameter.lower).hull(endpoint(parameter.upper));
        self.check_profile_span(span)?;
        Ok(span)
    }

    fn check_profile_span(&self, span: ValidatedInterval) -> Result<(), ValidatedGeometryError> {
        if !span.is_finite()
            || span.lower < self.profile_domain[0]
            || span.upper > self.profile_domain[1]
        {
            Err(ValidatedGeometryError::ParameterOutOfRange)
        } else {
            Ok(())
        }
    }

    pub(super) fn profile_endpoint(&self, back: bool) -> f64 {
        self.profile_parameter[usize::from(back)]
    }

    pub(super) fn covers(&self, source_index: usize, cell: [[f64; 2]; 2]) -> bool {
        source_index == 0
            && self.profile_parameter[0].min(self.profile_parameter[1]) <= cell[0][0]
            && cell[0][1] <= self.profile_parameter[0].max(self.profile_parameter[1])
    }

    pub(super) fn is_smooth(
        &self,
        curve: &Curve,
        parameter: ValidatedInterval,
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> Result<bool, ValidatedGeometryError> {
        curve_span_smooth(
            profile(curve)?,
            self.profile_span(parameter)?,
            1,
            false,
            budget,
            cancel,
        )
    }

    /// Encloses the exact original carrier intersection in original profile u.
    /// The correlated linear coefficients retain varying profile height.
    pub(super) fn profile_jet(
        &self,
        curve: &Curve,
        span: ValidatedInterval,
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> Result<(ValidatedCurveJet, [[ValidatedInterval; 2]; 2]), ValidatedGeometryError> {
        budget.charge(128, cancel)?;
        self.check_profile_span(span)?;
        let source = validated_curve_piecewise_jet(
            profile(curve)?,
            [span.lower, span.upper],
            budget,
            cancel,
        )?;
        let apply = |value: [ValidatedInterval; 3]| self.linear.map(|row| dot3(row, value));
        let position = add3(apply(source.position), self.offset);
        let residual = subtract3(self.origin, source.position);
        let coordinates = self.inverse.map(|row| dot3(row, residual));
        let jet = ValidatedCurveJet {
            position,
            first: apply(source.first),
            second: apply(source.second),
            third: apply(source.third),
        };
        if ![jet.position, jet.first, jet.second, jet.third]
            .into_iter()
            .flatten()
            .chain(coordinates)
            .all(ValidatedInterval::is_finite)
        {
            return Err(ValidatedGeometryError::ConditioningFailure);
        }
        Ok((
            jet,
            [[span, coordinates[0]], [coordinates[1], coordinates[2]]],
        ))
    }

    pub(super) fn certify(
        &self,
        curve: &Curve,
        parameter: ValidatedInterval,
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> Result<Tube, ValidatedGeometryError> {
        let (mut jet, charts) =
            self.profile_jet(curve, self.profile_span(parameter)?, budget, cancel)?;
        jet.first = jet.first.map(|value| value * self.scale[0]);
        jet.second = jet.second.map(|value| value * self.scale[1]);
        jet.third = jet.third.map(|value| value * self.scale[2]);
        if ![jet.first, jet.second, jet.third]
            .into_iter()
            .flatten()
            .all(ValidatedInterval::is_finite)
        {
            return Err(ValidatedGeometryError::ConditioningFailure);
        }
        let domain = flatten(charts);
        Ok(Tube {
            domain,
            rooted_domain: domain,
            authority: TubeAuthority::Constructive(jet),
        })
    }
}

/// Returns no proposal for an inapplicable operator; all successful pieces have
/// a complete constructive proof on the retained original source and range.
pub(super) fn prepare(
    curve: &Curve,
    range: [f64; 2],
    anchor: Option<ValidatedFacePlaneSourceAnchor>,
    options: &ValidatedCurveOptions,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Option<(Graph, Vec<ValidatedCurvePiece>)>, ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    let Some(ValidatedFacePlaneSourceAnchor::Both { front, back }) = anchor else {
        return Ok(None);
    };
    let Surface::ExtrusionSurface(source) = carriers(curve)?.0 else {
        return Ok(None);
    };
    if bounded_profile_domain(source.entity_curve()).is_err() {
        return Ok(None);
    }
    let base = size_of::<PreparedValidatedCurve<'_>>()
        .checked_add(size_of::<Graph>())
        .ok_or(ValidatedGeometryError::ResourceLimit)?;
    if base
        .checked_add(size_of::<ValidatedCurvePiece>())
        .is_none_or(|size| size > options.max_retained_bytes)
    {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    let graph = match Graph::new(curve, range, [front, back], budget, cancel) {
        Ok(graph) => graph,
        Err(error) if retryable(error) || error == ValidatedGeometryError::Unsupported => {
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    let (surface, _, leader) = carriers(curve)?;
    for (index, chart) in [front, back].into_iter().enumerate() {
        budget.charge(1, cancel)?;
        let endpoint = ValidatedInterval::point(range[index]);
        let tube = graph.certify(curve, endpoint, budget, cancel)?;
        let jet = implicit_jet(&tube)?;
        let source =
            validated_surface_piecewise_jet(surface, chart.map(|t| [t, t]), budget, cancel)?;
        let guide = validated_projection_leader_jet(leader, [range[index]; 2], budget, cancel)?;
        if magnitude(subtract3(source.position, jet.position))? > options.endpoint_tolerance
            || magnitude(subtract3(guide.position, jet.position))? > options.endpoint_tolerance
            || dot3(guide.first, jet.first).lower <= 0.0
        {
            return Ok(None);
        }
    }
    let mut pieces = Vec::new();
    let mut pending = [(range, 0usize); 49];
    let mut count = 1;
    while count != 0 {
        budget.charge(1, cancel)?;
        count -= 1;
        let (part, depth) = pending[count];
        let tube = graph.certify(curve, interval_range(part)?, budget, cancel)?;
        let jet = implicit_jet(&tube)?;
        if !jet
            .first
            .into_iter()
            .any(|value| value.lower > 0.0 || value.upper < 0.0)
        {
            let middle = part[0] * 0.5 + part[1] * 0.5;
            if depth >= options.max_depth || middle <= part[0] || middle >= part[1] {
                return Err(ValidatedGeometryError::NonTransverse);
            }
            pending[count] = ([middle, part[1]], depth + 1);
            pending[count + 1] = ([part[0], middle], depth + 1);
            count += 2;
            continue;
        }
        reserve_native_vec(
            &mut pieces,
            options.max_retained_bytes,
            base,
            budget,
            cancel,
        )?;
        pieces.push(ValidatedCurvePiece {
            parameter: part,
            jet,
            source_parameters: Some(unflatten(tube.domain)),
            contraction_bound: tube.contraction_bound(),
            surface_residual: tube.surface_residual(),
            projection: Projection::SourceProfile,
        });
    }
    Ok(Some((graph, pieces)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BsplineCurve, Curve2D, Edge, ExtrusionSurface, Face, KnotVector, Line, Point2, Point3,
        TrimDomainLimits, Vector3, Wire, builder, prepare_face_parameter_domain_with_uncertainty,
    };
    use monstertruck_topology::trimmed::TrimmedFace;

    fn fixture(middle_height: f64) -> (Surface, Plane, Curve, [Point3; 3]) {
        let controls = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, middle_height),
            Point3::new(2.0, 0.0, 0.0),
        ];
        let profile = Curve::BsplineCurve(BsplineCurve::new(
            KnotVector::bezier_knot(2),
            controls.to_vec(),
        ));
        let surface = Surface::ExtrusionSurface(ExtrusionSurface::by_extrusion(
            profile,
            Vector3::new(0.0, 0.0, 8.0),
        ));
        let plane = Plane::new(
            Point3::new(0.0, 0.0, 2.0),
            Point3::new(1.0, 0.0, 2.0),
            Point3::new(0.0, 1.0, 2.0),
        );
        // This chord supplies only the original nonunit gauge and orientation.
        // Interior physical positions must come from the complete source graph.
        let guide = Curve::BsplineCurve(BsplineCurve::new(
            KnotVector::from(vec![10.0, 10.0, 14.0, 14.0]),
            vec![Point3::new(2.0, 0.0, 2.0), Point3::new(0.0, 0.0, 2.0)],
        ));
        (surface, plane, guide, controls)
    }

    fn anchors() -> ValidatedFacePlaneSourceAnchor {
        ValidatedFacePlaneSourceAnchor::Both {
            front: [1.0, 0.25],
            back: [0.0, 0.25],
        }
    }

    #[test]
    fn finite_original_bspline_closed_endpoints_and_varying_profile_height_are_constructive() {
        for middle_height in [0.0, 1.0] {
            let (surface, plane, guide, controls) = fixture(middle_height);
            let cancel = AtomicBool::new(false);
            let mut budget = ValidatedGeometryBudget::new(2_000_000);
            let options = ValidatedCurveOptions::default();
            let prepared = prepared_face_plane_curve(
                &guide,
                ValidatedFacePlaneSource {
                    surface: &surface,
                    anchor: Some(anchors()),
                    endpoint_coordinate_planes: [None; 2],
                },
                &plane,
                [10.0, 14.0],
                &options,
                &mut budget,
                &cancel,
            )
            .expect("both genuine finite endpoints have complete original graph authority");
            for (parameter, expected) in [
                (10.0, [2.0, 0.0, 2.0]),
                (12.0, [1.0, 0.5, 2.0]),
                (14.0, [0.0, 0.0, 2.0]),
            ] {
                let actual = prepared.evaluate(parameter, &mut budget, &cancel).unwrap();
                for (actual, expected) in [actual.position.x, actual.position.y, actual.position.z]
                    .into_iter()
                    .zip(expected)
                {
                    assert!((actual - expected).abs() < 1.0e-10);
                }
            }
            let jet = prepared.jet([10.0, 14.0], &mut budget, &cancel).unwrap();
            assert!(jet.first[0].upper < 0.0);
            assert!(jet.position[2].lower <= 2.0 && 2.0 <= jet.position[2].upper);
            assert!(jet.position[2].width() < 1.0e-10);
            assert!(
                prepared
                    .is_smooth_on([10.0, 14.0], &mut budget, &cancel)
                    .unwrap()
            );
            assert!(matches!(
                prepared.jet([10.0, 14.1], &mut budget, &cancel),
                Err(ValidatedGeometryError::ParameterOutOfRange),
            ));

            // Build the actual original four-coedge circuit, including the
            // curved lower/upper source masters, rather than sampled chords.
            let upper: Vec<_> = controls
                .iter()
                .rev()
                .map(|point| *point + Vector3::new(0.0, 0.0, 8.0))
                .collect();
            let points = [
                controls[0],
                controls[2],
                controls[2] + Vector3::new(0.0, 0.0, 8.0),
                controls[0] + Vector3::new(0.0, 0.0, 8.0),
            ];
            let originals = [
                Curve::BsplineCurve(BsplineCurve::new(
                    KnotVector::bezier_knot(2),
                    controls.to_vec(),
                )),
                Curve::Line(Line(points[1], points[2])),
                Curve::BsplineCurve(BsplineCurve::new(KnotVector::bezier_knot(2), upper)),
                Curve::Line(Line(points[3], points[0])),
            ];
            let vertices = points.map(builder::vertex);
            let wire: Wire = (0..4)
                .map(|index| {
                    Edge::new(
                        &vertices[index],
                        &vertices[(index + 1) % 4],
                        originals[index].clone(),
                    )
                })
                .collect();
            let uv = [
                Point2::new(0.0, 0.0),
                Point2::new(1.0, 0.0),
                Point2::new(1.0, 1.0),
                Point2::new(0.0, 1.0),
            ];
            let trims = (0..4)
                .map(|index| Some(Curve2D::Line(Line(uv[index], uv[(index + 1) % 4]))))
                .collect();
            let trimmed = TrimmedFace::new(Face::new(vec![wire], surface.clone()), vec![trims]);
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
            let boundaries: Vec<_> = originals
                .iter()
                .map(|curve| {
                    prepared_validated_curve(curve, [0.0, 1.0], &options, &mut budget, &cancel)
                        .unwrap()
                })
                .collect();
            let bindings = [(1, false, 0.25), (3, true, 0.75)].map(|(index, back, cut)| {
                ValidatedPlaneTrimEndpointBinding {
                    coedge_index: index,
                    original_edge_id: domain.coedges()[index].identity.edge_id,
                    boundary: &boundaries[index],
                    boundary_range: [0.0, 1.0],
                    boundary_scale: 1.0,
                    candidate_index: 0,
                    endpoint_is_back: back,
                    source_cut_parameter: cut,
                    source_chart_error_mm: 0.0,
                }
            });
            let coverage_options = ValidatedPlaneCoverageOptions {
                maximum_endpoint_extension_mm: 1.0e-7,
                ..Default::default()
            };
            let coverage = validated_face_plane_coverage(
                &domain,
                &plane,
                &[&prepared],
                &bindings,
                &coverage_options,
                &mut budget,
                &cancel,
            )
            .expect("complete original cap incidence and the whole finite source graph");
            assert!(coverage.covered_cells > 0);
            assert_eq!(coverage.endpoint_strata.len(), 2);
            assert!(
                coverage
                    .endpoint_strata
                    .iter()
                    .all(|receipt| receipt.maximum_extension_mm()
                        <= coverage_options.maximum_endpoint_extension_mm)
            );
            assert!(
                validated_face_plane_coverage(
                    &domain,
                    &plane,
                    &[&prepared],
                    &[],
                    &coverage_options,
                    &mut ValidatedGeometryBudget::new(2_000_000),
                    &cancel,
                )
                .is_err(),
                "constructive bodies never substitute for original trim-root ownership"
            );
            if middle_height > 0.0 {
                // The original lower curved coedge has two additional,
                // off-grid plane roots. Valid carrier graphs must not turn
                // those unowned interior cap/port events into complete faces.
                let low_plane = Plane::new(
                    Point3::new(0.0, 0.0, 0.25),
                    Point3::new(1.0, 0.0, 0.25),
                    Point3::new(0.0, 1.0, 0.25),
                );
                let low_guide = Curve::BsplineCurve(BsplineCurve::new(
                    KnotVector::from(vec![10.0, 10.0, 14.0, 14.0]),
                    vec![Point3::new(2.0, 0.0, 0.25), Point3::new(0.0, 0.0, 0.25)],
                ));
                let low = prepared_face_plane_curve(
                    &low_guide,
                    ValidatedFacePlaneSource {
                        surface: &surface,
                        anchor: Some(ValidatedFacePlaneSourceAnchor::Both {
                            front: [1.0, 0.03125],
                            back: [0.0, 0.03125],
                        }),
                        endpoint_coordinate_planes: [None; 2],
                    },
                    &low_plane,
                    [10.0, 14.0],
                    &options,
                    &mut ValidatedGeometryBudget::new(2_000_000),
                    &cancel,
                )
                .unwrap();
                let low_bindings = bindings.map(|mut binding| {
                    binding.source_cut_parameter = if binding.endpoint_is_back {
                        0.96875
                    } else {
                        0.03125
                    };
                    binding
                });
                assert!(
                    validated_face_plane_coverage(
                        &domain,
                        &low_plane,
                        &[&low],
                        &low_bindings,
                        &coverage_options,
                        &mut ValidatedGeometryBudget::new(2_000_000),
                        &cancel,
                    )
                    .is_err(),
                    "complete original off-grid roots remain mandatory"
                );
            }

            // Same endpoints and guide, but a different original profile interior.
            // A candidate's job-local source authority cannot migrate to this face.
            let mut wrong_controls = controls;
            wrong_controls[1].y += 0.5;
            let wrong_source = Surface::ExtrusionSurface(ExtrusionSurface::by_extrusion(
                Curve::BsplineCurve(BsplineCurve::new(
                    KnotVector::bezier_knot(2),
                    wrong_controls.to_vec(),
                )),
                Vector3::new(0.0, 0.0, 8.0),
            ));
            let wrong = prepared_face_plane_curve(
                &guide,
                ValidatedFacePlaneSource {
                    surface: &wrong_source,
                    anchor: Some(anchors()),
                    endpoint_coordinate_planes: [None; 2],
                },
                &plane,
                [10.0, 14.0],
                &options,
                &mut ValidatedGeometryBudget::new(2_000_000),
                &cancel,
            )
            .unwrap();
            assert!(
                validated_face_plane_coverage(
                    &domain,
                    &plane,
                    &[&wrong],
                    &bindings,
                    &coverage_options,
                    &mut ValidatedGeometryBudget::new(2_000_000),
                    &cancel,
                )
                .is_err(),
                "endpoint agreement cannot replace the whole original source binding"
            );
        }
    }

    #[test]
    fn constructive_original_graph_rejects_wrong_anchors_domain_winding_and_ill_conditioning() {
        let (surface, plane, guide, _) = fixture(1.0);
        let owned = Curve::IntersectionCurve(SurfaceCurve::with_boundaries(
            Box::new(surface.clone()),
            Box::new(Surface::Plane(plane)),
            Box::new(guide.clone()),
            None,
            None,
        ));
        let cancel = AtomicBool::new(false);
        let options = ValidatedCurveOptions::default();
        for (front, back) in [
            ([1.0, 0.3], [0.0, 0.25]),
            ([1.01, 0.25], [0.0, 0.25]),
            ([0.0, 0.25], [1.0, 0.25]),
            ([1.0, 0.25], [1.0, 0.25]),
        ] {
            assert!(
                prepare(
                    &owned,
                    [10.0, 14.0],
                    Some(ValidatedFacePlaneSourceAnchor::Both { front, back }),
                    &options,
                    &mut ValidatedGeometryBudget::new(2_000_000),
                    &cancel,
                )
                .unwrap()
                .is_none()
            );
        }
        let (Surface::ExtrusionSurface(source), _, _) = carriers(&owned).unwrap() else {
            panic!()
        };
        for (wrong_surface, wrong_plane) in [
            (
                Surface::ExtrusionSurface(ExtrusionSurface::by_extrusion(
                    source.entity_curve().clone(),
                    Vector3::new(0.5, 0.0, 8.0),
                )),
                plane,
            ),
            (
                surface.clone(),
                Plane::new(
                    Point3::new(0.0, 0.0, 2.125),
                    Point3::new(1.0, 0.0, 2.125),
                    Point3::new(0.0, 1.0, 2.125),
                ),
            ),
        ] {
            let wrong = Curve::IntersectionCurve(SurfaceCurve::with_boundaries(
                Box::new(wrong_surface),
                Box::new(Surface::Plane(wrong_plane)),
                Box::new(guide.clone()),
                None,
                None,
            ));
            assert!(
                prepare(
                    &wrong,
                    [10.0, 14.0],
                    Some(anchors()),
                    &options,
                    &mut ValidatedGeometryBudget::new(2_000_000),
                    &cancel,
                )
                .unwrap()
                .is_none(),
                "wrong physical direction or plane offset is not authority"
            );
        }
        let near_parallel = Curve::IntersectionCurve(SurfaceCurve::with_boundaries(
            Box::new(Surface::ExtrusionSurface(ExtrusionSurface::by_extrusion(
                source.entity_curve().clone(),
                Vector3::new(1.0, 0.0, f64::from_bits(1)),
            ))),
            Box::new(Surface::Plane(plane)),
            Box::new(guide.clone()),
            None,
            None,
        ));
        assert!(
            Graph::new(
                &near_parallel,
                [10.0, 14.0],
                [[1.0, 0.25], [0.0, 0.25]],
                &mut ValidatedGeometryBudget::new(2_000_000),
                &cancel,
            )
            .is_err()
        );
        let proposal = ValidatedFacePlaneSource {
            surface: &surface,
            anchor: Some(anchors()),
            endpoint_coordinate_planes: [None; 2],
        };
        assert!(matches!(
            prepared_face_plane_curve(
                &guide,
                proposal,
                &plane,
                [10.0, 14.0],
                &options,
                &mut ValidatedGeometryBudget::new(0),
                &cancel,
            ),
            Err(ValidatedGeometryError::ResourceLimit)
        ));
        assert!(matches!(
            prepared_face_plane_curve(
                &guide,
                proposal,
                &plane,
                [10.0, 14.0],
                &options,
                &mut ValidatedGeometryBudget::new(2_000_000),
                &AtomicBool::new(true),
            ),
            Err(ValidatedGeometryError::Cancelled)
        ));
        assert!(matches!(
            prepared_face_plane_curve(
                &guide,
                proposal,
                &plane,
                [10.0, 14.0],
                &ValidatedCurveOptions {
                    max_retained_bytes: 0,
                    ..options
                },
                &mut ValidatedGeometryBudget::new(2_000_000),
                &cancel,
            ),
            Err(ValidatedGeometryError::ResourceLimit)
        ));
        assert!(matches!(
            prepared_face_plane_curve(
                &guide,
                ValidatedFacePlaneSource {
                    anchor: Some(ValidatedFacePlaneSourceAnchor::Both {
                        front: [1.0, 0.25],
                        back: [0.0, f64::NAN],
                    }),
                    ..proposal
                },
                &plane,
                [10.0, 14.0],
                &options,
                &mut ValidatedGeometryBudget::new(2_000_000),
                &cancel,
            ),
            Err(ValidatedGeometryError::InvalidInput)
        ));
    }
}

#[cfg(test)]
mod bounded_profile_tests {
    use super::*;
    use crate::{
        BsplineCurve, ExtrusionSurface, KnotVector, Line, Point3, Processor, TrimmedCurve, Vector3,
    };

    #[test]
    fn original_trim_bounds_not_bare_line_drawing_range_own_constructive_profile_support() {
        let line = Curve::Line(Line(Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 0.0, 0.0)));
        assert!(matches!(
            bounded_profile_domain(&line),
            Err(ValidatedGeometryError::Unsupported)
        ));
        let profile = Curve::TrimmedCurve(Processor::new(TrimmedCurve::new(
            Box::new(line),
            (-2.0, 3.0),
        )));
        assert_eq!(bounded_profile_domain(&profile).unwrap(), [-2.0, 3.0]);
        let source = Surface::ExtrusionSurface(ExtrusionSurface::by_extrusion(
            profile,
            Vector3::new(0.0, 0.0, 8.0),
        ));
        let plane = Plane::new(
            Point3::new(0.0, 0.0, 2.0),
            Point3::new(1.0, 0.0, 2.0),
            Point3::new(0.0, 1.0, 2.0),
        );
        let guide = Curve::BsplineCurve(BsplineCurve::new(
            KnotVector::from(vec![10.0, 10.0, 14.0, 14.0]),
            vec![Point3::new(6.0, 0.0, 2.0), Point3::new(-4.0, 0.0, 2.0)],
        ));
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(2_000_000);
        let prepared = prepared_face_plane_curve(
            &guide,
            ValidatedFacePlaneSource {
                surface: &source,
                anchor: Some(ValidatedFacePlaneSourceAnchor::Both {
                    front: [3.0, 0.25],
                    back: [-2.0, 0.25],
                }),
                endpoint_coordinate_planes: [None; 2],
            },
            &plane,
            [10.0, 14.0],
            &ValidatedCurveOptions::default(),
            &mut budget,
            &cancel,
        )
        .unwrap();
        for (parameter, x) in [(10.0, 6.0), (14.0, -4.0)] {
            let point = prepared.evaluate(parameter, &mut budget, &cancel).unwrap();
            assert!((point.position.x - x).abs() < 1.0e-10);
            assert!((point.position.z - 2.0).abs() < 1.0e-10);
        }
    }
}
