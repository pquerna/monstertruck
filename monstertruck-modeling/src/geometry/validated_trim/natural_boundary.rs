use super::super::validated_interval::{
    ProjectionExecution, ScalarProjection, Taylor, projected_spline,
};
use super::*;
use crate::{KnotVector, Line, Point2};

/// Original-source authorization for one natural-boundary chart completion.
///
/// The caller must establish from the original source chart/topology that these
/// two material-oriented uses meet at `vertex_id` along the selected native
/// natural boundary. Endpoint nearness alone is not that authority. Native
/// preparation checks the occurrence identities, shared vertex, clamped boundary,
/// endpoint support and whole connector/tube image against the numeric allowance.
#[derive(Clone, Copy, Debug)]
pub struct NaturalBoundaryJointEvidence {
    /// Original shared native vertex occurrence.
    pub vertex_id: VertexId<Point3>,
    /// Incoming original material-oriented coedge occurrence.
    pub incoming: ChartCoedgeIdentity<Curve, Surface>,
    /// Outgoing original material-oriented coedge occurrence.
    pub outgoing: ChartCoedgeIdentity<Curve, Surface>,
    /// Original chart's explicitly selected native tensor natural boundary.
    pub boundary: NativeSurfaceBoundary,
    /// Certified permitted absolute world-coordinate displacement from the vertex.
    /// Includes source-map, native representation and vertex-position uncertainty.
    pub max_vertex_deviation: [f64; 3],
}

/// Proves a literal native UV endpoint coordinate by exact coefficient arithmetic.
///
/// `true` means homogeneous numerator equals `value * denominator` exactly, with
/// nonzero denominator. Clamped polynomial/rational endpoints, line/polyline
/// endpoints and bounded nested literal processor factors are supported,
/// retaining every independent inverse anchor. No sampled or interval-nearness
/// equality is used. `false` also covers an unsupported exact representation;
/// checked arithmetic/cancellation/resource failures remain errors.
pub fn certified_curve2d_endpoint_coordinate(
    curve: &Curve2D,
    parameter: f64,
    axis: usize,
    value: f64,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<bool, TrimDomainError> {
    budget.charge(1, cancel)?;
    if axis >= 2 || !parameter.is_finite() || !value.is_finite() {
        return Err(TrimDomainError::InvalidChart);
    }
    let range = curve
        .try_range_tuple()
        .ok_or(TrimDomainError::InvalidChart)?;
    if parameter != range.0 && parameter != range.1 {
        return Ok(false);
    }
    let number = |value| {
        planar::Dyadic::scalar(value).ok_or(TrimDomainError::Geometry(
            ValidatedGeometryError::Unresolved,
        ))
    };
    let point = match extrusion::exact_uv_point(curve, number(parameter)?, 0, budget, cancel) {
        Ok(point) => point,
        Err(TrimDomainError::Geometry(ValidatedGeometryError::Unsupported)) => return Ok(false),
        Err(error) => return Err(error),
    };
    if point[2] == number(0.0)? {
        return Ok(false);
    }
    let expected = number(value)?
        .multiply(point[2])
        .ok_or(TrimDomainError::Geometry(
            ValidatedGeometryError::Unresolved,
        ))?;
    Ok(point[axis] == expected)
}

/// Proves that a retained native UV branch is a simple regular straight line.
///
/// A literal line or single-span degree-one polynomial/rational spline leaf,
/// every retained homogeneous `Mat3` factor and independent inverse anchor
/// establish homogeneous coordinates affine in the original scalar parameter.
/// The original span length stays homogeneous, without a rounded reciprocal.
/// Exact endpoint denominator signs
/// then exclude a pole on the whole interval; a nonzero projective tangent
/// numerator proves regularity and strict monotonicity along the image. A
/// singular matrix is allowed when that actual one-dimensional image is regular.
/// Either orientation of a finite active subrange is accepted. `false` also
/// covers unsupported leaves. This supplies no source/3D correspondence,
/// trim membership or domain-coverage authority.
pub fn certified_curve2d_is_straight_regular(
    curve: &Curve2D,
    range: [f64; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<bool, TrimDomainError> {
    budget.charge(1, cancel)?;
    if !range.into_iter().all(f64::is_finite) {
        return Err(TrimDomainError::InvalidChart);
    }
    if range[0] == range[1] {
        return Ok(false);
    }
    let domain = curve
        .try_range_tuple()
        .ok_or(TrimDomainError::InvalidChart)?;
    if !domain.0.is_finite() || !domain.1.is_finite() || domain.0 >= domain.1 {
        return Err(TrimDomainError::InvalidChart);
    }
    if range[0].min(range[1]) < domain.0 || range[0].max(range[1]) > domain.1 {
        return Err(ValidatedGeometryError::ParameterOutOfRange.into());
    }
    let mut basis = curve;
    let mut depth = 0_usize;
    while let Curve2D::TrimmedCurve(wrapper) = basis {
        budget.charge(1, cancel)?;
        depth += 1;
        if depth > super::super::MAX_NATIVE_GEOMETRY_NESTING {
            return Err(ValidatedGeometryError::ResourceLimit.into());
        }
        basis = wrapper.entity().curve().as_ref();
    }
    let affine = match basis {
        Curve2D::Line(_) => true,
        Curve2D::BsplineCurve(curve) => {
            curve.control_points().len() == 2 && curve.knot_vector().len() == 4
        }
        Curve2D::NurbsCurve(curve) => {
            curve.control_points().len() == 2 && curve.knot_vector().len() == 4
        }
        _ => false,
    };
    if !affine {
        return Ok(false);
    }
    let number = |value| {
        planar::Dyadic::scalar(value).ok_or(TrimDomainError::Geometry(
            ValidatedGeometryError::Unresolved,
        ))
    };
    let a = extrusion::exact_uv_point(curve, number(range[0])?, 0, budget, cancel)?;
    let b = extrusion::exact_uv_point(curve, number(range[1])?, 0, budget, cancel)?;
    let sign = a[2].compare(number(0.0)?);
    if sign.is_eq() || b[2].compare(number(0.0)?) != sign {
        return Ok(false);
    }
    for axis in 0..2 {
        budget.charge(1, cancel)?;
        // This cross product is (b - a) times the constant homogeneous
        // tangent numerator, not the determinant of the ambient matrix.
        let left = a[axis].multiply(b[2]).ok_or(TrimDomainError::Geometry(
            ValidatedGeometryError::Unresolved,
        ))?;
        let right = b[axis].multiply(a[2]).ok_or(TrimDomainError::Geometry(
            ValidatedGeometryError::Unresolved,
        ))?;
        if left != right {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(super) fn same_use(
    a: ChartCoedgeIdentity<Curve, Surface>,
    b: ChartCoedgeIdentity<Curve, Surface>,
) -> bool {
    a.face_id == b.face_id
        && a.wire_index == b.wire_index
        && a.coedge_index == b.coedge_index
        && a.edge_id == b.edge_id
        && a.orientation == b.orientation
}

#[derive(Clone, Copy)]
struct Node {
    range: [f64; 2],
    depth: usize,
}

pub(super) fn completion(
    face: &Face,
    vertex: Point3,
    evidence: &NaturalBoundaryJointEvidence,
    endpoints: [(&Curve2D, f64); 2],
    neighborhood: poles::JointNeighborhood<'_>,
    resources: (TrimDomainLimits, usize),
    execution: (&mut ValidatedGeometryBudget, &AtomicBool),
) -> Result<TrimJointKind, TrimDomainError> {
    let poles::JointNeighborhood {
        enclosure,
        front,
        back,
        margin,
    } = neighborhood;
    let (limits, scratch_bytes) = resources;
    let (budget, cancel) = execution;
    budget.charge(1, cancel)?;
    if !evidence
        .max_vertex_deviation
        .into_iter()
        .all(|v| v.is_finite() && v >= 0.0)
        || !(0..3).all(|axis| vertex[axis].is_finite())
    {
        return Err(TrimDomainError::InvalidChart);
    }
    admit_bytes(
        size_of::<[Node; 64]>().saturating_add(size_of::<[(Taylor, usize); 49]>()),
        scratch_bytes,
    )?;
    face.with_surface(|surface| match surface {
        Surface::BsplineSurface(carrier) => {
            let controls = carrier.control_points();
            let columns = controls.first().ok_or(TrimDomainError::InvalidChart)?.len();
            budget.charge(controls.len(), cancel)?;
            if columns == 0 || !controls.iter().all(|row| row.len() == columns) {
                return Err(TrimDomainError::InvalidChart);
            }
            prove(
                surface,
                carrier.knot_vectors(),
                [controls.len(), columns],
                |i, j| {
                    let p = controls[i][j];
                    [p.x, p.y, p.z, 1.0]
                },
                vertex,
                endpoints,
                evidence,
                enclosure,
                front,
                back,
                margin,
                limits,
                budget,
                cancel,
            )
        }
        Surface::NurbsSurface(carrier) => {
            let controls = carrier.control_points();
            let columns = controls.first().ok_or(TrimDomainError::InvalidChart)?.len();
            budget.charge(controls.len(), cancel)?;
            if columns == 0 || !controls.iter().all(|row| row.len() == columns) {
                return Err(TrimDomainError::InvalidChart);
            }
            prove(
                surface,
                carrier.knot_vectors(),
                [controls.len(), columns],
                |i, j| {
                    let p = controls[i][j];
                    [p.x, p.y, p.z, p.w]
                },
                vertex,
                endpoints,
                evidence,
                enclosure,
                front,
                back,
                margin,
                limits,
                budget,
                cancel,
            )
        }
        _ => Err(ValidatedGeometryError::Unsupported.into()),
    })
}

#[allow(clippy::too_many_arguments)]
fn prove(
    surface: &Surface,
    knots: &(KnotVector, KnotVector),
    count: [usize; 2],
    control: impl Fn(usize, usize) -> [f64; 4],
    vertex: Point3,
    endpoints: [(&Curve2D, f64); 2],
    evidence: &NaturalBoundaryJointEvidence,
    enclosure: &mut [ValidatedInterval; 2],
    front: [ValidatedInterval; 2],
    back: [ValidatedInterval; 2],
    margin: [f64; 2],
    limits: TrimDomainLimits,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<TrimJointKind, TrimDomainError> {
    budget.charge(knots.0.len().saturating_add(knots.1.len()), cancel)?;
    let (du, u) = isocurve::knot_domain(&knots.0, count[0])?;
    let (dv, v) = isocurve::knot_domain(&knots.1, count[1])?;
    if du > 32 || dv > 32 {
        return Err(ValidatedGeometryError::Unsupported.into());
    }
    let (fixed_axis, fixed_index, fixed_parameter) = match evidence.boundary {
        NativeSurfaceBoundary::UFront => (0, 0, u[0]),
        NativeSurfaceBoundary::UBack => (0, count[0] - 1, u[1]),
        NativeSurfaceBoundary::VFront => (1, 0, v[0]),
        NativeSurfaceBoundary::VBack => (1, count[1] - 1, v[1]),
    };
    for (curve, parameter) in endpoints {
        if !certified_curve2d_endpoint_coordinate(
            curve,
            parameter,
            fixed_axis,
            fixed_parameter,
            budget,
            cancel,
        )? {
            return Err(TrimDomainError::InvalidChart);
        }
    }
    let varying = 1 - fixed_axis;
    let domains = [u, v];
    let tube = [
        enclosure[0] + ValidatedInterval::new(-margin[0], margin[0])?,
        enclosure[1] + ValidatedInterval::new(-margin[1], margin[1])?,
    ];
    let allowed_fixed = ValidatedInterval::new(-margin[fixed_axis], margin[fixed_axis])?;
    for endpoint in [front, back] {
        let fixed_support = endpoint[fixed_axis] + allowed_fixed;
        if fixed_support.lower > fixed_parameter || fixed_support.upper < fixed_parameter {
            return Err(TrimDomainError::InvalidChart);
        }
    }
    // The source evidence names a natural-boundary completion within the actual
    // finite tensor domain. Outward arithmetic spill beyond its endpoints is not
    // an extrapolated spline patch and is intersected with that known support.
    let mut supported = [ValidatedInterval::point(0.0); 2];
    for axis in 0..2 {
        supported[axis] = tube[axis]
            .intersection(ValidatedInterval::new(domains[axis][0], domains[axis][1])?)
            .ok_or(TrimDomainError::InvalidChart)?;
    }
    let variable_range = [supported[varying].lower, supported[varying].upper];
    if variable_range[0] >= variable_range[1] {
        return Err(TrimDomainError::InvalidChart);
    }
    let row_knots = if varying == 0 { &knots.0 } else { &knots.1 };
    let row_control = |index| {
        if fixed_axis == 0 {
            control(fixed_index, index)
        } else {
            control(index, fixed_index)
        }
    };
    let drift = supported[fixed_axis] - ValidatedInterval::point(fixed_parameter);
    let mut pending = [Node {
        range: variable_range,
        depth: 0,
    }; 64];
    let mut length = 1;
    let mut nodes = 0_usize;
    let mut residual: Option<[ValidatedInterval; 3]> = None;
    while length != 0 {
        budget.charge(1, cancel)?;
        nodes = nodes.checked_add(1).ok_or(TrimDomainError::ResourceLimit)?;
        if nodes > limits.max_subdivisions {
            return Err(TrimDomainError::ResourceLimit);
        }
        length -= 1;
        let node = pending[length];
        let interval = ValidatedInterval::new(node.range[0], node.range[1])?;
        let parameter = Taylor::variable(interval, 0);
        let result = (|| -> Result<[ValidatedInterval; 3], ValidatedGeometryError> {
            let mut local = [ValidatedInterval::point(0.0); 3];
            for axis in 0..3 {
                let mut direction = [ValidatedInterval::point(0.0); 4];
                direction[axis] = ValidatedInterval::point(1.0);
                direction[3] = ValidatedInterval::point(-vertex[axis]);
                // Fold N - P_vertex W at each original homogeneous coefficient
                // before de Boor and division; no rounded residual control net.
                local[axis] = projected_spline(
                    row_knots,
                    count[varying],
                    parameter,
                    row_control,
                    ScalarProjection {
                        direction,
                        offset: ValidatedInterval::point(0.0),
                        center: None,
                    },
                    true,
                    &mut ProjectionExecution { budget, cancel },
                )?
                .0[0][0];
            }
            let mut rectangle = supported.map(|x| [x.lower, x.upper]);
            rectangle[varying] = node.range;
            let jet = super::super::validated_surface_jet(surface, rectangle, budget, cancel)?;
            let transverse = if fixed_axis == 0 { jet.du } else { jet.dv };
            for axis in 0..3 {
                local[axis] = local[axis] + transverse[axis] * drift;
                if !local[axis].is_finite() {
                    return Err(ValidatedGeometryError::ConditioningFailure);
                }
            }
            Ok(local)
        })();
        match result {
            Ok(local)
                if (0..3).all(|axis| {
                    local[axis].lower >= -evidence.max_vertex_deviation[axis]
                        && local[axis].upper <= evidence.max_vertex_deviation[axis]
                }) =>
            {
                residual = Some(match residual {
                    None => local,
                    Some(previous) => std::array::from_fn(|axis| previous[axis].hull(local[axis])),
                });
            }
            Ok(local)
                if (0..3).any(|axis| {
                    local[axis].lower > evidence.max_vertex_deviation[axis]
                        || local[axis].upper < -evidence.max_vertex_deviation[axis]
                }) =>
            {
                return Err(ValidatedGeometryError::Unresolved.into());
            }
            Ok(_)
            | Err(
                ValidatedGeometryError::ConditioningFailure
                | ValidatedGeometryError::NonSmooth
                | ValidatedGeometryError::DenominatorContainsZero,
            ) => {
                let middle = interval.midpoint();
                if node.depth >= 48
                    || length + 2 > pending.len()
                    || middle <= node.range[0]
                    || middle >= node.range[1]
                {
                    return Err(ValidatedGeometryError::Unresolved.into());
                }
                pending[length] = Node {
                    range: [middle, node.range[1]],
                    depth: node.depth + 1,
                };
                pending[length + 1] = Node {
                    range: [node.range[0], middle],
                    depth: node.depth + 1,
                };
                length += 2;
            }
            Err(error) => return Err(error.into()),
        }
    }
    let vertex_residual = residual.ok_or(TrimDomainError::InvalidChart)?;
    let world_enclosure =
        std::array::from_fn(|axis| vertex_residual[axis] + ValidatedInterval::point(vertex[axis]));
    let mut a = front.map(ValidatedInterval::midpoint);
    let mut b = back.map(ValidatedInterval::midpoint);
    a[fixed_axis] = fixed_parameter;
    b[fixed_axis] = fixed_parameter;
    a[varying] = a[varying].clamp(domains[varying][0], domains[varying][1]);
    b[varying] = b[varying].clamp(domains[varying][0], domains[varying][1]);
    if a[varying] == b[varying] {
        return Err(TrimDomainError::InvalidChart);
    }
    *enclosure = tube;
    Ok(TrimJointKind::NaturalBoundary {
        boundary: evidence.boundary,
        fixed_axis,
        fixed_parameter,
        variable_range,
        connector: Line(Point2::new(a[0], a[1]), Point2::new(b[0], b[1])),
        supported_parameter_box: supported.map(|x| [x.lower, x.upper]),
        vertex_residual,
        world_enclosure,
        max_vertex_deviation: evidence.max_vertex_deviation,
    })
}

#[cfg(test)]
mod natural_boundary_regressions {
    use super::*;
    use crate::{BsplineSurface, NurbsSurface, ParameterCurve, Tolerance, Vector4};
    use monstertruck_topology::{Edge, Vertex, Wire};

    fn fixture() -> (
        TrimmedFace<Point3, Curve, Surface, Curve2D>,
        NaturalBoundaryJointEvidence,
    ) {
        let vertex = Point3::new(100.0, -30.0, 7.0);
        // Mixed weights, positive denominator, deliberately non-rank-one row.
        // Literal rounded H controls, not a synthetic exact collapsed boundary.
        let row = [
            Vector4::new(100.0 + 1.0e-9, -30.0, 7.0, 1.0),
            Vector4::new(-12.5 - 2.0e-9, 3.75, -0.875, -0.125),
            Vector4::new(100.0 + 3.0e-9, -30.0, 7.0, 1.0),
        ];
        let surface = Surface::NurbsSurface(NurbsSurface::new(BsplineSurface::new(
            (
                KnotVector::from(vec![0.0, 0.0, 1.0, 1.0]),
                KnotVector::from(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            ),
            vec![
                row.to_vec(),
                vec![
                    Vector4::new(101.0, -30.0, 7.0, 1.0),
                    Vector4::new(101.0, -29.0, 7.0, 1.0),
                    Vector4::new(101.0, -28.0, 7.0, 1.0),
                ],
            ],
        )));
        let vertices = [
            Vertex::new(vertex),
            Vertex::new(Point3::new(101.0, -30.0, 7.0)),
            Vertex::new(Point3::new(101.0, -28.0, 7.0)),
        ];
        let points = [
            (Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)),
            (Point2::new(1.0, 0.0), Point2::new(1.0, 1.0)),
            (Point2::new(1.0, 1.0), Point2::new(0.0, 1.0)),
        ];
        let uv = points.map(|(a, b)| Curve2D::Line(Line(a, b)));
        let wire: Wire<Point3, Curve> = (0..3)
            .map(|index| {
                Edge::new(
                    &vertices[index],
                    &vertices[(index + 1) % 3],
                    Curve::ParameterCurve(ParameterCurve::new(
                        uv[index].clone(),
                        Box::new(surface.clone()),
                    )),
                )
            })
            .collect();
        let trimmed = TrimmedFace::new(
            Face::new(vec![wire], surface),
            vec![uv.into_iter().map(Some).collect()],
        );
        let uses: Vec<_> = trimmed
            .chart_coedges()
            .map(|coedge| coedge.identity)
            .collect();
        let evidence = NaturalBoundaryJointEvidence {
            vertex_id: vertices[0].id(),
            incoming: uses[2],
            outgoing: uses[0],
            boundary: NativeSurfaceBoundary::UFront,
            max_vertex_deviation: [1.0e-6; 3],
        };
        (trimmed, evidence)
    }

    fn limits() -> TrimDomainLimits {
        TrimDomainLimits {
            max_coedges: 8,
            max_subdivisions: 2048,
            max_retained_bytes: 1_048_576,
        }
    }

    #[test]
    fn mixed_weight_rounded_pole_row_has_quantified_nonspherical_completion() {
        let (trimmed, evidence) = fixture();
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(10_000_000);
        let prepared = prepare_face_parameter_domain_with_natural_boundaries(
            &trimmed,
            [0.0; 2],
            &[evidence],
            limits(),
            &mut budget,
            &cancel,
        )
        .unwrap();
        assert_eq!(prepared.joints().len(), 1);
        let joint = &prepared.joints()[0];
        assert_eq!(joint.vertex_id, evidence.vertex_id);
        assert!(same_use(joint.incoming, evidence.incoming));
        assert!(same_use(joint.outgoing, evidence.outgoing));
        let TrimJointKind::NaturalBoundary {
            fixed_axis,
            fixed_parameter,
            connector,
            vertex_residual,
            supported_parameter_box,
            ..
        } = joint.kind
        else {
            panic!("missing generic native natural-boundary completion")
        };
        assert_eq!((fixed_axis, fixed_parameter), (0, 0.0));
        assert!(connector.0.y > connector.1.y);
        assert_eq!(supported_parameter_box[1], [0.0, 1.0]);
        assert!(vertex_residual[0].upper > 0.0);
        assert!(
            (0..3).all(|axis| vertex_residual[axis].lower >= -1.0e-6
                && vertex_residual[axis].upper <= 1.0e-6)
        );
        assert_eq!(
            prepared
                .classify_parameter_point([0.5, 0.5], &mut budget, &cancel)
                .unwrap(),
            ParameterBoxClassification::Inside
        );
    }

    #[test]
    fn wrong_natural_boundary_or_original_use_is_refused() {
        let (trimmed, mut evidence) = fixture();
        evidence.boundary = NativeSurfaceBoundary::UBack;
        assert!(
            prepare_face_parameter_domain_with_natural_boundaries(
                &trimmed,
                [0.0; 2],
                &[evidence],
                limits(),
                &mut ValidatedGeometryBudget::new(10_000_000),
                &AtomicBool::new(false)
            )
            .is_err()
        );
        evidence.boundary = NativeSurfaceBoundary::UFront;
        evidence.incoming.coedge_index = 1;
        assert!(
            prepare_face_parameter_domain_with_natural_boundaries(
                &trimmed,
                [0.0; 2],
                &[evidence],
                limits(),
                &mut ValidatedGeometryBudget::new(10_000_000),
                &AtomicBool::new(false)
            )
            .is_err()
        );
    }

    #[test]
    fn genuine_nonzero_residual_is_not_an_exact_collapse_or_free_error() {
        let (trimmed, mut evidence) = fixture();
        evidence.max_vertex_deviation = [1.0e-12; 3];
        assert!(
            prepare_face_parameter_domain_with_natural_boundaries(
                &trimmed,
                [0.0; 2],
                &[evidence],
                limits(),
                &mut ValidatedGeometryBudget::new(10_000_000),
                &AtomicBool::new(false)
            )
            .is_err()
        );
        assert!(matches!(
            prepare_face_parameter_domain_with_natural_boundaries(
                &trimmed,
                [0.0; 2],
                &[evidence],
                limits(),
                &mut ValidatedGeometryBudget::new(10_000_000),
                &AtomicBool::new(true)
            ),
            Err(TrimDomainError::Geometry(ValidatedGeometryError::Cancelled))
        ));
    }

    #[test]
    fn endpoint_coordinate_identity_preserves_rational_weight_pose_and_inverse_anchor() {
        use crate::{
            BsplineCurve, Invertible, Matrix3, NurbsCurve, Processor, TrimmedCurve, Vector3,
        };
        let basis = Curve2D::NurbsCurve(NurbsCurve::new(BsplineCurve::new(
            KnotVector::from(vec![10.0, 10.0, 14.0, 14.0]),
            vec![Vector3::new(0.0, 6.0, 2.0), Vector3::new(0.0, 20.0, 4.0)],
        )));
        let mut posed = Processor::with_transform(
            TrimmedCurve::new(Box::new(basis), (10.0, 14.0)),
            Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 80.0, 0.0, 40.0),
        );
        posed.invert();
        let curve = Curve2D::TrimmedCurve(posed);
        let mut budget = ValidatedGeometryBudget::new(100_000);
        let cancel = AtomicBool::new(false);
        for parameter in [10.0, 14.0] {
            assert!(
                certified_curve2d_endpoint_coordinate(
                    &curve,
                    parameter,
                    0,
                    2.0,
                    &mut budget,
                    &cancel
                )
                .unwrap()
            );
            assert!(
                !certified_curve2d_endpoint_coordinate(
                    &curve,
                    parameter,
                    0,
                    2.0_f64.next_up(),
                    &mut budget,
                    &cancel
                )
                .unwrap()
            );
        }
        assert!(
            certified_curve2d_endpoint_coordinate(&curve, 10.0, 1, 0.125, &mut budget, &cancel)
                .unwrap()
        );
        assert!(
            !certified_curve2d_endpoint_coordinate(&curve, 12.0, 0, 2.0, &mut budget, &cancel)
                .unwrap()
        );
    }

    fn line_operator(matrix: crate::Matrix3, range: (f64, f64)) -> Curve2D {
        Curve2D::TrimmedCurve(crate::Processor::with_transform(
            crate::TrimmedCurve::new(
                Box::new(Curve2D::Line(Line(
                    Point2::new(0.0, 0.0),
                    Point2::new(1.0, 0.0),
                ))),
                range,
            ),
            matrix,
        ))
    }

    #[test]
    fn projective_line_is_regular_even_when_ambient_matrix_is_singular() {
        let curve = line_operator(
            crate::Matrix3::new(1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 3.0, 4.0, 2.0),
            (0.0, 1.0),
        );
        assert!(
            certified_curve2d_is_straight_regular(
                &curve,
                [0.0, 1.0],
                &mut ValidatedGeometryBudget::new(1000),
                &AtomicBool::new(false)
            )
            .unwrap()
        );
        assert!(matches!(
            certified_curve2d_is_straight_regular(
                &curve,
                [0.0, 1.0],
                &mut ValidatedGeometryBudget::new(1000),
                &AtomicBool::new(true)
            ),
            Err(TrimDomainError::Geometry(ValidatedGeometryError::Cancelled))
        ));
        assert!(
            certified_curve2d_is_straight_regular(
                &curve,
                [0.0, 1.0],
                &mut ValidatedGeometryBudget::new(0),
                &AtomicBool::new(false)
            )
            .is_err()
        );
    }

    #[test]
    fn zero_projective_tangent_and_zero_active_range_are_not_regular() {
        let curve = line_operator(
            crate::Matrix3::new(2.0, 4.0, 1.0, 0.0, 0.0, 0.0, 2.0, 4.0, 1.0),
            (0.0, 1.0),
        );
        assert!(
            !certified_curve2d_is_straight_regular(
                &curve,
                [0.0, 1.0],
                &mut ValidatedGeometryBudget::new(1000),
                &AtomicBool::new(false)
            )
            .unwrap()
        );
        assert!(
            !certified_curve2d_is_straight_regular(
                &curve,
                [0.5, 0.5],
                &mut ValidatedGeometryBudget::new(1000),
                &AtomicBool::new(false)
            )
            .unwrap()
        );
    }

    #[test]
    fn reversed_nested_line_operators_keep_their_original_scalar_anchors() {
        use crate::Invertible;
        let Curve2D::TrimmedCurve(mut inner) = line_operator(
            crate::Matrix3::new(1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 3.0, 4.0, 10.0),
            (-2.0, 7.0),
        ) else {
            unreachable!()
        };
        inner.invert();
        let mut outer = crate::Processor::new(crate::TrimmedCurve::new(
            Box::new(Curve2D::TrimmedCurve(inner)),
            (-1.0, 2.0),
        ));
        outer.invert();
        assert!(
            certified_curve2d_is_straight_regular(
                &Curve2D::TrimmedCurve(outer),
                [2.0, -1.0],
                &mut ValidatedGeometryBudget::new(1000),
                &AtomicBool::new(false)
            )
            .unwrap()
        );
    }

    #[test]
    fn denominator_zero_at_boundary_or_between_finite_endpoints_is_refused() {
        for constant in [0.0, 1.0] {
            let curve = line_operator(
                crate::Matrix3::new(1.0, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0, 0.0, constant),
                (0.0, 1.0),
            );
            assert!(
                !certified_curve2d_is_straight_regular(
                    &curve,
                    [0.0, 1.0],
                    &mut ValidatedGeometryBudget::new(1000),
                    &AtomicBool::new(false)
                )
                .unwrap()
            );
        }
    }

    #[test]
    fn affine_spline_span_keeps_original_knots_poses_and_independent_inverse_anchors() {
        use crate::{BsplineCurve, Invertible, Matrix3, Processor, TrimmedCurve};
        // Span length three has no finite dyadic reciprocal. Its homogeneous
        // scale must survive rather than fitting endpoints or dividing by three.
        let basis = Curve2D::BsplineCurve(BsplineCurve::new(
            KnotVector::from(vec![2.0, 2.0, 5.0, 5.0]),
            vec![Point2::new(7.0, -2.0), Point2::new(13.0, 4.0)],
        ));
        let mut inner = Processor::with_transform(
            TrimmedCurve::new(Box::new(basis), (2.0, 5.0)),
            Matrix3::new(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 5.0, -7.0, 1.0),
        );
        inner.invert();
        let mut outer = Processor::with_transform(
            TrimmedCurve::new(Box::new(Curve2D::TrimmedCurve(inner)), (2.5, 4.0)),
            Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -3.0, 11.0, 1.0),
        );
        outer.invert();
        let curve = Curve2D::TrimmedCurve(outer);
        let mut budget = ValidatedGeometryBudget::new(10_000);
        let cancel = AtomicBool::new(false);
        assert!(
            certified_curve2d_is_straight_regular(&curve, [4.0, 2.5], &mut budget, &cancel,)
                .unwrap()
        );
        for t in [2.5, 3.25, 4.0] {
            assert!(
                curve
                    .subs(t)
                    .near(&Point2::new(4.0 * t + 10.0, 6.0 * t - 11.0))
            );
        }
        assert!(
            certified_curve2d_endpoint_coordinate(&curve, 2.5, 0, 20.0, &mut budget, &cancel,)
                .unwrap()
        );
        assert!(
            certified_curve2d_endpoint_coordinate(&curve, 4.0, 1, 13.0, &mut budget, &cancel,)
                .unwrap()
        );
    }

    fn homogeneous_affine_span(controls: [crate::Vector3; 2]) -> Curve2D {
        use crate::{BsplineCurve, NurbsCurve};
        Curve2D::NurbsCurve(NurbsCurve::new(BsplineCurve::new(
            KnotVector::from(vec![2.0, 2.0, 5.0, 5.0]),
            Vec::from(controls),
        )))
    }

    #[test]
    fn rational_affine_spans_preserve_unequal_positive_and_negative_equivalent_weights() {
        use crate::Vector3;
        for sign in [1.0, -1.0] {
            let curve = homogeneous_affine_span([
                Vector3::new(0.0, 0.0, 2.0) * sign,
                Vector3::new(6.0, 3.0, 3.0) * sign,
            ]);
            assert!(
                certified_curve2d_is_straight_regular(
                    &curve,
                    [2.5, 4.5],
                    &mut ValidatedGeometryBudget::new(1000),
                    &AtomicBool::new(false),
                )
                .unwrap()
            );
            for t in [2.5, 3.0, 4.5] {
                let expected =
                    Point2::new(6.0 * (t - 2.0) / (t + 4.0), 3.0 * (t - 2.0) / (t + 4.0));
                assert!(curve.subs(t).near(&expected));
            }
        }
    }

    #[test]
    fn rational_affine_span_separates_actual_subrange_poles_and_rejects_knot_domain_escape() {
        use crate::{Processor, TrimmedCurve, Vector3};
        let curve =
            homogeneous_affine_span([Vector3::new(0.0, 0.0, -1.0), Vector3::new(1.0, 0.0, 1.0)]);
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(10_000);
        assert!(
            !certified_curve2d_is_straight_regular(&curve, [2.0, 5.0], &mut budget, &cancel,)
                .unwrap()
        );
        assert!(
            certified_curve2d_is_straight_regular(&curve, [4.0, 5.0], &mut budget, &cancel,)
                .unwrap()
        );
        assert!(curve.subs(4.0).near(&Point2::new(2.0, 0.0)));
        assert!(curve.subs(5.0).near(&Point2::new(1.0, 0.0)));
        let escaped = Curve2D::TrimmedCurve(Processor::new(TrimmedCurve::new(
            Box::new(curve),
            (1.0, 3.0),
        )));
        assert!(matches!(
            certified_curve2d_is_straight_regular(&escaped, [1.5, 2.5], &mut budget, &cancel,),
            Err(TrimDomainError::Geometry(
                ValidatedGeometryError::ParameterOutOfRange
            ))
        ));
    }

    #[test]
    fn endpoint_collinearity_does_not_admit_curved_or_multiple_spline_spans() {
        use crate::BsplineCurve;
        for (knots, middle) in [
            (vec![2.0, 2.0, 2.0, 5.0, 5.0, 5.0], 0.5),
            (vec![2.0, 2.0, 3.5, 5.0, 5.0], 1.0),
        ] {
            let curve = Curve2D::BsplineCurve(BsplineCurve::new(
                KnotVector::from(knots),
                vec![
                    Point2::new(0.0, 0.0),
                    Point2::new(1.0, 1.0),
                    Point2::new(2.0, 0.0),
                ],
            ));
            assert!(curve.subs(3.5).near(&Point2::new(1.0, middle)));
            assert!(
                !certified_curve2d_is_straight_regular(
                    &curve,
                    [2.0, 5.0],
                    &mut ValidatedGeometryBudget::new(1000),
                    &AtomicBool::new(false),
                )
                .unwrap()
            );
        }
    }
}
