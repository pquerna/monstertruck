use super::*;
use crate::{Line, Matrix3, Point2, Processor, TrimmedCurve, UnitCircle, validated_surface_jet};
use std::f64::consts::PI;

/// Singular stratum of a genuinely analytic native spherical carrier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SphericalPole {
    /// The sphere's positive local polar axis.
    North,
    /// The sphere's negative local polar axis.
    South,
}

/// Authority and numerical uncertainty of a native chart completion.
#[derive(Clone, Copy, Debug)]
pub enum TrimJointKind {
    /// No singular-stratum assertion; all local endpoint completions are retained.
    EndpointUncertainty,
    /// A bounded completion adjacent to a declared analytic spherical pole stratum.
    ///
    /// The connector is a covering-chart stratum, not an invented source edge.
    /// Its entire carrier image and native-vertex residual are enclosed. In
    /// particular, binary64 `PI` is never asserted to be the real south pole.
    /// The original vertex need not be asserted exactly equal to that stratum:
    /// endpoint conditioning and all possible carrier displacement remain in
    /// `polar_coordinate`, `world_enclosure` and `vertex_residual`.
    SphericalPole {
        /// Native chart axis containing the polar coordinate, after processor inversion.
        polar_axis: usize,
        /// Declared analytic pole stratum.
        pole: SphericalPole,
        /// Exact finite nominal UV completion with native domain `[0, 1]`.
        connector: Line<Point2>,
        /// Inclusive polar-coordinate enclosure, including source-map error.
        polar_coordinate: ValidatedInterval,
        /// World image of every completion in the complete UV joint envelope.
        world_enclosure: [ValidatedInterval; 3],
        /// World image minus the original native shared vertex position.
        vertex_residual: [ValidatedInterval; 3],
    },
    /// Original-source-authorized completion on an actual clamped tensor boundary.
    ///
    /// The whole supported UV joint tube, including transverse uncertainty,
    /// is enclosed relative to the shared vertex. This is not an exact collapse,
    /// surface-normal or mesh-coverage certificate.
    NaturalBoundary {
        /// Explicit original native natural-boundary branch.
        boundary: NativeSurfaceBoundary,
        /// Native chart coordinate fixed on that boundary.
        fixed_axis: usize,
        /// Exact stored knot endpoint of the fixed coordinate.
        fixed_parameter: f64,
        /// Complete variable-coordinate range used by the residual proof.
        variable_range: [f64; 2],
        /// Nominal boundary connector, material incoming to outgoing, domain `[0, 1]`.
        connector: Line<Point2>,
        /// Whole joint tube intersected with the actual native tensor knot domain.
        supported_parameter_box: [[f64; 2]; 2],
        /// Whole supported tube image minus the original shared vertex.
        vertex_residual: [ValidatedInterval; 3],
        /// Whole supported tube image in native world coordinates.
        world_enclosure: [ValidatedInterval; 3],
        /// Caller-certified permitted per-coordinate world displacement.
        max_vertex_deviation: [f64; 3],
    },
}

/// Endpoint support and permitted UV margin for a shared trim joint.
pub(super) struct JointNeighborhood<'a> {
    pub enclosure: &'a mut [ValidatedInterval; 2],
    pub front: [ValidatedInterval; 2],
    pub back: [ValidatedInterval; 2],
    pub margin: [f64; 2],
}

/// Certifies a bounded joint of two original parameter-curve uses.
///
/// The caller must bind both endpoint enclosures to the same original shared
/// vertex and complete, independently proved source curves. This reuses the
/// native face-domain ordinary endpoint rectangle or singular-stratum proof,
/// enclosing the entire connector image and its residual against that vertex.
/// Ordinary rectangles must remain strictly within one half-period; a period
/// alias is not a joint. The whole Euclidean residual must remain within
/// `maximum_vertex_deviation`. This does not replace whole-face trim preparation
/// or coverage, and does not assert that a rounded vertex is exactly on-carrier.
pub fn certified_surface_parameter_joint(
    surface: &Surface,
    vertex: Point3,
    endpoints: [[ValidatedInterval; 2]; 2],
    margin_uv: [f64; 2],
    maximum_vertex_deviation: f64,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Option<TrimJointKind>, TrimDomainError> {
    budget.charge(1, cancel)?;
    if !maximum_vertex_deviation.is_finite()
        || maximum_vertex_deviation <= 0.0
        || margin_uv.into_iter().any(|x| !x.is_finite() || x < 0.0)
        || endpoints.into_iter().flatten().any(|x| !x.is_finite())
        || [vertex.x, vertex.y, vertex.z]
            .into_iter()
            .any(|x| !x.is_finite())
    {
        return Err(TrimDomainError::InvalidChart);
    }
    let [front, back] = endpoints;
    let mut enclosure = [front[0].hull(back[0]), front[1].hull(back[1])];
    let proof = pole_completion_on(
        surface,
        vertex,
        JointNeighborhood {
            enclosure: &mut enclosure,
            front,
            back,
            margin: margin_uv,
        },
        budget,
        cancel,
    )?;
    let (kind, residual) = if let Some(kind) = proof {
        let TrimJointKind::SphericalPole {
            vertex_residual, ..
        } = kind
        else {
            return Ok(None);
        };
        (kind, vertex_residual)
    } else {
        let enclosure = [
            enclosure[0] + ValidatedInterval::new(-margin_uv[0], margin_uv[0])?,
            enclosure[1] + ValidatedInterval::new(-margin_uv[1], margin_uv[1])?,
        ];
        if !ordinary_joint_supported(enclosure, [surface.period_u(), surface.period_v()]) {
            return Ok(None);
        }
        let (_, residual) = joint_image_residual(surface, vertex, enclosure, budget, cancel)?;
        (TrimJointKind::EndpointUncertainty, residual)
    };
    budget.charge(4, cancel)?;
    let squared = residual
        .iter()
        .fold(ValidatedInterval::point(0.0), |sum, axis| {
            sum + *axis * *axis
        });
    let limit = ValidatedInterval::point(maximum_vertex_deviation);
    Ok((squared.upper <= (limit * limit).lower).then_some(kind))
}

/// Matches the ordinary endpoint-joint covering-chart gate used in preparation.
pub(super) fn ordinary_joint_supported(
    enclosure: [ValidatedInterval; 2],
    periods: [Option<f64>; 2],
) -> bool {
    (0..2).all(|axis| {
        periods[axis].is_none_or(|period| {
            period.is_finite() && period > 0.0 && enclosure[axis].width() < period * 0.5
        })
    })
}

fn joint_image_residual(
    surface: &Surface,
    vertex: Point3,
    enclosure: [ValidatedInterval; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<([ValidatedInterval; 3], [ValidatedInterval; 3]), TrimDomainError> {
    let bounds = enclosure.map(|axis| [axis.lower, axis.upper]);
    let world = validated_surface_jet(surface, bounds, budget, cancel)?.position;
    let residual = std::array::from_fn(|axis| world[axis] - ValidatedInterval::point(vertex[axis]));
    Ok((world, residual))
}

pub(super) fn pole_completion(
    face: &Face,
    vertex: Point3,
    neighborhood: JointNeighborhood<'_>,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Option<TrimJointKind>, TrimDomainError> {
    face.with_surface(|surface| pole_completion_on(surface, vertex, neighborhood, budget, cancel))
}

fn pole_completion_on(
    surface: &Surface,
    vertex: Point3,
    neighborhood: JointNeighborhood<'_>,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Option<TrimJointKind>, TrimDomainError> {
    let JointNeighborhood {
        enclosure,
        front,
        back,
        margin: margin_uv,
    } = neighborhood;
    let Surface::SphericalSurface(processor) = surface else {
        return Ok(None);
    };
    budget.charge(1, cancel)?;
    let polar_axis = if processor.orientation() { 0 } else { 1 };
    let latitude = enclosure[polar_axis];
    let angular_axis = 1 - polar_axis;
    let margin = ValidatedInterval::new(-margin_uv[angular_axis], margin_uv[angular_axis])?;
    let difference = (front[angular_axis] + margin) - (back[angular_axis] + margin);
    if !difference.is_finite() {
        return Err(TrimDomainError::InvalidChart);
    }
    if difference.contains_zero() {
        return Ok(None);
    }
    let angles = Curve2D::Conic(Conic2D::Ellipse(Processor::with_transform(
        TrimmedCurve::new(
            UnitCircle::<Point2>::new(),
            (difference.lower, difference.upper),
        ),
        Matrix3::from_scale(1.0),
    )));
    let phase = validated_curve2d_jet(
        &angles,
        [difference.lower, difference.upper],
        budget,
        cancel,
    )?
    .position;
    // A possible periodic alias is not authority for a pole completion.
    if phase[0].lower <= 1.0 && phase[0].upper >= 1.0 && phase[1].contains_zero() {
        return Ok(None);
    }
    let candidates = [
        (SphericalPole::North, ValidatedInterval::point(0.0), 0.0),
        (
            SphericalPole::South,
            ValidatedInterval::new(PI.next_down(), PI.next_up())?,
            PI,
        ),
    ];
    for (pole, exact_pole_enclosure, nominal) in candidates {
        budget.charge(1, cancel)?;
        let polar_tube = exact_pole_enclosure
            + ValidatedInterval::new(-margin_uv[polar_axis], margin_uv[polar_axis])?;
        if latitude.intersection(polar_tube).is_none() {
            continue;
        }
        let mut joint = [
            enclosure[0] + ValidatedInterval::new(-margin_uv[0], margin_uv[0])?,
            enclosure[1] + ValidatedInterval::new(-margin_uv[1], margin_uv[1])?,
        ];
        joint[polar_axis] = joint[polar_axis].hull(polar_tube);
        let (world_enclosure, vertex_residual) =
            joint_image_residual(surface, vertex, joint, budget, cancel)?;
        if !vertex_residual
            .into_iter()
            .all(ValidatedInterval::contains_zero)
        {
            continue;
        }
        let angular_axis = 1 - polar_axis;
        let mut a = front.map(|axis| axis.lower);
        let mut b = back.map(|axis| axis.lower);
        a[polar_axis] = nominal;
        b[polar_axis] = nominal;
        if a[angular_axis] == b[angular_axis] {
            continue;
        }
        *enclosure = joint;
        return Ok(Some(TrimJointKind::SphericalPole {
            polar_axis,
            pole,
            connector: Line(Point2::new(a[0], a[1]), Point2::new(b[0], b[1])),
            polar_coordinate: joint[polar_axis],
            world_enclosure,
            vertex_residual,
        }));
    }
    Ok(None)
}
