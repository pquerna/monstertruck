//! Elimination of an actual affine carrier, without changing the guide gauge.
//!
//! A strict two-variable inclusion proves the original common roots: the
//! nonsingular affine chart reconstructs their other two coordinates inside
//! the original four-variable domain. No intersection with that domain may
//! replace containment. All spatial jets use the same closed parameter span.
//!
//! Certification consumes continuation's original-bound local tube. An
//! arbitrary broader chart box need not have a contractive interval Jacobian;
//! public preparation still owns its unchanged numerical proposals, charged
//! retries, subdivision, endpoint seals and full requested parameter coverage.
//! Native coordinate jets retain the complete unchanged leader and evaluate
//! its equation and partial time derivatives on the correlated displacement
//! from that leader. Independent world-origin offsets are never constructed.
//!
//! Contraction evaluates the two original source-coefficient functionals, with
//! shared native tensor/profile bases and unchanged piecewise/pole admission.
//! Native center residuals additionally intersect an independent time mean-value
//! enclosure: the anchor uses one original midpoint leader jet, while its time
//! derivative and all inclusion/derivative premises retain the whole raw-t span.
//! Only the whole-span inclusion image supplies the mandatory reconstructed
//! spatial jet and transversality proof. If its midpoint is outside that image,
//! centered reconstruction additionally evaluates the full original convex-box
//! spatial gradients; no root-image clipping substitutes for that path proof.

use super::super::validated_interval::{
    ProjectionExecution, SurfaceFirstJet, TENSOR_EVALUATOR_FIXED_BYTES, ValidatedSurfaceFunctional,
    ValidatedSurfaceFunctionalJet, surface_piecewise_first_jet, surface_piecewise_position,
    validated_surface_functional_jets, validated_surface_functional_values,
};
use super::*;

pub(super) struct PlaneKrawczykReceipt {
    pub(super) jet: ValidatedCurveJet,
    pub(super) residual: [ValidatedInterval; 3],
    pub(super) rho: f64,
}

#[derive(Clone, Copy)]
struct Chart {
    origin: [ValidatedInterval; 3],
    axes: [[ValidatedInterval; 3]; 2],
    normal: [ValidatedInterval; 3],
    rows: [usize; 2],
    inverse: [[ValidatedInterval; 2]; 2],
}

#[derive(Clone, Copy)]
struct PlaneSystem {
    // Contraction needs only the original two scalar equations and gradients.
    value: [ValidatedInterval; 2],
    jacobian: [[ValidatedInterval; 2]; 2],
}

struct CoordinateJet {
    parameter: ValidatedInterval,
    jet: PlaneJet,
}

// Bound retained proof locals separately from the shared source evaluator's
// bounded scratch. Keep the original conservative two-system/two-Hessian
// allowance: rooted Hessians remain mandatory, while convex-domain gradients
// and their extracted derivative vectors fit the other spatial-jet slot.
const PLANE_WORKSPACE_BYTES: usize = 2 * size_of::<PlaneSystem>()
    + 2 * size_of::<ValidatedSurfaceJet>()
    + 2 * size_of::<ValidatedSurfaceFunctional>()
    + 6 * size_of::<ValidatedInterval>()
    + size_of::<Chart>()
    + size_of::<CoordinateJet>()
    + size_of::<PlaneJet>()
    + size_of::<PlaneKrawczykReceipt>()
    + 20 * size_of::<[ValidatedInterval; 2]>()
    + 12 * size_of::<[ValidatedInterval; 3]>()
    + 3 * size_of::<[[ValidatedInterval; 2]; 2]>()
    + 2 * size_of::<[[f64; 2]; 2]>()
    + 4 * size_of::<[ValidatedInterval; 4]>()
    + size_of::<Tube>()
    + 512;
const ORIGINAL_WORKSPACE_BYTES: usize = 3 * size_of::<System>()
    + size_of::<Tube>()
    + size_of::<[[f64; 4]; 4]>()
    + 2 * size_of::<[[ValidatedInterval; 4]; 4]>()
    + 8 * size_of::<[ValidatedInterval; 4]>();
const _: () = assert!(PLANE_WORKSPACE_BYTES <= ORIGINAL_WORKSPACE_BYTES);
const _: () = assert!(size_of::<PlaneKrawczykReceipt>() <= size_of::<System>());
// The midpoint leader uses the existing projection-jet temporary slot. Compute
// the original defect before the center residual, so its paired source result
// can reuse the dead enclosure slot rather than grow the proof-frame allowance.
const _: () = assert!(size_of::<ValidatedCurveJet>() <= size_of::<PlaneJet>());
const _: () = assert!(size_of::<[ValidatedSurfaceFunctionalJet; 2]>() <= size_of::<PlaneSystem>());
const _: () = assert!(
    size_of::<SurfaceFirstJet>() + 2 * size_of::<[ValidatedInterval; 3]>()
        <= size_of::<ValidatedSurfaceJet>()
);

// Both proof paths call the same fixed source evaluator. Track its concrete
// tensor state (including original-coefficient partition scratch) on BOTH sides,
// not as an uncharged saving of the reduced proof. max_retained_bytes admits
// owned source/proof allocations; it is not an RSS or compiler-stack limit.
const _: () = assert!(
    PLANE_WORKSPACE_BYTES + TENSOR_EVALUATOR_FIXED_BYTES
        <= ORIGINAL_WORKSPACE_BYTES + TENSOR_EVALUATOR_FIXED_BYTES
);

fn operands(
    curve: &Curve,
) -> Result<Option<(&Surface, &Plane, &Curve, usize)>, ValidatedGeometryError> {
    let (first, second, guide) = carriers(curve)?;
    Ok(match (first, second) {
        (_, Surface::Plane(plane)) => Some((first, plane, guide, 0)),
        (Surface::Plane(plane), _) => Some((second, plane, guide, 2)),
        _ => None,
    })
}

impl Chart {
    fn new(
        plane: &Plane,
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> Result<Self, ValidatedGeometryError> {
        budget.charge(1, cancel)?;
        let (a, b) = coverage::plane_axes(plane);
        let normal = cross3(a, b);
        let mut best = None;
        for rows in [[0, 1], [0, 2], [1, 2]] {
            budget.charge(1, cancel)?;
            let [i, j] = rows;
            let determinant = a[i] * b[j] - a[j] * b[i];
            if determinant.is_finite() && !determinant.contains_zero() {
                let separation = determinant.lower.abs().min(determinant.upper.abs());
                if best.is_none_or(|(_, _, previous)| separation > previous) {
                    best = Some((rows, determinant, separation));
                }
            }
        }
        let (rows, determinant, _) = best.ok_or(ValidatedGeometryError::Singular)?;
        let [i, j] = rows;
        let reciprocal = determinant.reciprocal()?;
        let chart = Self {
            origin: std::array::from_fn(|i| ValidatedInterval::point(plane.origin()[i])),
            axes: [a, b],
            normal,
            rows,
            inverse: [
                [b[j] * reciprocal, -b[i] * reciprocal],
                [-a[j] * reciprocal, a[i] * reciprocal],
            ],
        };
        if !chart
            .origin
            .into_iter()
            .chain(a)
            .chain(b)
            .chain(normal)
            .all(ValidatedInterval::is_finite)
        {
            return Err(ValidatedGeometryError::ConditioningFailure);
        }
        Ok(chart)
    }

    fn vector_parameters(&self, vector: [ValidatedInterval; 3]) -> [ValidatedInterval; 2] {
        interval_product(self.inverse, self.rows.map(|i| vector[i]))
    }

    fn parameters(&self, point: [ValidatedInterval; 3]) -> [ValidatedInterval; 2] {
        self.vector_parameters(subtract3(point, self.origin))
    }

    fn mean_value_parameters(
        &self,
        center: [ValidatedInterval; 2],
        du: [ValidatedInterval; 3],
        dv: [ValidatedInterval; 3],
        delta: [ValidatedInterval; 2],
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> Result<[ValidatedInterval; 2], ValidatedGeometryError> {
        budget.charge(1, cancel)?;
        budget.charge(2, cancel)?;
        let du = self.vector_parameters(du);
        let dv = self.vector_parameters(dv);
        let mut parameters = [ZERO; 2];
        for i in 0..2 {
            budget.charge(1, cancel)?;
            parameters[i] = center[i] + du[i] * delta[0] + dv[i] * delta[1];
        }
        Ok(parameters)
    }

    fn vector(&self, uv: [ValidatedInterval; 2]) -> [ValidatedInterval; 3] {
        add3(
            multiply3(self.axes[0], uv[0]),
            multiply3(self.axes[1], uv[1]),
        )
    }

    fn position(&self, uv: [ValidatedInterval; 2]) -> [ValidatedInterval; 3] {
        add3(
            add3(self.origin, multiply3(self.axes[0], uv[0])),
            multiply3(self.axes[1], uv[1]),
        )
    }
}

impl Chart {
    fn functionals(&self) -> [ValidatedSurfaceFunctional; 2] {
        std::array::from_fn(|i| {
            let mut direction = [ZERO; 3];
            for j in 0..2 {
                direction[self.rows[j]] = self.inverse[i][j];
            }
            ValidatedSurfaceFunctional {
                direction,
                origin: self.origin,
                offset: ZERO,
            }
        })
    }
}

fn native_functionals(chart: &Chart, guide: &ValidatedCurveJet) -> [ValidatedSurfaceFunctional; 2] {
    [
        ValidatedSurfaceFunctional {
            direction: chart.normal,
            origin: chart.origin,
            offset: ZERO,
        },
        ValidatedSurfaceFunctional {
            direction: guide.first,
            origin: guide.position,
            offset: ZERO,
        },
    ]
}

fn system(
    source: &Surface,
    chart: &Chart,
    coordinate: &CoordinateJet,
    domain: [ValidatedInterval; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<PlaneSystem, ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    let (value, jacobian) = match &coordinate.jet {
        PlaneJet::Native(guide) => {
            let projected = validated_surface_functional_jets(
                source,
                domain.map(|bound| [bound.lower, bound.upper]),
                native_functionals(chart, guide),
                budget,
                cancel,
            )?;
            (
                projected.map(|jet| jet.position),
                projected.map(|jet| [jet.du, jet.dv]),
            )
        }
        PlaneJet::BetweenPlanes { .. } => {
            // Retain the exact raw-t affine coefficient law and endpoint
            // branches; never distribute independent interval time factors.
            let source = surface_piecewise_first_jet(
                source,
                domain.map(|bound| [bound.lower, bound.upper]),
                budget,
                cancel,
            )?;
            (
                [
                    dot3(chart.normal, subtract3(source.position, chart.origin)),
                    coordinate
                        .jet
                        .residual(coordinate.parameter, source.position),
                ],
                [chart.normal, coordinate.jet.normal()]
                    .map(|normal| [dot3(normal, source.du), dot3(normal, source.dv)]),
            )
        }
    };
    let system = PlaneSystem { value, jacobian };
    if !system
        .value
        .into_iter()
        .chain(system.jacobian.into_iter().flatten())
        .all(ValidatedInterval::is_finite)
    {
        return Err(ValidatedGeometryError::ConditioningFailure);
    }
    Ok(system)
}

// Center residuals do not consume a Jacobian or grant derivative authority.
fn system_values(
    source: &Surface,
    chart: &Chart,
    coordinate: &CoordinateJet,
    domain: [ValidatedInterval; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<[ValidatedInterval; 2], ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    let value = match &coordinate.jet {
        PlaneJet::Native(guide) => validated_surface_functional_values(
            source,
            domain.map(|bound| [bound.lower, bound.upper]),
            native_functionals(chart, guide),
            budget,
            cancel,
        )?,
        PlaneJet::BetweenPlanes { .. } => {
            let position = surface_piecewise_position(
                source,
                domain.map(|bound| [bound.lower, bound.upper]),
                budget,
                cancel,
            )?;
            [
                dot3(chart.normal, subtract3(position, chart.origin)),
                coordinate.jet.residual(coordinate.parameter, position),
            ]
        }
    };
    if value.into_iter().all(ValidatedInterval::is_finite) {
        Ok(value)
    } else {
        Err(ValidatedGeometryError::ConditioningFailure)
    }
}

/// Independent time mean-value enclosure at the unchanged source center.
/// Only the residual's anchor is queried at one time; its derivative uses the
/// original whole-span leader jet and original source functional coefficients.
fn native_center_residual(
    source: &Surface,
    leader: &Curve,
    coordinate: &CoordinateJet,
    center: [ValidatedInterval; 2],
    execution: &mut ProjectionExecution<'_>,
) -> Result<Option<ValidatedInterval>, ValidatedGeometryError> {
    let PlaneJet::Native(guide) = &coordinate.jet else {
        return Ok(None);
    };
    let middle = coordinate.parameter.midpoint();
    let anchor = validated_projection_leader_jet(
        leader,
        [middle, middle],
        execution.budget,
        execution.cancel,
    )?;
    execution.budget.charge(3, execution.cancel)?;
    let speed_squared = guide
        .first
        .into_iter()
        .map(ValidatedInterval::square)
        .fold(ZERO, |a, b| a + b);
    let residuals = validated_surface_functional_values(
        source,
        center.map(|bound| [bound.lower, bound.upper]),
        [
            ValidatedSurfaceFunctional {
                direction: anchor.first,
                origin: anchor.position,
                offset: ZERO,
            },
            ValidatedSurfaceFunctional {
                direction: guide.second,
                origin: guide.position,
                offset: -speed_squared,
            },
        ],
        execution.budget,
        execution.cancel,
    )?;
    execution.budget.charge(1, execution.cancel)?;
    Ok(Some(
        residuals[0] + residuals[1] * (coordinate.parameter - ValidatedInterval::point(middle)),
    ))
}

pub(super) fn certify(
    curve: &Curve,
    projection: Projection,
    parameter: ValidatedInterval,
    domain: [ValidatedInterval; 4],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Option<Tube>, ValidatedGeometryError> {
    let Some((source, plane, leader, source_index)) = operands(curve)? else {
        return Ok(None);
    };
    budget.charge(1, cancel)?;
    let chart = Chart::new(plane, budget, cancel)?;
    // The original full-span projection jet is shared, never resampled or
    // evaluated on a smaller interval for the derivative certificate.
    let coordinate = CoordinateJet {
        parameter,
        jet: projection_jet(leader, projection, parameter, budget, cancel)?,
    };
    let source_domain = [domain[source_index], domain[source_index + 1]];
    let enclosure = system(source, &chart, &coordinate, source_domain, budget, cancel)?;
    let middle = source_domain.map(ValidatedInterval::midpoint);
    budget.charge(1, cancel)?;
    let inverse = inverse(
        enclosure
            .jacobian
            .map(|row| row.map(ValidatedInterval::midpoint)),
    )?;
    let defect = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            ValidatedInterval::point(if i == j { 1.0 } else { 0.0 })
                - (0..2)
                    .map(|k| ValidatedInterval::point(inverse[i][k]) * enclosure.jacobian[k][j])
                    .fold(ZERO, |a, b| a + b)
        })
    });
    let rho = defect
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|entry| ValidatedInterval::point(entry.magnitude()))
                .fold(ZERO, |a, b| a + b)
                .upper
        })
        .fold(0.0, f64::max);
    if !rho.is_finite() || rho >= 1.0 {
        return Err(ValidatedGeometryError::ConditioningFailure);
    }
    let mut center = system_values(
        source,
        &chart,
        &coordinate,
        middle.map(ValidatedInterval::point),
        budget,
        cancel,
    )?;
    if let Some(time_centered) = native_center_residual(
        source,
        leader,
        &coordinate,
        middle.map(ValidatedInterval::point),
        &mut ProjectionExecution { budget, cancel },
    )? {
        budget.charge(1, cancel)?;
        center[1] = center[1]
            .intersection(time_centered)
            .ok_or(ValidatedGeometryError::ConditioningFailure)?;
    }
    let correction = product(inverse, center);
    let delta = std::array::from_fn(|i| source_domain[i] - ValidatedInterval::point(middle[i]));
    let error = interval_product(defect, delta);
    let base: [ValidatedInterval; 2] =
        std::array::from_fn(|i| ValidatedInterval::point(middle[i]) - correction[i]);
    let mut image = std::array::from_fn(|i| base[i] + error[i]);
    if !(0..2).all(|i| source_domain[i].strictly_contains(image[i])) {
        return Err(ValidatedGeometryError::Unresolved);
    }
    // Original-domain strict inclusion has already proved every parameter's
    // unique root. Each such root also satisfies this centered mean-value
    // equation with the SAME original Jacobian enclosure. Reapplying it to
    // the root image contracts only the enclosure, never source ownership or
    // the parameter span. The original convex box covers all center-to-root
    // segments even when a contracted image no longer contains the center.
    for _ in 0..3 {
        budget.charge(1, cancel)?;
        let previous = image;
        let delta = std::array::from_fn(|i| image[i] - ValidatedInterval::point(middle[i]));
        let remainder = interval_product(defect, delta);
        for i in 0..2 {
            image[i] = image[i]
                .intersection(base[i] + remainder[i])
                .ok_or(ValidatedGeometryError::ConditioningFailure)?;
        }
        if image == previous {
            break;
        }
    }
    let plane_index = 2 - source_index;
    let mut refinement_used = false;
    let mut original_gradients = None;
    let mut retained_center_chart = None;
    let (rooted, uv) = loop {
        // This full spatial jet covers every root for the unchanged whole t
        // span. Scalar inclusion cannot substitute for spatial derivatives.
        let rooted = validated_surface_piecewise_jet(
            source,
            image.map(|bound| [bound.lower, bound.upper]),
            budget,
            cancel,
        )?;
        let tangent = cross3(cross3(rooted.du, rooted.dv), chart.normal);
        let transverse = tangent
            .map(ValidatedInterval::square)
            .into_iter()
            .fold(ZERO, |a, b| a + b);
        if transverse.lower <= 0.0 {
            return Err(ValidatedGeometryError::NonTransverse);
        }
        budget.charge(1, cancel)?;
        let direct_uv = chart.parameters(rooted.position);
        // Differentiate the exact affine chart functional BEFORE multiplying
        // source-coordinate displacements. Rooted gradients cover the entire
        // center-to-root segment only if both center coordinates lie in the
        // image. Otherwise retain the original convex-box gradient hull.
        budget.charge(1, cancel)?;
        let center_in_image =
            (0..2).all(|i| image[i].lower <= middle[i] && middle[i] <= image[i].upper);
        let (du, dv) = if center_in_image {
            (rooted.du, rooted.dv)
        } else {
            let gradients = match &mut original_gradients {
                Some(gradients) => gradients,
                slot => slot.insert(surface_piecewise_first_jet(
                    source,
                    source_domain.map(|bound| [bound.lower, bound.upper]),
                    budget,
                    cancel,
                )?),
            };
            (gradients.du, gradients.dv)
        };
        let image_delta = std::array::from_fn(|i| image[i] - ValidatedInterval::point(middle[i]));
        let center_chart = match retained_center_chart {
            Some(center) => center,
            None => {
                let center = validated_surface_functional_values(
                    source,
                    middle.map(|value| [value; 2]),
                    chart.functionals(),
                    budget,
                    cancel,
                )?;
                retained_center_chart = Some(center);
                center
            }
        };
        let mean_uv =
            chart.mean_value_parameters(center_chart, du, dv, image_delta, budget, cancel)?;
        let mut uv = direct_uv;
        for i in 0..2 {
            budget.charge(1, cancel)?;
            uv[i] = direct_uv[i]
                .intersection(mean_uv[i])
                .ok_or(ValidatedGeometryError::ConditioningFailure)?;
        }
        // Reconstructed bounds must fit the immutable original plane domain.
        // Never clip those bounds into that domain to discard certified roots.
        if (0..2).all(|i| {
            uv[i].is_finite()
                && domain[plane_index + i].lower <= uv[i].lower
                && uv[i].upper <= domain[plane_index + i].upper
        }) {
            break (rooted, uv);
        }
        if refinement_used {
            // The hull of the unchanged center and the WHOLE root image is a
            // complete convex enclosure of every center-to-root segment. A
            // center outside the image does not require gradients over the
            // entire original box when this smaller proved hull is available.
            // This is another original-source path proof, not UV clipping,
            // an assumed smaller uniqueness domain or a time-point derivative.
            budget.charge(1, cancel)?;
            let segment_domain =
                std::array::from_fn(|i| image[i].hull(ValidatedInterval::point(middle[i])));
            if !center_in_image && segment_domain != source_domain {
                // The old whole-box gradients are dead after their mean-UV
                // enclosure; reuse that same accounted spatial-jet slot.
                original_gradients = None;
                let segment = original_gradients.insert(surface_piecewise_first_jet(
                    source,
                    segment_domain.map(|bound| [bound.lower, bound.upper]),
                    budget,
                    cancel,
                )?);
                let segment_uv = chart.mean_value_parameters(
                    center_chart,
                    segment.du,
                    segment.dv,
                    image_delta,
                    budget,
                    cancel,
                )?;
                for i in 0..2 {
                    budget.charge(1, cancel)?;
                    uv[i] = uv[i]
                        .intersection(segment_uv[i])
                        .ok_or(ValidatedGeometryError::ConditioningFailure)?;
                }
                if (0..2).all(|i| {
                    uv[i].is_finite()
                        && domain[plane_index + i].lower <= uv[i].lower
                        && uv[i].upper <= domain[plane_index + i].upper
                }) {
                    break (rooted, uv);
                }
            }
            return Err(ValidatedGeometryError::Unresolved);
        }
        refinement_used = true;
        // Only failed reconstruction needs stronger necessary-root bounds.
        // Isolate each diagonal in delta = -correction + defect * delta,
        // retaining the SAME original whole-domain/time Jacobian enclosure.
        // rho < 1 proves each denominator excludes zero; every update still
        // intersects the old genuine root image. Existence and uniqueness
        // remain the original strict Krawczyk proof, not this contraction.
        budget.charge(2, cancel)?;
        let inverse_diagonal = [
            (ONE - defect[0][0]).reciprocal()?,
            (ONE - defect[1][1]).reciprocal()?,
        ];
        let original_image = image;
        for _ in 0..3 {
            budget.charge(1, cancel)?;
            let previous = image;
            for i in 0..2 {
                let other = 1 - i;
                let remainder =
                    defect[i][other] * (image[other] - ValidatedInterval::point(middle[other]));
                let isolated = ValidatedInterval::point(middle[i])
                    + (-correction[i] + remainder) * inverse_diagonal[i];
                image[i] = image[i]
                    .intersection(isolated)
                    .ok_or(ValidatedGeometryError::ConditioningFailure)?;
            }
            if image == previous {
                break;
            }
        }
        if image == original_image {
            return Err(ValidatedGeometryError::Unresolved);
        }
        // Fresh original spatial, derivative and UV proofs must now admit the
        // contracted whole-time image; the old refused UV bounds grant nothing.
    };
    let plane_position = chart.position(uv);
    let position = intersect3(rooted.position, plane_position)?;
    let residual = if source_index == 0 {
        subtract3(rooted.position, plane_position)
    } else {
        subtract3(plane_position, rooted.position)
    };
    budget.charge(1, cancel)?;
    let guide = &coordinate.jet;
    let ft = guide.partial_first(position);
    let first_uv = solve(inverse, defect, rho, [ZERO, -ft], budget, cancel)?;
    let source_first = add3(
        multiply3(rooted.du, first_uv[0]),
        multiply3(rooted.dv, first_uv[1]),
    );
    let first = intersect3(
        source_first,
        chart.vector(chart.vector_parameters(source_first)),
    )?;
    let h = second_composition(rooted, first_uv[0], first_uv[1]);
    budget.charge(1, cancel)?;
    let last = dot3(guide.normal(), h)
        + dot3(guide.first(), first) * ValidatedInterval::point(2.0)
        + guide.partial_second(position);
    let second_uv = solve(
        inverse,
        defect,
        rho,
        [-dot3(chart.normal, h), -last],
        budget,
        cancel,
    )?;
    let source_second = add3(
        add3(
            multiply3(rooted.du, second_uv[0]),
            multiply3(rooted.dv, second_uv[1]),
        ),
        h,
    );
    let second = intersect3(
        source_second,
        chart.vector(chart.vector_parameters(source_second)),
    )?;
    if !position
        .into_iter()
        .chain(first)
        .chain(second)
        .chain(residual)
        .all(ValidatedInterval::is_finite)
    {
        return Err(ValidatedGeometryError::ConditioningFailure);
    }
    let jet = ValidatedCurveJet {
        position,
        first,
        second,
        third: [ValidatedInterval {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
        }; 3],
    };
    let mut rooted_domain = domain;
    rooted_domain[source_index..source_index + 2].copy_from_slice(&image);
    rooted_domain[plane_index..plane_index + 2].copy_from_slice(&uv);
    Ok(Some(Tube {
        authority: TubeAuthority::Plane(PlaneKrawczykReceipt { jet, residual, rho }),
        domain,
        rooted_domain,
    }))
}

fn inverse(matrix: [[f64; 2]; 2]) -> Result<[[f64; 2]; 2], ValidatedGeometryError> {
    let [[a, b], [c, d]] = matrix;
    let determinant = a * d - b * c;
    if !determinant.is_finite() || determinant == 0.0 {
        return Err(ValidatedGeometryError::Singular);
    }
    let inverse = [
        [d / determinant, -b / determinant],
        [-c / determinant, a / determinant],
    ];
    if inverse.into_iter().flatten().all(f64::is_finite) {
        Ok(inverse)
    } else {
        Err(ValidatedGeometryError::ConditioningFailure)
    }
}

fn product(matrix: [[f64; 2]; 2], vector: [ValidatedInterval; 2]) -> [ValidatedInterval; 2] {
    interval_product(matrix.map(|row| row.map(ValidatedInterval::point)), vector)
}

fn interval_product(
    matrix: [[ValidatedInterval; 2]; 2],
    vector: [ValidatedInterval; 2],
) -> [ValidatedInterval; 2] {
    matrix.map(|row| row[0] * vector[0] + row[1] * vector[1])
}

fn solve(
    inverse: [[f64; 2]; 2],
    defect: [[ValidatedInterval; 2]; 2],
    rho: f64,
    rhs: [ValidatedInterval; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<[ValidatedInterval; 2], ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    let base = product(inverse, rhs);
    let maximum = base
        .into_iter()
        .map(ValidatedInterval::magnitude)
        .fold(0.0, f64::max);
    let radius = (ValidatedInterval::point(rho)
        * ValidatedInterval::point(maximum)
        * (ONE - ValidatedInterval::point(rho)).reciprocal()?)
    .upper;
    let error = ValidatedInterval {
        lower: -radius,
        upper: radius,
    };
    let mut result = base.map(|value| value + error);
    for _ in 0..3 {
        budget.charge(1, cancel)?;
        let residual = interval_product(defect, result);
        result = std::array::from_fn(|i| {
            result[i]
                .intersection(base[i] + residual[i])
                .unwrap_or(result[i])
        });
    }
    if result.into_iter().all(ValidatedInterval::is_finite) {
        Ok(result)
    } else {
        Err(ValidatedGeometryError::ConditioningFailure)
    }
}

pub(super) fn numerical_root(
    curve: &Curve,
    projection: Projection,
    parameter: f64,
    mut center: [f64; 4],
    options: &ValidatedCurveOptions,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Option<[f64; 4]>, ValidatedGeometryError> {
    let Some((source, plane, leader, source_index)) = operands(curve)? else {
        return Ok(None);
    };
    let chart = Chart::new(plane, budget, cancel)?;
    let parameter = ValidatedInterval::point(parameter);
    let coordinate = CoordinateJet {
        parameter,
        jet: projection_jet(leader, projection, parameter, budget, cancel)?,
    };
    for _ in 0..options.max_iterations {
        budget.charge(1, cancel)?;
        let value = system(
            source,
            &chart,
            &coordinate,
            [center[source_index], center[source_index + 1]].map(ValidatedInterval::point),
            budget,
            cancel,
        )?;
        let inverse = inverse(
            value
                .jacobian
                .map(|row| row.map(ValidatedInterval::midpoint)),
        )?;
        let delta = product(inverse, value.value).map(ValidatedInterval::midpoint);
        for i in 0..2 {
            center[source_index + i] -= delta[i];
        }
        if !center.into_iter().all(f64::is_finite) {
            return Err(ValidatedGeometryError::ConditioningFailure);
        }
        if delta.into_iter().map(f64::abs).fold(0.0, f64::max) < 1.0e-12 {
            break;
        }
    }
    budget.charge(1, cancel)?;
    let point = surface_piecewise_position(
        source,
        [center[source_index], center[source_index + 1]].map(|value| [value, value]),
        budget,
        cancel,
    )?;
    let uv = chart.parameters(point).map(ValidatedInterval::midpoint);
    let plane_index = 2 - source_index;
    center[plane_index..plane_index + 2].copy_from_slice(&uv);
    if center.into_iter().all(f64::is_finite) {
        Ok(Some(center))
    } else {
        Err(ValidatedGeometryError::ConditioningFailure)
    }
}

#[cfg(test)]
mod regressions {
    use super::*;
    use crate::{BsplineCurve, BsplineSurface, Invertible, Line, NurbsSurface, Vector3, Vector4};
    use monstertruck_core::EuclideanSpace;

    fn parabola(
        posed: bool,
        reversed: bool,
        swapped: bool,
    ) -> (Curve, [ValidatedInterval; 4], [f64; 3]) {
        let (origin, scale) = if posed {
            ([7.0, 11.0, 13.0], [-2.0, 3.0, 4.0])
        } else {
            ([0.0; 3], [1.0; 3])
        };
        let point = |u: f64, y: f64, v: f64| {
            Point3::new(
                origin[0] + scale[0] * u,
                origin[1] + scale[1] * y,
                origin[2] + scale[2] * v,
            )
        };
        // The independent polynomial is S(u,v)=(u,u^2,v-1/2), followed
        // by the displayed diagonal affine pose, including its reflection.
        let source = Surface::BsplineSurface(BsplineSurface::new(
            (KnotVector::bezier_knot(2), KnotVector::bezier_knot(1)),
            [(0.0, 0.0), (0.5, 0.0), (1.0, 1.0)]
                .into_iter()
                .map(|(u, y)| vec![point(u, y, -0.5), point(u, y, 0.5)])
                .collect(),
        ));
        let plane = Surface::Plane(Plane::from_origin_vectors(
            Point3::from(origin),
            Vector3::new(scale[0], 0.0, 0.0),
            Vector3::new(0.5 * scale[0], scale[1], 0.0),
        ));
        let guide = Curve::BsplineCurve(BsplineCurve::new(
            KnotVector::bezier_knot(2),
            vec![
                point(0.0, 0.0, 0.0),
                point(0.5, 0.0, 0.0),
                point(1.0, 1.0, 0.0),
            ],
        ));
        let guide = if reversed {
            let mut wrapper = Processor::new(TrimmedCurve::new(Box::new(guide), (0.0, 1.0)));
            wrapper.invert();
            Curve::TrimmedCurve(wrapper)
        } else {
            guide
        };
        let source_domain = [
            interval_range([0.1, 0.6]).unwrap(),
            interval_range([0.3, 0.7]).unwrap(),
        ];
        let plane_domain = [
            interval_range([-0.1, 0.8]).unwrap(),
            interval_range([-0.1, 0.6]).unwrap(),
        ];
        let (first, second, domain) = if swapped {
            (
                plane,
                source,
                [
                    plane_domain[0],
                    plane_domain[1],
                    source_domain[0],
                    source_domain[1],
                ],
            )
        } else {
            (
                source,
                plane,
                [
                    source_domain[0],
                    source_domain[1],
                    plane_domain[0],
                    plane_domain[1],
                ],
            )
        };
        (
            Curve::IntersectionCurve(SurfaceCurve::with_boundaries(
                Box::new(first),
                Box::new(second),
                Box::new(guide),
                None,
                None,
            )),
            domain,
            scale,
        )
    }

    fn contains(bound: ValidatedInterval, value: f64) {
        assert!(
            bound.lower <= value && value <= bound.upper,
            "{bound:?} excludes {value}"
        );
    }

    fn prepare_native<'a>(
        curve: &'a Curve,
        range: [f64; 2],
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> PreparedValidatedCurve<'a> {
        prepared_validated_curve(
            curve,
            range,
            &ValidatedCurveOptions::default(),
            budget,
            cancel,
        )
        .unwrap()
    }

    #[test]
    fn diagonal_root_image_closure_preserves_both_original_branches_and_whole_raw_time() {
        let cancel = AtomicBool::new(false);
        for side in [-1.0, 1.0] {
            for weight in [-1.0, 1.0] {
                // The original tensor is S(u,v)=(2u-1,2v-1,(2u-1)^2).
                // The plane z=1/4 has two components; each original guide
                // selects one, over its unchanged affine raw time [-2,2].
                let source = Surface::NurbsSurface(NurbsSurface::new(BsplineSurface::new(
                    (KnotVector::bezier_knot(2), KnotVector::bezier_knot(1)),
                    [(-1.0, 1.0), (0.0, -1.0), (1.0, 1.0)]
                        .into_iter()
                        .map(|(x, z)| {
                            [-1.0, 1.0]
                                .into_iter()
                                .map(|y| Vector4::new(x * weight, y * weight, z * weight, weight))
                                .collect()
                        })
                        .collect(),
                )));
                let plane = Surface::Plane(Plane::new(
                    Point3::new(-1.0, -1.0, 0.25),
                    Point3::new(1.0, -1.0, 0.25),
                    Point3::new(-1.0, 1.0, 0.25),
                ));
                let x = side * 0.5;
                let u = if side < 0.0 { 0.25 } else { 0.75 };
                let curve = Curve::IntersectionCurve(SurfaceCurve::with_boundaries(
                    Box::new(source),
                    Box::new(plane),
                    // The whole v image excludes the original box midpoint;
                    // reconstructing it still needs the original gradient hull.
                    Box::new(Curve::Line(Line(
                        Point3::new(x, 0.5, 0.25),
                        Point3::new(x, 0.53125, 0.25),
                    ))),
                    None,
                    None,
                ));
                let original_u = interval_range([u - 0.21875, u + 0.21875]).unwrap();
                let original_v = interval_range([0.1, 0.9]).unwrap();
                let domain = [original_u, original_v, original_u, original_v];
                let mut work = ValidatedGeometryBudget::new(1_000_000);
                let tube = certify(
                    &curve,
                    Projection::Native,
                    interval_range([-2.0, 2.0]).unwrap(),
                    domain,
                    &mut work,
                    &cancel,
                )
                .unwrap()
                .unwrap();
                assert!(matches!(tube.authority, TubeAuthority::Plane(_)));
                assert_eq!(tube.domain, domain);
                assert!(tube.contraction_bound() > 0.8 && tube.contraction_bound() < 1.0);
                let jet = implicit_jet(&tube).unwrap();
                for carrier in [0, 2] {
                    contains(tube.rooted_domain[carrier], u);
                    assert!(tube.rooted_domain[carrier].width() < 1.0e-10);
                    for v in [0.71875, 0.78125] {
                        contains(tube.rooted_domain[carrier + 1], v);
                    }
                }
                contains(jet.position[0], x);
                contains(jet.position[1], 0.4375);
                contains(jet.position[1], 0.5625);
                contains(jet.position[2], 0.25);
                for (axis, first) in [0.0, 0.03125, 0.0].into_iter().enumerate() {
                    contains(jet.first[axis], first);
                    contains(jet.second[axis], 0.0);
                }
            }
        }
    }

    #[test]
    fn coupled_original_roots_reconstruct_plane_uvs_on_both_whole_time_branches() {
        let cancel = AtomicBool::new(false);
        for side in [-1.0, 1.0] {
            // Original inverted chart: S(u,v)=(2v-1,2u-1,(2v-1)^2+(2u-1)^2).
            // The plane cuts a circle. The original line coordinate selects
            // one of its two x branches over the complete raw time [-2,2].
            let coefficients = [(-1.0, 1.0), (0.0, -1.0), (1.0, 1.0)];
            let source = Surface::BsplineSurface(BsplineSurface::new(
                (KnotVector::bezier_knot(2), KnotVector::bezier_knot(2)),
                coefficients
                    .into_iter()
                    .map(|(x, xx)| {
                        coefficients
                            .into_iter()
                            .map(|(y, yy)| Point3::new(x, y, xx + yy))
                            .collect()
                    })
                    .collect(),
            ))
            .inverse();
            let plane = Surface::Plane(Plane::new(
                Point3::new(0.0, 0.0, 0.25),
                Point3::new(1.0, 0.0, 0.25),
                Point3::new(0.0, 1.0, 0.25),
            ));
            let speed = 1.0 / 128.0;
            // Declare the original affine guide's actual raw restriction.
            // The identity wrapper keeps its coefficients and parameter gauge.
            let guide = Curve::TrimmedCurve(Processor::new(TrimmedCurve::new(
                Box::new(Curve::Line(Line(
                    Point3::new(side * 0.3, 0.4, 0.25),
                    Point3::new(side * 0.3, 0.4 + speed, 0.25),
                ))),
                (-2.0, 2.0),
            )));
            let curve = Curve::IntersectionCurve(SurfaceCurve::with_boundaries(
                Box::new(source),
                Box::new(plane),
                Box::new(guide),
                None,
                None,
            ));
            let mut work = ValidatedGeometryBudget::new(1_000_000);
            let options = ValidatedCurveOptions::default();
            let prepared =
                prepared_validated_curve(&curve, [-2.0, 2.0], &options, &mut work, &cancel)
                    .unwrap();
            assert_eq!(prepared.range(), [-2.0, 2.0]);
            let mut covered = -2.0;
            for piece in prepared.pieces() {
                assert_eq!(piece.parameter[0], covered);
                covered = piece.parameter[1];
                let domain = flatten(piece.source_parameters.expect("original UV binding"));
                let tube = certify(
                    &curve,
                    piece.projection,
                    interval_range(piece.parameter).unwrap(),
                    domain,
                    &mut work,
                    &cancel,
                )
                .unwrap()
                .unwrap();
                assert_eq!(tube.domain, domain);
                for t in piece.parameter {
                    let y = 0.4 + speed * t;
                    let x = side * (0.25_f64 - y * y).sqrt();
                    contains(tube.rooted_domain[0], (y + 1.0) * 0.5);
                    contains(tube.rooted_domain[1], (x + 1.0) * 0.5);
                    contains(tube.rooted_domain[2], x);
                    contains(tube.rooted_domain[3], y);
                }
            }
            assert_eq!(covered, 2.0);
            let jet = prepared.jet([-2.0, 2.0], &mut work, &cancel).unwrap();
            assert!(if side > 0.0 {
                jet.position[0].lower > 0.0
            } else {
                jet.position[0].upper < 0.0
            });
            for t in [-2.0, 0.0, 2.0] {
                let y = 0.4 + speed * t;
                let x = side * (0.25_f64 - y * y).sqrt();
                contains(jet.position[0], x);
                contains(jet.position[1], y);
                contains(jet.position[2], 0.25);
                contains(jet.first[0], -y * speed / x);
                contains(jet.first[1], speed);
                contains(jet.first[2], 0.0);
                contains(jet.second[0], -0.25 * speed * speed / (x * x * x));
                contains(jet.second[1], 0.0);
                contains(jet.second[2], 0.0);
                let point = prepared.evaluate(t, &mut work, &cancel).unwrap();
                for (axis, expected) in [x, y, 0.25].into_iter().enumerate() {
                    contains(point.error[axis], expected);
                    contains(point.error[axis], point.position[axis]);
                    assert!(point.error[axis].width() <= options.endpoint_tolerance);
                }
            }
            for outside in [-2.25, 2.25] {
                assert!(matches!(
                    prepared.evaluate(outside, &mut work, &cancel),
                    Err(ValidatedGeometryError::ParameterOutOfRange)
                ));
            }
            assert!(matches!(
                prepared.evaluate(0.0, &mut work, &AtomicBool::new(true)),
                Err(ValidatedGeometryError::Cancelled)
            ));
        }
    }

    #[test]
    fn full_span_native_quadratic_jets_preserve_reflection_reversal_and_carrier_order() {
        let cancel = AtomicBool::new(false);
        for posed in [false, true] {
            for reversed in [false, true] {
                for swapped in [false, true] {
                    let (curve, _, scale) = parabola(posed, reversed, swapped);
                    let parameter = if reversed {
                        [87.0 / 128.0, 88.0 / 128.0]
                    } else {
                        [40.0 / 128.0, 41.0 / 128.0]
                    };
                    let mut work = ValidatedGeometryBudget::new(2_000_000);
                    let prepared = prepared_validated_curve(
                        &curve, parameter, &ValidatedCurveOptions::default(), &mut work, &cancel,
                    ).unwrap_or_else(|error| panic!(
                        "Original native preparation refused: posed={posed}, reversed={reversed}, swapped={swapped}, range={parameter:?}, consumed={}, remaining={}, error={error:?}",
                        work.consumed(), work.remaining(),
                    ));
                    let full_jet = prepared.jet(parameter, &mut work, &cancel).unwrap();
                    let origin = if posed { [7.0, 11.0, 13.0] } else { [0.0; 3] };
                    let sign = if reversed { -1.0 } else { 1.0 };
                    let mut covered = parameter[0];
                    for piece in prepared.pieces() {
                        assert_eq!(piece.parameter[0], covered);
                        covered = piece.parameter[1];
                        // These are continuation's actual original-bound tubes,
                        // not arbitrary broad boxes around sampled roots.
                        let domain = flatten(piece.source_parameters.unwrap());
                        let tube = certify(
                            &curve,
                            piece.projection,
                            interval_range(piece.parameter).unwrap(),
                            domain,
                            &mut work,
                            &cancel,
                        )
                        .unwrap()
                        .unwrap();
                        assert!(matches!(tube.authority, TubeAuthority::Plane(_)));
                        assert_eq!(tube.domain, domain);
                        assert!(tube.contraction_bound() < 1.0);
                        assert!(
                            tube.surface_residual()
                                .into_iter()
                                .all(ValidatedInterval::contains_zero)
                        );
                        let jet = implicit_jet(&tube).unwrap();
                        // Monotone endpoint extrema describe each WHOLE piece.
                        for t in piece.parameter {
                            let u = if reversed { 1.0 - t } else { t };
                            for (axis, expected) in [
                                origin[0] + scale[0] * u,
                                origin[1] + scale[1] * u * u,
                                origin[2],
                            ]
                            .into_iter()
                            .enumerate()
                            {
                                contains(jet.position[axis], expected);
                                contains(full_jet.position[axis], expected);
                            }
                            for (axis, expected) in
                                [sign * scale[0], sign * 2.0 * scale[1] * u, 0.0]
                                    .into_iter()
                                    .enumerate()
                            {
                                contains(jet.first[axis], expected);
                                contains(full_jet.first[axis], expected);
                            }
                            for (axis, expected) in
                                [0.0, 2.0 * scale[1], 0.0].into_iter().enumerate()
                            {
                                contains(jet.second[axis], expected);
                                contains(full_jet.second[axis], expected);
                            }
                            let index = if swapped { 2 } else { 0 };
                            contains(tube.rooted_domain[index], u);
                            contains(tube.rooted_domain[index + 1], 0.5);
                            contains(tube.rooted_domain[2 - index], u - 0.5 * u * u);
                            contains(tube.rooted_domain[3 - index], u * u);
                        }
                    }
                    assert_eq!(covered, parameter[1]);
                }
            }
        }
    }

    #[test]
    fn changing_coordinate_planes_use_the_original_raw_gauge_and_analytic_second_jet() {
        let (curve, _, _) = parabola(false, false, false);
        let span: [f64; 2] = [0.3, 0.32];
        let (source, Surface::Plane(plane), guide) = carriers(&curve).unwrap() else {
            panic!("Actual plane carrier")
        };
        let root = 2.0 * span[0] / (1.0 + (1.0 + 4.0 * span[0] * span[0]).sqrt());
        let cancel = AtomicBool::new(false);
        let mut work = ValidatedGeometryBudget::new(2_000_000);
        let prepared = prepared_face_plane_curve(
            guide,
            ValidatedFacePlaneSource {
                surface: source,
                anchor: Some(ValidatedFacePlaneSourceAnchor::Front([root, 0.5])),
                endpoint_coordinate_planes: span.map(|t| {
                    Some(CoordinatePlane {
                        normal: [1.0, t, 0.0],
                        offset: t,
                    })
                }),
            },
            plane,
            span,
            &ValidatedCurveOptions::default(),
            &mut work,
            &cancel,
        )
        .unwrap();
        let jet = prepared.jet(span, &mut work, &cancel).unwrap();
        for piece in prepared.pieces() {
            let tube = certify(
                prepared.native_curve().unwrap(),
                piece.projection,
                interval_range(piece.parameter).unwrap(),
                flatten(piece.source_parameters.unwrap()),
                &mut work,
                &cancel,
            )
            .unwrap()
            .unwrap();
            assert!(matches!(tube.authority, TubeAuthority::Plane(_)));
        }
        // Solve u+t*u^2=t independently, not by calling the old system or
        // reading any leader point. Differentiate this scalar identity twice.
        for t in [span[0], 0.31, span[1]] {
            let u = 2.0 * t / (1.0 + (1.0 + 4.0 * t * t).sqrt());
            let first = (1.0 - u * u) / (1.0 + 2.0 * t * u);
            let second = (-4.0 * u * first - 2.0 * t * first * first) / (1.0 + 2.0 * t * u);
            for (axis, expected) in [u, u * u, 0.0].into_iter().enumerate() {
                contains(jet.position[axis], expected);
            }
            for (axis, expected) in [first, 2.0 * u * first, 0.0].into_iter().enumerate() {
                contains(jet.first[axis], expected);
            }
            for (axis, expected) in [second, 2.0 * first * first + 2.0 * u * second, 0.0]
                .into_iter()
                .enumerate()
            {
                contains(jet.second[axis], expected);
            }
        }
    }

    #[test]
    fn reconstructed_plane_domain_is_containment_not_root_clipping() {
        let (curve, _, _) = parabola(false, false, false);
        let cancel = AtomicBool::new(false);
        let mut work = ValidatedGeometryBudget::new(2_000_000);
        let prepared = prepare_native(&curve, [0.3, 0.32], &mut work, &cancel);
        let piece = &prepared.pieces()[0];
        let mut domain = flatten(piece.source_parameters.unwrap());
        let span = interval_range(piece.parameter).unwrap();
        domain[2] = interval_range([2.0, 3.0]).unwrap();
        for _ in 0..2 {
            let before = work.consumed();
            assert!(matches!(
                certify(&curve, piece.projection, span, domain, &mut work, &cancel),
                Err(ValidatedGeometryError::Unresolved)
            ));
            assert!(work.consumed() > before);
        }
        assert!(
            super::super::certify(&curve, piece.projection, span, domain, &mut work, &cancel)
                .is_err()
        );
    }

    #[test]
    fn degenerate_parallel_and_rational_pole_carriers_never_gain_authority() {
        let cancel = AtomicBool::new(false);
        let guide = || {
            Box::new(Curve::Line(Line(
                Point3::origin(),
                Point3::new(1.0, 0.0, 0.0),
            )))
        };
        let domain = [interval_range([0.4, 0.6]).unwrap(); 4];
        for plane in [
            Plane::xy(),
            Plane::new(
                Point3::origin(),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(2.0, 0.0, 0.0),
            ),
        ] {
            let curve = Curve::IntersectionCurve(SurfaceCurve::with_boundaries(
                Box::new(Surface::Plane(Plane::xy())),
                Box::new(Surface::Plane(plane)),
                guide(),
                None,
                None,
            ));
            assert!(matches!(
                certify(
                    &curve,
                    Projection::Native,
                    ZERO,
                    domain,
                    &mut ValidatedGeometryBudget::new(1_000_000),
                    &cancel
                ),
                Err(ValidatedGeometryError::Singular | ValidatedGeometryError::NonTransverse)
            ));
        }
        let pole = Surface::NurbsSurface(NurbsSurface::new(BsplineSurface::new(
            (KnotVector::bezier_knot(1), KnotVector::bezier_knot(1)),
            vec![
                vec![
                    Vector4::new(0.0, 0.0, 0.0, -1.0),
                    Vector4::new(0.0, 1.0, 1.0, -1.0),
                ],
                vec![
                    Vector4::new(1.0, 0.0, 0.0, 1.0),
                    Vector4::new(1.0, 1.0, 1.0, 1.0),
                ],
            ],
        )));
        let curve = Curve::IntersectionCurve(SurfaceCurve::with_boundaries(
            Box::new(pole),
            Box::new(Surface::Plane(Plane::xy())),
            guide(),
            None,
            None,
        ));
        assert!(matches!(
            certify(
                &curve,
                Projection::Native,
                ZERO,
                domain,
                &mut ValidatedGeometryBudget::new(1_000_000),
                &cancel
            ),
            Err(ValidatedGeometryError::DenominatorContainsZero)
        ));
    }

    #[test]
    fn new_work_is_cumulative_and_cancellation_never_falls_through_to_generic() {
        let (curve, _, _) = parabola(false, false, false);
        let cancel = AtomicBool::new(false);
        let mut setup = ValidatedGeometryBudget::new(2_000_000);
        let prepared = prepare_native(&curve, [0.3, 0.32], &mut setup, &cancel);
        let piece = &prepared.pieces()[0];
        let domain = flatten(piece.source_parameters.unwrap());
        let span = interval_range(piece.parameter).unwrap();
        let mut complete = ValidatedGeometryBudget::new(1_000_000);
        certify(
            &curve,
            Projection::Native,
            span,
            domain,
            &mut complete,
            &cancel,
        )
        .unwrap()
        .unwrap();
        // Every prefix is a real bounded attempt, including reconstruction and
        // both derivative solves; no exhausted attempt can return a receipt.
        for limit in 0..complete.consumed() {
            let mut work = ValidatedGeometryBudget::new(limit);
            assert!(matches!(
                super::super::certify(&curve, Projection::Native, span, domain, &mut work, &cancel),
                Err(ValidatedGeometryError::ResourceLimit)
            ));
            assert_eq!(work.consumed(), limit);
        }
        assert!(matches!(
            super::super::certify(
                &curve,
                Projection::Native,
                span,
                domain,
                &mut complete,
                &AtomicBool::new(true)
            ),
            Err(ValidatedGeometryError::Cancelled)
        ));
        assert_eq!(
            PLANE_WORKSPACE_BYTES.min(ORIGINAL_WORKSPACE_BYTES),
            PLANE_WORKSPACE_BYTES
        );
    }

    #[test]
    fn consumer_binding_keeps_native_endpoint_and_retained_storage_contracts() {
        let (curve, _, _) = parabola(false, true, false);
        let cancel = AtomicBool::new(false);
        let options = ValidatedCurveOptions::default();
        let mut work = ValidatedGeometryBudget::new(2_000_000);
        let range = [0.65, 0.7];
        let prepared =
            prepared_validated_curve(&curve, range, &options, &mut work, &cancel).unwrap();
        assert_eq!(prepared.range(), range);
        assert!(std::ptr::eq(prepared.native_curve().unwrap(), &curve));
        assert!(prepared.retained_bytes() <= options.max_retained_bytes);
        let mut covered = range[0];
        for piece in prepared.pieces() {
            assert_eq!(piece.parameter[0], covered);
            covered = piece.parameter[1];
            assert!(piece.source_parameters.is_some());
            assert!(
                piece
                    .surface_residual
                    .into_iter()
                    .all(ValidatedInterval::contains_zero)
            );
        }
        assert_eq!(covered, range[1]);
        for t in range {
            let point = prepared.evaluate(t, &mut work, &cancel).unwrap();
            let u = 1.0 - t;
            for (axis, expected) in [u, u * u, 0.0].into_iter().enumerate() {
                contains(point.error[axis], expected);
                assert!(point.error[axis].width() <= options.endpoint_tolerance);
            }
        }
        let (source, Surface::Plane(plane), guide) = carriers(&curve).unwrap() else {
            panic!("Actual plane carrier")
        };
        let generated = prepared_face_plane_curve(
            guide,
            ValidatedFacePlaneSource {
                surface: source,
                anchor: Some(ValidatedFacePlaneSourceAnchor::Front([0.35, 0.5])),
                endpoint_coordinate_planes: [None; 2],
            },
            plane,
            range,
            &options,
            &mut work,
            &cancel,
        )
        .unwrap();
        let (retained_source, retained_plane, _) =
            carriers(generated.native_curve().unwrap()).unwrap();
        assert!(
            validated_same_surface_coefficients(source, retained_source, &mut work, &cancel)
                .unwrap()
        );
        assert!(matches!(retained_plane, Surface::Plane(retained) if retained == plane));
        assert_eq!(generated.range(), range);
        for t in range {
            let point = generated.evaluate(t, &mut work, &cancel).unwrap();
            let u = 1.0 - t;
            contains(point.error[0], u);
            contains(point.error[1], u * u);
            contains(point.error[2], 0.0);
        }
        let too_small = ValidatedCurveOptions {
            max_retained_bytes: size_of::<PreparedValidatedCurve<'_>>() - 1,
            ..options
        };
        assert!(matches!(
            prepared_validated_curve(&curve, range, &too_small, &mut work, &cancel),
            Err(ValidatedGeometryError::ResourceLimit)
        ));
    }

    #[test]
    fn only_actual_affine_plane_coefficients_select_elimination() {
        let (original, _, _) = parabola(false, false, false);
        let (source, _, leader) = carriers(&original).unwrap();
        let second = Surface::BsplineSurface(BsplineSurface::new(
            (KnotVector::bezier_knot(1), KnotVector::bezier_knot(1)),
            vec![
                vec![Point3::origin(), Point3::new(0.5, 1.0, 0.0)],
                vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.5, 1.0, 0.0)],
            ],
        ));
        let curve = Curve::IntersectionCurve(SurfaceCurve::with_boundaries(
            Box::new(source.clone()),
            Box::new(second),
            Box::new(leader.clone()),
            None,
            None,
        ));
        let cancel = AtomicBool::new(false);
        let mut work = ValidatedGeometryBudget::new(2_000_000);
        let range = [0.3, 0.32];
        let prepared = prepare_native(&curve, range, &mut work, &cancel);
        let jet = prepared.jet(range, &mut work, &cancel).unwrap();
        let mut covered = range[0];
        for piece in prepared.pieces() {
            assert_eq!(piece.parameter[0], covered);
            covered = piece.parameter[1];
            let domain = flatten(piece.source_parameters.unwrap());
            let span = interval_range(piece.parameter).unwrap();
            assert!(
                certify(&curve, piece.projection, span, domain, &mut work, &cancel)
                    .unwrap()
                    .is_none()
            );
            let tube =
                super::super::certify(&curve, piece.projection, span, domain, &mut work, &cancel)
                    .unwrap();
            assert!(matches!(tube.authority, TubeAuthority::Krawczyk { .. }));
        }
        assert_eq!(covered, range[1]);
        for t in range {
            contains(jet.position[0], t);
            contains(jet.position[1], t * t);
        }
        // Stored vectors, not their rounded translated defining points, are
        // the actual coefficients of a vector-defined native plane.
        let plane = Plane::from_origin_vectors(
            Point3::new(1.0e16, 1.0e16, 1.0e16),
            Vector3::unit_x(),
            Vector3::unit_y(),
        );
        let chart = Chart::new(&plane, &mut work, &cancel).unwrap();
        assert_eq!(chart.axes[0], [ONE, ZERO, ZERO]);
        assert_eq!(chart.axes[1], [ZERO, ONE, ZERO]);
        assert_eq!(chart.vector_parameters([ONE, ZERO, ZERO]), [ONE, ZERO]);
    }

    #[test]
    fn native_center_residual_encloses_an_independent_cubic_law_in_raw_time() {
        let source = Surface::Plane(Plane::from_origin_vectors(
            Point3::new(4.0, -3.0, 5.0),
            Vector3::new(2.0, 1.0, 0.0),
            Vector3::new(-1.0, 3.0, 1.0),
        ));
        let basis = Curve::BsplineCurve(BsplineCurve::new(
            KnotVector::from(vec![2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0]),
            vec![
                Point3::new(1.0, 2.0, 3.0),
                Point3::new(3.0, 2.0, 4.0),
                Point3::new(5.0, 2.0, 5.0),
                Point3::new(7.0, 0.0, 6.0),
            ],
        ));
        for reversed in [false, true] {
            let mut processor =
                Processor::new(TrimmedCurve::new(Box::new(basis.clone()), (2.0, 4.0)));
            if reversed {
                processor.invert();
            }
            let leader = Curve::TrimmedCurve(processor);
            for range in [[2.6, 2.64], [2.25, 3.75]] {
                let parameter = interval_range(range).unwrap();
                for [u, v] in [[0.25, -0.5], [1.0, 2.0]] {
                    let cancel = AtomicBool::new(false);
                    let mut budget = ValidatedGeometryBudget::new(1_000_000);
                    let coordinate = CoordinateJet {
                        parameter,
                        jet: projection_jet(
                            &leader,
                            Projection::Native,
                            parameter,
                            &mut budget,
                            &cancel,
                        )
                        .unwrap(),
                    };
                    let centered = native_center_residual(
                        &source,
                        &leader,
                        &coordinate,
                        [u, v].map(ValidatedInterval::point),
                        &mut ProjectionExecution {
                            budget: &mut budget,
                            cancel: &cancel,
                        },
                    )
                    .unwrap()
                    .unwrap();
                    let point = [4.0 + 2.0 * u - v, -3.0 + u + 3.0 * v, 5.0 + v];
                    let direct = coordinate
                        .jet
                        .residual(parameter, point.map(ValidatedInterval::point));
                    let combined = direct.intersection(centered).unwrap();
                    for fraction in [0.0, 0.25, 0.5, 0.75, 1.0] {
                        let t = range[0] + fraction * (range[1] - range[0]);
                        let s = if reversed {
                            (4.0 - t) / 2.0
                        } else {
                            (t - 2.0) / 2.0
                        };
                        let sign = if reversed { -1.0 } else { 1.0 };
                        let residual = sign
                            * (3.0 * (point[0] - (1.0 + 6.0 * s))
                                - 3.0 * s * s * (point[1] - (2.0 - 2.0 * s.powi(3)))
                                + 1.5 * (point[2] - (3.0 + 3.0 * s)));
                        contains(centered, residual);
                        contains(combined, residual);
                    }
                }
            }
        }
    }

    #[test]
    fn native_center_residual_still_admits_original_source_poles_for_zero_functionals() {
        let knots = KnotVector::bezier_knot(1);
        let source = Surface::NurbsSurface(NurbsSurface::new(BsplineSurface::new(
            (knots.clone(), knots),
            vec![
                vec![Vector4::new(0.0, 0.0, 0.0, -1.0); 2],
                vec![Vector4::new(0.0, 0.0, 0.0, 1.0); 2],
            ],
        )));
        let leader = Curve::Line(Line(Point3::origin(), Point3::origin()));
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(1_000_000);
        let parameter = interval_range([0.0, 1.0]).unwrap();
        let coordinate = CoordinateJet {
            parameter,
            jet: projection_jet(&leader, Projection::Native, parameter, &mut budget, &cancel)
                .unwrap(),
        };
        assert!(matches!(
            native_center_residual(
                &source,
                &leader,
                &coordinate,
                [ValidatedInterval::point(0.5); 2],
                &mut ProjectionExecution {
                    budget: &mut budget,
                    cancel: &cancel
                },
            ),
            Err(ValidatedGeometryError::DenominatorContainsZero)
        ));
    }

    #[test]
    fn native_cubic_guide_retains_nonzero_second_projection_normal() {
        let source = Surface::BsplineSurface(BsplineSurface::new(
            (KnotVector::bezier_knot(3), KnotVector::bezier_knot(1)),
            [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 1.0)]
                .into_iter()
                .map(|(x, y)| vec![Point3::new(x, y, -0.5), Point3::new(x, y, 0.5)])
                .collect(),
        ));
        let guide = Curve::BsplineCurve(BsplineCurve::new(
            KnotVector::bezier_knot(3),
            vec![
                Point3::origin(),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(2.0, 0.0, 0.0),
                Point3::new(3.0, 1.0, 0.0),
            ],
        ));
        let curve = Curve::IntersectionCurve(SurfaceCurve::with_boundaries(
            Box::new(source),
            Box::new(Surface::Plane(Plane::xy())),
            Box::new(guide),
            None,
            None,
        ));
        let span = [0.3, 0.32];
        let cancel = AtomicBool::new(false);
        let mut work = ValidatedGeometryBudget::new(2_000_000);
        let prepared = prepare_native(&curve, span, &mut work, &cancel);
        let jet = prepared.jet(span, &mut work, &cancel).unwrap();
        for piece in prepared.pieces() {
            let tube = certify(
                &curve,
                piece.projection,
                interval_range(piece.parameter).unwrap(),
                flatten(piece.source_parameters.unwrap()),
                &mut work,
                &cancel,
            )
            .unwrap()
            .unwrap();
            assert!(matches!(tube.authority, TubeAuthority::Plane(_)));
        }
        // Exact binary64 Bezier coefficients give x=3t and y=t^3;
        // the projection normal's second derivative is (0,6,0).
        for t in span {
            contains(jet.position[0], 3.0 * t);
            contains(jet.position[1], t * t * t);
            contains(jet.first[0], 3.0);
            contains(jet.first[1], 3.0 * t * t);
            contains(jet.second[0], 0.0);
            contains(jet.second[1], 6.0 * t);
        }
    }

    #[test]
    fn original_gradient_hull_covers_a_reduced_image_that_excludes_the_center() {
        let cancel = AtomicBool::new(false);
        let mut work = ValidatedGeometryBudget::new(1_000);
        let plane = Plane::from_origin_vectors(
            Point3::origin(),
            Vector3::unit_x(),
            Vector3::new(0.5, 1.0, 0.0),
        );
        let chart = Chart::new(&plane, &mut work, &cancel).unwrap();
        let span = interval_range([-1.0, 1.0]).unwrap();
        // The nonlinear image excludes its center at u=v=0. Only gradients
        // on the ORIGINAL box [-1,1]^2 cover every connecting segment:
        // S=(u^2+v/2,v,0), hence the actual plane coordinates are (u^2,v).
        // Gradients restricted to u in [1/2,1] would instead lose u=1/2's
        // actual coordinate 1/4 when integrated from the outside center.
        let curved = ValidatedSurfaceJet {
            position: [interval_range([-0.5, 1.5]).unwrap(), span, ZERO],
            du: [interval_range([-2.0, 2.0]).unwrap(), ZERO, ZERO],
            dv: [ValidatedInterval::point(0.5), ONE, ZERO],
            duu: [ValidatedInterval::point(2.0), ZERO, ZERO],
            duv: [ZERO; 3],
            dvv: [ZERO; 3],
        };
        let image_delta = [interval_range([0.5, 1.0]).unwrap(), span];
        let mean = chart
            .mean_value_parameters(
                [ZERO; 2],
                curved.du,
                curved.dv,
                image_delta,
                &mut work,
                &cancel,
            )
            .unwrap();
        contains(mean[0], 0.25);
        contains(mean[0], 1.0);
        contains(mean[1], -1.0);
        contains(mean[1], 1.0);
    }

    #[test]
    fn generated_native_parabola_keeps_translated_full_span_affordable_and_sealed() {
        let (original, _, _) = parabola(true, false, false);
        let (source, _, guide) = carriers(&original).unwrap();
        let plane = Plane::from_origin_vectors(
            Point3::new(7.0, 11.0, 13.0),
            Vector3::new(-2.0, 0.0, 0.0),
            Vector3::new(0.0, 3.0, 0.0),
        );
        let range = [0.3, 0.32];
        let cancel = AtomicBool::new(false);
        let options = ValidatedCurveOptions::default();
        let mut work = ValidatedGeometryBudget::new(2_000_000);
        let prepared = prepared_face_plane_curve(
            guide,
            ValidatedFacePlaneSource {
                surface: source,
                anchor: Some(ValidatedFacePlaneSourceAnchor::Front([range[0], 0.5])),
                endpoint_coordinate_planes: [None; 2],
            },
            &plane,
            range,
            &options,
            &mut work,
            &cancel,
        )
        .unwrap();
        assert_eq!(prepared.range(), range);
        let (retained_source, retained_plane, _) =
            carriers(prepared.native_curve().unwrap()).unwrap();
        assert!(
            validated_same_surface_coefficients(source, retained_source, &mut work, &cancel)
                .unwrap()
        );
        assert!(matches!(retained_plane, Surface::Plane(retained) if retained == &plane));
        let jet = prepared.jet(range, &mut work, &cancel).unwrap();
        let mut covered = range[0];
        for piece in prepared.pieces() {
            assert_eq!(piece.parameter[0], covered);
            covered = piece.parameter[1];
            assert_eq!(piece.projection, Projection::Native);
            let uv = piece.source_parameters.unwrap();
            // The actual original source/plane roots and both parameter
            // inventories cover every closed piece, without world snapping.
            for t in piece.parameter {
                contains(uv[0][0], t);
                contains(uv[0][1], 0.5);
                contains(uv[1][0], t);
                contains(uv[1][1], t * t);
            }
        }
        assert_eq!(covered, range[1]);
        // These exact monotone polynomials give the whole requested span's
        // extrema, independent of the implementation's continuation partition.
        for t in range {
            for (axis, value) in [7.0 - 2.0 * t, 11.0 + 3.0 * t * t, 13.0]
                .into_iter()
                .enumerate()
            {
                contains(jet.position[axis], value);
            }
            for (axis, value) in [-2.0, 6.0 * t, 0.0].into_iter().enumerate() {
                contains(jet.first[axis], value);
            }
            for (axis, value) in [0.0, 6.0, 0.0].into_iter().enumerate() {
                contains(jet.second[axis], value);
            }
            let endpoint = prepared.evaluate(t, &mut work, &cancel).unwrap();
            for (axis, value) in [7.0 - 2.0 * t, 11.0 + 3.0 * t * t, 13.0]
                .into_iter()
                .enumerate()
            {
                contains(endpoint.error[axis], value);
                assert!(endpoint.error[axis].width() <= options.endpoint_tolerance);
            }
        }
        assert!(work.remaining() > 0);
    }

    #[test]
    fn scalar_native_system_keeps_displaced_reparameterized_guide_equations_and_time_partials() {
        let (curve, _, _) = parabola(false, false, false);
        let (source, Surface::Plane(plane), _) = carriers(&curve).unwrap() else {
            panic!("Actual original affine carrier")
        };
        let cancel = AtomicBool::new(false);
        let mut work = ValidatedGeometryBudget::new(1_000_000);
        let chart = Chart::new(plane, &mut work, &cancel).unwrap();
        // The independent leader is L(t)=(s,s^3,0), s=a+b*t.
        // It is deliberately NOT the intersection's native quadratic guide.
        for (a, b) in [(0.0, 1.0), (0.125, 2.0), (0.75, -1.0)] {
            let parameter = interval_range([0.25, 0.375]).unwrap();
            let s = ValidatedInterval::point(a) + ValidatedInterval::point(b) * parameter;
            let guide = ValidatedCurveJet {
                position: [s, s * s * s, ZERO],
                first: [
                    ValidatedInterval::point(b),
                    ValidatedInterval::point(3.0 * b) * s * s,
                    ZERO,
                ],
                second: [ZERO, ValidatedInterval::point(6.0 * b * b) * s, ZERO],
                third: [ZERO, ValidatedInterval::point(6.0 * b * b * b), ZERO],
            };
            let coordinate = CoordinateJet {
                parameter,
                jet: PlaneJet::Native(guide),
            };
            let domain = [
                interval_range([0.2, 0.7]).unwrap(),
                interval_range([0.4, 0.6]).unwrap(),
            ];
            let reduced = system(source, &chart, &coordinate, domain, &mut work, &cancel).unwrap();
            for t in [0.25_f64, 0.3125, 0.375] {
                let s = a + b * t;
                for u in [0.2_f64, 0.45, 0.7] {
                    for v in [0.4_f64, 0.5, 0.6] {
                        let point = [u, u * u, v - 0.5].map(ValidatedInterval::point);
                        contains(reduced.value[0], v - 0.5);
                        contains(
                            reduced.value[1],
                            b * (u - s) + 3.0 * b * s * s * (u * u - s * s * s),
                        );
                        contains(reduced.jacobian[0][0], 0.0);
                        contains(reduced.jacobian[0][1], 1.0);
                        contains(reduced.jacobian[1][0], b + 6.0 * b * s * s * u);
                        contains(reduced.jacobian[1][1], 0.0);
                        contains(
                            coordinate.jet.partial_first(point),
                            6.0 * b * b * s * (u * u - s * s * s) - b * b * (1.0 + 9.0 * s.powi(4)),
                        );
                        contains(
                            coordinate.jet.partial_second(point),
                            6.0 * b * b * b * (u * u - s * s * s) - 54.0 * b * b * b * s * s * s,
                        );
                    }
                }
            }
        }
    }
}
