use super::{
    ONE, Taylor, ValidatedCurveJet, ValidatedCurveJet2D, ValidatedGeometryBudget,
    ValidatedGeometryError, ValidatedInterval, ValidatedSurfaceJet, curve_jet_from_taylor,
    interval_range,
};
use crate::{
    Conic2D, Curve, Curve2D, KnotVector, Matrix3, Matrix4, ParametricCurve, Plane, Surface,
    geometry::MAX_NATIVE_GEOMETRY_NESTING,
};
use monstertruck_geometry::prelude::BoundaryCurve2D;
use std::sync::atomic::AtomicBool;

const MAX_DEGREE: usize = 32;

/// Encloses an explicit native spatial curve and three native derivatives.
///
/// Implicit intersections are intentionally refused here: use
/// [`super::super::validated_curve::prepared_validated_curve`] and its immutable
/// branch binding instead. A range crossing a knot below first-order
/// continuity returns [`ValidatedGeometryError::NonSmooth`]. At other knots the
/// second and third bounds include both one-sided derivatives.
pub fn validated_curve_jet(
    curve: &Curve,
    range: [f64; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<ValidatedCurveJet, ValidatedGeometryError> {
    let value = curve_taylor(
        curve,
        Taylor::variable(interval_range(range)?, 0),
        0,
        budget,
        cancel,
    )?;
    let [position, first, second, third] = curve_jet_from_taylor(value);
    Ok(ValidatedCurveJet {
        position,
        first,
        second,
        third,
    })
}

/// Zeroth-order native image bounds across continuous C0 source knots.
/// This uses the same outward basis and denominator arithmetic, with constant
/// Taylor inputs so derivative overflow cannot obstruct a position-only query.
pub(crate) fn validated_curve_position_bounds(
    curve: &Curve,
    range: [f64; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<[ValidatedInterval; 3], ValidatedGeometryError> {
    let value = curve_taylor_with_policy(
        curve,
        Taylor::interval(interval_range(range)?),
        0,
        false,
        true,
        budget,
        cancel,
    )?;
    Ok(value.map(|v| v.0[0][0]))
}

/// Piecewise/one-sided native jets; not a C1 or Taylor qualification.
pub(crate) fn validated_curve_piecewise_jet(
    curve: &Curve,
    range: [f64; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<ValidatedCurveJet, ValidatedGeometryError> {
    let value = curve_taylor_with_policy(
        curve,
        Taylor::variable(interval_range(range)?, 0),
        0,
        false,
        true,
        budget,
        cancel,
    )?;
    let [position, first, second, third] = curve_jet_from_taylor(value);
    Ok(ValidatedCurveJet {
        position,
        first,
        second,
        third,
    })
}

/// Encloses an implicit guide's analytic continuation without reanchoring it.
pub(crate) fn validated_projection_leader_jet(
    curve: &Curve,
    range: [f64; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<ValidatedCurveJet, ValidatedGeometryError> {
    let value = curve_taylor_with_extension(
        curve,
        Taylor::variable(interval_range(range)?, 0),
        0,
        true,
        budget,
        cancel,
    )?;
    let [position, first, second, third] = curve_jet_from_taylor(value);
    Ok(ValidatedCurveJet {
        position,
        first,
        second,
        third,
    })
}

/// Borrows one immutable original leader's admitted knot structure.
///
/// Coordinate-plane proposals query many knots of the same original spline.
/// Admit its complete requested range once, then retain the identical closed
/// incident-span basis/evidence arithmetic at each point. This receipt carries
/// no root, containment, smoothness-across-kinks or geometry-replacement authority.
pub(crate) struct ProjectionLeaderKnotEvaluator<'a> {
    curve: &'a Curve,
    range: ValidatedInterval,
    degree: Option<usize>,
}

impl<'a> ProjectionLeaderKnotEvaluator<'a> {
    pub(crate) fn new(
        curve: &'a Curve,
        range: [f64; 2],
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> Result<Self, ValidatedGeometryError> {
        budget.charge(1, cancel)?;
        let range = interval_range(range)?;
        let coefficients = match curve {
            Curve::BsplineCurve(curve) => Some((curve.knot_vector(), curve.control_points().len())),
            Curve::NurbsCurve(curve) => Some((curve.knot_vector(), curve.control_points().len())),
            _ => None,
        };
        let degree = coefficients
            .map(|(knots, count)| {
                validate_knots_with_policy(knots, count, range, false, true, budget, cancel)
            })
            .transpose()?;
        Ok(Self {
            curve,
            range,
            degree,
        })
    }

    pub(crate) fn jet(
        &self,
        parameter: f64,
        budget: &mut ValidatedGeometryBudget,
        cancel: &AtomicBool,
    ) -> Result<ValidatedCurveJet, ValidatedGeometryError> {
        budget.charge(1, cancel)?;
        if !parameter.is_finite() || parameter < self.range.lower || parameter > self.range.upper {
            return Err(ValidatedGeometryError::ParameterOutOfRange);
        }
        let Some(degree) = self.degree else {
            return validated_projection_leader_jet(self.curve, [parameter; 2], budget, cancel);
        };
        let parameter = Taylor::variable(ValidatedInterval::point(parameter), 0);
        let values = match self.curve {
            Curve::BsplineCurve(curve) => {
                curve_sum_with_validated_knots(
                    curve.knot_vector(),
                    (curve.control_points().len(), degree),
                    parameter,
                    |i| {
                        let point = curve.control_points()[i];
                        [point.x, point.y, point.z]
                    },
                    None,
                    (budget, cancel),
                )?
                .values
            }
            Curve::NurbsCurve(curve) => {
                let homogeneous = curve_sum_with_validated_knots(
                    curve.knot_vector(),
                    (curve.control_points().len(), degree),
                    parameter,
                    |i| {
                        let point = curve.control_points()[i];
                        [point.x, point.y, point.z, point.w]
                    },
                    Some(3),
                    (budget, cancel),
                )?;
                let values = normalized(homogeneous)?;
                [values[0], values[1], values[2]]
            }
            _ => return Err(ValidatedGeometryError::InvalidInput),
        };
        values
            .into_iter()
            .try_for_each(|value| value.finite().map(|_| ()))?;
        let [position, first, second, third] = curve_jet_from_taylor(values);
        Ok(ValidatedCurveJet {
            position,
            first,
            second,
            third,
        })
    }
}

/// Encloses a native parameter-space curve, including posed analytic conics.
pub fn validated_curve2d_jet(
    curve: &Curve2D,
    range: [f64; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<ValidatedCurveJet2D, ValidatedGeometryError> {
    curve2d_jet_with_policy(curve, range, false, budget, cancel)
}

/// Encloses a continuous piecewise-smooth original UV curve and all one-sided
/// derivatives, including both incident closed spans at an exact native knot.
/// Discontinuous knots, nonfinite coefficients and denominator poles refuse.
/// This supplies neither C1 continuity nor second-derivative continuity:
/// consumers of a normal-coordinate ribbon must independently qualify C1.
pub fn validated_curve2d_piecewise_jet(
    curve: &Curve2D,
    range: [f64; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<ValidatedCurveJet2D, ValidatedGeometryError> {
    curve2d_jet_with_policy(curve, range, true, budget, cancel)
}

fn curve2d_jet_with_policy(
    curve: &Curve2D,
    range: [f64; 2],
    allow_c0: bool,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<ValidatedCurveJet2D, ValidatedGeometryError> {
    let value = curve2d_taylor_with_policy(
        curve,
        Taylor::variable(interval_range(range)?, 0),
        0,
        false,
        allow_c0,
        budget,
        cancel,
    )?;
    let [position, first, second, third] = curve_jet_from_taylor(value);
    Ok(ValidatedCurveJet2D {
        position,
        first,
        second,
        third,
    })
}

/// Encloses an exact imported boundary without converting or sampling it.
pub fn validated_boundary_jet(
    curve: &BoundaryCurve2D,
    range: [f64; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<ValidatedCurveJet2D, ValidatedGeometryError> {
    let t = Taylor::variable(interval_range(range)?, 0);
    let value = match curve {
        BoundaryCurve2D::Line(line) => line_taylor([line.0.x, line.0.y], [line.1.x, line.1.y], t)?,
        BoundaryCurve2D::BsplineCurve(curve) => spline_curve(
            curve.knot_vector(),
            curve.control_points().len(),
            t,
            |i| [curve.control_points()[i].x, curve.control_points()[i].y],
            false,
            budget,
            cancel,
        )?,
        BoundaryCurve2D::NurbsCurve(curve) => rational_curve(
            curve.knot_vector(),
            curve.control_points().len(),
            t,
            |i| {
                [
                    curve.control_points()[i].x,
                    curve.control_points()[i].y,
                    curve.control_points()[i].z,
                ]
            },
            false,
            budget,
            cancel,
        )?,
    };
    value
        .into_iter()
        .try_for_each(|jet| jet.finite().map(|_| ()))?;
    let [position, first, second, third] = curve_jet_from_taylor(value);
    Ok(ValidatedCurveJet2D {
        position,
        first,
        second,
        third,
    })
}

/// Encloses a native source surface in its actual, possibly inverted chart.
///
/// Rational denominators are evaluated on each knot-span rectangle, so mixed
/// homogeneous weight signs are admitted when their actual denominator is
/// separated from zero. T-spline arithmetic is not admitted. Processor matrices
/// must be finite. Homogeneous quotients include their parameter derivatives;
/// a denominator not separated from zero is a typed refusal.
pub fn validated_surface_jet(
    surface: &Surface,
    range: [[f64; 2]; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<ValidatedSurfaceJet, ValidatedGeometryError> {
    surface_jet_with_policy(surface, range, false, budget, cancel)
}

/// Encloses continuous piecewise-smooth sources and their one-sided jets.
/// Position and first-derivative ranges support continuous-source image and
/// first-order mean-value bounds. Hessian unions are not Taylor bounds or a
/// gradient Jacobian without a separately qualified C1 source box. Successful
/// evaluation grants neither regularity nor a unique normal at a crease.
pub fn validated_surface_piecewise_jet(
    surface: &Surface,
    range: [[f64; 2]; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<ValidatedSurfaceJet, ValidatedGeometryError> {
    surface_jet_with_policy(surface, range, true, budget, cancel)
}

fn surface_jet_with_policy(
    surface: &Surface,
    range: [[f64; 2]; 2],
    allow_c0: bool,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<ValidatedSurfaceJet, ValidatedGeometryError> {
    let value = surface_taylor_with_policy(
        surface,
        Taylor::variable(interval_range(range[0])?, 0),
        Taylor::variable(interval_range(range[1])?, 1),
        0,
        allow_c0,
        budget,
        cancel,
    )?;
    let two = ValidatedInterval::point(2.0);
    Ok(ValidatedSurfaceJet {
        position: value.map(|v| v.0[0][0]),
        du: value.map(|v| v.0[1][0]),
        dv: value.map(|v| v.0[0][1]),
        duu: value.map(|v| v.0[2][0] * two),
        duv: value.map(|v| v.0[1][1]),
        dvv: value.map(|v| v.0[0][2] * two),
    })
}

pub(crate) fn curve_taylor(
    curve: &Curve,
    t: Taylor,
    nesting: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<[Taylor; 3], ValidatedGeometryError> {
    curve_taylor_with_extension(curve, t, nesting, false, budget, cancel)
}

pub(super) fn curve_taylor_with_extension(
    curve: &Curve,
    t: Taylor,
    nesting: usize,
    conic_extension: bool,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<[Taylor; 3], ValidatedGeometryError> {
    curve_taylor_with_policy(curve, t, nesting, conic_extension, false, budget, cancel)
}

fn curve_taylor_with_policy(
    curve: &Curve,
    t: Taylor,
    nesting: usize,
    conic_extension: bool,
    allow_c0: bool,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<[Taylor; 3], ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    if nesting > MAX_NATIVE_GEOMETRY_NESTING {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    let value = match curve {
        Curve::Line(line) => line_taylor(
            [line.0.x, line.0.y, line.0.z],
            [line.1.x, line.1.y, line.1.z],
            t,
        )?,
        Curve::BsplineCurve(curve) => spline_curve(
            curve.knot_vector(),
            curve.control_points().len(),
            t,
            |i| {
                [
                    curve.control_points()[i].x,
                    curve.control_points()[i].y,
                    curve.control_points()[i].z,
                ]
            },
            allow_c0,
            budget,
            cancel,
        )?,
        Curve::NurbsCurve(curve) => {
            let homogeneous = rational_spline(
                curve.knot_vector(),
                curve.control_points().len(),
                t,
                |i| {
                    let p = curve.control_points()[i];
                    [p.x, p.y, p.z, p.w]
                },
                allow_c0,
                budget,
                cancel,
            )?;
            std::array::from_fn(|i| homogeneous[i])
        }
        Curve::ParameterCurve(curve) => {
            let uv = curve2d_taylor_with_policy(
                curve.curve(),
                t,
                nesting + 1,
                conic_extension,
                allow_c0,
                budget,
                cancel,
            )?;
            surface_taylor_with_policy(
                curve.surface(),
                uv[0],
                uv[1],
                nesting + 1,
                allow_c0,
                budget,
                cancel,
            )?
        }
        Curve::TrimmedCurve(curve) => {
            let parameter =
                processed_parameter(curve.orientation(), Some(curve.entity().range()), t)?;
            let point = curve_taylor_with_policy(
                curve.entity().curve(),
                parameter,
                nesting + 1,
                true,
                allow_c0,
                budget,
                cancel,
            )?;
            transform3(*curve.transform(), point)?
        }
        Curve::IntersectionCurve(_) => return Err(ValidatedGeometryError::Unsupported),
    };
    value
        .into_iter()
        .try_for_each(|jet| jet.finite().map(|_| ()))?;
    Ok(value)
}

pub(super) fn curve2d_taylor(
    curve: &Curve2D,
    t: Taylor,
    nesting: usize,
    conic_extension: bool,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<[Taylor; 2], ValidatedGeometryError> {
    curve2d_taylor_with_policy(curve, t, nesting, conic_extension, false, budget, cancel)
}

pub(super) fn curve2d_taylor_with_policy(
    curve: &Curve2D,
    t: Taylor,
    nesting: usize,
    conic_extension: bool,
    allow_c0: bool,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<[Taylor; 2], ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    if nesting > MAX_NATIVE_GEOMETRY_NESTING {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    let value = match curve {
        Curve2D::Line(line) => line_taylor([line.0.x, line.0.y], [line.1.x, line.1.y], t)?,
        Curve2D::BsplineCurve(curve) => spline_curve(
            curve.knot_vector(),
            curve.control_points().len(),
            t,
            |i| [curve.control_points()[i].x, curve.control_points()[i].y],
            allow_c0,
            budget,
            cancel,
        )?,
        Curve2D::NurbsCurve(curve) => rational_curve(
            curve.knot_vector(),
            curve.control_points().len(),
            t,
            |i| {
                [
                    curve.control_points()[i].x,
                    curve.control_points()[i].y,
                    curve.control_points()[i].z,
                ]
            },
            allow_c0,
            budget,
            cancel,
        )?,
        Curve2D::Polyline(curve) => {
            if curve.len() < 2 {
                return Err(ValidatedGeometryError::InvalidInput);
            }
            let range = t.0[0][0];
            if range.lower < 0.0 || range.upper > (curve.len() - 1) as f64 {
                return Err(ValidatedGeometryError::ParameterOutOfRange);
            }
            if allow_c0 {
                let mut result: Option<[Taylor; 2]> = None;
                for index in 0..curve.len() - 1 {
                    budget.charge(1, cancel)?;
                    let Some(part) = clipped_parameter(t, index as f64, (index + 1) as f64) else {
                        continue;
                    };
                    let (p, q) = (curve[index], curve[index + 1]);
                    let value = line_taylor(
                        [p.x, p.y],
                        [q.x, q.y],
                        part - Taylor::constant(index as f64),
                    )?;
                    result = Some(match result {
                        None => value,
                        Some(previous) => std::array::from_fn(|i| previous[i].hull(value[i])),
                    });
                }
                result.ok_or(ValidatedGeometryError::ParameterOutOfRange)?
            } else {
                if range.lower.floor() < range.upper.ceil() - 1.0 {
                    return Err(ValidatedGeometryError::NonSmooth);
                }
                let index = (range.midpoint().floor() as usize).min(curve.len() - 2);
                let p = curve[index];
                let q = curve[index + 1];
                line_taylor([p.x, p.y], [q.x, q.y], t - Taylor::constant(index as f64))?
            }
        }
        Curve2D::Conic(conic) => {
            let (parameter, transform, kind) = match conic {
                Conic2D::Ellipse(curve) => (
                    processed_parameter_with_extension(
                        curve.orientation(),
                        curve.try_range_tuple(),
                        t,
                        conic_extension,
                    )?,
                    *curve.transform(),
                    0,
                ),
                Conic2D::Hyperbola(curve) => (
                    processed_parameter_with_extension(
                        curve.orientation(),
                        curve.try_range_tuple(),
                        t,
                        conic_extension,
                    )?,
                    *curve.transform(),
                    1,
                ),
                Conic2D::Parabola(curve) => (
                    processed_parameter_with_extension(
                        curve.orientation(),
                        curve.try_range_tuple(),
                        t,
                        conic_extension,
                    )?,
                    *curve.transform(),
                    2,
                ),
            };
            let point = match kind {
                0 => {
                    let (sine, cosine) = parameter.sin_cos();
                    [cosine, sine]
                }
                1 => {
                    let positive = parameter.exp()?;
                    let negative = (-parameter).exp()?;
                    [
                        (positive + negative) * Taylor::constant(0.5),
                        (positive - negative) * Taylor::constant(0.5),
                    ]
                }
                _ => [parameter * parameter, parameter * Taylor::constant(2.0)],
            };
            transform2(transform, point)?
        }
        Curve2D::TrimmedCurve(curve) => {
            let parameter =
                processed_parameter(curve.orientation(), Some(curve.entity().range()), t)?;
            let point = curve2d_taylor_with_policy(
                curve.entity().curve(),
                parameter,
                nesting + 1,
                true,
                allow_c0,
                budget,
                cancel,
            )?;
            transform2(*curve.transform(), point)?
        }
    };
    value
        .into_iter()
        .try_for_each(|jet| jet.finite().map(|_| ()))?;
    Ok(value)
}

pub(crate) fn processed_parameter(
    orientation: bool,
    range: Option<(f64, f64)>,
    t: Taylor,
) -> Result<Taylor, ValidatedGeometryError> {
    processed_parameter_with_extension(orientation, range, t, false)
}

pub(super) fn processed_parameter_with_extension(
    orientation: bool,
    range: Option<(f64, f64)>,
    t: Taylor,
    extension: bool,
) -> Result<Taylor, ValidatedGeometryError> {
    let (lower, upper) = range.ok_or(ValidatedGeometryError::InvalidInput)?;
    let range = interval_range([lower, upper])?;
    let contained = t.0[0][0].lower >= lower && t.0[0][0].upper <= upper;
    if !extension && !contained {
        return Err(ValidatedGeometryError::ParameterOutOfRange);
    }
    if orientation {
        return Ok(t);
    }
    let anchor = ValidatedInterval::point(lower) + ValidatedInterval::point(upper);
    let mut reflected = Taylor::interval(anchor) - t;
    // Reflection is decreasing: evaluate each endpoint with the existing
    // outward point arithmetic, whose TwoSum proof retains exact differences.
    // Ranged subtraction would widen even exactly representable endpoint
    // images across a coordinate joint. Inexact anchors and images still
    // retain their full uncertainty; no knot or range endpoint is snapped.
    reflected.0[0][0] = ValidatedInterval {
        lower: (anchor - ValidatedInterval::point(t.0[0][0].upper)).lower,
        upper: (anchor - ValidatedInterval::point(t.0[0][0].lower)).upper,
    };
    if contained {
        // Reflection maps its complete declared domain back into that domain.
        reflected.0[0][0] = reflected.0[0][0]
            .intersection(range)
            .ok_or(ValidatedGeometryError::ConditioningFailure)?;
    }
    Ok(reflected)
}

pub(super) fn surface_taylor_with_policy(
    surface: &Surface,
    u: Taylor,
    v: Taylor,
    nesting: usize,
    allow_c0: bool,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<[Taylor; 3], ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    if nesting > MAX_NATIVE_GEOMETRY_NESTING {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    let value = match surface {
        Surface::Plane(plane) => plane_taylor(plane, u, v),
        Surface::ExtrusionSurface(surface) => {
            let point = curve_taylor_with_policy(
                surface.entity_curve(),
                u,
                nesting + 1,
                false,
                allow_c0,
                budget,
                cancel,
            )?;
            let direction = surface.extruding_vector();
            if !(0..3).all(|i| direction[i].is_finite()) {
                return Err(ValidatedGeometryError::InvalidInput);
            }
            std::array::from_fn(|i| point[i] + v * Taylor::constant(direction[i]))
        }
        Surface::BsplineSurface(surface) => spline_surface(
            (surface.knot_vectors(), surface.control_points()),
            [u, v],
            |p| [p.x, p.y, p.z],
            allow_c0,
            (budget, cancel),
        )?,
        Surface::NurbsSurface(surface) => {
            let controls = surface.control_points();
            let control = |p: &crate::Vector4| [p.x, p.y, p.z, p.w];
            if let Some(value) = rational_affine_factor(
                (surface.knot_vectors(), controls),
                [u, v],
                1,
                &control,
                allow_c0,
                (budget, cancel),
            )? {
                value
            } else if let Some(value) = rational_affine_factor(
                (surface.knot_vectors(), controls),
                [u, v],
                0,
                &control,
                allow_c0,
                (budget, cancel),
            )? {
                value
            } else {
                let h = rational_surface(
                    (surface.knot_vectors(), controls),
                    [u, v],
                    control,
                    allow_c0,
                    (budget, cancel),
                )?;
                let mut value: [Taylor; 3] = std::array::from_fn(|i| h[i]);
                for (coordinate, component) in value.iter_mut().enumerate() {
                    if let Some(reduced) = rational_rank_one_coordinate(
                        (surface.knot_vectors(), controls),
                        [u, v],
                        coordinate,
                        0,
                        &control,
                        allow_c0,
                        (budget, cancel),
                    )? {
                        *component = reduced;
                    } else if let Some(reduced) = rational_rank_one_coordinate(
                        (surface.knot_vectors(), controls),
                        [u, v],
                        coordinate,
                        1,
                        &control,
                        allow_c0,
                        (budget, cancel),
                    )? {
                        *component = reduced;
                    }
                }
                value
            }
        }
        Surface::SphericalSurface(processor) => {
            let (u, v) = if processor.orientation() {
                (u, v)
            } else {
                (v, u)
            };
            let sphere = processor.entity();
            let (su, cu) = u.sin_cos();
            let (sv, cv) = v.sin_cos();
            let radius = Taylor::constant(sphere.radius());
            let center = sphere.center();
            let point = [su * cv, su * sv, cu];
            transform3(
                *processor.transform(),
                std::array::from_fn(|i| Taylor::constant(center[i]) + radius * point[i]),
            )?
        }
        Surface::ToroidalSurface(processor) => {
            let (u, v) = if processor.orientation() {
                (u, v)
            } else {
                (v, u)
            };
            let torus = processor.entity();
            let (su, cu) = u.sin_cos();
            let (sv, cv) = v.sin_cos();
            let minor = Taylor::constant(torus.small_radius());
            let radial = Taylor::constant(torus.large_radius()) + minor * cv;
            let center = torus.center();
            transform3(
                *processor.transform(),
                [
                    Taylor::constant(center.x) + radial * cu,
                    Taylor::constant(center.y) + radial * su,
                    Taylor::constant(center.z) + minor * sv,
                ],
            )?
        }
        Surface::RevolutionSurface(processor) => {
            let (u, v) = if processor.orientation() {
                (u, v)
            } else {
                (v, u)
            };
            let revolution = processor.entity();
            let origin = revolution.origin();
            let axis = revolution.axis();
            if !(0..3).all(|i| axis[i].is_finite() && origin[i].is_finite()) {
                return Err(ValidatedGeometryError::InvalidInput);
            }
            let point = curve_taylor_with_policy(
                revolution.entity_curve(),
                u,
                nesting + 1,
                false,
                allow_c0,
                budget,
                cancel,
            )?;
            let delta: [Taylor; 3] =
                std::array::from_fn(|i| point[i] - Taylor::constant(origin[i]));
            let (sine, cosine) = v.sin_cos();
            let cross = [
                delta[2] * Taylor::constant(axis.y) - delta[1] * Taylor::constant(axis.z),
                delta[0] * Taylor::constant(axis.z) - delta[2] * Taylor::constant(axis.x),
                delta[1] * Taylor::constant(axis.x) - delta[0] * Taylor::constant(axis.y),
            ];
            // Form the axial and radial coefficients before applying them to
            // the original profile. This is the same Rodrigues operation,
            // without a spurious full-period oscillation from correlated axial
            // terms or subtraction of two whole profile-position intervals.
            let point = std::array::from_fn(|i| {
                let rotated = (0..3)
                    .map(|j| {
                        let axial =
                            ValidatedInterval::point(axis[i]) * ValidatedInterval::point(axis[j]);
                        let radial =
                            ValidatedInterval::point(if i == j { 1.0 } else { 0.0 }) - axial;
                        delta[j] * (Taylor::interval(axial) + Taylor::interval(radial) * cosine)
                    })
                    .fold(Taylor::ZERO, |a, b| a + b);
                Taylor::constant(origin[i]) + rotated + cross[i] * sine
            });
            transform3(*processor.transform(), point)?
        }
        Surface::TsplineSurface(_) => return Err(ValidatedGeometryError::Unsupported),
    };
    value
        .into_iter()
        .try_for_each(|jet| jet.finite().map(|_| ()))?;
    Ok(value)
}

fn plane_taylor(plane: &Plane, u: Taylor, v: Taylor) -> [Taylor; 3] {
    let [origin, one, another] = plane.defining_points();
    std::array::from_fn(|i| {
        let origin = Taylor::constant(origin[i]);
        let (a, b) = match plane.stored_axes() {
            Some([a, b]) => (Taylor::constant(a[i]), Taylor::constant(b[i])),
            None => (
                Taylor::constant(one[i]) - origin,
                Taylor::constant(another[i]) - origin,
            ),
        };
        origin + u * a + v * b
    })
}

fn line_taylor<const N: usize>(
    a: [f64; N],
    b: [f64; N],
    t: Taylor,
) -> Result<[Taylor; N], ValidatedGeometryError> {
    if !a.into_iter().chain(b).all(f64::is_finite) {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    let mut result: [Taylor; N] = std::array::from_fn(|i| {
        Taylor::constant(a[i]) + t * (Taylor::constant(b[i]) - Taylor::constant(a[i]))
    });
    if t.0[0][0].lower >= 0.0 && t.0[0][0].upper <= 1.0 {
        for i in 0..N {
            let hull = ValidatedInterval {
                lower: a[i].min(b[i]),
                upper: a[i].max(b[i]),
            };
            result[i].0[0][0] = result[i].0[0][0]
                .intersection(hull)
                .ok_or(ValidatedGeometryError::ConditioningFailure)?;
        }
    }
    Ok(result)
}

fn transform2(matrix: Matrix3, point: [Taylor; 2]) -> Result<[Taylor; 2], ValidatedGeometryError> {
    if !(0..3).all(|i| (0..3).all(|j| matrix[i][j].is_finite())) {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    let coordinates: [Taylor; 2] = std::array::from_fn(|i| {
        Taylor::constant(matrix[2][i])
            + Taylor::constant(matrix[0][i]) * point[0]
            + Taylor::constant(matrix[1][i]) * point[1]
    });
    // cgmath's native Matrix3/Point2 action truncates, without a homogeneous
    // divide. Rational control-net weights are handled at their original leaf.
    Ok(coordinates)
}
fn transform3(matrix: Matrix4, point: [Taylor; 3]) -> Result<[Taylor; 3], ValidatedGeometryError> {
    if !(0..4).all(|i| (0..4).all(|j| matrix[i][j].is_finite())) {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    let coordinates: [Taylor; 3] = std::array::from_fn(|i| {
        (0..3)
            .map(|j| Taylor::constant(matrix[j][i]) * point[j])
            .fold(Taylor::constant(matrix[3][i]), |a, b| a + b)
    });
    if (0..3).all(|i| matrix[i][3] == 0.0) && matrix[3][3] == 1.0 {
        return Ok(coordinates);
    }
    let denominator = (0..3)
        .map(|j| Taylor::constant(matrix[j][3]) * point[j])
        .fold(Taylor::constant(matrix[3][3]), |a, b| a + b);
    let inverse = denominator.reciprocal()?;
    Ok(coordinates.map(|value| value * inverse))
}

pub(super) fn validate_knots(
    knots: &KnotVector,
    count: usize,
    parameter: ValidatedInterval,
    extrapolate: bool,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<usize, ValidatedGeometryError> {
    validate_knots_with_policy(knots, count, parameter, extrapolate, false, budget, cancel)
}

pub(super) fn validate_knots_with_policy(
    knots: &KnotVector,
    count: usize,
    parameter: ValidatedInterval,
    extrapolate: bool,
    allow_c0: bool,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<usize, ValidatedGeometryError> {
    if count == 0 || knots.len() <= count {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    let degree = knots.len() - count - 1;
    if degree > MAX_DEGREE {
        return Err(ValidatedGeometryError::Unsupported);
    }
    for (index, knot) in knots.iter().copied().enumerate() {
        budget.charge(1, cancel)?;
        if !knot.is_finite() || index > 0 && knot < knots[index - 1] {
            return Err(ValidatedGeometryError::InvalidInput);
        }
    }
    if knots[0] >= knots[knots.len() - 1] {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    if !extrapolate && (parameter.lower < knots[0] || parameter.upper > knots[knots.len() - 1]) {
        return Err(ValidatedGeometryError::ParameterOutOfRange);
    }
    let mut index = 0;
    while index < knots.len() {
        let knot = knots[index];
        let start = index;
        while index < knots.len() && knots[index] == knot {
            budget.charge(1, cancel)?;
            index += 1;
        }
        let multiplicity = index - start;
        let touches_jump =
            parameter.lower <= knot && knot <= parameter.upper && multiplicity > degree;
        let crosses_kink = !allow_c0
            && parameter.lower < knot
            && knot < parameter.upper
            && multiplicity > degree.saturating_sub(1);
        if knots[0] < knot && knot < knots[knots.len() - 1] && (touches_jump || crosses_kink) {
            return Err(ValidatedGeometryError::NonSmooth);
        }
    }
    Ok(degree)
}

pub(super) fn basis(
    knots: &KnotVector,
    count: usize,
    degree: usize,
    span: usize,
    parameter: Taylor,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<(usize, usize, [Taylor; MAX_DEGREE + 2]), ValidatedGeometryError> {
    let start = span.saturating_sub(degree);
    let mut values = [Taylor::ZERO; MAX_DEGREE + 2];
    values[span - start] = Taylor::constant(1.0);
    for order in 1..=degree {
        budget.charge(order + 1, cancel)?;
        let end = span.min(knots.len() - order - 2);
        for index in start..=end {
            let mut a = if knots[index + order] == knots[index] {
                Taylor::ZERO
            } else {
                (parameter - Taylor::constant(knots[index]))
                    * Taylor::interval(
                        (ValidatedInterval::point(knots[index + order])
                            - ValidatedInterval::point(knots[index]))
                        .reciprocal()?,
                    )
            };
            let mut b = if knots[index + order + 1] == knots[index + 1] {
                Taylor::ZERO
            } else {
                (Taylor::constant(knots[index + order + 1]) - parameter)
                    * Taylor::interval(
                        (ValidatedInterval::point(knots[index + order + 1])
                            - ValidatedInterval::point(knots[index + 1]))
                        .reciprocal()?,
                    )
            };
            // Preserve the exact affine endpoint identities before recurrence:
            // (b-a)/(b-a) = 1 and (a-a)/(b-a) = 0. Independent outward
            // reciprocal products need not have an exact endpoint midpoint;
            // knot-plane proposals must not acquire that arithmetic drift.
            // Only the value is simplified; Taylor derivatives keep their
            // original outward coefficient arithmetic.
            let point = parameter.0[0][0];
            if point.lower == point.upper {
                if knots[index + order] != knots[index] {
                    if point.lower == knots[index] {
                        a.0[0][0] = super::ZERO;
                    } else if point.lower == knots[index + order] {
                        a.0[0][0] = ONE;
                    }
                }
                if knots[index + order + 1] != knots[index + 1] {
                    if point.lower == knots[index + order + 1] {
                        b.0[0][0] = super::ZERO;
                    } else if point.lower == knots[index + 1] {
                        b.0[0][0] = ONE;
                    }
                }
            }
            let offset = index - start;
            values[offset] = a * values[offset] + b * values[offset + 1];
        }
        if end < span {
            values[end + 1 - start] = Taylor::ZERO;
        }
    }
    let length = count.saturating_sub(start).min(degree + 1);
    Ok((start, length, values))
}

pub(super) fn clipped_parameter(parameter: Taylor, lower: f64, upper: f64) -> Option<Taylor> {
    let mut result = parameter;
    result.0[0][0] = result.0[0][0].intersection(ValidatedInterval { lower, upper })?;
    Some(result)
}

fn spline_curve<const N: usize>(
    knots: &KnotVector,
    count: usize,
    parameter: Taylor,
    control: impl Fn(usize) -> [f64; N],
    allow_c0: bool,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<[Taylor; N], ValidatedGeometryError> {
    curve_sum(
        knots,
        count,
        parameter,
        control,
        None,
        allow_c0,
        (budget, cancel),
    )
    .map(|value| value.values)
}

fn curve_sum<const N: usize>(
    knots: &KnotVector,
    count: usize,
    parameter: Taylor,
    control: impl Fn(usize) -> [f64; N],
    weight_axis: Option<usize>,
    allow_c0: bool,
    (budget, cancel): (&mut ValidatedGeometryBudget, &AtomicBool),
) -> Result<ControlSum<N>, ValidatedGeometryError> {
    let degree = validate_knots_with_policy(
        knots,
        count,
        parameter.0[0][0],
        false,
        allow_c0,
        budget,
        cancel,
    )?;
    curve_sum_with_validated_knots(
        knots,
        (count, degree),
        parameter,
        control,
        weight_axis,
        (budget, cancel),
    )
}

// The caller owns complete immutable knot admission for this same range.
// Binary searches retain both incident closed spans at an exact knot.
fn curve_sum_with_validated_knots<const N: usize>(
    knots: &KnotVector,
    (count, degree): (usize, usize),
    parameter: Taylor,
    control: impl Fn(usize) -> [f64; N],
    weight_axis: Option<usize>,
    (budget, cancel): (&mut ValidatedGeometryBudget, &AtomicBool),
) -> Result<ControlSum<N>, ValidatedGeometryError> {
    let bound = parameter.0[0][0];
    let mut boundary = |strict: bool| -> Result<usize, ValidatedGeometryError> {
        let (mut low, mut high) = (0, knots.len());
        while low < high {
            budget.charge(1, cancel)?;
            let middle = low + (high - low) / 2;
            if if strict {
                knots[middle] <= bound.upper
            } else {
                knots[middle] < bound.lower
            } {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        Ok(low)
    };
    let first = if bound.lower == knots[0] {
        0
    } else {
        boundary(false)?.saturating_sub(1)
    };
    let end = if bound.upper == knots[knots.len() - 1] {
        knots.len() - 1
    } else {
        boundary(true)?.min(knots.len() - 1)
    };
    let mut result: Option<ControlSum<N>> = None;
    for span in first..end {
        budget.charge(1, cancel)?;
        if knots[span] == knots[span + 1] {
            continue;
        }
        let Some(parameter) = clipped_parameter(parameter, knots[span], knots[span + 1]) else {
            continue;
        };
        let (start, length, values) = basis(knots, count, degree, span, parameter, budget, cancel)?;
        let partition_unity = span >= degree && span < count;
        let mut value = [Taylor::ZERO; N];
        let mut evidence = ControlEvidence::new(weight_axis);
        for (index, basis_value) in values.iter().take(length).enumerate() {
            budget.charge(1, cancel)?;
            let p = control(start + index);
            evidence.include(p, *basis_value, partition_unity, &mut value)?;
        }
        let value = evidence.finish(value, partition_unity, true)?;
        result = Some(match result {
            None => value,
            Some(previous) => previous.hull(value),
        });
    }
    result.ok_or(ValidatedGeometryError::ParameterOutOfRange)
}

fn rational_curve(
    knots: &KnotVector,
    count: usize,
    parameter: Taylor,
    control: impl Fn(usize) -> [f64; 3],
    allow_c0: bool,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<[Taylor; 2], ValidatedGeometryError> {
    let h = rational_spline(knots, count, parameter, control, allow_c0, budget, cancel)?;
    Ok([h[0], h[1]])
}

fn spline_surface<P, const N: usize>(
    (knots, controls): (&(KnotVector, KnotVector), &[Vec<P>]),
    [u, v]: [Taylor; 2],
    control: impl Fn(&P) -> [f64; N],
    allow_c0: bool,
    (budget, cancel): (&mut ValidatedGeometryBudget, &AtomicBool),
) -> Result<[Taylor; N], ValidatedGeometryError> {
    surface_sum(
        (knots, controls),
        [u, v],
        control,
        None,
        allow_c0,
        (budget, cancel),
    )
    .map(|value| value.values)
}

/// Original-coefficient tensor identity, admitted only for two complete bases.
///
/// In i-major order the first row is retained once; all other original controls
/// are visited once. Differences precede any scalar functional, so an additive
/// source does not acquire artificial cross-axis Taylor dependence.
pub(super) struct TensorCoefficients<const N: usize> {
    partition: bool,
    origin: [f64; N],
    row: [f64; N],
    columns: [[ValidatedInterval; N]; MAX_DEGREE + 1],
}

// Actual maximum homogeneous layout, including the first-row interval buffer.
pub(crate) const TENSOR_COEFFICIENT_WORKSPACE_BYTES: usize = size_of::<TensorCoefficients<4>>();

// Shared named tensor state for full and K=1/2 projected evaluation: two basis
// arrays, coefficient identity, original evidence, both independent output
// arrays, span hull and the existing rational subdivision stack. This is not an
// RSS/compiler-frame bound. Recursive native geometry still obeys
// MAX_NATIVE_GEOMETRY_NESTING.
pub(crate) const TENSOR_EVALUATOR_FIXED_BYTES: usize = 2 * size_of::<[Taylor; MAX_DEGREE + 2]>()
    + TENSOR_COEFFICIENT_WORKSPACE_BYTES
    + size_of::<ControlEvidence<4>>()
    + 2 * size_of::<[Taylor; 4]>()
    + size_of::<Option<ControlSum<4>>>()
    + size_of::<[([Taylor; 2], usize); 49]>();

impl<const N: usize> TensorCoefficients<N> {
    pub(super) fn new(partition: bool) -> Self {
        Self {
            partition,
            origin: [0.0; N],
            row: [0.0; N],
            columns: [[super::ZERO; N]; MAX_DEGREE + 1],
        }
    }

    pub(super) fn origin_difference(&self, point: [f64; N]) -> [ValidatedInterval; N] {
        std::array::from_fn(|axis| {
            ValidatedInterval::point(point[axis]) - ValidatedInterval::point(self.origin[axis])
        })
    }

    // Original hull/evidence collection has already validated this coefficient.
    pub(super) fn term(
        &mut self,
        point: [f64; N],
        [i, j]: [usize; 2],
        [a, b]: [Taylor; 2],
        product: Taylor,
    ) -> ([ValidatedInterval; N], Taylor) {
        if !self.partition {
            return (point.map(ValidatedInterval::point), product);
        }
        if i == 0 && j == 0 {
            self.origin = point;
            self.row = point;
            return (point.map(ValidatedInterval::point), Taylor::constant(1.0));
        }
        let difference = |origin: [f64; N]| {
            std::array::from_fn(|axis| {
                ValidatedInterval::point(point[axis]) - ValidatedInterval::point(origin[axis])
            })
        };
        if i == 0 {
            self.columns[j] = difference(self.origin);
            return (self.columns[j], b);
        }
        if j == 0 {
            self.row = point;
            return (difference(self.origin), a);
        }
        let row_difference: [ValidatedInterval; N] = difference(self.row);
        let mixed = std::array::from_fn(|axis| row_difference[axis] - self.columns[j][axis]);
        let factor = if mixed.iter().any(|coefficient| *coefficient != super::ZERO) {
            product
        } else {
            Taylor::ZERO
        };
        (mixed, factor)
    }
}

pub(super) fn add_tensor_term<const K: usize>(
    values: &mut [Taylor; K],
    coefficients: [ValidatedInterval; K],
    basis: Taylor,
) {
    for (value, coefficient) in values.iter_mut().zip(coefficients) {
        if coefficient != super::ZERO {
            *value = *value + Taylor::interval(coefficient) * basis;
        }
    }
}

/// Intersect independent Taylor proofs on the identical complete source cell.
/// An inconsistent coefficient is a failure, never a choice of one proof.
pub(super) fn intersect_tensor_values<const K: usize>(
    values: &mut [Taylor; K],
    original: &[Taylor; K],
) -> Result<(), ValidatedGeometryError> {
    for (value, original) in values.iter_mut().zip(original) {
        for i in 0..4 {
            for j in 0..4 - i {
                value.0[i][j] = value.0[i][j]
                    .intersection(original.0[i][j])
                    .ok_or(ValidatedGeometryError::ConditioningFailure)?;
            }
        }
    }
    Ok(())
}

// Keep each tensor's source data, UV Taylor pair, and execution context together.
fn surface_sum<P, const N: usize>(
    (knots, controls): (&(KnotVector, KnotVector), &[Vec<P>]),
    [u, v]: [Taylor; 2],
    control: impl Fn(&P) -> [f64; N],
    weight_axis: Option<usize>,
    allow_c0: bool,
    (budget, cancel): (&mut ValidatedGeometryBudget, &AtomicBool),
) -> Result<ControlSum<N>, ValidatedGeometryError> {
    if controls.is_empty() || controls[0].is_empty() {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    let nu = controls.len();
    let nv = controls[0].len();
    for row in controls {
        budget.charge(1, cancel)?;
        if row.len() != nv {
            return Err(ValidatedGeometryError::InvalidInput);
        }
    }
    let pu = validate_knots_with_policy(&knots.0, nu, u.0[0][0], true, allow_c0, budget, cancel)?;
    let pv = validate_knots_with_policy(&knots.1, nv, v.0[0][0], true, allow_c0, budget, cancel)?;
    let mut result: Option<ControlSum<N>> = None;
    for su in 0..knots.0.len() - 1 {
        budget.charge(1, cancel)?;
        if knots.0[su] == knots.0[su + 1] {
            continue;
        }
        let lower = if knots.0[su] == knots.0[0] {
            u.0[0][0].lower.min(knots.0[su])
        } else {
            knots.0[su]
        };
        let upper = if knots.0[su + 1] == knots.0[knots.0.len() - 1] {
            u.0[0][0].upper.max(knots.0[su + 1])
        } else {
            knots.0[su + 1]
        };
        let Some(u) = clipped_parameter(u, lower, upper) else {
            continue;
        };
        let (iu, lu, bu) = basis(&knots.0, nu, pu, su, u, budget, cancel)?;
        for sv in 0..knots.1.len() - 1 {
            budget.charge(1, cancel)?;
            if knots.1[sv] == knots.1[sv + 1] {
                continue;
            }
            let lower = if knots.1[sv] == knots.1[0] {
                v.0[0][0].lower.min(knots.1[sv])
            } else {
                knots.1[sv]
            };
            let upper = if knots.1[sv + 1] == knots.1[knots.1.len() - 1] {
                v.0[0][0].upper.max(knots.1[sv + 1])
            } else {
                knots.1[sv + 1]
            };
            let Some(v) = clipped_parameter(v, lower, upper) else {
                continue;
            };
            let (iv, lv, bv) = basis(&knots.1, nv, pv, sv, v, budget, cancel)?;
            let partition_unity = su >= pu && su < nu && sv >= pv && sv < nv;
            let mut value = [Taylor::ZERO; N];
            let mut original = [Taylor::ZERO; N];
            let mut evidence = ControlEvidence::new(weight_axis);
            let mut tensor = TensorCoefficients::new(partition_unity);
            for i in 0..lu {
                for j in 0..lv {
                    budget.charge(1, cancel)?;
                    let p = control(&controls[iu + i][iv + j]);
                    evidence.observe(p)?;
                    let product = bu[i] * bv[j];
                    let (coefficient, basis) = tensor.term(p, [i, j], [bu[i], bv[j]], product);
                    if partition_unity {
                        budget.charge(1, cancel)?;
                        if i == 0 && j == 0 {
                            original = p.map(Taylor::constant);
                        } else {
                            let residual = if i == 0 || j == 0 {
                                coefficient
                            } else {
                                tensor.origin_difference(p)
                            };
                            add_tensor_term(&mut original, residual, product);
                        }
                    }
                    add_tensor_term(&mut value, coefficient, basis);
                }
            }
            if partition_unity {
                budget.charge(10 * N, cancel)?;
                intersect_tensor_values(&mut value, &original)?;
            }
            let nonnegative = knots.0[su] <= u.0[0][0].lower
                && u.0[0][0].upper <= knots.0[su + 1]
                && knots.1[sv] <= v.0[0][0].lower
                && v.0[0][0].upper <= knots.1[sv + 1];
            let value = evidence.finish(value, partition_unity, nonnegative)?;
            result = Some(match result {
                None => value,
                Some(previous) => previous.hull(value),
            });
        }
    }
    result.ok_or(ValidatedGeometryError::ParameterOutOfRange)
}

#[derive(Clone, Copy)]
struct Ratio {
    numerator: f64,
    denominator: f64,
}
impl Ratio {
    fn equal(self, other: Self) -> bool {
        exact_product(self.numerator, other.denominator)
            == exact_product(other.numerator, self.denominator)
    }
    fn enclosure(self) -> Result<ValidatedInterval, ValidatedGeometryError> {
        let candidate = self.numerator / self.denominator;
        if candidate.is_finite()
            && exact_product(candidate, self.denominator) == exact_product(self.numerator, 1.0)
        {
            Ok(ValidatedInterval::point(candidate))
        } else {
            Ok(ValidatedInterval::point(self.numerator)
                * ValidatedInterval::point(self.denominator).reciprocal()?)
        }
    }
}

fn exact_product(a: f64, b: f64) -> (bool, u128, i32) {
    fn parts(value: f64) -> (bool, u64, i32) {
        let bits = value.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i32;
        let significand = bits & ((1_u64 << 52) - 1);
        if exponent == 0 {
            (bits >> 63 != 0, significand, -1074)
        } else {
            (
                bits >> 63 != 0,
                significand | (1_u64 << 52),
                exponent - 1075,
            )
        }
    }
    let (sa, a, ea) = parts(a);
    let (sb, b, eb) = parts(b);
    let product = u128::from(a) * u128::from(b);
    if product == 0 {
        (false, 0, 0)
    } else {
        let shift = product.trailing_zeros();
        (sa != sb, product >> shift, ea + eb + shift as i32)
    }
}

type ExactDyadic = (bool, u128, i32);

fn exact_difference(a: f64, b: f64) -> Option<ExactDyadic> {
    exact_dyadic_difference(exact_product(a, 1.0), exact_product(b, 1.0))
}

fn exact_dyadic_difference(a: ExactDyadic, mut b: ExactDyadic) -> Option<ExactDyadic> {
    if b.1 != 0 {
        b.0 = !b.0;
    }
    if a.1 == 0 {
        return Some(b);
    }
    if b.1 == 0 {
        return Some(a);
    }
    let exponent = a.2.min(b.2);
    let scale = |value: ExactDyadic| {
        let shift = u32::try_from(value.2 - exponent).ok()?;
        if shift >= 128 || value.1.leading_zeros() < shift {
            return None;
        }
        Some(value.1 << shift)
    };
    let (a_mantissa, b_mantissa) = (scale(a)?, scale(b)?);
    let (sign, magnitude) = if a.0 == b.0 {
        (a.0, a_mantissa.checked_add(b_mantissa)?)
    } else if a_mantissa >= b_mantissa {
        (a.0, a_mantissa - b_mantissa)
    } else {
        (b.0, b_mantissa - a_mantissa)
    };
    if magnitude == 0 {
        return Some((false, 0, 0));
    }
    let shift = magnitude.trailing_zeros();
    Some((sign, magnitude >> shift, exponent + shift as i32))
}

fn exact_dyadic_product(a: ExactDyadic, b: ExactDyadic) -> Option<ExactDyadic> {
    if a.1 == 0 || b.1 == 0 {
        return Some((false, 0, 0));
    }
    Some((a.0 != b.0, a.1.checked_mul(b.1)?, a.2 + b.2))
}

fn exact_dyadic_products_equal(
    a: ExactDyadic,
    b: ExactDyadic,
    c: ExactDyadic,
    d: ExactDyadic,
) -> bool {
    if a.1 == 0 || b.1 == 0 {
        return c.1 == 0 || d.1 == 0;
    }
    if c.1 == 0 || d.1 == 0 || (a.0 != b.0) != (c.0 != d.0) || a.2 + b.2 != c.2 + d.2 {
        return false;
    }
    // Cancel cross factors instead of allocating a wide product or overflowing
    // u128. Every input mantissa is normalized and odd.
    let mut left = [a.1, b.1];
    let mut right = [c.1, d.1];
    for x in &mut left {
        for y in &mut right {
            let (mut p, mut q) = (*x, *y);
            while q != 0 {
                (p, q) = (q, p % q);
            }
            *x /= p;
            *y /= p;
        }
    }
    left == [1, 1] && right == [1, 1]
}

/// Proves physical first-jet continuity at a multiplicity-degree knot.
/// Inputs are original homogeneous controls and original neighboring knots;
/// neither knot differences nor tangent numerator differences are rounded.
/// Checked dyadic difference overflow conservatively declines the proof.
pub(crate) fn homogeneous_join_is_c1<const N: usize>(
    previous: [f64; N],
    shared: [f64; N],
    next: [f64; N],
    left_knot: f64,
    knot: f64,
    right_knot: f64,
) -> bool {
    if N < 2
        || !previous
            .into_iter()
            .chain(shared)
            .chain(next)
            .all(f64::is_finite)
        || ![left_knot, knot, right_knot]
            .into_iter()
            .all(f64::is_finite)
        || !(left_knot < knot && knot < right_knot)
        || shared[N - 1] == 0.0
    {
        return false;
    }
    let (Some(left_span), Some(right_span)) = (
        exact_difference(knot, left_knot),
        exact_difference(right_knot, knot),
    ) else {
        return false;
    };
    for coordinate in 0..N - 1 {
        let Some(left) = exact_dyadic_difference(
            exact_product(shared[coordinate], previous[N - 1]),
            exact_product(previous[coordinate], shared[N - 1]),
        ) else {
            return false;
        };
        let Some(right) = exact_dyadic_difference(
            exact_product(next[coordinate], shared[N - 1]),
            exact_product(shared[coordinate], next[N - 1]),
        ) else {
            return false;
        };
        if !exact_dyadic_products_equal(left, right_span, right, left_span) {
            return false;
        }
    }
    true
}

// Prove a degree-one tensor factor by homogeneous coefficient identities.
// Checked arithmetic overflow declines this reduction, never approximates it.
fn rational_affine_factor<P>(
    (knots, controls): (&(KnotVector, KnotVector), &[Vec<P>]),
    [u, v]: [Taylor; 2],
    axis: usize,
    control: &impl Fn(&P) -> [f64; 4],
    allow_c0: bool,
    (budget, cancel): (&mut ValidatedGeometryBudget, &AtomicBool),
) -> Result<Option<[Taylor; 3]>, ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    if controls.is_empty() || controls[0].is_empty() {
        return Ok(None);
    }
    let columns = controls[0].len();
    let (base_knots, factor_knots, base_parameter, factor_parameter, base_count) = if axis == 1 {
        if columns != 2 {
            return Ok(None);
        }
        (&knots.0, &knots.1, u, v, controls.len())
    } else {
        if controls.len() != 2 {
            return Ok(None);
        }
        (&knots.1, &knots.0, v, u, columns)
    };
    if factor_knots.len() != 4 || base_knots.is_empty() {
        return Ok(None);
    }
    budget.charge(4, cancel)?;
    if !factor_knots.iter().all(|value| value.is_finite()) {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    let (start, end) = (factor_knots[0], factor_knots[3]);
    if factor_knots[1] != start || factor_knots[2] != end || start >= end {
        return Ok(None);
    }
    if base_parameter.0[0][0].lower < base_knots[0]
        || base_parameter.0[0][0].upper > base_knots[base_knots.len() - 1]
    {
        return Ok(None);
    }
    for row in controls {
        budget.charge(1, cancel)?;
        if row.len() != columns {
            return Err(ValidatedGeometryError::InvalidInput);
        }
    }
    let get = |index: usize, side: usize| {
        if axis == 1 {
            control(&controls[index][side])
        } else {
            control(&controls[side][index])
        }
    };
    let mut reference: Option<([f64; 4], [f64; 4])> = None;
    for index in 0..base_count {
        budget.charge(4, cancel)?;
        let (a, b) = (get(index, 0), get(index, 1));
        if !a.into_iter().chain(b).all(f64::is_finite) {
            return Err(ValidatedGeometryError::InvalidInput);
        }
        if a[3] != b[3] {
            return Ok(None);
        }
        if a[3] == 0.0 {
            if a != b {
                return Ok(None);
            }
            continue;
        }
        if let Some((first, second)) = reference {
            for coordinate in 0..3 {
                let Some(left) = exact_difference(b[coordinate], a[coordinate])
                    .and_then(|delta| exact_dyadic_product(delta, exact_product(first[3], 1.0)))
                else {
                    return Ok(None);
                };
                let Some(right) = exact_difference(second[coordinate], first[coordinate])
                    .and_then(|delta| exact_dyadic_product(delta, exact_product(a[3], 1.0)))
                else {
                    return Ok(None);
                };
                if left != right {
                    return Ok(None);
                }
            }
        } else {
            reference = Some((a, b));
        }
    }
    let Some((first, second)) = reference else {
        return Ok(None);
    };
    let span = ValidatedInterval::point(end) - ValidatedInterval::point(start);
    let inverse_span = match span.reciprocal() {
        Ok(value) => value,
        Err(_) => return Ok(None),
    };
    let inverse_weight = ValidatedInterval::point(first[3]).reciprocal()?;
    let displacement: [ValidatedInterval; 3] = std::array::from_fn(|coordinate| {
        (ValidatedInterval::point(second[coordinate]) - ValidatedInterval::point(first[coordinate]))
            * inverse_weight
    });
    let direction = displacement.map(|value| value * inverse_span);
    if !direction.into_iter().all(ValidatedInterval::is_finite) {
        return Ok(None);
    }
    let base = rational_spline(
        base_knots,
        base_count,
        base_parameter,
        |index| get(index, 0),
        allow_c0,
        budget,
        cancel,
    )?;
    let factor = factor_parameter - Taylor::constant(start);
    let mut value = [Taylor::ZERO; 3];
    for coordinate in 0..3 {
        let mut translation = factor * Taylor::interval(direction[coordinate]);
        if factor_parameter.0[0][0].lower >= start && factor_parameter.0[0][0].upper <= end {
            translation.0[0][0] = translation.0[0][0]
                .intersection(super::ZERO.hull(displacement[coordinate]))
                .ok_or(ValidatedGeometryError::ConditioningFailure)?;
        }
        value[coordinate] = base[coordinate] + translation;
    }
    Ok(Some(value))
}

// The caller has already admitted the full tensor denominator on the box.
// Cross-products prove common tensor factors, not approximate low rank.
fn rational_rank_one_coordinate<P>(
    (knots, controls): (&(KnotVector, KnotVector), &[Vec<P>]),
    [u, v]: [Taylor; 2],
    coordinate: usize,
    cancelled_axis: usize,
    control: &impl Fn(&P) -> [f64; 4],
    allow_c0: bool,
    (budget, cancel): (&mut ValidatedGeometryBudget, &AtomicBool),
) -> Result<Option<Taylor>, ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    let (nu, nv) = (controls.len(), controls[0].len());
    let (outer_count, inner_count, base_knots, parameter) = if cancelled_axis == 0 {
        (nu, nv, &knots.1, v)
    } else {
        (nv, nu, &knots.0, u)
    };
    if base_knots.is_empty()
        || parameter.0[0][0].lower < base_knots[0]
        || parameter.0[0][0].upper > base_knots[base_knots.len() - 1]
    {
        return Ok(None);
    }
    let get = |outer: usize, inner: usize| {
        if cancelled_axis == 0 {
            control(&controls[outer][inner])
        } else {
            control(&controls[inner][outer])
        }
    };
    let mut pivot = None;
    'find: for outer in 0..outer_count {
        for inner in 0..inner_count {
            budget.charge(1, cancel)?;
            let point = get(outer, inner);
            if !point.into_iter().all(f64::is_finite) {
                return Err(ValidatedGeometryError::InvalidInput);
            }
            if point[3] != 0.0 {
                pivot = Some((outer, inner, point[3]));
                break 'find;
            }
        }
    }
    let Some((reference, column, weight)) = pivot else {
        return Ok(None);
    };
    for outer in 0..outer_count {
        budget.charge(1, cancel)?;
        let factor = get(outer, column);
        if !factor.into_iter().all(f64::is_finite) {
            return Err(ValidatedGeometryError::InvalidInput);
        }
        for inner in 0..inner_count {
            budget.charge(4, cancel)?;
            let point = get(outer, inner);
            let source = get(reference, inner);
            if !point.into_iter().chain(source).all(f64::is_finite) {
                return Err(ValidatedGeometryError::InvalidInput);
            }
            if exact_product(point[coordinate], weight)
                != exact_product(source[coordinate], factor[3])
                || exact_product(point[3], weight) != exact_product(source[3], factor[3])
            {
                return Ok(None);
            }
        }
    }
    let value = rational_spline(
        base_knots,
        inner_count,
        parameter,
        |index| {
            let point = get(reference, index);
            [point[coordinate], point[3]]
        },
        allow_c0,
        budget,
        cancel,
    )?;
    Ok(Some(value[0]))
}

#[derive(Clone, Copy)]
struct ControlSum<const N: usize> {
    values: [Taylor; N],
    ratios: [Option<Ratio>; N],
    euclidean_hulls: [Option<ValidatedInterval>; N],
}
impl<const N: usize> ControlSum<N> {
    fn hull(self, other: Self) -> Self {
        Self {
            values: std::array::from_fn(|i| self.values[i].hull(other.values[i])),
            ratios: std::array::from_fn(|i| match (self.ratios[i], other.ratios[i]) {
                (Some(a), Some(b)) if a.equal(b) => Some(a),
                _ => None,
            }),
            euclidean_hulls: std::array::from_fn(|i| {
                match (self.euclidean_hulls[i], other.euclidean_hulls[i]) {
                    (Some(a), Some(b)) => Some(a.hull(b)),
                    _ => None,
                }
            }),
        }
    }
}
struct ControlEvidence<const N: usize> {
    hulls: [Option<ValidatedInterval>; N],
    ratios: [Option<Ratio>; N],
    possible: [bool; N],
    weight: Option<usize>,
    euclidean_hulls: [Option<ValidatedInterval>; N],
    origin: Option<[f64; N]>,
}
impl<const N: usize> ControlEvidence<N> {
    fn new(weight: Option<usize>) -> Self {
        Self {
            hulls: [None; N],
            ratios: [None; N],
            possible: [true; N],
            weight,
            euclidean_hulls: [None; N],
            origin: None,
        }
    }
    fn include(
        &mut self,
        point: [f64; N],
        basis: Taylor,
        partition_unity: bool,
        values: &mut [Taylor; N],
    ) -> Result<(), ValidatedGeometryError> {
        self.observe(point)?;
        // Curves need only the one-axis origin identity.
        let first = self.origin.is_none();
        let origin = self
            .origin
            .get_or_insert(if partition_unity { point } else { [0.0; N] });
        if first {
            *values = origin.map(Taylor::constant);
        }
        for i in 0..N {
            let coefficient = if point[i] == origin[i] {
                ValidatedInterval::point(0.0)
            } else {
                ValidatedInterval::point(point[i]) - ValidatedInterval::point(origin[i])
            };
            values[i] = values[i] + Taylor::interval(coefficient) * basis;
        }
        Ok(())
    }
    fn observe(&mut self, point: [f64; N]) -> Result<(), ValidatedGeometryError> {
        if !point.into_iter().all(f64::is_finite) {
            return Err(ValidatedGeometryError::InvalidInput);
        }
        for i in 0..N {
            let value = ValidatedInterval::point(point[i]);
            self.hulls[i] = Some(self.hulls[i].map_or(value, |previous| previous.hull(value)));
            if let Some(weight) = self.weight {
                if point[weight] == 0.0 {
                    self.possible[i] &= point[i] == 0.0;
                } else {
                    let ratio = Ratio {
                        numerator: point[i],
                        denominator: point[weight],
                    };
                    let projected = ratio.enclosure()?;
                    self.euclidean_hulls[i] = Some(
                        self.euclidean_hulls[i]
                            .map_or(projected, |previous| previous.hull(projected)),
                    );
                    if let Some(previous) = self.ratios[i] {
                        self.possible[i] &= previous.equal(ratio);
                    } else {
                        self.ratios[i] = Some(ratio);
                    }
                }
            }
        }
        Ok(())
    }
    fn finish(
        self,
        mut values: [Taylor; N],
        partition_unity: bool,
        nonnegative: bool,
    ) -> Result<ControlSum<N>, ValidatedGeometryError> {
        if partition_unity {
            for (i, value) in values.iter_mut().enumerate() {
                if let Some(hull) = self.hulls[i] {
                    if hull.lower == hull.upper {
                        // Identical coefficients are a constant polynomial by partition unity.
                        *value = Taylor::interval(hull);
                    } else if nonnegative {
                        value.0[0][0] = value.0[0][0]
                            .intersection(hull)
                            .ok_or(ValidatedGeometryError::ConditioningFailure)?;
                    }
                }
            }
        }
        let convex = nonnegative
            && self
                .weight
                .and_then(|axis| self.hulls[axis])
                .is_some_and(|hull| !hull.contains_zero());
        Ok(ControlSum {
            values,
            ratios: std::array::from_fn(|i| {
                if self.possible[i] {
                    self.ratios[i]
                } else {
                    None
                }
            }),
            euclidean_hulls: if convex {
                self.euclidean_hulls
            } else {
                [None; N]
            },
        })
    }
}

fn normalized<const N: usize>(
    homogeneous: ControlSum<N>,
) -> Result<[Taylor; N], ValidatedGeometryError> {
    let inverse = homogeneous.values[N - 1].reciprocal()?;
    let mut value = [Taylor::ZERO; N];
    for (i, component) in value.iter_mut().enumerate() {
        *component = if i + 1 == N {
            Taylor::constant(1.0)
        } else if let Some(ratio) = homogeneous.ratios[i] {
            // Exact dyadic coefficient cross-products prove the quotient constant.
            Taylor::interval(ratio.enclosure()?)
        } else {
            homogeneous.values[i] * inverse
        };
        if let Some(hull) = homogeneous.euclidean_hulls[i] {
            component.0[0][0] = component.0[0][0]
                .intersection(hull)
                .ok_or(ValidatedGeometryError::ConditioningFailure)?;
        }
        component.finite()?;
    }
    Ok(value)
}

fn rational_spline<const N: usize>(
    knots: &KnotVector,
    count: usize,
    parameter: Taylor,
    control: impl Fn(usize) -> [f64; N],
    allow_c0: bool,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<[Taylor; N], ValidatedGeometryError> {
    let mut pending = [(parameter, 0usize); 49];
    let mut length = 1;
    let mut result: Option<[Taylor; N]> = None;
    while length != 0 {
        budget.charge(1, cancel)?;
        length -= 1;
        let (part, depth) = pending[length];
        let homogeneous = curve_sum(
            knots,
            count,
            part,
            &control,
            Some(N - 1),
            allow_c0,
            (budget, cancel),
        )?;
        match normalized(homogeneous) {
            Ok(value) => {
                result = Some(match result {
                    None => value,
                    Some(previous) => std::array::from_fn(|i| previous[i].hull(value[i])),
                });
            }
            Err(ValidatedGeometryError::DenominatorContainsZero) => {
                let middle = part.0[0][0].midpoint();
                if depth == 48 || middle <= part.0[0][0].lower || middle >= part.0[0][0].upper {
                    return Err(ValidatedGeometryError::DenominatorContainsZero);
                }
                let mut lower = part;
                let mut upper = part;
                lower.0[0][0].upper = middle;
                upper.0[0][0].lower = middle;
                budget.charge(2, cancel)?;
                pending[length] = (upper, depth + 1);
                pending[length + 1] = (lower, depth + 1);
                length += 2;
            }
            Err(error) => return Err(error),
        }
    }
    result.ok_or(ValidatedGeometryError::ParameterOutOfRange)
}

fn rational_surface<P, const N: usize>(
    (knots, controls): (&(KnotVector, KnotVector), &[Vec<P>]),
    [u, v]: [Taylor; 2],
    control: impl Fn(&P) -> [f64; N],
    allow_c0: bool,
    (budget, cancel): (&mut ValidatedGeometryBudget, &AtomicBool),
) -> Result<[Taylor; N], ValidatedGeometryError> {
    let mut pending = [([u, v], 0usize); 49];
    let mut length = 1;
    let mut result: Option<[Taylor; N]> = None;
    while length != 0 {
        budget.charge(1, cancel)?;
        length -= 1;
        let (part, depth) = pending[length];
        let homogeneous = surface_sum(
            (knots, controls),
            part,
            &control,
            Some(N - 1),
            allow_c0,
            (budget, cancel),
        )?;
        match normalized(homogeneous) {
            Ok(value) => {
                result = Some(match result {
                    None => value,
                    Some(previous) => std::array::from_fn(|i| previous[i].hull(value[i])),
                });
            }
            Err(ValidatedGeometryError::DenominatorContainsZero) => {
                let axis = usize::from(part[1].0[0][0].width() > part[0].0[0][0].width());
                let middle = part[axis].0[0][0].midpoint();
                if depth == 48
                    || middle <= part[axis].0[0][0].lower
                    || middle >= part[axis].0[0][0].upper
                {
                    return Err(ValidatedGeometryError::DenominatorContainsZero);
                }
                let mut lower = part;
                let mut upper = part;
                lower[axis].0[0][0].upper = middle;
                upper[axis].0[0][0].lower = middle;
                budget.charge(2, cancel)?;
                pending[length] = (upper, depth + 1);
                pending[length + 1] = (lower, depth + 1);
                length += 2;
            }
            Err(error) => return Err(error),
        }
    }
    result.ok_or(ValidatedGeometryError::ParameterOutOfRange)
}

#[cfg(test)]
mod denominator_regressions {
    use super::*;
    use crate::{BsplineCurve, NurbsCurve, Point3, Vector3, Vector4};

    #[test]
    fn borrowed_original_knot_queries_retain_identical_closed_jets_at_knots_and_between_spans() {
        let cancel = AtomicBool::new(false);
        for rational in [false, true] {
            let knots = KnotVector::from(
                std::iter::once(0.0)
                    .chain((0..=128).map(|i| f64::from(i) / 128.0))
                    .chain(std::iter::once(1.0))
                    .collect::<Vec<_>>(),
            );
            let points = (0..=128)
                .map(|i| {
                    let t = f64::from(i) / 128.0;
                    Point3::new(t, t * t, t * t * t)
                })
                .collect::<Vec<_>>();
            let curve = if rational {
                Curve::NurbsCurve(NurbsCurve::new(BsplineCurve::new(
                    knots,
                    points
                        .iter()
                        .enumerate()
                        .map(|(i, point)| {
                            let weight = -1.0 - i as f64 / 256.0;
                            Vector4::new(
                                point.x * weight,
                                point.y * weight,
                                point.z * weight,
                                weight,
                            )
                        })
                        .collect(),
                )))
            } else {
                Curve::BsplineCurve(BsplineCurve::new(knots, points))
            };
            let mut reused_work = ValidatedGeometryBudget::new(1_000_000);
            let mut repeated_work = ValidatedGeometryBudget::new(1_000_000);
            let queries =
                ProjectionLeaderKnotEvaluator::new(&curve, [0.0, 1.0], &mut reused_work, &cancel)
                    .unwrap();
            for i in 0..=256 {
                let t = f64::from(i) / 256.0;
                let reused = queries.jet(t, &mut reused_work, &cancel).unwrap();
                let repeated =
                    validated_projection_leader_jet(&curve, [t; 2], &mut repeated_work, &cancel)
                        .unwrap();
                assert_eq!(reused.position, repeated.position);
                assert_eq!(reused.first, repeated.first);
                assert_eq!(reused.second, repeated.second);
                assert_eq!(reused.third, repeated.third);
                if !rational && i > 0 && i < 256 && i % 2 == 0 {
                    // The independent piecewise-linear y=t² interpolant has
                    // distinct adjacent slopes; a closed knot query keeps both.
                    assert!(reused.first[1].lower <= 2.0 * t - 1.0 / 128.0);
                    assert!(reused.first[1].upper >= 2.0 * t + 1.0 / 128.0);
                }
            }
            for outside in [-1.0 / 128.0, 1.0 + 1.0 / 128.0] {
                assert!(matches!(
                    queries.jet(outside, &mut reused_work, &cancel),
                    Err(ValidatedGeometryError::ParameterOutOfRange)
                ));
            }
            assert!(matches!(
                queries.jet(0.5, &mut reused_work, &AtomicBool::new(true)),
                Err(ValidatedGeometryError::Cancelled)
            ));
            assert!(matches!(
                queries.jet(0.5, &mut ValidatedGeometryBudget::new(0), &cancel),
                Err(ValidatedGeometryError::ResourceLimit)
            ));
        }
    }

    #[test]
    fn borrowed_original_knot_queries_preserve_poles_and_discontinuous_source_refusals() {
        let cancel = AtomicBool::new(false);
        let pole = Curve::NurbsCurve(NurbsCurve::new(BsplineCurve::new(
            KnotVector::from(vec![0.0, 0.0, 1.0, 1.0]),
            vec![
                Vector4::new(1.0, 0.0, 0.0, 1.0),
                Vector4::new(1.0, 0.0, 0.0, -1.0),
            ],
        )));
        let mut work = ValidatedGeometryBudget::new(100_000);
        let queries =
            ProjectionLeaderKnotEvaluator::new(&pole, [0.0, 1.0], &mut work, &cancel).unwrap();
        assert!(matches!(
            queries.jet(0.5, &mut work, &cancel),
            Err(ValidatedGeometryError::DenominatorContainsZero)
        ));
        let jump = Curve::BsplineCurve(BsplineCurve::new(
            KnotVector::from(vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0]),
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.5, 0.0, 0.0),
                Point3::new(0.5, 1.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
            ],
        ));
        assert!(matches!(
            ProjectionLeaderKnotEvaluator::new(&jump, [0.0, 1.0], &mut work, &cancel),
            Err(ValidatedGeometryError::NonSmooth)
        ));
        for knots in [
            KnotVector::from(vec![0.0, 1.0]),
            KnotVector::from(vec![0.0, 0.0, f64::INFINITY, f64::INFINITY]),
        ] {
            let invalid = Curve::BsplineCurve(BsplineCurve::new_unchecked(
                knots,
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
            ));
            assert!(matches!(
                ProjectionLeaderKnotEvaluator::new(&invalid, [0.0, 1.0], &mut work, &cancel),
                Err(ValidatedGeometryError::InvalidInput)
            ));
        }
    }

    #[test]
    fn unit_weight_span_and_negative_equivalent_have_nonzero_denominators() {
        for sign in [1.0, -1.0] {
            let curve = Curve2D::NurbsCurve(NurbsCurve::new(BsplineCurve::new(
                KnotVector::from(vec![0.0, 0.0, 1.0, 1.0]),
                vec![Vector3::new(0.0, 0.0, sign), Vector3::new(sign, 0.0, sign)],
            )));
            let jet = validated_curve2d_jet(
                &curve,
                [0.0, 1.0],
                &mut ValidatedGeometryBudget::new(100_000),
                &AtomicBool::new(false),
            )
            .unwrap();
            assert!(jet.position[0].lower <= 0.0 && jet.position[0].upper >= 1.0);
            assert!(jet.first[0].lower <= 1.0 && jet.first[0].upper >= 1.0);
        }
    }

    #[test]
    fn mixed_weights_with_separated_actual_denominator_subdivide() {
        let curve = Curve2D::NurbsCurve(NurbsCurve::new(BsplineCurve::new(
            KnotVector::from(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            vec![
                Vector3::new(0.0, 0.0, 1.0),
                Vector3::new(-0.05, 0.0, -0.1),
                Vector3::new(1.0, 0.0, 1.0),
            ],
        )));
        let jet = validated_curve2d_jet(
            &curve,
            [0.0, 1.0],
            &mut ValidatedGeometryBudget::new(1_000_000),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(
            jet.position
                .into_iter()
                .chain(jet.first)
                .chain(jet.second)
                .all(ValidatedInterval::is_finite)
        );
    }
}

#[cfg(test)]
mod constant_coordinate_regressions {
    use super::*;
    use crate::{BsplineSurface, NurbsSurface, Vector4};

    #[test]
    fn full_rational_patch_retains_constant_height_and_zero_height_derivatives() {
        for weights in [[1.0; 4], [1.0, 0.5, 2.0, 1.5]] {
            let knots = KnotVector::from(vec![0.0, 0.0, 1.0, 1.0]);
            let points = [(0.0, 0.0), (0.0, 4.0), (10.0, 0.0), (10.0, 4.0)];
            let controls: Vec<_> = points
                .into_iter()
                .zip(weights)
                .map(|((x, y), w)| Vector4::new(x * w, y * w, 8.0 * w, w))
                .collect();
            let surface = Surface::NurbsSurface(NurbsSurface::new(BsplineSurface::new(
                (knots.clone(), knots),
                vec![controls[..2].to_vec(), controls[2..].to_vec()],
            )));
            let jet = validated_surface_jet(
                &surface,
                [[0.0, 1.0]; 2],
                &mut ValidatedGeometryBudget::new(1_000_000),
                &AtomicBool::new(false),
            )
            .expect("positive denominator and exact coefficient identity must certify");
            assert_eq!(
                jet.position[2],
                ValidatedInterval {
                    lower: 8.0,
                    upper: 8.0
                }
            );
            for derivative in [jet.du, jet.dv, jet.duu, jet.duv, jet.dvv] {
                assert_eq!(
                    derivative[2],
                    ValidatedInterval {
                        lower: 0.0,
                        upper: 0.0
                    }
                );
            }
            assert!(jet.position[0].lower >= 0.0 && jet.position[0].upper <= 10.0);
            assert!(jet.position[1].lower >= 0.0 && jet.position[1].upper <= 4.0);
        }
    }
}

#[cfg(test)]
mod native_extrusion_jet_tests {
    use super::*;
    use crate::geometry::TryIntoHomogeneousBsplineSurface;
    use crate::*;

    #[test]
    fn ellipse_extrusion_retains_analytic_coefficients_and_refuses_rational_claim() {
        let conic = Curve2D::Conic(Conic2D::Ellipse(Processor::with_transform(
            TrimmedCurve::new(UnitCircle::<Point2>::new(), (0.0, std::f64::consts::TAU)),
            Matrix3::from_nonuniform_scale(2.0, 3.0),
        )));
        let profile = Curve::ParameterCurve(ParameterCurve::new(
            conic,
            Box::new(Surface::Plane(Plane::new(
                Point3::origin(),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ))),
        ));
        let direction = Vector3::new(0.125, -0.25, 7.0);
        let surface = Surface::ExtrusionSurface(ExtrusionSurface::by_extrusion(profile, direction));
        assert!(surface.try_into_homogeneous_bspline_surface().is_none());
        let jet = validated_surface_jet(
            &surface,
            [[0.25, 0.5], [-3.0, 8.0]],
            &mut ValidatedGeometryBudget::new(100_000),
            &AtomicBool::new(false),
        )
        .unwrap();
        for i in 0..3 {
            assert!(jet.dv[i].lower <= direction[i] && jet.dv[i].upper >= direction[i]);
            assert!(jet.duv[i].contains_zero());
            assert!(jet.dvv[i].contains_zero());
        }
        assert!(jet.position[2].lower <= -21.0 && jet.position[2].upper >= 56.0);
        assert!(matches!(
            validated_surface_jet(
                &surface,
                [[0.25, 0.5], [-3.0, 8.0]],
                &mut ValidatedGeometryBudget::new(0),
                &AtomicBool::new(false)
            ),
            Err(ValidatedGeometryError::ResourceLimit)
        ));
        assert!(
            validated_surface_jet(
                &surface,
                [[0.25, 0.5], [-3.0, 8.0]],
                &mut ValidatedGeometryBudget::new(100_000),
                &AtomicBool::new(true)
            )
            .is_err()
        );
    }
}

#[cfg(test)]
mod tensor_factor_regressions {
    use super::*;
    use crate::{BsplineSurface, NurbsSurface, Point3, Vector4};

    fn contains(bound: ValidatedInterval, value: f64) {
        assert!(
            bound.lower <= value && value <= bound.upper,
            "{bound:?} excludes {value}"
        );
    }

    #[test]
    fn dual_tensor_proof_retains_one_origin_bounds_without_rereading_controls() {
        let knot = KnotVector::bezier_knot(3);
        let knots = (knot.clone(), knot);
        let controls: Vec<Vec<[f64; 3]>> = (0..4)
            .map(|i| {
                (0..4)
                    .map(|j| {
                        let (i, j) = (i as f64, j as f64);
                        [
                            i + j * j,
                            3.0 * i * i - j,
                            5.0 * i * j - 2.0 * i.powi(3) + j.powi(3),
                        ]
                    })
                    .collect()
            })
            .collect();
        let t = Taylor::variable(ValidatedInterval::new(0.125, 0.375).unwrap(), 0);
        let cases = [
            [t, Taylor::constant(0.125) + t + t * t],
            [
                Taylor::variable(ValidatedInterval::new(0.1, 0.9).unwrap(), 0),
                Taylor::variable(ValidatedInterval::new(0.2, 0.8).unwrap(), 1),
            ],
        ];
        for uv in cases {
            let cancel = AtomicBool::new(false);
            let mut budget = ValidatedGeometryBudget::new(1_000_000);
            let visits = std::cell::Cell::new(0);
            let value = surface_sum(
                (&knots, &controls),
                uv,
                |point| {
                    visits.set(visits.get() + 1);
                    *point
                },
                None,
                false,
                (&mut budget, &cancel),
            )
            .unwrap()
            .values;
            assert_eq!(visits.get(), 16);
            let (_, lu, bu) = basis(&knots.0, 4, 3, 3, uv[0], &mut budget, &cancel).unwrap();
            let (_, lv, bv) = basis(&knots.1, 4, 3, 3, uv[1], &mut budget, &cancel).unwrap();
            let mut original = [Taylor::ZERO; 3];
            let mut evidence = ControlEvidence::new(None);
            for (i, a) in bu[..lu].iter().enumerate() {
                for (j, b) in bv[..lv].iter().enumerate() {
                    evidence
                        .include(controls[i][j], *a * *b, true, &mut original)
                        .unwrap();
                }
            }
            let original = evidence.finish(original, true, true).unwrap().values;
            for (value, original) in value.into_iter().zip(original) {
                for i in 0..4 {
                    for j in 0..4 - i {
                        assert!(original.0[i][j].lower <= value.0[i][j].lower);
                        assert!(value.0[i][j].upper <= original.0[i][j].upper);
                    }
                }
            }
        }
    }

    #[test]
    fn additive_original_tensor_retains_axis_hessians_on_wide_boxes() {
        let q = [0.25, -0.25, 0.25];
        for offset in [0.0, 1_073_741_824.0] {
            for (a, b) in [(1.0, 1.0), (-4.0, 8.0), (8.0, -4.0)] {
                let knots = KnotVector::bezier_knot(2);
                let surface = Surface::BsplineSurface(BsplineSurface::new(
                    (knots.clone(), knots),
                    (0..3)
                        .map(|i| {
                            (0..3)
                                .map(|j| {
                                    Point3::new(
                                        i as f64 / 2.0,
                                        j as f64 / 2.0,
                                        offset + a * q[i] + b * q[j],
                                    )
                                })
                                .collect()
                        })
                        .collect(),
                ));
                let mut previous = None;
                for v in [[0.1, 0.9], [0.25, 0.5], [-1.0, 2.0]] {
                    let jet = validated_surface_jet(
                        &surface,
                        [[0.1, 0.9], v],
                        &mut ValidatedGeometryBudget::new(1_000_000),
                        &AtomicBool::new(false),
                    )
                    .unwrap();
                    assert_eq!(jet.duv, [super::super::ZERO; 3]);
                    contains(jet.duu[2], 2.0 * a);
                    contains(jet.dvv[2], 2.0 * b);
                    assert!(jet.duu[2].width() < 1.0e-10);
                    assert!(jet.dvv[2].width() < 1.0e-10);
                    if let Some(previous) = previous {
                        assert_eq!(jet.duu, previous);
                    }
                    previous = Some(jet.duu);
                    for u in [0.1_f64, 0.5, 0.9] {
                        for v in [v[0], v[1]] {
                            contains(
                                jet.position[2],
                                offset + a * (u - 0.5).powi(2) + b * (v - 0.5).powi(2),
                            );
                            contains(jet.du[2], 2.0 * a * (u - 0.5));
                            contains(jet.dv[2], 2.0 * b * (v - 0.5));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn mixed_original_tensor_encloses_an_independent_polynomial_law() {
        for (a, b, c) in [(2.0, 3.0, -4.0), (-1.0, -5.0, 8.0)] {
            let knots = KnotVector::bezier_knot(2);
            let square = [0.0, 0.0, 1.0];
            let surface = Surface::BsplineSurface(BsplineSurface::new(
                (knots.clone(), knots),
                (0..3)
                    .map(|i| {
                        (0..3)
                            .map(|j| {
                                let (u, v) = (i as f64 / 2.0, j as f64 / 2.0);
                                Point3::new(u, v, a * square[i] + b * u * v + c * square[j])
                            })
                            .collect()
                    })
                    .collect(),
            ));
            let jet = validated_surface_jet(
                &surface,
                [[0.1, 0.9], [0.2, 0.8]],
                &mut ValidatedGeometryBudget::new(1_000_000),
                &AtomicBool::new(false),
            )
            .unwrap();
            contains(jet.duv[2], b);
            contains(jet.duu[2], 2.0 * a);
            contains(jet.dvv[2], 2.0 * c);
            for u in [0.1, 0.5, 0.9] {
                for v in [0.2, 0.5, 0.8] {
                    contains(jet.position[2], a * u * u + b * u * v + c * v * v);
                    contains(jet.du[2], 2.0 * a * u + b * v);
                    contains(jet.dv[2], b * u + 2.0 * c * v);
                }
            }
        }
    }

    #[test]
    fn additive_hessians_keep_every_original_smooth_knot_span() {
        let knots = KnotVector::from(vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]);
        let q = [0.25, 0.0, 0.0, 0.25];
        let linear = [0.0, 0.25, 0.75, 1.0];
        let surface = Surface::BsplineSurface(BsplineSurface::new(
            (knots.clone(), knots),
            (0..4)
                .map(|i| {
                    (0..4)
                        .map(|j| Point3::new(linear[i], linear[j], q[i] + q[j]))
                        .collect()
                })
                .collect(),
        ));
        let jet = validated_surface_jet(
            &surface,
            [[0.1, 0.9]; 2],
            &mut ValidatedGeometryBudget::new(1_000_000),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(jet.duv, [super::super::ZERO; 3]);
        contains(jet.duu[2], 2.0);
        contains(jet.dvv[2], 2.0);
        assert!(jet.duu[2].width() < 1.0e-10);
        assert!(jet.dvv[2].width() < 1.0e-10);
        for u in [0.1_f64, 0.5, 0.9] {
            for v in [0.1_f64, 0.5, 0.9] {
                contains(jet.position[2], (u - 0.5).powi(2) + (v - 0.5).powi(2));
                contains(jet.du[2], 2.0 * (u - 0.5));
                contains(jet.dv[2], 2.0 * (v - 0.5));
            }
        }
    }

    #[test]
    fn incomplete_source_partition_keeps_the_original_flat_basis_sum() {
        let knots = (
            KnotVector::from(vec![0.0, 1.0, 2.0, 3.0, 4.0]),
            KnotVector::bezier_knot(1),
        );
        let controls = vec![vec![[7.0], [7.0]], vec![[7.0], [7.0]]];
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(1_000_000);
        let u = Taylor::variable(ValidatedInterval::new(0.25, 0.75).unwrap(), 0);
        let v = Taylor::variable(ValidatedInterval::new(0.1, 0.9).unwrap(), 1);
        let value = surface_sum(
            (&knots, &controls),
            [u, v],
            |p| *p,
            None,
            false,
            (&mut budget, &cancel),
        )
        .unwrap()
        .values[0];
        // On the first original unclamped span only B_0,2(u)=u²/2 survives.
        for u in [0.25, 0.5, 0.75] {
            contains(value.0[0][0], 3.5 * u * u);
            contains(value.0[1][0], 7.0 * u);
            contains(value.0[2][0], 3.5);
        }
        assert!(value.0[0][0].upper < 7.0);
    }

    #[test]
    fn affine_tensor_factor_keeps_mixed_and_second_derivatives_exact_zero() {
        let h = std::f64::consts::FRAC_1_SQRT_2;
        for weights in [[1.0, h, 1.0], [1.0, -0.125, 1.0]] {
            for swapped in [false, true] {
                let controls: Vec<Vec<_>> = weights
                    .into_iter()
                    .enumerate()
                    .map(|(i, w)| {
                        let x = i as f64 * 0.5 * w;
                        vec![
                            Vector4::new(x, 0.0, 0.0, w),
                            Vector4::new(x, 0.0, 2.0 * w, w),
                        ]
                    })
                    .collect();
                let controls = if swapped {
                    (0..2)
                        .map(|j| (0..3).map(|i| controls[i][j]).collect())
                        .collect()
                } else {
                    controls
                };
                let quadratic = KnotVector::from(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
                let linear = KnotVector::from(vec![0.0, 0.0, 1.0, 1.0]);
                let knots = if swapped {
                    (linear, quadratic)
                } else {
                    (quadratic, linear)
                };
                let surface =
                    Surface::NurbsSurface(NurbsSurface::new(BsplineSurface::new(knots, controls)));
                let range = if swapped {
                    [[-2.0, 3.0], [0.0, 1.0]]
                } else {
                    [[0.0, 1.0], [-2.0, 3.0]]
                };
                let jet = validated_surface_jet(
                    &surface,
                    range,
                    &mut ValidatedGeometryBudget::new(1_000_000),
                    &AtomicBool::new(false),
                )
                .expect(
                    "An exact affine homogeneous factor must retain its derivative identities.",
                );
                assert_eq!(jet.duv, [super::super::ZERO; 3]);
                assert_eq!(
                    if swapped { jet.duu } else { jet.dvv },
                    [super::super::ZERO; 3]
                );
                let factor_first = if swapped { jet.du } else { jet.dv };
                assert_eq!(factor_first[0], super::super::ZERO);
                assert_eq!(factor_first[1], super::super::ZERO);
                assert!(factor_first[2].lower <= 2.0 && factor_first[2].upper >= 2.0);
                assert!(jet.position[2].lower <= -4.0 && jet.position[2].upper >= 6.0);
            }
        }
    }

    #[test]
    fn degree_two_tensor_coordinate_cancellation_preserves_native_axis_zeros() {
        for a in [[1.0, 2.0, 1.0], [1.0, -0.125, 1.0]] {
            let b = [1.0, 2.0, 1.0];
            let numerator_u = [0.0, 1.0, 3.0];
            let numerator_v = [0.0, 1.0, 0.0];
            let controls: Vec<Vec<_>> = (0..3)
                .map(|i| {
                    (0..3)
                        .map(|j| {
                            let weight = a[i] * b[j];
                            Vector4::new(
                                (i as f64 - j as f64) * weight,
                                numerator_u[i] * b[j],
                                a[i] * numerator_v[j],
                                weight,
                            )
                        })
                        .collect()
                })
                .collect();
            let knots = KnotVector::from(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
            let surface = Surface::NurbsSurface(NurbsSurface::new(BsplineSurface::new(
                (knots.clone(), knots),
                controls,
            )));
            let jet = validated_surface_jet(
                &surface,
                [[0.0, 1.0]; 2],
                &mut ValidatedGeometryBudget::new(1_000_000),
                &AtomicBool::new(false),
            )
            .expect("Exact whole-coefficient factors cancel after full denominator admission.");
            assert_eq!([jet.du[2], jet.duu[2], jet.duv[2]], [super::super::ZERO; 3]);
            assert_eq!([jet.dv[1], jet.dvv[1], jet.duv[1]], [super::super::ZERO; 3]);
            assert!(
                jet.position
                    .into_iter()
                    .chain(jet.du)
                    .chain(jet.dv)
                    .chain(jet.duu)
                    .chain(jet.duv)
                    .chain(jet.dvv)
                    .all(ValidatedInterval::is_finite)
            );
        }
    }
}

#[cfg(test)]
mod retained_trim_regressions {
    use super::*;
    use crate::{
        Invertible, Line, Matrix3, Matrix4, Point2, Point3, Processor, Tolerance, TrimmedCurve,
        UnitCircle,
    };

    #[test]
    fn line_support_hulls_do_not_clamp_retained_analytic_continuation() {
        let line = Curve::Line(Line(Point3::new(0.0, 0.0, 0.0), Point3::new(5.0, 0.0, 0.0)));
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(100_000);
        let jet = validated_curve_jet(&line, [0.0, 1.0], &mut budget, &cancel).unwrap();
        assert_eq!((jet.position[0].lower, jet.position[0].upper), (0.0, 5.0));
        let scalar = super::super::validated_curve_projected_jet(
            &line,
            [0.0, 1.0],
            [0.0; 3],
            [1.0, 0.0, 0.0],
            &mut budget,
            &cancel,
        )
        .unwrap();
        assert_eq!((scalar.position.lower, scalar.position.upper), (0.0, 5.0));
        let extended = Curve::TrimmedCurve(Processor::with_transform(
            TrimmedCurve::new(Box::new(line), (-1.0, 2.0)),
            Matrix4::from_scale(1.0),
        ));
        let jet = validated_curve_jet(&extended, [-1.0, 2.0], &mut budget, &cancel).unwrap();
        assert!(jet.position[0].lower <= -5.0 && jet.position[0].upper >= 10.0);
        let scalar = super::super::validated_curve_projected_jet(
            &extended,
            [-1.0, 2.0],
            [0.0; 3],
            [1.0, 0.0, 0.0],
            &mut budget,
            &cancel,
        )
        .unwrap();
        assert!(scalar.position.lower <= -5.0 && scalar.position.upper >= 10.0);
    }

    #[test]
    fn nested_conic_uses_each_original_inverse_anchor_outside_placeholder_ranges() {
        let mut inner = Processor::new(TrimmedCurve::new(UnitCircle::<Point2>::new(), (1.0, 2.0)));
        inner.invert();
        let basis = Curve2D::Conic(Conic2D::Ellipse(inner));
        let extended = Curve2D::TrimmedCurve(Processor::new(TrimmedCurve::new(
            Box::new(basis),
            (2.5, 3.0),
        )));
        let mut outer = Processor::new(TrimmedCurve::new(Box::new(extended), (2.75, 3.0)));
        outer.invert();
        let curve = Curve2D::TrimmedCurve(outer);
        let mut budget = ValidatedGeometryBudget::new(100_000);
        let jet = validated_curve2d_jet(&curve, [2.75, 2.75], &mut budget, &AtomicBool::new(false))
            .unwrap();
        for (axis, expected) in [1.0, 0.0].into_iter().enumerate() {
            assert!(jet.position[axis].lower <= expected && expected <= jet.position[axis].upper);
        }
        assert!(jet.first[1].lower <= 1.0 && 1.0 <= jet.first[1].upper);
    }

    #[test]
    fn retained_uv_jets_enclose_actual_native_affine_points_and_derivatives() {
        let cases = [
            (
                Line(Point2::new(0.0, 0.0), Point2::new(5.0, 10.0)),
                Matrix3::new(80.0, 0.0, 0.0, 0.0, 120.0, 0.0, 40.0, 80.0, 40.0),
                Line(Point2::new(40.0, 80.0), Point2::new(440.0, 1280.0)),
            ),
            (
                Line(Point2::new(0.0, 0.0), Point2::new(1.0, 2.0)),
                Matrix3::new(1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, -0.25),
                Line(Point2::new(0.0, 0.0), Point2::new(1.0, 2.0)),
            ),
        ];
        for (basis, matrix, expected) in cases {
            let curve = Curve2D::TrimmedCurve(Processor::with_transform(
                TrimmedCurve::new(Box::new(Curve2D::Line(basis)), (0.0, 1.0)),
                matrix,
            ));
            let jet = validated_curve2d_jet(
                &curve,
                [0.0, 1.0],
                &mut ValidatedGeometryBudget::new(100_000),
                &AtomicBool::new(false),
            )
            .unwrap();
            for t in [0.0, 0.5, 1.0] {
                let point = expected.subs(t);
                let derivative = expected.der(t);
                assert!(curve.subs(t).near(&point));
                assert!(curve.der(t).near(&derivative));
                for (axis, value) in [point.x, point.y].into_iter().enumerate() {
                    assert!(jet.position[axis].lower <= value && value <= jet.position[axis].upper);
                }
                for (axis, value) in [derivative.x, derivative.y].into_iter().enumerate() {
                    assert!(jet.first[axis].lower <= value && value <= jet.first[axis].upper);
                    assert_eq!(jet.second[axis], super::super::ZERO);
                }
            }
        }
    }

    #[test]
    fn rational_uv_leaf_pole_is_not_repaired_by_unused_pose_row() {
        use crate::{BsplineCurve, NurbsCurve, Vector3};
        let basis = Curve2D::NurbsCurve(NurbsCurve::new(BsplineCurve::new(
            KnotVector::from(vec![0.0, 0.0, 1.0, 1.0]),
            vec![Vector3::new(0.0, 0.0, -0.5), Vector3::new(1.0, 0.0, 0.5)],
        )));
        let curve = Curve2D::TrimmedCurve(Processor::with_transform(
            TrimmedCurve::new(Box::new(basis), (0.0, 1.0)),
            Matrix3::new(1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, -2.0),
        ));
        assert!(!curve.subs(0.5).x.is_finite());
        assert!(matches!(
            validated_curve2d_jet(
                &curve,
                [0.5, 0.5],
                &mut ValidatedGeometryBudget::new(100_000),
                &AtomicBool::new(false),
            ),
            Err(ValidatedGeometryError::DenominatorContainsZero)
        ));
    }
}

#[cfg(test)]
mod c0_source_policy_regressions {
    use super::*;
    use crate::{BsplineSurface, Point3};

    #[test]
    fn uv_piecewise_policy_keeps_closed_knot_sides_and_refuses_jumps() {
        use crate::{BsplineCurve, Point2};
        let curve = Curve2D::BsplineCurve(BsplineCurve::new(
            KnotVector::from(vec![0.0, 0.0, 1.0, 2.0, 2.0]),
            vec![
                Point2::new(0.0, 0.0),
                Point2::new(1.0, 0.0),
                Point2::new(1.0, 2.0),
            ],
        ));
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(100_000);
        assert!(matches!(
            validated_curve2d_jet(&curve, [0.5, 1.5], &mut budget, &cancel),
            Err(ValidatedGeometryError::NonSmooth)
        ));
        for range in [[0.5, 1.5], [1.0, 1.0]] {
            let jet = validated_curve2d_piecewise_jet(&curve, range, &mut budget, &cancel).unwrap();
            for (derivative, maximum) in jet.first.into_iter().zip([1.0, 2.0]) {
                assert!(derivative.lower <= 0.0 && maximum <= derivative.upper);
            }
        }
        let jump = Curve2D::BsplineCurve(BsplineCurve::new(
            KnotVector::from(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]),
            (0..4).map(|i| Point2::new(i as f64, 0.0)).collect(),
        ));
        assert!(matches!(
            validated_curve2d_piecewise_jet(&jump, [1.0, 1.0], &mut budget, &cancel),
            Err(ValidatedGeometryError::NonSmooth)
        ));
    }

    #[test]
    fn continuous_piecewise_source_policy_does_not_admit_discontinuities() {
        let v = KnotVector::from(vec![0.0, 0.0, 1.0, 1.0]);
        let controls = vec![
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 1.0)],
            vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 0.0, 1.0)],
            vec![Point3::new(2.0, 0.25, 0.0), Point3::new(2.0, 0.25, 1.0)],
        ];
        let surface = Surface::BsplineSurface(BsplineSurface::new(
            (KnotVector::from(vec![0.0, 0.0, 1.0, 2.0, 2.0]), v.clone()),
            controls,
        ));
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(100_000);
        assert!(matches!(
            validated_surface_jet(&surface, [[0.5, 1.5], [0.0, 1.0]], &mut budget, &cancel),
            Err(ValidatedGeometryError::NonSmooth)
        ));
        let jet = validated_surface_piecewise_jet(
            &surface,
            [[0.5, 1.5], [0.0, 1.0]],
            &mut budget,
            &cancel,
        )
        .unwrap();
        assert!(
            jet.position
                .into_iter()
                .chain(jet.du)
                .chain(jet.dv)
                .all(ValidatedInterval::is_finite)
        );
        assert!(
            !super::super::surface_box_smooth(
                &surface,
                ValidatedInterval::new(0.5, 1.5).unwrap(),
                ValidatedInterval::new(0.0, 1.0).unwrap(),
                &mut budget,
                &cancel
            )
            .unwrap()
        );
        assert!(
            super::super::surface_box_smooth(
                &surface,
                ValidatedInterval::new(0.0, 1.0).unwrap(),
                ValidatedInterval::new(0.0, 1.0).unwrap(),
                &mut budget,
                &cancel
            )
            .unwrap()
        );
        let controls = (0..4)
            .map(|i| {
                vec![
                    Point3::new(i as f64, 0.0, 0.0),
                    Point3::new(i as f64, 0.0, 1.0),
                ]
            })
            .collect();
        let discontinuous = Surface::BsplineSurface(BsplineSurface::new(
            (KnotVector::from(vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0]), v),
            controls,
        ));
        assert!(matches!(
            validated_surface_piecewise_jet(
                &discontinuous,
                [[0.25, 0.75], [0.0, 1.0]],
                &mut budget,
                &cancel
            ),
            Err(ValidatedGeometryError::NonSmooth)
        ));
    }
}
