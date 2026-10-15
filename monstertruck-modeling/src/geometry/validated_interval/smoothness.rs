use super::explicit::{
    curve2d_taylor_with_policy, homogeneous_join_is_c1, processed_parameter, validate_knots,
    validate_knots_with_policy,
};
use super::{
    Taylor, ValidatedGeometryBudget as Budget, ValidatedGeometryError as E, ValidatedInterval as I,
    interval_range,
};
use crate::{Curve, Curve2D, KnotVector, Surface, geometry::MAX_NATIVE_GEOMETRY_NESTING};
use std::sync::atomic::AtomicBool;

/// Qualifies source continuity, not merely the finiteness of piecewise jets.
/// Order one proves C1 with bounded piecewise second derivatives; order two
/// proves C2 with bounded piecewise third derivatives. Positive-width spans
/// admit selected one-sided endpoints; point queries at interior knots and
/// uncertain bands straddling knots are not rounded to a smooth span.
pub(crate) fn curve_span_smooth(
    curve: &Curve,
    range: I,
    order: usize,
    conic_extension: bool,
    budget: &mut Budget,
    cancel: &AtomicBool,
) -> Result<bool, E> {
    interval_range([range.lower, range.upper])?;
    curve_smooth(
        curve,
        Taylor::interval(range),
        order,
        conic_extension,
        0,
        budget,
        cancel,
    )
}

/// Qualifies the actual retained source UV box, including nested sweep/profile
/// and parameter-curve images evaluated by the same outward native arithmetic.
pub(crate) fn surface_box_smooth(
    surface: &Surface,
    u: I,
    v: I,
    budget: &mut Budget,
    cancel: &AtomicBool,
) -> Result<bool, E> {
    interval_range([u.lower, u.upper])?;
    interval_range([v.lower, v.upper])?;
    surface_smooth(
        surface,
        Taylor::interval(u),
        Taylor::interval(v),
        1,
        0,
        budget,
        cancel,
    )
}

fn enter(depth: usize, order: usize, budget: &mut Budget, cancel: &AtomicBool) -> Result<(), E> {
    budget.charge(1, cancel)?;
    if depth > MAX_NATIVE_GEOMETRY_NESTING {
        return Err(E::ResourceLimit);
    }
    if !(1..=2).contains(&order) {
        return Err(E::InvalidInput);
    }
    Ok(())
}

fn knots_smooth(
    knots: &KnotVector,
    count: usize,
    parameter: I,
    order: usize,
    extrapolate: bool,
    budget: &mut Budget,
    cancel: &AtomicBool,
) -> Result<bool, E> {
    let degree = match validate_knots(knots, count, parameter, extrapolate, budget, cancel) {
        Ok(degree) => degree,
        Err(E::NonSmooth) => return Ok(false),
        Err(error) => return Err(error),
    };
    let mut index = 0;
    while index < knots.len() {
        budget.charge(1, cancel)?;
        let start = index;
        let knot = knots[index];
        while index < knots.len() && knots[index] == knot {
            budget.charge(1, cancel)?;
            index += 1;
        }
        if knots[0] < knot
            && knot < knots[knots.len() - 1]
            && (parameter.lower < knot && knot < parameter.upper
                || parameter.lower == parameter.upper && parameter.lower == knot)
            && degree < (index - start).saturating_add(order)
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn curve_knots_smooth<const N: usize>(
    knots: &KnotVector,
    count: usize,
    parameter: I,
    order: usize,
    control: impl Fn(usize) -> [f64; N],
    budget: &mut Budget,
    cancel: &AtomicBool,
) -> Result<bool, E> {
    let degree =
        match validate_knots_with_policy(knots, count, parameter, false, true, budget, cancel) {
            Ok(degree) => degree,
            Err(E::NonSmooth) => return Ok(false),
            Err(error) => return Err(error),
        };
    let mut index = 0;
    while index < knots.len() {
        budget.charge(1, cancel)?;
        let start = index;
        let knot = knots[index];
        while index < knots.len() && knots[index] == knot {
            budget.charge(1, cancel)?;
            index += 1;
        }
        let crosses = parameter.lower < knot && knot < parameter.upper
            || parameter.lower == parameter.upper && parameter.lower == knot;
        if knots[0] < knot
            && knot < knots[knots.len() - 1]
            && crosses
            && degree < (index - start).saturating_add(order)
        {
            if order != 1
                || index - start != degree
                || start < 2
                || start >= count
                || index >= knots.len()
            {
                return Ok(false);
            }
            budget.charge(3, cancel)?;
            if !homogeneous_join_is_c1(
                control(start - 2),
                control(start - 1),
                control(start),
                knots[start - 1],
                knot,
                knots[index],
            ) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn curve_smooth(
    curve: &Curve,
    parameter: Taylor,
    order: usize,
    extension: bool,
    depth: usize,
    budget: &mut Budget,
    cancel: &AtomicBool,
) -> Result<bool, E> {
    enter(depth, order, budget, cancel)?;
    match curve {
        Curve::Line(_) => Ok(true),
        Curve::BsplineCurve(curve) => curve_knots_smooth(
            curve.knot_vector(),
            curve.control_points().len(),
            parameter.0[0][0],
            order,
            |i| {
                let p = curve.control_points()[i];
                [p.x, p.y, p.z, 1.0]
            },
            budget,
            cancel,
        ),
        Curve::NurbsCurve(curve) => curve_knots_smooth(
            curve.knot_vector(),
            curve.control_points().len(),
            parameter.0[0][0],
            order,
            |i| {
                let p = curve.control_points()[i];
                [p.x, p.y, p.z, p.w]
            },
            budget,
            cancel,
        ),
        Curve::TrimmedCurve(wrapper) => {
            let parameter = processed_parameter(
                wrapper.orientation(),
                Some(wrapper.entity().range()),
                parameter,
            )?;
            curve_smooth(
                wrapper.entity().curve(),
                parameter,
                order,
                true,
                depth + 1,
                budget,
                cancel,
            )
        }
        Curve::ParameterCurve(curve) => {
            if !curve2d_smooth(curve.curve(), parameter, order, depth + 1, budget, cancel)? {
                return Ok(false);
            }
            let uv = curve2d_taylor_with_policy(
                curve.curve(),
                parameter,
                depth + 1,
                extension,
                true,
                budget,
                cancel,
            )?;
            surface_smooth(
                curve.surface(),
                uv[0],
                uv[1],
                order,
                depth + 1,
                budget,
                cancel,
            )
        }
        Curve::IntersectionCurve(_) => Ok(false),
    }
}

fn curve2d_smooth(
    curve: &Curve2D,
    parameter: Taylor,
    order: usize,
    depth: usize,
    budget: &mut Budget,
    cancel: &AtomicBool,
) -> Result<bool, E> {
    enter(depth, order, budget, cancel)?;
    match curve {
        Curve2D::Line(_) | Curve2D::Conic(_) => Ok(true),
        Curve2D::BsplineCurve(curve) => curve_knots_smooth(
            curve.knot_vector(),
            curve.control_points().len(),
            parameter.0[0][0],
            order,
            |i| {
                let p = curve.control_points()[i];
                [p.x, p.y, 1.0]
            },
            budget,
            cancel,
        ),
        Curve2D::NurbsCurve(curve) => curve_knots_smooth(
            curve.knot_vector(),
            curve.control_points().len(),
            parameter.0[0][0],
            order,
            |i| {
                let p = curve.control_points()[i];
                [p.x, p.y, p.z]
            },
            budget,
            cancel,
        ),
        Curve2D::TrimmedCurve(wrapper) => {
            let parameter = processed_parameter(
                wrapper.orientation(),
                Some(wrapper.entity().range()),
                parameter,
            )?;
            curve2d_smooth(
                wrapper.entity().curve(),
                parameter,
                order,
                depth + 1,
                budget,
                cancel,
            )
        }
        Curve2D::Polyline(curve) => {
            if curve.len() < 2 {
                return Err(E::InvalidInput);
            }
            let range = parameter.0[0][0];
            if range.lower < 0.0 || range.upper > (curve.len() - 1) as f64 {
                return Err(E::ParameterOutOfRange);
            }
            for index in 1..curve.len() - 1 {
                budget.charge(1, cancel)?;
                let knot = index as f64;
                if range.lower < knot && knot < range.upper
                    || range.lower == range.upper && range.lower == knot
                {
                    if order != 1 {
                        return Ok(false);
                    }
                    let (a, b, c) = (curve[index - 1], curve[index], curve[index + 1]);
                    if !homogeneous_join_is_c1(
                        [a.x, a.y, 1.0],
                        [b.x, b.y, 1.0],
                        [c.x, c.y, 1.0],
                        (index - 1) as f64,
                        knot,
                        (index + 1) as f64,
                    ) {
                        return Ok(false);
                    }
                }
            }
            Ok(true)
        }
    }
}

fn surface_smooth(
    surface: &Surface,
    u: Taylor,
    v: Taylor,
    order: usize,
    depth: usize,
    budget: &mut Budget,
    cancel: &AtomicBool,
) -> Result<bool, E> {
    enter(depth, order, budget, cancel)?;
    match surface {
        Surface::Plane(_) | Surface::SphericalSurface(_) | Surface::ToroidalSurface(_) => Ok(true),
        Surface::BsplineSurface(surface) => {
            let controls = surface.control_points();
            Ok(knots_smooth(
                &surface.knot_vectors().0,
                controls.len(),
                u.0[0][0],
                order,
                true,
                budget,
                cancel,
            )? && knots_smooth(
                &surface.knot_vectors().1,
                controls.first().map_or(0, Vec::len),
                v.0[0][0],
                order,
                true,
                budget,
                cancel,
            )?)
        }
        Surface::NurbsSurface(surface) => {
            let controls = surface.control_points();
            Ok(knots_smooth(
                &surface.knot_vectors().0,
                controls.len(),
                u.0[0][0],
                order,
                true,
                budget,
                cancel,
            )? && knots_smooth(
                &surface.knot_vectors().1,
                controls.first().map_or(0, Vec::len),
                v.0[0][0],
                order,
                true,
                budget,
                cancel,
            )?)
        }
        Surface::ExtrusionSurface(surface) => curve_smooth(
            surface.entity_curve(),
            u,
            order,
            false,
            depth + 1,
            budget,
            cancel,
        ),
        Surface::RevolutionSurface(surface) => {
            let profile = if surface.orientation() { u } else { v };
            curve_smooth(
                surface.entity().entity_curve(),
                profile,
                order,
                false,
                depth + 1,
                budget,
                cancel,
            )
        }
        Surface::TsplineSurface(_) => Ok(false),
    }
}

#[cfg(test)]
mod source_smoothness_regressions {
    use super::*;
    use crate::{
        BsplineCurve, BsplineSurface, Invertible, Line, ParameterCurve, Point2, Point3, Processor,
        TrimmedCurve,
    };

    #[test]
    fn nested_inverse_qualification_keeps_original_anchors() {
        let source = Curve::BsplineCurve(BsplineCurve::new(
            KnotVector::from(vec![10.0, 10.0, 11.0, 14.0, 14.0]),
            vec![
                Point3::new(10.0, 0.0, 0.0),
                Point3::new(11.0, 0.0, 0.0),
                Point3::new(14.0, 1.0, 0.0),
            ],
        ));
        let mut inner = Processor::new(TrimmedCurve::new(Box::new(source), (10.0, 14.0)));
        inner.invert();
        let source = Curve::TrimmedCurve(inner);
        let cancel = AtomicBool::new(false);
        let mut budget = Budget::new(100_000);
        assert!(
            curve_span_smooth(
                &source,
                I::new(10.0, 11.0).unwrap(),
                1,
                false,
                &mut budget,
                &cancel
            )
            .unwrap()
        );
        assert!(
            !curve_span_smooth(
                &source,
                I::new(12.0, 14.0).unwrap(),
                1,
                false,
                &mut budget,
                &cancel
            )
            .unwrap()
        );
        let mut outer = Processor::new(TrimmedCurve::new(Box::new(source), (10.0, 12.0)));
        outer.invert();
        assert!(
            curve_span_smooth(
                &Curve::TrimmedCurve(outer),
                I::new(10.0, 12.0).unwrap(),
                1,
                false,
                &mut budget,
                &cancel
            )
            .unwrap()
        );
    }

    #[test]
    fn parameter_curve_qualification_uses_whole_underlying_uv_image() {
        let surface = Surface::BsplineSurface(BsplineSurface::new(
            (
                KnotVector::from(vec![0.0, 0.0, 0.5, 1.0, 1.0]),
                KnotVector::from(vec![0.0, 0.0, 1.0, 1.0]),
            ),
            vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 1.0)],
                vec![Point3::new(0.5, 0.0, 0.0), Point3::new(0.5, 0.0, 1.0)],
                vec![Point3::new(1.0, 0.25, 0.0), Point3::new(1.0, 0.25, 1.0)],
            ],
        ));
        let source = Curve::ParameterCurve(ParameterCurve::new(
            Curve2D::Line(Line(Point2::new(0.25, 0.0), Point2::new(0.75, 0.0))),
            Box::new(surface),
        ));
        let cancel = AtomicBool::new(false);
        let mut budget = Budget::new(100_000);
        assert!(
            !curve_span_smooth(
                &source,
                I::new(0.0, 1.0).unwrap(),
                1,
                false,
                &mut budget,
                &cancel
            )
            .unwrap()
        );
        assert!(
            curve_span_smooth(
                &source,
                I::new(0.0, 0.25).unwrap(),
                1,
                false,
                &mut budget,
                &cancel
            )
            .unwrap()
        );
    }
}

#[cfg(test)]
mod physical_c1_regressions {
    use super::*;
    use crate::{
        BsplineCurve, Invertible, Matrix4, NurbsCurve, Processor, TrimmedCurve,
        ValidatedCurveOptions, Vector4, prepared_validated_curve,
    };

    fn rational_join(right_scale: f64, first_knot: f64, last_knot: f64) -> Curve {
        let h = std::f64::consts::FRAC_1_SQRT_2;
        Curve::NurbsCurve(NurbsCurve::new(BsplineCurve::new(
            KnotVector::from(vec![
                first_knot, first_knot, first_knot, 1.0, 1.0, last_knot, last_knot, last_knot,
            ]),
            vec![
                Vector4::new(1.0, 0.0, 0.0, 1.0),
                Vector4::new(h, h, 0.0, h),
                Vector4::new(0.0, 1.0, 0.0, 1.0),
                Vector4::new(-right_scale * h, h, 0.0, h),
                Vector4::new(-right_scale, 0.0, 0.0, 1.0),
            ],
        )))
    }

    #[test]
    fn original_homogeneous_tangents_qualify_a_real_c1_join_and_refuse_a_cusp() {
        let cancel = AtomicBool::new(false);
        let mut budget = Budget::new(1_000_000);
        for (scale, expected) in [(2.0, true), (1.0, false)] {
            let curve = rational_join(scale, 0.0, 3.0);
            let prepared = prepared_validated_curve(
                &curve,
                [0.0, 3.0],
                &ValidatedCurveOptions::default(),
                &mut budget,
                &cancel,
            )
            .unwrap();
            assert_eq!(
                prepared
                    .is_smooth_on([0.0, 3.0], &mut budget, &cancel)
                    .unwrap(),
                expected
            );
            assert!(
                prepared
                    .position_bounds([0.0, 3.0], &mut budget, &cancel)
                    .unwrap()
                    .into_iter()
                    .all(I::is_finite)
            );
        }
    }

    #[test]
    fn exact_c1_proof_does_not_round_original_knot_differences() {
        let tiny = 2.0_f64.powi(-55);
        assert_eq!(1.0 - tiny, 1.0);
        let curve = rational_join(1.0, tiny, 2.0);
        assert!(
            !curve_span_smooth(
                &curve,
                I::new(tiny, 2.0).unwrap(),
                1,
                false,
                &mut Budget::new(100_000),
                &AtomicBool::new(false)
            )
            .unwrap()
        );
    }

    #[test]
    fn full_rational_circle_c1_survives_retained_tilted_inverted_pose() {
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let curve = Curve::NurbsCurve(NurbsCurve::new(BsplineCurve::new(
            KnotVector::from(vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0,
            ]),
            vec![
                Vector4::new(1.0, 0.0, 0.0, 1.0),
                Vector4::new(h, h, 0.0, h),
                Vector4::new(0.0, 1.0, 0.0, 1.0),
                Vector4::new(-h, h, 0.0, h),
                Vector4::new(-1.0, 0.0, 0.0, 1.0),
                Vector4::new(-h, -h, 0.0, h),
                Vector4::new(0.0, -1.0, 0.0, 1.0),
                Vector4::new(h, -h, 0.0, h),
                Vector4::new(1.0, 0.0, 0.0, 1.0),
            ],
        )));
        let pose = Matrix4::new(
            1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 7.0, 11.0, 13.0, 1.0,
        );
        let mut wrapper =
            Processor::with_transform(TrimmedCurve::new(Box::new(curve), (0.0, 4.0)), pose);
        wrapper.invert();
        let curve = Curve::TrimmedCurve(wrapper);
        let cancel = AtomicBool::new(false);
        let mut budget = Budget::new(1_000_000);
        let prepared = prepared_validated_curve(
            &curve,
            [0.0, 4.0],
            &ValidatedCurveOptions::default(),
            &mut budget,
            &cancel,
        )
        .unwrap();
        assert!(
            prepared
                .is_smooth_on([0.0, 4.0], &mut budget, &cancel)
                .unwrap()
        );
        assert!(
            prepared
                .jet([0.0, 4.0], &mut budget, &cancel)
                .unwrap()
                .second
                .into_iter()
                .all(I::is_finite)
        );
    }
}
