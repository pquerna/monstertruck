//! Outward enclosures of original STEP conic coefficients.

use super::*;
use monstertruck_modeling::{
    ValidatedCurveJet, ValidatedGeometryBudget, ValidatedGeometryError, ValidatedInterval,
    validated_curve_jet,
};
use std::result::Result;
use std::sync::atomic::AtomicBool;

/// Encloses an original STEP conic and its first three source-parameter derivatives.
///
/// The analytic base is enclosed before applying the original processor matrix.
/// No reconstructed plane frame, fitted leader or rationalized parameter is
/// substituted for the source coefficients. Affine processor matrices and
/// bounded trims are supported; projective frames refuse explicitly.
pub fn validated_step_conic_jet(
    conic: &Conic3D,
    range: [f64; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<ValidatedCurveJet, ValidatedGeometryError> {
    budget.charge(2, cancel)?;
    let (frame, orientation, source_range) = match conic {
        Conic3D::Ellipse(curve) => (
            *curve.transform(),
            curve.orientation(),
            curve.entity().range_tuple(),
        ),
        Conic3D::Hyperbola(curve) => (
            *curve.transform(),
            curve.orientation(),
            curve.entity().range_tuple(),
        ),
        Conic3D::Parabola(curve) => (
            *curve.transform(),
            curve.orientation(),
            curve.entity().range_tuple(),
        ),
    };
    if frame.x.w != 0.0 || frame.y.w != 0.0 || frame.z.w != 0.0 || frame.w.w != 1.0 {
        return Err(ValidatedGeometryError::Unsupported);
    }
    if !source_range.0.is_finite()
        || !source_range.1.is_finite()
        || range[0] < source_range.0
        || range[1] > source_range.1
    {
        return Err(ValidatedGeometryError::ParameterOutOfRange);
    }
    let mut base = match conic {
        Conic3D::Ellipse(_) => ModelingConic2D::Ellipse(Processor::new(TrimmedCurve::new(
            UnitCircle::<Point2>::new(),
            source_range,
        ))),
        Conic3D::Hyperbola(_) => ModelingConic2D::Hyperbola(Processor::new(TrimmedCurve::new(
            UnitHyperbola::<Point2>::new(),
            source_range,
        ))),
        Conic3D::Parabola(_) => ModelingConic2D::Parabola(Processor::new(TrimmedCurve::new(
            UnitParabola::<Point2>::new(),
            source_range,
        ))),
    };
    if !orientation {
        base.invert();
    }
    let base = ModelingCurve::ParameterCurve(ParameterCurve::new(
        ModelingCurve2D::Conic(base),
        Box::new(ModelingSurface::Plane(Plane::xy())),
    ));
    let jet = validated_curve_jet(&base, range, budget, cancel)?;
    let zero = ValidatedInterval::new(0.0, 0.0)?;
    let coefficients: [[ValidatedInterval; 4]; 3] = [
        [frame.x.x, frame.y.x, frame.z.x, frame.w.x],
        [frame.x.y, frame.y.y, frame.z.y, frame.w.y],
        [frame.x.z, frame.y.z, frame.z.z, frame.w.z],
    ]
    .map(|row| {
        row.map(|value| ValidatedInterval {
            lower: value,
            upper: value,
        })
    });
    if coefficients
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    let apply = |value: [ValidatedInterval; 3], position: bool| {
        std::array::from_fn(|i| {
            coefficients[i][0] * value[0]
                + coefficients[i][1] * value[1]
                + coefficients[i][2] * value[2]
                + if position { coefficients[i][3] } else { zero }
        })
    };
    let output = ValidatedCurveJet {
        position: apply(jet.position, true),
        first: apply(jet.first, false),
        second: apply(jet.second, false),
        third: apply(jet.third, false),
    };
    if [output.position, output.first, output.second, output.third]
        .iter()
        .flatten()
        .all(|value| value.is_finite())
    {
        Ok(output)
    } else {
        Err(ValidatedGeometryError::ConditioningFailure)
    }
}
