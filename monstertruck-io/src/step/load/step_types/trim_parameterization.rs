//! Numeric STEP trim selection, separate from decoded native parameter gauges.
//!
//! Endpoint/sense ordering follows OCCT 7.9 StepToGeom::MakeTrimmedCurve
//! (StepToGeom.cxx 2389–2395) and Geom_TrimmedCurve::SetTrim/Reverse
//! (Geom_TrimmedCurve.cxx 91–155). A reversed STEP trim swaps its selectors
//! before SetTrim; its canonical parameter uses the root-family reversal.

use super::*;
use crate::step::save::StepCurve;
use crate::step::{ParameterChart, ParameterEnclosure};

#[derive(Clone, Copy, Debug)]
pub(super) struct BasisChart {
    raw_to_internal: ParameterChart,
    raw_domain: Option<ParameterEnclosure>,
    internal_domain: (f64, f64),
    reversal: ParameterChart,
    period: Option<f64>,
    origin: f64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct TrimSelection {
    pub internal_range: (f64, f64),
    pub reversed: bool,
    pub raw_to_internal: ParameterChart,
    pub raw_range: ParameterEnclosure,
    pub selected_basis_range: ParameterEnclosure,
    pub raw_parameter_bounds: [ParameterEnclosure; 2],
    pub selected_basis_bounds: [ParameterEnclosure; 2],
}

fn arithmetic<T>(value: Option<T>) -> Result<T, StepConvertingError> {
    value.ok_or_else(|| ExactConversionError::UnrepresentableTrimParameterChart.into())
}

fn point(value: f64) -> Result<ParameterEnclosure, StepConvertingError> {
    arithmetic(ParameterEnclosure::point(value))
}

fn period_shift(turns: f64, period: f64) -> Result<ParameterChart, StepConvertingError> {
    arithmetic(crate::step::periodic_parameter_shift(turns, period))
}

fn reflection(range: (f64, f64)) -> Result<ParameterChart, StepConvertingError> {
    arithmetic(ParameterChart::reflection(range))
}

fn root_chart<C: BoundedCurve + StepCurve>(
    source: &CurveAny,
    decoded: &C,
) -> Result<BasisChart, StepConvertingError> {
    let internal_domain = decoded.range_tuple();
    let raw_domain = matches!(source, CurveAny::BoundedCurve(_))
        .then(|| ParameterEnclosure::new(internal_domain.0, internal_domain.1))
        .flatten();
    let period = decoded.period();
    let origin = match source {
        CurveAny::Conic(_) => 0.0,
        _ => internal_domain.0,
    };
    Ok(BasisChart {
        raw_to_internal: ParameterChart::identity(),
        raw_domain,
        internal_domain,
        reversal: arithmetic(decoded.step_reversed_parameter_chart())?,
        period,
        origin,
    })
}

fn declared_chart<C: BoundedCurve + StepCurve, M>(
    decoded: &SourceTrimmedCurve<C, M>,
) -> Result<BasisChart, StepConvertingError>
where SourceTrimmedCurve<C, M>: BoundedCurve + StepCurve {
    let SourceParameterGauge::Declared {
        raw_to_internal,
        raw_range,
        ..
    } = decoded.raw_parameter_gauge()?
    else {
        return Err(ExactConversionError::UnrepresentableTrimParameterChart.into());
    };
    Ok(BasisChart {
        raw_to_internal,
        raw_domain: Some(raw_range),
        internal_domain: decoded.range_tuple(),
        reversal: arithmetic(decoded.step_reversed_parameter_chart())?,
        // An explicit STEP TRIMMED_CURVE is bounded and nonperiodic, even
        // though the native decorator forwards its underlying period().
        period: None,
        origin: raw_range.lower,
    })
}

pub(super) fn basis_2d(
    source: &CurveAny,
    decoded: &Curve2D,
) -> Result<BasisChart, StepConvertingError> {
    match (source, decoded) {
        (CurveAny::TrimmedCurve(_), Curve2D::TrimmedCurve(decoded)) => declared_chart(decoded),
        _ => root_chart(source, decoded),
    }
}

pub(super) fn basis_3d(
    source: &CurveAny,
    decoded: &Curve3D,
) -> Result<BasisChart, StepConvertingError> {
    match (source, decoded) {
        (CurveAny::TrimmedCurve(_), Curve3D::TrimmedCurve(decoded)) => declared_chart(decoded),
        (CurveAny::SurfaceCurve(source), Curve3D::SurfaceCurve(decoded)) => {
            basis_3d(&source.curve_3d, decoded.leader())
        }
        (CurveAny::SurfaceCurve(source), Curve3D::IntersectionCurve(decoded)) => {
            basis_3d(&source.curve_3d, decoded.leader().as_ref())
        }
        (CurveAny::Pcurve(source), Curve3D::ParameterCurve(decoded)) => {
            let source = source
                .reference_to_curve
                .representation_item
                .first()
                .ok_or(ExactConversionError::InvalidTrimParameterSelectors)?;
            basis_2d(source, decoded.curve())
        }
        _ => root_chart(source, decoded),
    }
}
impl BasisChart {
    fn internal_selector(self, value: f64) -> Result<f64, StepConvertingError> {
        if let Some(domain) = self.raw_domain {
            // These are named endpoints of the same admitted restriction, not
            // a tolerance/nearest-point clamp of an arbitrary source selector.
            if value == domain.lower {
                return Ok(if self.raw_to_internal.scale > 0.0 {
                    self.internal_domain.0
                } else {
                    self.internal_domain.1
                });
            }
            if value == domain.upper {
                return Ok(if self.raw_to_internal.scale > 0.0 {
                    self.internal_domain.1
                } else {
                    self.internal_domain.0
                });
            }
        }
        arithmetic(self.raw_to_internal.nominal(value))
    }
}

pub(super) fn select(
    selectors: (f64, f64),
    sense: bool,
    basis: BasisChart,
) -> Result<TrimSelection, StepConvertingError> {
    let (first, second) = selectors;
    if let Some(domain) = basis.raw_domain
        && (!domain.contains(first) || !domain.contains(second))
    {
        return Err(ExactConversionError::InvalidTrimParameterSelectors.into());
    }
    let source_reversed;
    let selected;
    let selected_basis_bounds;
    let canonical_values;
    let canonical_maps;
    let branch_shift;
    if let Some(period) = basis.period {
        if !period.is_finite() || period <= 0.0 {
            return Err(ExactConversionError::UnrepresentableTrimParameterChart.into());
        }
        source_reversed = !sense;
        // SetTrim's increasing interval starts at trim1 for T, trim2 for F.
        let base = if sense { first } else { second };
        let end = if sense { second } else { first };
        let canonical_turns = ((base - basis.origin) / period).floor();
        let normalize = period_shift(-canonical_turns, period)?;
        let mut turns = ((base - end) / period).floor() + 1.0;
        let mut lift = period_shift(turns, period)?;
        let mut lifted_end = arithmetic(lift.nominal(end))?;
        // Equal selectors, or an explicit period representative, select a full
        // period as in StepToGeom's AdjustPeriodic path, never a guessed zero.
        if lifted_end <= base {
            turns += 1.0;
            lift = period_shift(turns, period)?;
            lifted_end = arithmetic(lift.nominal(end))?;
        }
        let length = lifted_end - base;
        if !length.is_finite() || length <= 0.0 || length > period.next_up() {
            return Err(ExactConversionError::UnsupportedPeriodicTrimSelection.into());
        }
        let lifted_bounds = arithmetic(lift.enclose(end))?;
        canonical_values = [base, end];
        canonical_maps = [normalize, arithmetic(normalize.compose(lift))?];
        if sense {
            selected = (first, lifted_end);
            selected_basis_bounds = [point(first)?, lifted_bounds];
            branch_shift = arithmetic(normalize.inverse())?;
        } else {
            let shift = period_shift(-turns, period)?;
            selected = (arithmetic(shift.nominal(second))?, first);
            selected_basis_bounds = [arithmetic(shift.enclose(second))?, point(first)?];
            branch_shift = arithmetic(shift.compose(arithmetic(normalize.inverse())?))?;
        }
    } else {
        if first == second {
            return Err(ExactConversionError::InvalidTrimParameterSelectors.into());
        }
        // Swapping selectors for F and SetTrim's nonperiodic sorting cancel:
        // the physical endpoints are trim1 then trim2 for either sense flag.
        source_reversed = first > second;
        selected = (first.min(second), first.max(second));
        canonical_values = [selected.0, selected.1];
        canonical_maps = [ParameterChart::identity(); 2];
        selected_basis_bounds = [point(selected.0)?, point(selected.1)?];
        branch_shift = ParameterChart::identity();
    }
    let mut internal_range = (
        basis.internal_selector(selected.0)?,
        basis.internal_selector(selected.1)?,
    );
    if internal_range.0 > internal_range.1 {
        internal_range = (internal_range.1, internal_range.0);
    }
    if internal_range.0 == internal_range.1 {
        return Err(ExactConversionError::UnrepresentableTrimParameterChart.into());
    }
    if basis.raw_domain.is_some()
        && (internal_range.0 < basis.internal_domain.0
            || internal_range.1 > basis.internal_domain.1)
    {
        return Err(ExactConversionError::UnrepresentableTrimParameterChart.into());
    }
    let mapped_basis = arithmetic(basis.raw_to_internal.compose(branch_shift))?;
    // Compose the periodic adjustment and family reversal before evaluating
    // endpoints. This retains coefficient uncertainty without losing exact
    // nominal cancellations such as TAU - (TAU - selector).
    let (to_basis, endpoint_maps, endpoint_values) = if source_reversed {
        (
            arithmetic(mapped_basis.compose(basis.reversal))?,
            [
                arithmetic(basis.reversal.compose(canonical_maps[1]))?,
                arithmetic(basis.reversal.compose(canonical_maps[0]))?,
            ],
            [canonical_values[1], canonical_values[0]],
        )
    } else {
        (mapped_basis, canonical_maps, canonical_values)
    };
    let raw_range = arithmetic(ParameterEnclosure::new(
        arithmetic(endpoint_maps[0].nominal(endpoint_values[0]))?,
        arithmetic(endpoint_maps[1].nominal(endpoint_values[1]))?,
    ))?;
    let raw_parameter_bounds = [
        arithmetic(endpoint_maps[0].enclose(endpoint_values[0]))?,
        arithmetic(endpoint_maps[1].enclose(endpoint_values[1]))?,
    ];
    let reversed = source_reversed != (basis.raw_to_internal.scale < 0.0);
    let raw_to_internal = if reversed {
        arithmetic(reflection(internal_range)?.compose(to_basis))?
    } else {
        to_basis
    };
    Ok(TrimSelection {
        internal_range,
        reversed,
        raw_to_internal,
        raw_range,
        raw_parameter_bounds,
        selected_basis_bounds,
        selected_basis_range: arithmetic(ParameterEnclosure::new(selected.0, selected.1))?,
    })
}
