//! Raw STEP charts are not the decoded geometry's internal coordinates.

use super::*;
use crate::step::{ParameterChart, ParameterEnclosure, SurfaceParameterChart};
use std::result::Result;

fn inherited(
    gauge: SourceParameterGauge,
    range: (f64, f64),
    forward: bool,
) -> Result<SourceParameterGauge, ExactConversionError> {
    if forward {
        return Ok(gauge);
    }
    let SourceParameterGauge::Declared {
        raw_to_internal,
        raw_range,
        selected_basis_range,
        raw_parameter_bounds,
        selected_basis_bounds,
    } = gauge
    else {
        return Ok(gauge);
    };
    let reflection = ParameterEnclosure::point(range.0)
        .and_then(|point| ParameterChart::new(-1.0, range.0, point))
        .and_then(|reverse| {
            ParameterEnclosure::point(-range.1)
                .and_then(|point| ParameterChart::translation(-range.1, point))
                .and_then(|shift| reverse.compose(shift))
        })
        .and_then(|reverse| reverse.compose(raw_to_internal))
        .ok_or(ExactConversionError::UnrepresentableTrimParameterChart)?;
    Ok(SourceParameterGauge::Declared {
        raw_to_internal: reflection,
        raw_range,
        selected_basis_range,
        raw_parameter_bounds,
        selected_basis_bounds,
    })
}

impl Curve2D {
    /// Returns the raw STEP gauge separately from this curve's decoded chart.
    /// Synthetic restrictions retain the declared ancestor's chart, not a new
    /// source occurrence or a claim of trim coverage.
    pub fn raw_parameter_gauge(&self) -> Result<SourceParameterGauge, ExactConversionError> {
        match self {
            Self::TrimmedCurve(curve) => match curve.raw_parameter_gauge()? {
                SourceParameterGauge::Internal => inherited(
                    curve.entity().curve().raw_parameter_gauge()?,
                    curve.range_tuple(),
                    curve.orientation(),
                ),
                gauge => Ok(gauge),
            },
            _ => Ok(SourceParameterGauge::Internal),
        }
    }

    /// Maps raw canonical STEP parameters to decoded internal coordinates.
    pub fn raw_parameter_chart(&self) -> Result<Option<ParameterChart>, ExactConversionError> {
        Ok(match self.raw_parameter_gauge()? {
            SourceParameterGauge::Declared {
                raw_to_internal, ..
            } => Some(raw_to_internal),
            _ => None,
        })
    }
}

impl Curve3D {
    /// Returns retained raw STEP gauge metadata, never an implicit phase identity.
    pub fn raw_parameter_gauge(&self) -> Result<SourceParameterGauge, ExactConversionError> {
        match self {
            Self::TrimmedCurve(curve) => match curve.raw_parameter_gauge()? {
                SourceParameterGauge::Internal => inherited(
                    curve.entity().curve().raw_parameter_gauge()?,
                    curve.range_tuple(),
                    curve.orientation(),
                ),
                gauge => Ok(gauge),
            },
            Self::SurfaceCurve(curve) => curve.leader().raw_parameter_gauge(),
            Self::ParameterCurve(curve) => curve.curve().raw_parameter_gauge(),
            _ => Ok(SourceParameterGauge::Internal),
        }
    }

    /// Maps raw canonical STEP parameters to the decoded internal curve chart.
    /// This coefficient enclosure is not a geometric correspondence certificate.
    pub fn raw_parameter_chart(&self) -> Result<Option<ParameterChart>, ExactConversionError> {
        Ok(match self.raw_parameter_gauge()? {
            SourceParameterGauge::Declared {
                raw_to_internal, ..
            } => Some(raw_to_internal),
            _ => None,
        })
    }
}

impl Surface {
    /// Maps raw STEP UV to this source surface's decoded internal UV.
    ///
    /// Swept profiles retain canonical raw STEP trim gauges independently of
    /// their native active domains. Nonzero offset enclosures must be carried
    /// into a bounded correspondence proof, not treated as exact phase shifts.
    pub fn raw_surface_parameter_chart(
        &self,
    ) -> Result<SurfaceParameterChart, ExactConversionError> {
        let mut chart = SurfaceParameterChart::identity();
        match self {
            Self::SweepSurface(SweepSurface::ExtrusionSurface(surface)) => {
                if let Some(profile) = surface.entity_curve().raw_parameter_chart()? {
                    chart.axes[0] = profile;
                }
            }
            Self::SweepSurface(SweepSurface::RevolutionSurface(surface)) => {
                if let Some(profile) = surface.entity().entity_curve().raw_parameter_chart()? {
                    chart.axes[usize::from(!surface.orientation())] = profile;
                }
            }
            _ => {}
        }
        Ok(chart)
    }
}
