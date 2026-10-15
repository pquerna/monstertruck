//! Decoded trim geometry with its separately retained raw STEP parameter gauge.

use super::*;
use crate::step::{ParameterChart, ParameterEnclosure};
use std::ops::Deref;

/// Raw STEP canonical coordinates are separate from the decoded curve chart.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum SourceParameterGauge {
    /// A synthetic restriction has no independent raw STEP declaration.
    Internal,
    /// A source restriction retains its original chart and selected basis span.
    Declared {
        /// Canonical raw STEP parameter to decoded internal parameter.
        raw_to_internal: ParameterChart,
        /// Nominal canonical active domain, checked before nested selections.
        raw_range: ParameterEnclosure,
        /// Selected interval in the basis curve's raw parameter chart.
        selected_basis_range: ParameterEnclosure,
        /// Canonical endpoint enclosures, retaining period and arithmetic error.
        raw_parameter_bounds: [ParameterEnclosure; 2],
        /// Selected basis endpoint enclosures, not just rounded nominal bounds.
        selected_basis_bounds: [ParameterEnclosure; 2],
    },
    /// A subsequent operation could not represent the chart arithmetic finitely.
    Unavailable,
}

/// A native-compatible trim and its original STEP gauge, without changing its basis.
///
/// Evaluation, differentiation and `parameter_range` use the decoded internal
/// chart. `raw_parameter_gauge` is the separate, bounded raw STEP correspondence;
/// neither its nominal offset nor a rounded period is an exact phase identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SourceTrimmedCurve<C, M> {
    curve: Processor<TrimmedCurve<Box<C>>, M>,
    gauge: SourceParameterGauge,
}

impl<C, M> SourceTrimmedCurve<C, M> {
    /// Constructs a synthetic restriction, not a new STEP source occurrence.
    pub fn internal(curve: Processor<TrimmedCurve<Box<C>>, M>) -> Self {
        Self {
            curve,
            gauge: SourceParameterGauge::Internal,
        }
    }

    pub(crate) fn declared(
        curve: Processor<TrimmedCurve<Box<C>>, M>,
        raw_to_internal: ParameterChart,
        raw_range: ParameterEnclosure,
        selected_basis_range: ParameterEnclosure,
        raw_parameter_bounds: [ParameterEnclosure; 2],
        selected_basis_bounds: [ParameterEnclosure; 2],
    ) -> Self {
        Self {
            curve,
            gauge: SourceParameterGauge::Declared {
                raw_to_internal,
                raw_range,
                selected_basis_range,
                raw_parameter_bounds,
                selected_basis_bounds,
            },
        }
    }

    /// Returns the raw source chart without asserting geometric correspondence.
    pub fn raw_parameter_gauge(
        &self,
    ) -> std::result::Result<SourceParameterGauge, ExactConversionError> {
        if self.gauge == SourceParameterGauge::Unavailable {
            Err(ExactConversionError::UnrepresentableTrimParameterChart)
        } else {
            Ok(self.gauge)
        }
    }

    pub(crate) fn split_forward(&mut self, parameter: f64) -> Self
    where Processor<TrimmedCurve<Box<C>>, M>: Cut {
        Self {
            curve: self.curve.cut(parameter),
            gauge: self.gauge,
        }
    }
}

impl<C, M> Deref for SourceTrimmedCurve<C, M> {
    type Target = Processor<TrimmedCurve<Box<C>>, M>;
    fn deref(&self) -> &Self::Target { &self.curve }
}

impl<C: Clone, M: Clone> ParametricCurve for SourceTrimmedCurve<C, M>
where Processor<TrimmedCurve<Box<C>>, M>: ParametricCurve
{
    type Point = <Processor<TrimmedCurve<Box<C>>, M> as ParametricCurve>::Point;
    type Vector = <Processor<TrimmedCurve<Box<C>>, M> as ParametricCurve>::Vector;
    fn evaluate(&self, parameter: f64) -> Self::Point { self.curve.evaluate(parameter) }
    fn derivative(&self, parameter: f64) -> Self::Vector { self.curve.derivative(parameter) }
    fn derivative_2(&self, parameter: f64) -> Self::Vector { self.curve.derivative_2(parameter) }
    fn derivative_n(&self, order: usize, parameter: f64) -> Self::Vector {
        self.curve.derivative_n(order, parameter)
    }
    fn derivatives(&self, order: usize, parameter: f64) -> CurveDerivatives<Self::Vector> {
        self.curve.derivatives(order, parameter)
    }
    fn parameter_range(&self) -> ParameterRange { self.curve.parameter_range() }
    // A finite STEP restriction is not periodic admission authority, even
    // though the native generic decorator can evaluate its periodic basis.
    fn period(&self) -> Option<f64> { None }
}

impl<C: Clone, M: Clone> BoundedCurve for SourceTrimmedCurve<C, M> where Processor<TrimmedCurve<Box<C>>, M>: BoundedCurve {}

impl<C: Clone, M: Clone> Invertible for SourceTrimmedCurve<C, M>
where Processor<TrimmedCurve<Box<C>>, M>: BoundedCurve + Invertible
{
    fn invert(&mut self) {
        if let SourceParameterGauge::Declared {
            raw_to_internal,
            raw_range,
            selected_basis_range,
            raw_parameter_bounds,
            selected_basis_bounds,
        } = self.gauge
        {
            let range = self.curve.range_tuple();
            let reflection = ParameterChart::reflection(range)
                .and_then(|reflection| reflection.compose(raw_to_internal));
            self.gauge = match reflection {
                Some(raw_to_internal) => SourceParameterGauge::Declared {
                    raw_to_internal,
                    raw_range,
                    selected_basis_range,
                    raw_parameter_bounds,
                    selected_basis_bounds,
                },
                None => SourceParameterGauge::Unavailable,
            };
        }
        self.curve.invert();
    }
}

impl<C: Clone, M: Clone> ParameterDivision1D for SourceTrimmedCurve<C, M>
where Processor<TrimmedCurve<Box<C>>, M>: ParameterDivision1D
{
    type Point = <Processor<TrimmedCurve<Box<C>>, M> as ParameterDivision1D>::Point;
    fn try_parameter_division(
        &self,
        range: (f64, f64),
        tolerance: f64,
    ) -> Option<(Vec<f64>, Vec<Self::Point>)> {
        self.curve.try_parameter_division(range, tolerance)
    }
    fn parameter_division(
        &self,
        range: (f64, f64),
        tolerance: f64,
    ) -> (Vec<f64>, Vec<Self::Point>) {
        self.curve.parameter_division(range, tolerance)
    }
}

impl<C: Clone, M: Clone> SearchParameter<CurveParameter> for SourceTrimmedCurve<C, M>
where Processor<TrimmedCurve<Box<C>>, M>: SearchParameter<CurveParameter>
{
    type Point = <Processor<TrimmedCurve<Box<C>>, M> as SearchParameter<CurveParameter>>::Point;
    fn search_parameter<H: Into<SearchParameterHint1D>>(
        &self,
        point: Self::Point,
        hint: H,
        trials: usize,
    ) -> Option<f64> {
        self.curve.search_parameter(point, hint, trials)
    }
}

impl<C: Clone, M: Clone> SearchNearestParameter<CurveParameter> for SourceTrimmedCurve<C, M>
where Processor<TrimmedCurve<Box<C>>, M>: SearchNearestParameter<CurveParameter>
{
    type Point =
        <Processor<TrimmedCurve<Box<C>>, M> as SearchNearestParameter<CurveParameter>>::Point;
    fn search_nearest_parameter<H: Into<SearchParameterHint1D>>(
        &self,
        point: Self::Point,
        hint: H,
        trials: usize,
    ) -> Option<f64> {
        self.curve.search_nearest_parameter(point, hint, trials)
    }
}

impl<C: Clone, M: Clone> Transformed<M> for SourceTrimmedCurve<C, M>
where Processor<TrimmedCurve<Box<C>>, M>: Transformed<M>
{
    fn transform_by(&mut self, transform: M) { self.curve.transform_by(transform); }
}

impl<C, M> save::StepFormat for SourceTrimmedCurve<C, M>
where Processor<TrimmedCurve<Box<C>>, M>: save::StepFormat
{
    fn fmt(&self, index: usize, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        save::StepFormat::fmt(&self.curve, index, formatter)
    }
}

impl<C, M> save::StepLength for SourceTrimmedCurve<C, M>
where Processor<TrimmedCurve<Box<C>>, M>: save::StepLength
{
    fn step_length(&self) -> usize { save::StepLength::step_length(&self.curve) }
}

impl<C, M> save::StepCurve for SourceTrimmedCurve<C, M>
where Processor<TrimmedCurve<Box<C>>, M>: save::StepCurve
{
    fn same_sense(&self) -> bool { save::StepCurve::same_sense(&self.curve) }
    fn step_parameter_chart(&self) -> Option<ParameterChart> {
        save::StepCurve::step_parameter_chart(&self.curve)
    }
    fn step_reversed_parameter_chart(&self) -> Option<ParameterChart> {
        save::StepCurve::step_reversed_parameter_chart(&self.curve)
    }
}

macro_rules! source_step_curve {
    ($ty:ident; $($variant:ident),+ $(,)?) => {
        impl save::StepCurve for $ty {
            fn same_sense(&self) -> bool {
                match self { $(Self::$variant(curve) => save::StepCurve::same_sense(curve),)+ }
            }
            fn step_parameter_chart(&self) -> Option<ParameterChart> {
                match self { $(Self::$variant(curve) => save::StepCurve::step_parameter_chart(curve),)+ }
            }
            fn step_reversed_parameter_chart(&self) -> Option<ParameterChart> {
                match self { $(Self::$variant(curve) => save::StepCurve::step_reversed_parameter_chart(curve),)+ }
            }
            fn step_parameter_period(&self) -> Option<f64> {
                match self { $(Self::$variant(curve) => save::StepCurve::step_parameter_period(curve),)+ }
            }
        }
    };
}

source_step_curve!(Conic2D; Ellipse, Hyperbola, Parabola);
source_step_curve!(Conic3D; Ellipse, Hyperbola, Parabola);
source_step_curve!(Curve2D; Line, Polyline, Conic, BsplineCurve, NurbsCurve, TrimmedCurve);
source_step_curve!(Curve3D; Line, Polyline, Conic, BsplineCurve, ParameterCurve, SurfaceCurve,
    IntersectionCurve, NurbsCurve, TrimmedCurve);
