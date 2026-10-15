use super::*;
use crate::step::{ParameterChart, SurfaceParameterChart};
use monstertruck_geometry::prelude::ExactTrimBoundary2D;
use std::result::Result;

type SourcePair = monstertruck_geometry::prelude::ParameterCurve<Box<Curve2D>, Box<Surface>>;

/// A source parameter curve with a frozen, admitted input-to-carrier chart.
///
/// `new` composes decoded UV with the decoded carrier. `try_from_raw` admits
/// actual STEP UV and caches its bounded chart before geometry can be evaluated.
/// The carrier is owned and cannot be mutated through a borrowed accessor.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StepParameterCurve {
    source: SourcePair,
    input_chart: SurfaceParameterChart,
}

impl StepParameterCurve {
    /// Composes UV already expressed in the decoded carrier's internal chart.
    pub fn new(curve: Box<Curve2D>, surface: Box<Surface>) -> Self {
        Self {
            source: SourcePair::new(curve, surface),
            input_chart: SurfaceParameterChart::identity(),
        }
    }

    /// Admits raw STEP UV, rejecting unavailable or unrepresentable charts.
    pub fn try_from_raw(
        curve: Box<Curve2D>,
        surface: Box<Surface>,
    ) -> Result<Self, ExactConversionError> {
        curve.raw_parameter_gauge()?;
        let input_chart = surface.raw_surface_parameter_chart()?;
        Ok(Self {
            source: SourcePair::new(curve, surface),
            input_chart,
        })
    }

    /// Reassembles explicitly charted source data without silently dropping its chart.
    pub fn try_from_parts(
        curve: Box<Curve2D>,
        surface: Box<Surface>,
        input_chart: SurfaceParameterChart,
    ) -> Result<Self, ExactConversionError> {
        if !input_chart.axes.iter().all(|axis| {
            ParameterChart::new(axis.scale, axis.offset, axis.offset_enclosure).is_some()
        }) {
            return Err(ExactConversionError::UnrepresentableTrimParameterChart);
        }
        Ok(Self {
            source: SourcePair::new(curve, surface),
            input_chart,
        })
    }

    /// Borrows stored UV in its input chart: raw for `try_from_raw`, decoded for `new`.
    pub fn curve(&self) -> &Curve2D { self.source.curve() }
    /// Borrows the frozen source carrier, whose geometry methods use decoded coordinates.
    pub fn surface(&self) -> &Surface { self.source.surface() }
    /// The admitted stored-UV-to-decoded-carrier map, not geometric authorization.
    pub fn input_parameter_chart(&self) -> SurfaceParameterChart { self.input_chart }
    /// Extracts all geometry and chart metadata; no bare pair loses the chart.
    pub fn decompose(self) -> (Box<Curve2D>, Box<Surface>, SurfaceParameterChart) {
        let (curve, surface) = self.source.decompose();
        (curve, surface, self.input_chart)
    }

    pub(crate) fn map_curve(self, map: impl FnOnce(Box<Curve2D>) -> Box<Curve2D>) -> Self {
        let (curve, surface) = self.source.decompose();
        Self {
            source: SourcePair::new(map(curve), surface),
            input_chart: self.input_chart,
        }
    }

    pub(crate) fn split_with(
        self,
        split: impl FnOnce(Curve2D) -> (Curve2D, Curve2D),
    ) -> (Self, Self) {
        let (curve, surface) = self.source.decompose();
        let (first, second) = split(*curve);
        (
            Self {
                source: SourcePair::new(Box::new(first), surface.clone()),
                input_chart: self.input_chart,
            },
            Self {
                source: SourcePair::new(Box::new(second), surface),
                input_chart: self.input_chart,
            },
        )
    }

    fn emitted_uv_chart(&self) -> Option<SurfaceParameterChart> {
        save::StepSurface::step_parameter_chart(self.surface())?.compose(self.input_chart)
    }

    pub(crate) fn input_affine_transform(&self) -> Matrix3 {
        let chart = self.input_chart;
        let jacobian = chart.jacobian();
        Matrix3::from_cols(
            Vector3::new(jacobian[0][0], jacobian[1][0], 0.0),
            Vector3::new(jacobian[0][1], jacobian[1][1], 0.0),
            Vector3::new(chart.axes[0].offset, chart.axes[1].offset, 1.0),
        )
    }
}

impl<'de> Deserialize<'de> for StepParameterCurve {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct ChartedSource {
            source: SourcePair,
            input_chart: SurfaceParameterChart,
        }
        let value = ChartedSource::deserialize(deserializer)?;
        let (curve, surface) = value.source.decompose();
        Self::try_from_parts(curve, surface, value.input_chart).map_err(serde::de::Error::custom)
    }
}

/// Borrowed adapter: evaluating a coedge never clones its carrier or UV data.
#[derive(Clone, Copy)]
struct DecodedUv<'a> {
    curve: &'a Curve2D,
    chart: SurfaceParameterChart,
    identity: bool,
}

impl DecodedUv<'_> {
    fn vector(&self, vector: Vector2) -> Vector2 {
        let vector = if self.chart.swapped {
            Vector2::new(vector.y, vector.x)
        } else {
            vector
        };
        Vector2::new(
            self.chart.axes[0].scale * vector.x,
            self.chart.axes[1].scale * vector.y,
        )
    }
    fn point(&self, point: Point2) -> Point2 {
        let vector = self.vector(point.to_vec());
        Point2::new(
            vector.x + self.chart.axes[0].offset,
            vector.y + self.chart.axes[1].offset,
        )
    }
}

impl ParametricCurve for DecodedUv<'_> {
    type Point = Point2;
    type Vector = Vector2;
    fn evaluate(&self, parameter: f64) -> Point2 {
        let point = self.curve.evaluate(parameter);
        if self.identity {
            point
        } else {
            self.point(point)
        }
    }
    fn derivative(&self, parameter: f64) -> Vector2 {
        let vector = self.curve.derivative(parameter);
        if self.identity {
            vector
        } else {
            self.vector(vector)
        }
    }
    fn derivative_2(&self, parameter: f64) -> Vector2 {
        let vector = self.curve.derivative_2(parameter);
        if self.identity {
            vector
        } else {
            self.vector(vector)
        }
    }
    fn derivative_n(&self, order: usize, parameter: f64) -> Vector2 {
        let vector = self.curve.derivative_n(order, parameter);
        if self.identity {
            vector
        } else if order == 0 {
            self.point(Point2::from_vec(vector)).to_vec()
        } else {
            self.vector(vector)
        }
    }
    fn derivatives(&self, order: usize, parameter: f64) -> CurveDerivatives<Vector2> {
        let mut derivatives = self.curve.derivatives(order, parameter);
        if !self.identity {
            derivatives[0] = self.point(Point2::from_vec(derivatives[0])).to_vec();
            for vector in &mut derivatives[1..] {
                *vector = self.vector(*vector);
            }
        }
        derivatives
    }
    fn parameter_range(&self) -> ParameterRange { self.curve.parameter_range() }
    fn period(&self) -> Option<f64> { self.curve.period() }
}

impl BoundedCurve for DecodedUv<'_> {}

impl ParameterDivision1D for DecodedUv<'_> {
    type Point = Point2;

    fn try_parameter_division(
        &self,
        range: (f64, f64),
        tolerance: f64,
    ) -> Option<(Vec<f64>, Vec<Point2>)> {
        let scale = self.chart.axes[0]
            .scale
            .abs()
            .max(self.chart.axes[1].scale.abs());
        let (parameters, mut points) = self
            .curve
            .try_parameter_division(range, tolerance / scale)?;
        if !self.identity {
            points
                .iter_mut()
                .for_each(|point| *point = self.point(*point));
        }
        Some((parameters, points))
    }

    fn parameter_division(&self, range: (f64, f64), tolerance: f64) -> (Vec<f64>, Vec<Point2>) {
        let scale = self.chart.axes[0]
            .scale
            .abs()
            .max(self.chart.axes[1].scale.abs());
        let (parameters, mut points) = self.curve.parameter_division(range, tolerance / scale);
        if !self.identity {
            points
                .iter_mut()
                .for_each(|point| *point = self.point(*point));
        }
        (parameters, points)
    }
}
impl SearchParameter<CurveParameter> for DecodedUv<'_> {
    type Point = Point2;
    fn search_parameter<H: Into<SearchParameterHint1D>>(
        &self,
        point: Point2,
        hint: H,
        trials: usize,
    ) -> Option<f64> {
        let point = if self.identity {
            point
        } else {
            let point = self.chart.inverse()?.nominal([point.x, point.y])?;
            Point2::new(point[0], point[1])
        };
        self.curve.search_parameter(point, hint, trials)
    }
}

fn decoded_parameter_curve(curve: &StepParameterCurve) -> ParameterCurve<DecodedUv<'_>, &Surface> {
    let chart = curve.input_chart;
    ParameterCurve::new(
        DecodedUv {
            curve: curve.curve(),
            chart,
            identity: !chart.swapped
                && chart
                    .axes
                    .iter()
                    .all(|axis| axis.scale == 1.0 && axis.offset == 0.0),
        },
        curve.surface(),
    )
}

impl ParametricCurve for StepParameterCurve {
    type Point = Point3;
    type Vector = Vector3;
    fn evaluate(&self, parameter: f64) -> Point3 {
        decoded_parameter_curve(self).evaluate(parameter)
    }
    fn derivative(&self, parameter: f64) -> Vector3 {
        decoded_parameter_curve(self).derivative(parameter)
    }
    fn derivative_2(&self, parameter: f64) -> Vector3 {
        decoded_parameter_curve(self).derivative_2(parameter)
    }
    fn derivative_n(&self, order: usize, parameter: f64) -> Vector3 {
        decoded_parameter_curve(self).derivative_n(order, parameter)
    }
    fn derivatives(&self, order: usize, parameter: f64) -> CurveDerivatives<Vector3> {
        decoded_parameter_curve(self).derivatives(order, parameter)
    }
    fn parameter_range(&self) -> ParameterRange { self.curve().parameter_range() }
    fn period(&self) -> Option<f64> { self.curve().period() }
}
impl BoundedCurve for StepParameterCurve {}
impl SearchParameter<CurveParameter> for StepParameterCurve {
    type Point = Point3;
    fn search_parameter<H: Into<SearchParameterHint1D>>(
        &self,
        point: Point3,
        hint: H,
        trials: usize,
    ) -> Option<f64> {
        decoded_parameter_curve(self).search_parameter(point, hint, trials)
    }
}
impl SearchNearestParameter<CurveParameter> for StepParameterCurve {
    type Point = Point3;
    fn search_nearest_parameter<H: Into<SearchParameterHint1D>>(
        &self,
        point: Point3,
        hint: H,
        trials: usize,
    ) -> Option<f64> {
        decoded_parameter_curve(self).search_nearest_parameter(point, hint, trials)
    }
}
impl ParameterDivision1D for StepParameterCurve {
    type Point = Point3;
    fn try_parameter_division(
        &self,
        range: (f64, f64),
        tol: f64,
    ) -> Option<(Vec<f64>, Vec<Point3>)> {
        decoded_parameter_curve(self).try_parameter_division(range, tol)
    }
    fn parameter_division(&self, range: (f64, f64), tol: f64) -> (Vec<f64>, Vec<Point3>) {
        decoded_parameter_curve(self).parameter_division(range, tol)
    }
}
impl ExactTrimBoundary2D for StepParameterCurve {
    fn exact_trim_boundary_2d(&self, tolerance: f64) -> Vec<Point2> {
        decoded_parameter_curve(self).exact_trim_boundary_2d(tolerance)
    }

    fn project_boundary_point(&self, point: Point3, hint: Option<f64>) -> Option<(f64, Point2)> {
        decoded_parameter_curve(self).project_boundary_point(point, hint)
    }
}
impl Cut for StepParameterCurve {
    fn cut(&mut self, parameter: f64) -> Self {
        Self {
            source: self.source.cut(parameter),
            input_chart: self.input_chart,
        }
    }
}
impl Invertible for StepParameterCurve {
    fn invert(&mut self) { self.source.invert(); }
}
impl Transformed<Matrix4> for StepParameterCurve {
    fn transform_by(&mut self, transform: Matrix4) { self.source.transform_by(transform); }
}
impl save::StepFormat for StepParameterCurve {
    fn fmt(&self, index: usize, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let chart = self.emitted_uv_chart().ok_or(std::fmt::Error)?;
        save::fmt_parameter_curve_with_chart(self.curve(), self.surface(), chart, index, formatter)
    }
}
impl save::StepLength for StepParameterCurve {
    fn step_length(&self) -> usize {
        let length = self
            .emitted_uv_chart()
            .and_then(|chart| {
                save::with_pcurve_step_curve(self.curve(), chart, save::StepLength::step_length)
            })
            .unwrap_or_else(|| save::StepLength::step_length(self.curve()));
        3 + length + save::StepLength::step_length(self.surface())
    }
}
impl save::StepCurve for StepParameterCurve {
    fn same_sense(&self) -> bool {
        self.emitted_uv_chart()
            .and_then(|chart| {
                save::with_pcurve_step_curve(self.curve(), chart, save::StepCurve::same_sense)
            })
            .unwrap_or(false)
    }
    fn step_parameter_chart(&self) -> Option<ParameterChart> {
        self.emitted_uv_chart()
            .and_then(|chart| {
                save::with_pcurve_step_curve(
                    self.curve(),
                    chart,
                    save::StepCurve::step_parameter_chart,
                )
            })
            .flatten()
    }
    fn step_reversed_parameter_chart(&self) -> Option<ParameterChart> {
        self.emitted_uv_chart()
            .and_then(|chart| {
                save::with_pcurve_step_curve(
                    self.curve(),
                    chart,
                    save::StepCurve::step_reversed_parameter_chart,
                )
            })
            .flatten()
    }
    fn step_parameter_period(&self) -> Option<f64> {
        self.emitted_uv_chart()
            .and_then(|chart| {
                save::with_pcurve_step_curve(
                    self.curve(),
                    chart,
                    save::StepCurve::step_parameter_period,
                )
            })
            .flatten()
    }
}

#[cfg(test)]
mod raw_pcurve_export_regressions {
    use super::*;
    use crate::step::ParameterEnclosure;
    use crate::step::load::{CurveAnyHolder, Table};
    use monstertruck_geometry::prelude::Sphere as GeometrySphere;
    use std::f64::consts::TAU;
    use step_p21::tables::EntityTable;

    #[test]
    fn cached_chart_samples_decoded_sphere_uv_and_refines_world_chords() {
        let axis = |scale, offset| {
            ParameterChart::new(
                scale,
                offset,
                ParameterEnclosure::point(offset).expect("finite chart offset"),
            )
            .expect("invertible chart axis")
        };
        let center = Point3::new(1.0, -2.0, 4.0);
        // Charts admit only orientation signs and translations, not arbitrary
        // scaling. The original UV line carries the full angular extent.
        let curve = StepParameterCurve::try_from_parts(
            Box::new(Curve2D::Line(Line(
                Point2::new(2.0, 3.0),
                Point2::new(2.0, 3.0 - TAU),
            ))),
            Box::new(Surface::ElementarySurface(ElementarySurface::Sphere(
                Processor::new(Sphere(GeometrySphere::new(center, 3.0))),
            ))),
            SurfaceParameterChart {
                axes: [axis(-1.0, 3.0), axis(-1.0, 2.0)],
                swapped: true,
            },
        )
        .expect("explicit admitted sphere chart");
        let world = |angle: f64| center + Vector3::new(3.0 * angle.cos(), 3.0 * angle.sin(), 0.0);
        let tolerance = 0.01;
        let samples = curve.exact_trim_boundary_2d(tolerance);
        // A straight stored UV line needs refinement only after lifting to the sphere.
        assert!(samples.len() > 8);
        assert!(
            samples
                .first()
                .expect("start UV")
                .near(&Point2::new(0.0, 0.0))
        );
        assert!(samples.last().expect("end UV").near(&Point2::new(TAU, 0.0)));
        samples.iter().for_each(|uv| {
            assert!(uv.y.abs() < 1.0e-10);
            assert!((0.0..=TAU).contains(&uv.x));
            let parameter = uv.x / TAU;
            let expected = world(uv.x);
            assert!(curve.evaluate(parameter).near(&expected));
            assert!(curve.surface().evaluate(uv.x, uv.y).near(&expected));
        });
        samples.windows(2).for_each(|pair| {
            assert!(pair[0].x < pair[1].x);
            let middle = world((pair[0].x + pair[1].x) * 0.5);
            let chord_middle = world(pair[0].x) + (world(pair[1].x) - world(pair[0].x)) * 0.5;
            assert!((middle - chord_middle).magnitude() <= tolerance + 1.0e-10);
        });
        for (parameter, hint) in [(0.0, None), (1.0, Some(1.0)), (0.37, Some(0.35))] {
            let (projected, uv) = curve
                .project_boundary_point(world(TAU * parameter), hint)
                .expect("world boundary point projects into decoded UV");
            assert!((projected - parameter).abs() < 1.0e-7);
            assert!(uv.near(&Point2::new(TAU * parameter, 0.0)));
        }
    }

    #[test]
    fn reflected_conic_uv_uses_the_emitted_scalar_chart_and_entity_count() {
        let conic = Curve2D::Conic(Conic2D::Ellipse(Processor::with_transform(
            TrimmedCurve::new(UnitCircle::new(), (0.0, std::f64::consts::TAU)),
            Matrix3::from_cols(
                Vector3::new(2.0, 0.0, 0.0),
                Vector3::new(0.0, 2.0, 0.0),
                Vector3::new(0.0, 0.0, 1.0),
            ),
        )));
        let uv = Curve2D::TrimmedCurve(SourceTrimmedCurve::internal(Processor::new(
            TrimmedCurve::new(Box::new(conic), (0.2, 1.4)),
        )));
        let input_chart = SurfaceParameterChart {
            axes: [ParameterChart::identity(); 2],
            swapped: true,
        };
        let pcurve = StepParameterCurve::try_from_parts(
            Box::new(uv),
            Box::new(Surface::ElementarySurface(ElementarySurface::Plane(
                Plane::xy(),
            ))),
            input_chart,
        )
        .expect("explicit reflected UV frame");
        let export_chart = save::StepCurve::step_parameter_chart(&pcurve)
            .expect("actual emitted reflected conic scalar chart");
        let emitted = format!("{}", save::StepDisplay::new(&pcurve, 100));
        let entities = emitted
            .lines()
            .filter(|line| line.trim_start().starts_with('#'))
            .count();
        assert_eq!(entities, save::StepLength::step_length(&pcurve));
        let table = Table::from_step(&format!(
            "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('reflected UV'),'2;1');\n\
             FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n{emitted}\nENDSEC;\nEND-ISO-10303-21;"
        )).expect("actual emitted entities parse");
        let owned = EntityTable::<CurveAnyHolder>::get_owned(&table, 100)
            .expect("all emitted PCURVE references resolve");
        let parsed = Curve3D::try_from(&owned).expect("checked emitted source admission");
        let input_chart = parsed
            .raw_parameter_chart()
            .expect("source canonical scalar chart")
            .expect("bounded affine source scalar chart");
        for parameter in [0.2, 0.7, 1.4] {
            let raw = export_chart
                .nominal(parameter)
                .expect("emitted scalar parameter");
            let decoded = input_chart
                .nominal(raw)
                .expect("decoded source scalar parameter");
            let expected = Point3::new(2.0 * parameter.sin(), 2.0 * parameter.cos(), 0.0);
            let derivative = Vector3::new(2.0 * parameter.cos(), -2.0 * parameter.sin(), 0.0);
            assert!(pcurve.subs(parameter).near(&expected));
            assert!(parsed.subs(decoded).near(&expected));
            let tangent = parsed.der(decoded) * (input_chart.scale * export_chart.scale);
            assert!(tangent.near(&derivative));
        }
    }
}
