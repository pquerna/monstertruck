//! Exact chart correspondence for STEP-to-modeling conversion.

use super::*;
use std::result::Result;

/// Exact STEP conversion refusals distinct from parse errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ExactConversionError {
    /// A compact exact curve conversion could not be represented safely.
    #[error(
        "STEP {family} curve has an unbounded, degenerate or unrepresentable conversion chart."
    )]
    CurveDomain { family: &'static str },
    /// Transforming a parameter curve requires a nonlinear composition.
    #[error("STEP parameter curve requires an exact nonlinear carrier-chart composition.")]
    NonlinearParameterCurveChart,
    /// A projected source curve lacks a validated native coordinate binding.
    #[error("STEP projected source curve requires validated native parameter correspondence.")]
    UnestablishedCurveCorrespondence,
    /// A periodic seam lacks one of its declared chart branches.
    #[error("STEP seam requires two explicit structurally matched pcurve branches.")]
    MissingSeamParameterBranch,
    /// Multiple stored trims cannot be assigned to one source occurrence.
    #[error("STEP source occurrence has ambiguous declared pcurve correspondence.")]
    AmbiguousStoredTrim,
    /// Cartesian-only or Cartesian-preferred trims lack an explicit parameter chart.
    #[error(
        "STEP trimmed curve requires declared PARAMETER_VALUE selectors; Cartesian trimming is unsupported."
    )]
    CartesianTrimSelectors,
    /// Endpoint parameter selectors are ambiguous, nonfinite or degenerate.
    #[error("STEP trimmed curve has ambiguous, nonfinite or degenerate parameter selectors.")]
    InvalidTrimParameterSelectors,
    /// Recursive source geometry exceeds the native geometry nesting cap.
    #[error("STEP source geometry exceeds the native geometry nesting limit.")]
    SourceGeometryNestingExceeded,
    /// A source geometry reference re-enters its active ancestor chain.
    #[error("STEP source geometry contains a cyclic reference to #{entity}.")]
    CyclicSourceGeometry { entity: u64 },
    /// Expanded source data exceeds the declared producer-profile resource policy.
    #[error(
        "STEP source expansion exceeds its resource profile ({expanded_bytes} bytes, {visits} visits)."
    )]
    SourceGeometryExpansionExceeded {
        expanded_bytes: usize,
        visits: usize,
    },
    /// A periodic branch exceeds finite, representable bounded arithmetic.
    #[error("STEP periodic trim selection has no representable finite branch chart.")]
    UnsupportedPeriodicTrimSelection,
    /// A source chart or a selected periodic branch has no finite enclosure.
    #[error("STEP trim parameter chart cannot be represented with finite bounded arithmetic.")]
    UnrepresentableTrimParameterChart,
}

/// The chart change made while converting a bounded STEP curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CurveParameterMapKind {
    /// The native parameter is the source parameter.
    Identity,
    /// The conversion has no established parameter correspondence.
    Unavailable { reason: &'static str },
}

/// Independent source-curve restriction, not a native convenience interval.
///
/// Carrier-surface and owning BREP restrictions must still be checked separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurveParameterDomain {
    /// Bare analytic primitive; its `0..1`/`0..TAU` native range is not authority.
    Unbounded,
    /// An explicit trim, polyline or spline declares a finite active interval.
    Bounded,
}

/// Decoded source-geometry and native intervals and their conversion formula.
///
/// `source_range` and `kind` refer to the already-decoded internal chart, not
/// raw ISO numeric selectors. `raw_step_gauge` retains that separate bounded
/// chart. Neither map alone certifies trim coverage or geometric equivalence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurveParameterMap {
    /// The already-decoded source geometry's internal parameter interval.
    pub source_range: (f64, f64),
    /// The converted native parameter interval.
    pub native_range: (f64, f64),
    /// The applied chart conversion.
    pub kind: CurveParameterMapKind,
    /// Raw STEP canonical coordinates to the decoded source chart, separately.
    pub raw_step_gauge: SourceParameterGauge,
    /// Whether the source independently declares finite parameter bounds.
    pub source_domain: CurveParameterDomain,
}

impl CurveParameterMap {
    /// Reports whether source and native interval orientations agree.
    pub fn orientation_preserved(&self) -> Option<bool> {
        match self.kind {
            CurveParameterMapKind::Unavailable { .. } => None,
            _ => Some(true),
        }
    }

    /// Evaluates the decoded formula, respecting independently declared bounds.
    pub fn source_to_native(&self, parameter: f64) -> Option<f64> {
        let (start, end) = self.source_range;
        if !parameter.is_finite()
            || (self.source_domain == CurveParameterDomain::Bounded
                && (parameter < start || parameter > end))
        {
            return None;
        }
        let value = match self.kind {
            CurveParameterMapKind::Identity => parameter,
            CurveParameterMapKind::Unavailable { .. } => return None,
        };
        value.is_finite().then_some(value)
    }

    /// Evaluates the native-to-source parameter formula within its interval.
    pub fn native_to_source(&self, parameter: f64) -> Option<f64> {
        let (start, end) = self.native_range;
        if !parameter.is_finite()
            || (self.source_domain == CurveParameterDomain::Bounded
                && (parameter < start || parameter > end))
        {
            return None;
        }
        let value = match self.kind {
            CurveParameterMapKind::Identity => parameter,
            CurveParameterMapKind::Unavailable { .. } => return None,
        };
        value.is_finite().then_some(value)
    }

    /// Returns the raw STEP-to-decoded coefficient map, including its enclosure.
    pub fn raw_step_parameter_chart(&self) -> Option<crate::step::ParameterChart> {
        match self.raw_step_gauge {
            SourceParameterGauge::Internal => Some(crate::step::ParameterChart::identity()),
            SourceParameterGauge::Declared {
                raw_to_internal, ..
            } => Some(raw_to_internal),
            SourceParameterGauge::Unavailable => None,
        }
    }

    /// Converts a raw STEP parameter nominally; not a coverage certificate.
    ///
    /// Retained endpoint enclosures permit coordinate evaluation despite
    /// nominal affine roundoff. They do not enlarge `raw_range` or establish
    /// active-domain membership; selector and coverage checks remain separate.
    pub fn raw_step_to_native(&self, parameter: f64) -> Option<f64> {
        if let SourceParameterGauge::Declared {
            raw_range,
            raw_parameter_bounds,
            ..
        } = self.raw_step_gauge
            && !raw_range.contains(parameter)
            && !raw_parameter_bounds
                .iter()
                .any(|bound| bound.contains(parameter))
        {
            return None;
        }
        let decoded = self.raw_step_parameter_chart()?.nominal(parameter)?;
        if matches!(self.raw_step_gauge, SourceParameterGauge::Declared { .. }) {
            // A nominal endpoint can round a fraction of an ulp outside the
            // decoded interval. Do not clamp or misreport a different chart:
            // the nominal interval and retained endpoint uncertainty above
            // remain separate from source selector/coverage authorization.
            match self.kind {
                CurveParameterMapKind::Identity => Some(decoded),
                CurveParameterMapKind::Unavailable { .. } => None,
            }
        } else {
            self.source_to_native(decoded)
        }
    }

    /// Converts the native internal chart back to the raw canonical gauge.
    pub fn native_to_raw_step(&self, parameter: f64) -> Option<f64> {
        let parameter = self.native_to_source(parameter)?;
        self.raw_step_parameter_chart()?
            .inverse()?
            .nominal(parameter)
    }
}

/// The exact source-to-modeling surface chart change.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SurfaceParameterMap {
    /// Both parameter axes are unchanged.
    Identity,
    /// An affine UV map in homogeneous two-dimensional coordinates.
    Affine(Matrix3),
    /// Only the extrusion profile parameter changes.
    Extrusion(CurveParameterMap),
    /// The profile parameter of a revolution changes, respecting axis inversion.
    Revolution {
        profile: CurveParameterMap,
        inverted: bool,
    },
    /// The conversion has no established surface chart correspondence.
    Unavailable { reason: &'static str },
}

impl SurfaceParameterMap {
    /// Evaluates the descriptive raw source-to-native coordinate formula.
    ///
    /// Profile axes accept their nominal interval and retained endpoint
    /// coefficient-error enclosures; unchanged axes may be unbounded. `Some`
    /// does not certify active-domain inclusion or trim coverage. No nearest
    /// projection, tolerance expansion or parameter clamping is performed.
    pub fn source_to_native(&self, parameter: (f64, f64)) -> Option<(f64, f64)> {
        if !parameter.0.is_finite() || !parameter.1.is_finite() {
            return None;
        }
        match self {
            Self::Identity => Some(parameter),
            Self::Affine(matrix) => {
                let point = matrix.transform_point(Point2::new(parameter.0, parameter.1));
                (point.x.is_finite() && point.y.is_finite()).then_some((point.x, point.y))
            }
            Self::Extrusion(profile) => {
                Some((profile.raw_step_to_native(parameter.0)?, parameter.1))
            }
            Self::Revolution {
                profile,
                inverted: false,
            } => Some((profile.raw_step_to_native(parameter.0)?, parameter.1)),
            Self::Revolution {
                profile,
                inverted: true,
            } => Some((parameter.0, profile.raw_step_to_native(parameter.1)?)),
            Self::Unavailable { .. } => None,
        }
    }

    /// Evaluates the native-to-source chart formula.
    pub fn native_to_source(&self, parameter: (f64, f64)) -> Option<(f64, f64)> {
        if !parameter.0.is_finite() || !parameter.1.is_finite() {
            return None;
        }
        match self {
            Self::Identity => Some(parameter),
            Self::Affine(matrix) => {
                let point = matrix
                    .invert()?
                    .transform_point(Point2::new(parameter.0, parameter.1));
                (point.x.is_finite() && point.y.is_finite()).then_some((point.x, point.y))
            }
            Self::Extrusion(profile) => {
                Some((profile.native_to_raw_step(parameter.0)?, parameter.1))
            }
            Self::Revolution {
                profile,
                inverted: false,
            } => Some((profile.native_to_raw_step(parameter.0)?, parameter.1)),
            Self::Revolution {
                profile,
                inverted: true,
            } => Some((parameter.0, profile.native_to_raw_step(parameter.1)?)),
            Self::Unavailable { .. } => None,
        }
    }

    /// Reports chart orientation parity, not material-side face orientation.
    pub fn orientation_preserved(&self) -> Option<bool> {
        match self {
            Self::Identity => Some(true),
            Self::Affine(matrix) => {
                let determinant = matrix.determinant();
                (determinant.is_finite() && determinant != 0.0).then_some(determinant > 0.0)
            }
            Self::Extrusion(profile) | Self::Revolution { profile, .. } => Some(
                profile.orientation_preserved()?
                    == (profile.raw_step_parameter_chart()?.scale > 0.0),
            ),
            Self::Unavailable { .. } => None,
        }
    }

    /// Returns the nominal affine UV map; coefficient enclosures remain in metadata.
    pub fn affine_transform(&self) -> Option<Matrix3> {
        match self {
            Self::Identity => Some(Matrix3::identity()),
            Self::Affine(matrix) => Some(*matrix),
            Self::Extrusion(profile)
            | Self::Revolution {
                profile,
                inverted: false,
            } if profile.kind == CurveParameterMapKind::Identity => {
                let chart = profile.raw_step_parameter_chart()?;
                Some(Matrix3::from_cols(
                    Vector3::new(chart.scale, 0.0, 0.0),
                    Vector3::new(0.0, 1.0, 0.0),
                    Vector3::new(chart.offset, 0.0, 1.0),
                ))
            }
            Self::Revolution {
                profile,
                inverted: true,
            } if profile.kind == CurveParameterMapKind::Identity => {
                let chart = profile.raw_step_parameter_chart()?;
                Some(Matrix3::from_cols(
                    Vector3::new(1.0, 0.0, 0.0),
                    Vector3::new(0.0, chart.scale, 0.0),
                    Vector3::new(0.0, chart.offset, 1.0),
                ))
            }
            _ => None,
        }
    }

    /// Decoded source-carrier UV to native UV, excluding the raw input stage.
    ///
    /// Compose this with an admitted pcurve input chart exactly once. The full
    /// `affine_transform` instead starts from raw STEP surface coordinates.
    pub fn internal_affine_transform(&self) -> Option<Matrix3> {
        match self {
            Self::Identity => Some(Matrix3::identity()),
            Self::Affine(matrix) => Some(*matrix),
            Self::Extrusion(profile) | Self::Revolution { profile, .. }
                if profile.kind == CurveParameterMapKind::Identity =>
            {
                Some(Matrix3::identity())
            }
            _ => None,
        }
    }
}

impl Curve3D {
    /// Retains only a declared source pcurve with structural carrier correspondence.
    ///
    /// `orientation` is the original oriented coedge use. Periodic seam
    /// branches are selected explicitly and missing/ambiguous branches refuse.
    /// An absent declared pcurve is not silently replaced by sampled recovery.
    pub fn declared_parameter_curve_on(
        &self,
        surface: &Surface,
        orientation: bool,
    ) -> Result<Option<StepParameterCurve>, ExactConversionError> {
        let selected = match self {
            Self::SurfaceCurve(curve) => {
                curve.declared_parameter_curve_ref_on(surface, orientation)?
            }
            Self::ParameterCurve(curve) if curve.surface() == surface => Some(curve),
            _ => None,
        };
        Ok(selected.map(|curve| {
            let mut curve = curve.clone();
            if !orientation {
                curve.invert();
            }
            curve
        }))
    }

    /// Converts geometry and reports the source/native chart correspondence.
    ///
    /// Implicit projected curves report unavailable correspondence rather than
    /// declaring their sampled leader an exact geometric certificate.
    /// For an explicit `SURFACE_CURVE`, the map describes the retained native
    /// leader function only. Original kind, master and PCURVE associations stay
    /// on the source object; their geometric equivalence needs a separate
    /// whole-domain incidence/correspondence certificate.
    pub fn to_modeling_with_parameter_map(
        &self,
    ) -> Result<(ModelingCurve, CurveParameterMap), StepConvertingError> {
        let curve = ModelingCurve::try_from(self)?;
        let map = self.parameter_map_for(curve.range_tuple())?;
        Ok((curve, map))
    }

    fn source_parameter_domain(&self) -> CurveParameterDomain {
        match self {
            Self::Line(_) | Self::Conic(_) => CurveParameterDomain::Unbounded,
            Self::SurfaceCurve(curve) => curve.leader().source_parameter_domain(),
            Self::IntersectionCurve(curve) => curve.leader().source_parameter_domain(),
            Self::ParameterCurve(curve) => match curve.curve() {
                Curve2D::Line(_) | Curve2D::Conic(_) => CurveParameterDomain::Unbounded,
                _ => CurveParameterDomain::Bounded,
            },
            _ => CurveParameterDomain::Bounded,
        }
    }

    fn parameter_map_for(
        &self,
        native_range: (f64, f64),
    ) -> Result<CurveParameterMap, StepConvertingError> {
        let source_range = self.range_tuple();
        let kind = match self {
            Self::TrimmedCurve(curve) => {
                curve.entity().curve().parameter_map_for(native_range)?.kind
            }
            Self::SurfaceCurve(curve) => {
                if curve.kind() == SurfaceCurveKind::IntersectionCurve {
                    CurveParameterMapKind::Unavailable {
                        reason: "Projected implicit curve equivalence requires validated continuation.",
                    }
                } else {
                    curve.leader().parameter_map_for(native_range)?.kind
                }
            }
            Self::IntersectionCurve(_) => CurveParameterMapKind::Unavailable {
                reason: "Projected implicit curve equivalence requires validated continuation.",
            },
            _ => CurveParameterMapKind::Identity,
        };
        Ok(CurveParameterMap {
            source_range,
            native_range,
            kind,
            raw_step_gauge: self.raw_parameter_gauge()?,
            source_domain: self.source_parameter_domain(),
        })
    }
}

impl Surface {
    /// Converts geometry and reports its exact source/native chart change.
    ///
    /// This map does not certify face trims or singular point regularity.
    pub fn to_modeling_with_parameter_map(
        &self,
    ) -> Result<(ModelingSurface, SurfaceParameterMap), StepConvertingError> {
        let surface = ModelingSurface::try_from(self)?;
        let map = match self {
            Self::ElementarySurface(ElementarySurface::Sphere(sphere))
                if matches!(&surface, ModelingSurface::SphericalSurface(_)) =>
            {
                let matrix = if sphere.orientation() {
                    Matrix3::from_cols(
                        Vector3::new(0.0, 1.0, 0.0),
                        Vector3::new(-1.0, 0.0, 0.0),
                        Vector3::new(std::f64::consts::FRAC_PI_2, 0.0, 1.0),
                    )
                } else {
                    Matrix3::from_cols(
                        Vector3::new(0.0, -1.0, 0.0),
                        Vector3::new(1.0, 0.0, 0.0),
                        Vector3::new(0.0, std::f64::consts::FRAC_PI_2, 1.0),
                    )
                };
                SurfaceParameterMap::Affine(matrix)
            }
            Self::SweepSurface(SweepSurface::ExtrusionSurface(extrusion)) => {
                let native_range = match &surface {
                    ModelingSurface::ExtrusionSurface(converted) => {
                        converted.entity_curve().range_tuple()
                    }
                    _ => {
                        return Ok((
                            surface,
                            SurfaceParameterMap::Unavailable {
                                reason: "Extrusion did not retain its original analytic profile.",
                            },
                        ));
                    }
                };
                SurfaceParameterMap::Extrusion(
                    extrusion.entity_curve().parameter_map_for(native_range)?,
                )
            }
            Self::SweepSurface(SweepSurface::RevolutionSurface(revolution)) => {
                let native_range = match &surface {
                    ModelingSurface::RevolutionSurface(converted) => {
                        converted.entity().entity_curve().range_tuple()
                    }
                    _ => {
                        return Ok((
                            surface,
                            SurfaceParameterMap::Unavailable {
                                reason: "Revolution did not retain its analytic profile chart.",
                            },
                        ));
                    }
                };
                let profile = revolution
                    .entity()
                    .entity_curve()
                    .parameter_map_for(native_range)?;
                SurfaceParameterMap::Revolution {
                    profile,
                    inverted: !revolution.orientation(),
                }
            }
            Self::BsplineSurface(_)
            | Self::NurbsSurface(_)
            | Self::ElementarySurface(
                ElementarySurface::Plane(_)
                | ElementarySurface::CylindricalSurface(_)
                | ElementarySurface::ConicalSurface(_)
                | ElementarySurface::ToroidalSurface(_),
            ) => SurfaceParameterMap::Identity,
            _ => SurfaceParameterMap::Unavailable {
                reason: "Analytic fallback rationalization has no established source chart map.",
            },
        };
        Ok((surface, map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_curve_correspondence(source: &Curve3D) {
        let (native, map) = source
            .to_modeling_with_parameter_map()
            .expect("exact compact curve");
        assert_eq!(map.orientation_preserved(), Some(true));
        assert_eq!(map.kind, CurveParameterMapKind::Identity);
        assert_eq!(map.source_range, map.native_range);
        assert!(map.source_to_native(f64::NAN).is_none());
        assert!(map.native_to_source(f64::INFINITY).is_none());
        let (start, end) = map.source_range;
        (0..=32).for_each(|index| {
            let source_parameter = start + (end - start) * index as f64 / 32.0;
            let native_parameter = map
                .source_to_native(source_parameter)
                .expect("covered source parameter");
            let roundtrip = map
                .native_to_source(native_parameter)
                .expect("covered native parameter");
            assert!((roundtrip - source_parameter).abs() < 1.0e-12);
            assert!(
                native
                    .subs(native_parameter)
                    .distance(source.subs(source_parameter))
                    < 1.0e-11
            );
        });
    }

    #[test]
    fn compact_native_conics_preserve_geometry_and_parameters() {
        let transform = Matrix4::from_translation(Vector3::new(2.0, -3.0, 5.0))
            * Matrix4::from_nonuniform_scale(2.0, 3.0, 1.0);
        let mut hyperbola = Processor::with_transform(
            TrimmedCurve::new(UnitHyperbola::<Point3>::new(), (-1.5, 2.0)),
            transform,
        );
        let mut parabola = Processor::with_transform(
            TrimmedCurve::new(UnitParabola::<Point3>::new(), (-2.0, 3.0)),
            transform,
        );
        assert_curve_correspondence(&Curve3D::Conic(Conic3D::Hyperbola(hyperbola)));
        assert_curve_correspondence(&Curve3D::Conic(Conic3D::Parabola(parabola)));
        hyperbola.invert();
        parabola.invert();
        assert_curve_correspondence(&Curve3D::Conic(Conic3D::Hyperbola(hyperbola)));
        assert_curve_correspondence(&Curve3D::Conic(Conic3D::Parabola(parabola)));
    }

    #[test]
    fn complete_circle_retains_original_analytic_parameter_and_orientation() {
        let mut circle = Processor::<_, Matrix4>::new(TrimmedCurve::new(
            UnitCircle::<Point3>::new(),
            (0.0, std::f64::consts::TAU),
        ));
        let source = Curve3D::Conic(Conic3D::Ellipse(circle));
        let net = source
            .try_into_homogeneous_bspline_curve()
            .expect("compact rational utility");
        assert!(net.control_points().iter().all(|point| point.w > 0.0));
        assert_eq!(net.degree(), 2);
        assert_curve_correspondence(&source);
        circle.invert();
        assert_curve_correspondence(&Curve3D::Conic(Conic3D::Ellipse(circle)));
    }

    #[test]
    fn native_polyline_preserves_every_corner_and_segment_parameter() {
        let source = Curve3D::Polyline(PolylineCurve(vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 2.0, -1.0),
            Point3::new(3.0, -2.0, 1.0),
            Point3::new(4.0, 0.0, 0.0),
        ]));
        assert_curve_correspondence(&source);
    }

    #[test]
    fn finite_hyperbola_does_not_require_a_saturating_half_angle_chart() {
        let source = Curve3D::Conic(Conic3D::Hyperbola(Processor::<_, Matrix4>::new(
            TrimmedCurve::new(UnitHyperbola::<Point3>::new(), (40.0, 41.0)),
        )));
        assert_curve_correspondence(&source);
        assert!(source.try_into_homogeneous_bspline_curve().is_none());
    }

    #[test]
    fn raw_translated_conic_frame_preserves_original_axes_and_derivatives() {
        let x = Vector3::new(1.0, 0.0, 0.0);
        let y = Vector3::new(0.0, 3.0, 0.0);
        let origin = Point3::new(1.0e16, -1.0e16, 7.0);
        let transform = Matrix4::from_cols(
            x.extend(0.0),
            y.extend(0.0),
            Vector3::unit_z().extend(0.0),
            origin.to_homogeneous(),
        );
        let source = Curve3D::Conic(Conic3D::Parabola(Processor::with_transform(
            TrimmedCurve::new(UnitParabola::<Point3>::new(), (-1.0, 2.0)),
            transform,
        )));
        let (native, map) = source
            .to_modeling_with_parameter_map()
            .expect("raw source frame");
        assert_eq!(map.kind, CurveParameterMapKind::Identity);
        let ModelingCurve::ParameterCurve(curve) = &native else {
            panic!("analytic source conic");
        };
        let ModelingSurface::Plane(plane) = curve.surface().as_ref() else {
            panic!("affine embedding");
        };
        assert_eq!(plane.origin(), origin);
        assert_eq!(plane.stored_axes(), Some([x, y]));
        assert_eq!(native.derivative(0.0), Vector3::new(0.0, 6.0, 0.0));
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let mut budget = monstertruck_modeling::ValidatedGeometryBudget::new(100_000);
        let jet =
            monstertruck_modeling::validated_curve_jet(&native, [0.0, 0.0], &mut budget, &cancel)
                .expect("raw coefficient jet");
        assert!(jet.first[1].lower <= 6.0 && jet.first[1].upper >= 6.0);
        assert_eq!(jet.first[0].lower, 0.0);
        assert_eq!(jet.first[0].upper, 0.0);
        assert!(source.to_modeling_with_parameter_map().is_ok());
    }

    #[test]
    fn source_ellipse_extrusion_retains_exact_profile_chart_and_unclamped_extent() {
        let profile = Curve3D::Conic(Conic3D::Ellipse(Processor::<_, Matrix4>::new(
            TrimmedCurve::new(UnitCircle::<Point3>::new(), (0.0, std::f64::consts::TAU)),
        )));
        let source = Surface::SweepSurface(SweepSurface::ExtrusionSurface(
            ExtrusionSurface::by_extrusion(Box::new(profile), Vector3::new(0.0, 0.0, 7.0)),
        ));
        let (native, map) = source
            .to_modeling_with_parameter_map()
            .expect("native exact extrusion");
        let ModelingSurface::ExtrusionSurface(extrusion) = &native else {
            panic!("native extrusion");
        };
        assert!(matches!(
            extrusion.entity_curve(),
            ModelingCurve::ParameterCurve(_)
        ));
        assert_eq!(extrusion.extruding_vector(), Vector3::new(0.0, 0.0, 7.0));
        for uv in [(0.0, -3.0), (0.37, 9.0), (std::f64::consts::TAU, 12.0)] {
            assert_eq!(map.source_to_native(uv), Some(uv));
            assert_eq!(native.evaluate(uv.0, uv.1), source.evaluate(uv.0, uv.1));
        }
    }

    #[test]
    fn sphere_affine_chart_conjugates_processor_inversion() {
        let mut sphere = Processor::<_, Matrix4>::new(Sphere(
            monstertruck_geometry::prelude::Sphere::new(Point3::new(1.0, 2.0, 3.0), 4.0),
        ));
        [false, true].into_iter().for_each(|invert| {
            if invert {
                sphere.invert();
            }
            let source = Surface::ElementarySurface(ElementarySurface::Sphere(sphere));
            let (native, map) = source
                .to_modeling_with_parameter_map()
                .expect("analytic sphere");
            assert_eq!(map.orientation_preserved(), Some(true));
            let source_uv = (0.37, 0.73);
            let native_uv = map
                .source_to_native(source_uv)
                .expect("sphere affine chart");
            assert!(
                source
                    .subs(source_uv.0, source_uv.1)
                    .distance(native.subs(native_uv.0, native_uv.1))
                    < 1.0e-12
            );
            let roundtrip = map
                .native_to_source(native_uv)
                .expect("inverse sphere chart");
            assert!((roundtrip.0 - source_uv.0).abs() < 1.0e-12);
            assert!((roundtrip.1 - source_uv.1).abs() < 1.0e-12);
        });
    }
}

#[cfg(test)]
mod raw_endpoint_coordinate_regressions {
    use super::*;
    use crate::step::{ParameterChart, ParameterEnclosure};

    #[test]
    fn endpoint_roundoff_coordinate_evaluation_does_not_expand_authoritative_domains() {
        let lower = 0.2f64;
        let upper = 5.7f64;
        let domain = ParameterEnclosure::new(lower, upper).unwrap();
        let map = CurveParameterMap {
            source_range: (lower, upper),
            native_range: (lower, upper),
            kind: CurveParameterMapKind::Identity,
            source_domain: CurveParameterDomain::Bounded,
            raw_step_gauge: SourceParameterGauge::Declared {
                raw_to_internal: ParameterChart::identity(),
                raw_range: domain,
                selected_basis_range: domain,
                raw_parameter_bounds: [
                    ParameterEnclosure::new(lower.next_down(), lower.next_up()).unwrap(),
                    ParameterEnclosure::point(upper).unwrap(),
                ],
                selected_basis_bounds: [
                    ParameterEnclosure::point(lower).unwrap(),
                    ParameterEnclosure::point(upper).unwrap(),
                ],
            },
        };
        let rounded = lower.next_down();
        assert!(!domain.contains(rounded));
        assert_eq!(map.source_to_native(rounded), None);
        assert_eq!(map.raw_step_to_native(rounded), Some(rounded));
        assert_eq!(map.raw_step_to_native(lower - 0.001), None);
        assert_eq!(map.source_range, (lower, upper));
        assert_eq!(map.native_range, (lower, upper));
        assert_eq!(
            SurfaceParameterMap::Extrusion(map).source_to_native((rounded, -3.0)),
            Some((rounded, -3.0))
        );
    }
}
