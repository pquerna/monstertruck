//! The 3-D [`Surface`] enum and every impl that belongs to it.

use super::parameter_curves::same_surface;
use super::*;

/// 3-dimensional surfaces
#[derive(
    Clone,
    Debug,
    Serialize,
    Deserialize,
    From,
    ParametricSurface,
    ParameterDivision2D,
    Invertible,
    SearchParameterD2,
)]
#[allow(clippy::large_enum_variant)]
pub enum Surface {
    /// Plane
    Plane(Plane),
    /// 3-dimensional B-spline surface
    BsplineSurface(BsplineSurface<Point3>),
    /// 3-dimensional NURBS Surface
    NurbsSurface(NurbsSurface<Vector4>),
    /// revoluted curve
    #[serde(alias = "RevolutedCurve")]
    RevolutionSurface(Processor<RevolutionSurface<Curve>, Matrix4>),
    /// Exact linear extrusion of its original profile.
    ExtrusionSurface(ExtrusionSurface<Curve, Vector3>),
    /// T-spline surface
    TsplineSurface(Tmesh<Point3>),
    /// Analytic sphere, posed by the processor's transform.
    ///
    /// Carried as the analytic type rather than its (exact) rational net so the
    /// CLOSED-FORM [`ParameterDivision2D`] survives the STEP conversion -- see
    /// the module note on `TryFrom<&Surface> for Surface` in
    /// `monstertruck-io/src/step/load/step_geometry/geom_impls/`. Spec 012 U1.2.
    SphericalSurface(Processor<Sphere, Matrix4>),
    /// Analytic torus, posed by the processor's transform. Same reason as
    /// [`Surface::SphericalSurface`]; spindle tori never reach this variant.
    ToroidalSurface(Processor<Torus, Matrix4>),
}

macro_rules! derive_surface_method {
    ($surface: expr, $method: expr, $($ver: ident),*) => {
        match $surface {
            Self::Plane(got) => $method(got, $($ver), *),
            Self::BsplineSurface(got) => $method(got, $($ver), *),
            Self::NurbsSurface(got) => $method(got, $($ver), *),
            Self::RevolutionSurface(got) => $method(got, $($ver), *),
            Self::ExtrusionSurface(got) => $method(got, $($ver), *),
            Self::TsplineSurface(got) => $method(got, $($ver), *),
            Self::SphericalSurface(got) => $method(got, $($ver), *),
            Self::ToroidalSurface(got) => $method(got, $($ver), *),
        }
    };
}

macro_rules! derive_surface_self_method {
    ($surface: expr, $method: expr, $($ver: ident),*) => {
        match $surface {
            Self::Plane(got) => Self::Plane($method(got, $($ver), *)),
            Self::BsplineSurface(got) => Self::BsplineSurface($method(got, $($ver), *)),
            Self::NurbsSurface(got) => Self::NurbsSurface($method(got, $($ver), *)),
            Self::RevolutionSurface(got) => Self::RevolutionSurface($method(got, $($ver), *)),
            Self::ExtrusionSurface(got) => Self::ExtrusionSurface($method(got, $($ver), *)),
            Self::TsplineSurface(got) => Self::TsplineSurface($method(got, $($ver), *)),
            Self::SphericalSurface(got) => Self::SphericalSurface($method(got, $($ver), *)),
            Self::ToroidalSurface(got) => Self::ToroidalSurface($method(got, $($ver), *)),
        }
    };
}

impl ParametricSurface3D for Surface {
    #[inline(always)]
    fn normal(&self, u: f64, v: f64) -> Vector3 {
        derive_surface_method!(self, ParametricSurface3D::normal, u, v)
    }
}

/// Trials the analytic containment test allows its inverse map, matching
/// `INCLUDE_CURVE_TRIALS` in `monstertruck-geometry`'s `NurbsSurface` impl --
/// the one [`Surface::SphericalSurface`] and [`Surface::ToroidalSurface`]
/// faces reached before spec 012 U1.2 gave them analytic variants.
const INCLUDE_CURVE_TRIALS: usize = 100;

/// Whether every sample of `curve` recovers a parameter on `surface`.
///
/// Samples the CURVE at the same density the `NurbsSurface` impl this replaces
/// used -- interior points of each knot span, `2 * degree` of them -- rather
/// than the control polygon, whose points are not on a rational curve at all.
fn sampled_surface_includes_curve<S>(surface: &S, curve: &Curve) -> bool
where S: ParametricSurface3D + SearchParameter<SurfaceParameter, Point = Point3> {
    let include_span = |start: f64, end: f64, density: usize| {
        start.is_finite()
            && end.is_finite()
            && start < end
            && (0..=density).all(|index| {
                let ratio = index as f64 / density as f64;
                let point = curve.evaluate(start * (1.0 - ratio) + end * ratio);
                point.x.is_finite()
                    && point.y.is_finite()
                    && point.z.is_finite()
                    && surface
                        .search_parameter(point, None, INCLUDE_CURVE_TRIALS)
                        .is_some_and(|(u, v)| surface.evaluate(u, v).near(&point))
            })
    };
    let include_knots = |knots: &KnotVector, degree: usize| {
        let mut found_span = false;
        for span in knots.windows(2) {
            if span[0] == span[1] {
                continue;
            }
            found_span = true;
            if !include_span(span[0], span[1], usize::max(degree * 2, 2)) {
                return false;
            }
        }
        found_span
    };
    // The leader supplies only the parameter partition. Its chord geometry is
    // not the projected intersection curve and must never be the witness.
    let mut partition = curve;
    while let Curve::IntersectionCurve(intersection) = partition {
        partition = intersection.leader().as_ref();
    }
    match partition {
        Curve::Line(line) => {
            let (start, end) = line.range_tuple();
            include_span(start, end, 2)
        }
        Curve::BsplineCurve(spline) => include_knots(spline.knot_vector(), spline.degree()),
        Curve::NurbsCurve(spline) => include_knots(spline.knot_vector(), spline.degree()),
        Curve::TrimmedCurve(trimmed) => {
            let (start, end) = trimmed.range_tuple();
            include_span(start, end, 8)
        }
        Curve::ParameterCurve(parameter) => match parameter.curve() {
            Curve2D::Line(line) => {
                let (start, end) = line.range_tuple();
                include_span(start, end, 2)
            }
            Curve2D::Polyline(polyline) => {
                polyline.len() > 1
                    && (0..polyline.len() - 1)
                        .all(|index| include_span(index as f64, (index + 1) as f64, 2))
            }
            Curve2D::Conic(conic) => {
                let (start, end) = conic.range_tuple();
                include_span(start, end, 4)
            }
            Curve2D::TrimmedCurve(trimmed) => {
                let (start, end) = trimmed.range_tuple();
                include_span(start, end, 8)
            }
            Curve2D::BsplineCurve(spline) => include_knots(spline.knot_vector(), spline.degree()),
            Curve2D::NurbsCurve(spline) => include_knots(spline.knot_vector(), spline.degree()),
        },
        Curve::IntersectionCurve(_) => unreachable!(),
    }
}

impl Transformed<Matrix4> for Surface {
    fn transform_by(&mut self, trans: Matrix4) {
        derive_surface_method!(self, Transformed::transform_by, trans);
    }
    fn transformed(&self, trans: Matrix4) -> Self {
        derive_surface_self_method!(self, Transformed::transformed, trans)
    }
}

impl IncludeCurve<Curve> for Surface {
    #[inline(always)]
    fn include(&self, curve: &Curve) -> bool {
        if let Curve::ParameterCurve(parameter) = curve {
            same_surface(parameter.surface().as_ref(), self)
                || sampled_surface_includes_curve(self, curve)
        } else if matches!(curve, Curve::IntersectionCurve(_) | Curve::TrimmedCurve(_)) {
            sampled_surface_includes_curve(self, curve)
        } else {
            match self {
                Surface::BsplineSurface(surface) => match curve {
                    &Curve::Line(curve) => surface.include(&BsplineCurve::from(curve)),
                    Curve::BsplineCurve(curve) => surface.include(curve),
                    Curve::NurbsCurve(curve) => surface.include(curve),
                    Curve::IntersectionCurve(_)
                    | Curve::ParameterCurve(_)
                    | Curve::TrimmedCurve(_) => unreachable!(),
                },
                Surface::NurbsSurface(surface) => match curve {
                    &Curve::Line(curve) => surface.include(&BsplineCurve::from(curve)),
                    Curve::BsplineCurve(curve) => surface.include(curve),
                    Curve::NurbsCurve(curve) => surface.include(curve),
                    Curve::IntersectionCurve(_)
                    | Curve::ParameterCurve(_)
                    | Curve::TrimmedCurve(_) => unreachable!(),
                },
                Surface::Plane(surface) => match curve {
                    &Curve::Line(curve) => surface.include(&BsplineCurve::from(curve)),
                    Curve::BsplineCurve(curve) => surface.include(curve),
                    Curve::NurbsCurve(curve) => surface.include(curve),
                    Curve::IntersectionCurve(_)
                    | Curve::ParameterCurve(_)
                    | Curve::TrimmedCurve(_) => unreachable!(),
                },
                Surface::TsplineSurface(surface) => {
                    curve.lift_up().control_points().iter().all(|v| {
                        let p = v.to_point();
                        surface.search_parameter(p, None, 1).is_some()
                    })
                }
                // The analytic variants answer containment through their own
                // exact inverse map. `Torus` carries no `IncludeCurve` impl at
                // all, so the pair shares one sampled test -- and it samples
                // the CURVE, like the `NurbsSurface` impl these faces used to
                // reach, not the control polygon like the T-mesh arm above. A
                // rational curve's control points are not on the curve, so
                // testing them would have been strictly more conservative than
                // what this replaces.
                Surface::SphericalSurface(surface) => {
                    sampled_surface_includes_curve(surface, curve)
                }
                Surface::ToroidalSurface(surface) => sampled_surface_includes_curve(surface, curve),
                Surface::ExtrusionSurface(surface) => {
                    sampled_surface_includes_curve(surface, curve)
                }
                // Preserve the posed original carrier. Reconstructing from its
                // entity curve drops the Processor's transform and inversion.
                Surface::RevolutionSurface(surface) => {
                    sampled_surface_includes_curve(surface, curve)
                }
            }
        }
    }
}

impl IncludeCurve<Curve> for Plane {
    fn include(&self, curve: &Curve) -> bool {
        if matches!(curve, Curve::IntersectionCurve(_)) {
            return sampled_surface_includes_curve(self, curve);
        }
        curve.lift_up().control_points().iter().all(|v| {
            let p = v.to_point();
            self.search_parameter(p, None, 1).is_some()
        })
    }
}

impl ToSameGeometry<Surface> for Plane {
    fn to_same_geometry(&self) -> Surface { (*self).into() }
}

impl ToSameGeometry<Surface> for RevolutionSurface<Curve> {
    fn to_same_geometry(&self) -> Surface {
        Surface::RevolutionSurface(Processor::new(self.clone()))
    }
}

fn transform_preserves_nearest_parameter(transform: &Matrix4) -> bool {
    let tol = 1.0e-10;
    let x = Vector3::new(transform.x.x, transform.x.y, transform.x.z);
    let y = Vector3::new(transform.y.x, transform.y.y, transform.y.z);
    let z = Vector3::new(transform.z.x, transform.z.y, transform.z.z);
    let scale2 = x.magnitude2();
    scale2 > tol
        && f64::abs(y.magnitude2() - scale2) <= tol
        && f64::abs(z.magnitude2() - scale2) <= tol
        && f64::abs(x.dot(y)) <= tol
        && f64::abs(x.dot(z)) <= tol
        && f64::abs(y.dot(z)) <= tol
        && f64::abs(transform.x.w) <= tol
        && f64::abs(transform.y.w) <= tol
        && f64::abs(transform.z.w) <= tol
        && f64::abs(transform.w.w - 1.0) <= tol
}

fn revolution_surface_nearest_parameter(
    surface: &Processor<RevolutionSurface<Curve>, Matrix4>,
    point: Point3,
    hint: SearchParameterHint2D,
    trials: usize,
) -> Option<(f64, f64)> {
    if env::var("MT_BOOL_DISABLE_REVOLUTION_NEAREST_FAST_PATH").is_err()
        && transform_preserves_nearest_parameter(surface.transform())
    {
        let inv = surface.transform().inverse_transform()?;
        let point = inv.transform_point(point);
        let uv = surface
            .entity()
            .search_nearest_parameter(point, hint, trials)?;
        Some(if surface.orientation() {
            uv
        } else {
            (uv.1, uv.0)
        })
    } else {
        surface.search_nearest_parameter(point, hint, trials)
    }
}

/// Separable presearch for a revolved-curve surface, bit-identical to the generic
/// [`algo::surface::presearch`] over the same `Processor<RevolutionSurface<..>, Matrix4>`.
///
/// `Processor::evaluate(u, v)` expands (per `orientation`) to
/// `transform(origin + rotation(angle) * (curve(param) - origin))`, where the
/// NURBS/basis `curve(param)` evaluation depends only on one grid axis and the
/// sincos-built `rotation(angle)` matrix only on the other. The generic grid scan
/// re-derives both `division + 1` times per grid line; this hoists each out of the
/// `O(division^2)` inner loop, computing every distinct `curve(param)` and
/// `rotation(angle)` exactly once and reusing it across the crossing line -- the
/// same waste `NurbsSurface::presearch_separable` removes for tensor-product NURBS,
/// which never covered the revolved-curve surface path.
///
/// Everything that decides the result is unchanged: the same grid nodes (`u`, `v`
/// from the identical expressions), the identical evaluation arithmetic
/// (`transform.transform_point(origin + rotation * (curve - origin))`), the same
/// `distance2` metric, and the same strict-`<` first-minimum tie-break. The
/// returned `(u, v)` -- and therefore every downstream Newton seed -- is
/// byte-for-byte identical to the generic presearch.
fn revolution_processor_presearch_separable(
    rotted: &Processor<RevolutionSurface<Curve>, Matrix4>,
    point: Point3,
    (urange, vrange): ((f64, f64), (f64, f64)),
    division: usize,
) -> (f64, f64) {
    let revolution = rotted.entity();
    let transform = rotted.transform();
    let origin = revolution.origin();
    let axis = revolution.axis();
    let curve = revolution.entity_curve();
    // Spec 014 W3: the same grid as `algo::surface::presearch`, charged in the
    // same unit. This path presearches at division 100 where the tensor-product
    // paths use 50, so nodes -- unlike a count of CALLS -- can tell a revolution
    // search apart from a NURBS one (10,201 vs 2,601).
    algo::surface::charge_presearch_nodes(algo::surface::presearch_nodes(division));
    let ((u0, u1), (v0, v1)) = (urange, vrange);
    // Identical to the generic presearch's grid-node expression.
    let node = |bound0: f64, bound1: f64, index: usize| {
        let t = index as f64 / division as f64;
        bound0 * (1.0 - t) + bound1 * t
    };
    // Each expensive separable factor is derived once per grid line and reused
    // across the crossing line (the hoist), then paired with its node value so the
    // loops iterate by value -- no range-indexing. `Processor::evaluate(u, v)` feeds
    // (curve arg, rotation arg) = (u, v) when `orientation`, else (v, u), so the two
    // factors bind to opposite axes per orientation; the returned node and the
    // outer=`u`/inner=`v` iteration order stay fixed to reproduce the generic
    // presearch's argmin and strict-`<` first-minimum tie-break exactly.
    let mut res = (0.0, 0.0);
    let mut min = f64::INFINITY;
    if rotted.orientation() {
        // evaluate(u, v) = transform(origin + rotation(v) * (curve(u) - origin)).
        let curve_by_u: Vec<(f64, Point3)> = (0..=division)
            .map(|i| {
                let u = node(u0, u1, i);
                (u, curve.evaluate(u))
            })
            .collect();
        let rotation_by_v: Vec<(f64, Matrix3)> = (0..=division)
            .map(|j| {
                let v = node(v0, v1, j);
                (v, Matrix3::from_axis_angle(axis, Rad(v)))
            })
            .collect();
        for &(u, curve_point) in &curve_by_u {
            for &(v, rotation) in &rotation_by_v {
                let dist = transform
                    .transform_point(origin + rotation * (curve_point - origin))
                    .distance2(point);
                if dist < min {
                    min = dist;
                    res = (u, v);
                }
            }
        }
    } else {
        // evaluate(u, v) = transform(origin + rotation(u) * (curve(v) - origin)).
        let rotation_by_u: Vec<(f64, Matrix3)> = (0..=division)
            .map(|i| {
                let u = node(u0, u1, i);
                (u, Matrix3::from_axis_angle(axis, Rad(u)))
            })
            .collect();
        let curve_by_v: Vec<(f64, Point3)> = (0..=division)
            .map(|j| {
                let v = node(v0, v1, j);
                (v, curve.evaluate(v))
            })
            .collect();
        for &(u, rotation) in &rotation_by_u {
            for &(v, curve_point) in &curve_by_v {
                let dist = transform
                    .transform_point(origin + rotation * (curve_point - origin))
                    .distance2(point);
                if dist < min {
                    min = dist;
                    res = (u, v);
                }
            }
        }
    }
    res
}

impl SearchNearestParameter<SurfaceParameter> for Surface {
    type Point = Point3;
    fn search_nearest_parameter<H: Into<SearchParameterHint2D>>(
        &self,
        point: Point3,
        hint: H,
        trials: usize,
    ) -> Option<(f64, f64)> {
        match self {
            Surface::Plane(plane) => plane.search_nearest_parameter(point, hint, trials),
            Surface::BsplineSurface(bspsurface) => {
                bspsurface.search_nearest_parameter(point, hint, trials)
            }
            Surface::NurbsSurface(surface) => surface.search_nearest_parameter(point, hint, trials),
            Surface::ExtrusionSurface(surface) => {
                surface.search_nearest_parameter(point, hint, trials)
            }
            Surface::TsplineSurface(surface) => {
                surface.search_nearest_parameter(point, hint, trials)
            }
            // The analytic inverse map, not a Newton descent. This is the half
            // of the sphere/torus conversion that the rational net never
            // carried, and (011-T1) the half that decides where trims land.
            Surface::SphericalSurface(surface) => {
                surface.search_nearest_parameter(point, hint, trials)
            }
            Surface::ToroidalSurface(surface) => {
                surface.search_nearest_parameter(point, hint, trials)
            }
            Surface::RevolutionSurface(rotted) => {
                let hint = hint.into();
                revolution_surface_nearest_parameter(rotted, point, hint, trials).or_else(|| {
                    let hint = match hint {
                        SearchParameterHint2D::Parameter(hint0, hint1) => (hint0, hint1),
                        SearchParameterHint2D::Range(x, y) => {
                            revolution_processor_presearch_separable(rotted, point, (x, y), 100)
                        }
                        SearchParameterHint2D::None => revolution_processor_presearch_separable(
                            rotted,
                            point,
                            rotted.range_tuple(),
                            100,
                        ),
                    };
                    algo::surface::search_nearest_parameter(rotted, point, hint, trials).or_else(
                        || {
                            let candidate = rotted.evaluate(hint.0, hint.1);
                            candidate.near(&point).then_some(hint)
                        },
                    )
                })
            }
        }
    }
}

impl TryIntoBsplineSurface for Surface {
    fn try_into_bspline_surface(&self) -> Option<BsplineSurface<Point3>> {
        match self {
            Surface::Plane(p) => p.try_into_bspline_surface(),
            Surface::BsplineSurface(b) => b.try_into_bspline_surface(),
            Surface::NurbsSurface(n) => n.try_into_bspline_surface(),
            Surface::RevolutionSurface(_) => None,
            Surface::ExtrusionSurface(surface) => {
                let curve = match surface.entity_curve() {
                    Curve::Line(line) => BsplineCurve::from(*line),
                    Curve::BsplineCurve(curve) => curve.clone(),
                    _ => return None,
                };
                let translated =
                    curve.transformed(Matrix4::from_translation(surface.extruding_vector()));
                Some(BsplineSurface::homotopy(curve, translated))
            }
            Surface::TsplineSurface(_) => None,
            // No exact NON-rational net exists for either; the exact form is
            // the homogeneous one below.
            Surface::SphericalSurface(_) | Surface::ToroidalSurface(_) => None,
        }
    }
}

impl TryIntoHomogeneousBsplineSurface for Surface {
    fn try_into_homogeneous_bspline_surface(&self) -> Option<BsplineSurface<Vector4>> {
        match self {
            Surface::Plane(p) => p.try_into_homogeneous_bspline_surface(),
            Surface::BsplineSurface(b) => b.try_into_homogeneous_bspline_surface(),
            Surface::NurbsSurface(n) => n.try_into_homogeneous_bspline_surface(),
            Surface::RevolutionSurface(r) => r.try_into_homogeneous_bspline_surface(),
            Surface::ExtrusionSurface(surface) => surface.try_into_homogeneous_bspline_surface(),
            Surface::TsplineSurface(_) => None,
            // The SAME call the STEP loader used to make eagerly, on the same
            // `Processor<_, Matrix4>`: the net the boolean's homogeneous path
            // prepares is byte-identical to the one it saw before this
            // variant existed. Only WHEN it is built has moved.
            Surface::SphericalSurface(s) => s.try_into_homogeneous_bspline_surface(),
            Surface::ToroidalSurface(t) => t.try_into_homogeneous_bspline_surface(),
        }
    }

    fn original_surface_coefficients(&self) -> Option<OriginalSurfaceCoefficients<'_>> {
        match self {
            Self::BsplineSurface(surface) => surface.original_surface_coefficients(),
            Self::NurbsSurface(surface) => surface.original_surface_coefficients(),
            Self::ExtrusionSurface(surface) => surface.original_surface_coefficients(),
            _ => None,
        }
    }

    fn try_into_homogeneous_bspline_surface_over(
        &self,
        parameter_range: Option<SurfaceParameterRectangle>,
    ) -> Option<HomogeneousSurfaceConversion> {
        match self {
            Surface::Plane(p) => p.try_into_homogeneous_bspline_surface_over(parameter_range),
            Surface::BsplineSurface(b) => {
                b.try_into_homogeneous_bspline_surface_over(parameter_range)
            }
            Surface::NurbsSurface(n) => {
                n.try_into_homogeneous_bspline_surface_over(parameter_range)
            }
            Surface::RevolutionSurface(r) => {
                r.try_into_homogeneous_bspline_surface_over(parameter_range)
            }
            Surface::ExtrusionSurface(surface) => {
                let Some((urange, vrange)) = parameter_range else {
                    return surface
                        .try_into_homogeneous_bspline_surface()
                        .map(Into::into);
                };
                if ![urange.0, urange.1, vrange.0, vrange.1]
                    .into_iter()
                    .all(f64::is_finite)
                    || urange.0 >= urange.1
                    || vrange.0 >= vrange.1
                {
                    return None;
                }
                let profile = surface
                    .entity_curve()
                    .try_into_homogeneous_bspline_curve_over(urange)
                    .or_else(|| surface.entity_curve().try_into_homogeneous_bspline_curve())?;
                let direction = surface.extruding_vector();
                let points = profile
                    .control_points()
                    .iter()
                    .map(|point| {
                        [vrange.0, vrange.1]
                            .into_iter()
                            .map(|v| {
                                Vector4::new(
                                    point.x + v * direction.x * point.w,
                                    point.y + v * direction.y * point.w,
                                    point.z + v * direction.z * point.w,
                                    point.w,
                                )
                            })
                            .collect()
                    })
                    .collect();
                Some(HomogeneousSurfaceConversion {
                    surface: BsplineSurface::new(
                        (
                            profile.knot_vector().clone(),
                            KnotVector::from(vec![vrange.0, vrange.0, vrange.1, vrange.1]),
                        ),
                        points,
                    ),
                    surface_frame_axes: (true, true),
                })
            }
            Surface::TsplineSurface(_) => None,
            Surface::SphericalSurface(s) => {
                s.try_into_homogeneous_bspline_surface_over(parameter_range)
            }
            Surface::ToroidalSurface(t) => {
                t.try_into_homogeneous_bspline_surface_over(parameter_range)
            }
        }
    }
}

impl SupportsExactPatchDomains for Surface {
    fn supports_exact_patch_domains(&self) -> bool {
        match self {
            Surface::Plane(p) => p.supports_exact_patch_domains(),
            Surface::BsplineSurface(b) => b.supports_exact_patch_domains(),
            Surface::NurbsSurface(n) => n.supports_exact_patch_domains(),
            Surface::RevolutionSurface(r) => r.supports_exact_patch_domains(),
            Surface::TsplineSurface(t) => t.supports_exact_patch_domains(),
            Surface::ExtrusionSurface(_) => false,
            // `false`, where the rational net answered `true`. This is the same
            // flip T22 made when cylinders and cones moved onto the analytic
            // revolution variant, and for the same reason: the patch domain of
            // an analytic surface is not a knot span.
            Surface::SphericalSurface(s) => s.supports_exact_patch_domains(),
            Surface::ToroidalSurface(t) => t.supports_exact_patch_domains(),
        }
    }
}

impl TryIntoAnalyticSurfaceKind for Surface {
    fn try_into_analytic_surface_kind(&self) -> Option<AnalyticSurfaceKind> {
        match self {
            Surface::Plane(p) => p.try_into_analytic_surface_kind(),
            Surface::BsplineSurface(b) => b.try_into_analytic_surface_kind(),
            Surface::NurbsSurface(n) => n.try_into_analytic_surface_kind(),
            Surface::RevolutionSurface(r) => r.try_into_analytic_surface_kind(),
            Surface::TsplineSurface(t) => t.try_into_analytic_surface_kind(),
            Surface::ExtrusionSurface(_) => None,
            // `None`, which is exactly what these faces answered as rational
            // nets: `NurbsSurface::try_into_analytic_surface_kind` only ever
            // recognises a plane or a homogeneous extrusion, and a sphere net
            // is neither. Teaching it `SphericalRevolution` would be a
            // BOOLEAN change, and this track is the tessellation one --
            // recorded as follow-on work instead of smuggled in here.
            Surface::SphericalSurface(s) => s.try_into_analytic_surface_kind(),
            Surface::ToroidalSurface(t) => t.try_into_analytic_surface_kind(),
        }
    }

    fn original_plane(&self) -> Option<&Plane> {
        match self {
            Self::Plane(plane) => Some(plane),
            _ => None,
        }
    }

    fn original_sphere(&self) -> Option<(&Sphere, Option<&Matrix4>)> {
        match self {
            Self::SphericalSurface(sphere) => sphere.original_sphere(),
            _ => None,
        }
    }

    fn original_torus(&self) -> Option<(&Torus, Option<&Matrix4>)> {
        match self {
            Self::ToroidalSurface(torus) => torus.original_torus(),
            _ => None,
        }
    }
}

impl ToSameGeometry<Surface> for HomotopySurface<Curve, Curve> {
    fn to_same_geometry(&self) -> Surface {
        let curve0 = self.first_curve().clone().lift_up();
        let curve1 = self.second_curve().clone().lift_up();
        NurbsSurface::new(BsplineSurface::homotopy(curve0, curve1)).into()
    }
}

impl ToSameGeometry<Surface> for ExtrusionSurface<Curve, Vector3> {
    fn to_same_geometry(&self) -> Surface { Surface::ExtrusionSurface(self.clone()) }
}

#[cfg(test)]
mod native_extrusion_tests {
    use super::*;

    #[test]
    fn native_extrusion_preserves_profile_and_actual_extent() {
        let profile = Curve::BsplineCurve(BsplineCurve::new(
            KnotVector::bezier_knot(2),
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 2.0, 0.0),
                Point3::new(2.0, 0.0, 0.0),
            ],
        ));
        let direction = Vector3::new(0.25, -0.5, 3.0);
        let native = ExtrusionSurface::by_extrusion(profile.clone(), direction);
        let surface = native.to_same_geometry();
        assert!(matches!(surface, Surface::ExtrusionSurface(_)));
        assert_eq!(
            surface.evaluate(0.25, 7.0),
            profile.evaluate(0.25) + direction * 7.0
        );
        assert_eq!(surface.derivative_v(0.25, 7.0), direction);
        assert_eq!(surface.derivative_uv(0.25, 7.0), Vector3::zero());
        let conversion = surface
            .try_into_homogeneous_bspline_surface_over(Some(((0.0, 1.0), (-2.0, 7.0))))
            .unwrap();
        assert_eq!(conversion.surface_frame_axes, (true, true));
        assert_near!(
            conversion.surface.evaluate(0.25, 7.0).to_point(),
            surface.evaluate(0.25, 7.0)
        );
        assert_near!(
            surface.inverse().evaluate(0.75, 7.0),
            surface.evaluate(0.25, 7.0)
        );
    }

    #[test]
    fn native_line_extrusion_is_not_replaced_by_a_plane() {
        let profile = Curve::Line(Line(Point3::origin(), Point3::new(2.0, 0.0, 0.0)));
        let surface = ExtrusionSurface::by_extrusion(profile, Vector3::unit_z()).to_same_geometry();
        assert!(matches!(surface, Surface::ExtrusionSurface(_)));
        let conversion = surface
            .try_into_homogeneous_bspline_surface_over(Some(((-3.0, 4.0), (-2.0, 7.0))))
            .unwrap();
        assert_near!(
            conversion.surface.evaluate(-3.0, 7.0).to_point(),
            surface.evaluate(-3.0, 7.0)
        );
    }
}

#[cfg(test)]
mod parameter_curve_containment_regressions {
    use super::*;
    use std::f64::consts::TAU;

    fn cylinder(radius: f64) -> Surface {
        Surface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
            Curve::Line(Line(
                Point3::new(radius, 0.0, 0.0),
                Point3::new(radius, 0.0, 1.0),
            )),
            Point3::origin(),
            Vector3::unit_z(),
        )))
    }

    #[test]
    fn original_parameter_circle_is_included_on_other_actual_carriers() {
        let radius = 5.0;
        let height = 3.0;
        let circle = Curve::ParameterCurve(ParameterCurve::new(
            Curve2D::Conic(Conic2D::Ellipse(Processor::new(TrimmedCurve::new(
                UnitCircle::<Point2>::new(),
                (0.0, TAU),
            )))),
            Box::new(Surface::Plane(Plane::new(
                Point3::new(0.0, 0.0, height),
                Point3::new(radius, 0.0, height),
                Point3::new(0.0, radius, height),
            ))),
        ));
        let cap = Surface::Plane(Plane::new(
            Point3::new(-7.0, 4.0, height),
            Point3::new(-5.0, 4.0, height),
            Point3::new(-7.0, 7.0, height),
        ));
        let unrelated_cap =
            cap.transformed(Matrix4::from_translation(Vector3::new(0.0, 0.0, 0.01)));
        let pose = Matrix4::from_translation(Vector3::new(11.0, -4.0, 7.0))
            * Matrix4::from_angle_x(Rad(0.37));
        let mut bounded = Curve::TrimmedCurve(Processor::new(TrimmedCurve::new(
            Box::new(circle.clone()),
            (0.3, 1.7),
        )));
        bounded.invert();
        for curve in [circle, bounded] {
            let actual = curve.transformed(pose);
            for mut carrier in [cylinder(radius), cap.clone()] {
                carrier.transform_by(pose);
                carrier.invert();
                assert!(carrier.include(&actual));
                for parameter in [actual.range_tuple().0, actual.range_tuple().1] {
                    let point = actual.evaluate(parameter);
                    let local = pose.invert().unwrap().transform_point(point);
                    assert!((local.x.hypot(local.y) - radius).abs() < 1.0e-10);
                    assert!((local.z - height).abs() < 1.0e-10);
                }
            }
            assert!(!cylinder(radius + 0.01).transformed(pose).include(&actual));
            assert!(!unrelated_cap.transformed(pose).include(&actual));
        }
    }
}
