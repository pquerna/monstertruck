//! The 3-D [`Curve`] enum, its 2-D parameter-space companions [`Curve2D`] and
//! [`Conic2D`], and every impl that belongs to them.

use super::domain_projection::sampled_parameter_boundary;
use super::parameter_curves::{
    boundary_matches_surface_curve, curve2d_from_sampled_boundary, direct_bspline_boundary_line,
    direct_nurbs_boundary_line, exact_bspline_boundary, exact_line_boundary, exact_nurbs_boundary,
    line_points, parameter_curve_points, same_surface,
};
use super::*;

type ModelSurfaceCurve = SurfaceCurve<
    Box<Curve>,
    Box<Surface>,
    Box<Surface>,
    ParameterCurve<Curve2D, Box<Surface>>,
    ParameterCurve<Curve2D, Box<Surface>>,
>;

/// 3-dimensional curve
#[derive(
    Clone,
    Debug,
    Serialize,
    Deserialize,
    From,
    ParametricCurve,
    BoundedCurve,
    Invertible,
    SearchNearestParameterD1,
    SearchParameterD1,
)]
pub enum Curve {
    /// line
    Line(Line<Point3>),
    /// 3-dimensional B-spline curve
    BsplineCurve(BsplineCurve<Point3>),
    /// 3-dimensional NURBS curve
    NurbsCurve(NurbsCurve<Vector4>),
    /// 3-dimensional curve carried by a 2-dimensional parameter curve on a surface
    #[allow(clippy::enum_variant_names)]
    ParameterCurve(ParameterCurve<Curve2D, Box<Surface>>),
    /// intersection curve
    IntersectionCurve(ModelSurfaceCurve),
    /// A bounded original basis with an independent affine pose and sense.
    TrimmedCurve(Processor<TrimmedCurve<Box<Curve>>, Matrix4>),
}

/// 2-dimensional curve used as a parameter-space trim.
#[derive(
    Clone,
    Debug,
    Serialize,
    Deserialize,
    From,
    ParametricCurve,
    BoundedCurve,
    ParameterDivision1D,
    Cut,
    Invertible,
    SearchNearestParameterD1,
    SearchParameterD1,
)]
pub enum Conic2D {
    /// ellipse
    Ellipse(Processor<TrimmedCurve<UnitCircle<Point2>>, Matrix3>),
    /// hyperbola
    Hyperbola(Processor<TrimmedCurve<UnitHyperbola<Point2>>, Matrix3>),
    /// parabola
    Parabola(Processor<TrimmedCurve<UnitParabola<Point2>>, Matrix3>),
}

/// 2-dimensional curve used as a parameter-space trim.
#[derive(
    Clone,
    Debug,
    Serialize,
    Deserialize,
    From,
    ParametricCurve,
    BoundedCurve,
    ParameterDivision1D,
    Invertible,
    SearchNearestParameterD1,
    SearchParameterD1,
)]
pub enum Curve2D {
    /// line
    Line(Line<Point2>),
    /// polyline
    Polyline(PolylineCurve<Point2>),
    /// conic
    Conic(Conic2D),
    /// 2-dimensional B-spline curve
    BsplineCurve(BsplineCurve<Point2>),
    /// 2-dimensional NURBS curve
    NurbsCurve(NurbsCurve<Vector3>),
    /// A bounded original parameter-space basis and its independent sense.
    TrimmedCurve(Processor<TrimmedCurve<Box<Curve2D>>, Matrix3>),
}

macro_rules! derive_curve_method {
    ($curve: expr, $method: expr, $($ver: ident),*) => {
        match $curve {
            Curve::Line(got) => $method(got, $($ver), *),
            Curve::BsplineCurve(got) => $method(got, $($ver), *),
            Curve::NurbsCurve(got) => $method(got, $($ver), *),
            Curve::ParameterCurve(got) => $method(got, $($ver), *),
            Curve::IntersectionCurve(got) => $method(got, $($ver), *),
            Curve::TrimmedCurve(got) => $method(got, $($ver), *),
        }
    };
}

macro_rules! derive_curve_self_method {
    ($curve: expr, $method: expr, $($ver: ident),*) => {
        match $curve {
            Curve::Line(got) => Curve::Line($method(got, $($ver), *)),
            Curve::BsplineCurve(got) => Curve::BsplineCurve($method(got, $($ver), *)),
            Curve::NurbsCurve(got) => Curve::NurbsCurve($method(got, $($ver), *)),
            Curve::ParameterCurve(got) => Curve::ParameterCurve($method(got, $($ver), *)),
            Curve::IntersectionCurve(got) => Curve::IntersectionCurve($method(got, $($ver), *)),
            Curve::TrimmedCurve(got) => Curve::TrimmedCurve($method(got, $($ver), *)),
        }
    };
}

fn sample_curve_to_nurbs(curve: &(impl ParametricCurve3D + BoundedCurve)) -> NurbsCurve<Vector4> {
    let (t0, t1) = curve.range_tuple();
    let samples = 16usize;
    let points: Vec<Point3> = (0..=samples)
        .map(|i| t0 + (t1 - t0) * (i as f64) / (samples as f64))
        .map(|t| curve.evaluate(t))
        .collect();
    let knots: Vec<f64> = (0..=samples).map(|i| i as f64 / samples as f64).collect();
    let knot_vec = KnotVector::from(
        iter::once(0.0)
            .chain(knots.iter().copied())
            .chain(iter::once(1.0))
            .collect::<Vec<_>>(),
    );
    NurbsCurve::from(BsplineCurve::new(knot_vec, points))
}

fn linear_bspline_division(
    curve: &BsplineCurve<Point3>,
    range: (f64, f64),
) -> Option<(Vec<f64>, Vec<Point3>)> {
    let curve_range = curve.range_tuple();
    (curve.degree() == 1 && curve_range.0.near(&range.0) && curve_range.1.near(&range.1)).then(
        || {
            (
                (1..=curve.control_points().len())
                    .map(|index| curve.knot(index))
                    .collect(),
                curve.control_points().clone(),
            )
        },
    )
}

impl Transformed<Matrix4> for Curve {
    fn transform_by(&mut self, trans: Matrix4) {
        derive_curve_method!(self, Transformed::transform_by, trans);
    }
    fn transformed(&self, trans: Matrix4) -> Self {
        derive_curve_self_method!(self, Transformed::transformed, trans)
    }
}

impl ParameterDivision1D for Curve {
    type Point = Point3;
    fn parameter_division(&self, range: (f64, f64), tol: f64) -> (Vec<f64>, Vec<Self::Point>) {
        let debug_profile = env::var("MT_PROFILE_CURVE_DIVISION").is_ok();
        let started = Instant::now();
        let result = match self {
            Curve::Line(curve) => curve.parameter_division(range, tol),
            Curve::BsplineCurve(curve) => linear_bspline_division(curve, range)
                .unwrap_or_else(|| curve.parameter_division(range, tol)),
            Curve::NurbsCurve(curve) => curve.parameter_division(range, tol),
            Curve::ParameterCurve(curve) => curve.parameter_division(range, tol),
            Curve::IntersectionCurve(curve) => curve.leader().parameter_division(range, tol),
            Curve::TrimmedCurve(curve) => curve.parameter_division(range, tol),
        };
        if debug_profile {
            let kind = match self {
                Curve::Line(_) => "Line",
                Curve::BsplineCurve(_) => "BsplineCurve",
                Curve::NurbsCurve(_) => "NurbsCurve",
                Curve::ParameterCurve(_) => "ParameterCurve",
                Curve::TrimmedCurve(_) => "TrimmedCurve",
                Curve::IntersectionCurve(curve) => match curve.leader().as_ref() {
                    Curve::Line(_) => "IntersectionCurve(Line)",
                    Curve::BsplineCurve(_) => "IntersectionCurve(BsplineCurve)",
                    Curve::NurbsCurve(_) => "IntersectionCurve(NurbsCurve)",
                    Curve::ParameterCurve(_) => "IntersectionCurve(ParameterCurve)",
                    Curve::IntersectionCurve(_) => "IntersectionCurve(IntersectionCurve)",
                    Curve::TrimmedCurve(_) => "IntersectionCurve(TrimmedCurve)",
                },
            };
            eprintln!(
                "trace bool model_curve_division kind={} points={} tol={} elapsed_ms={:.3}",
                kind,
                result.1.len(),
                tol,
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        result
    }
}

impl From<IntersectionCurve<BsplineCurve<Point3>, Surface, Surface>> for Curve {
    fn from(c: IntersectionCurve<BsplineCurve<Point3>, Surface, Surface>) -> Curve {
        let (surface0, surface1, leader) = c.destruct();
        Curve::IntersectionCurve(SurfaceCurve::with_boundaries(
            Box::new(surface0),
            Box::new(surface1),
            Box::new(leader.into()),
            None,
            None,
        ))
    }
}

fn boundary_curve_2d_to_model_curve(
    curve: ParameterCurve<BoundaryCurve2D, Surface>,
) -> ParameterCurve<Curve2D, Box<Surface>> {
    let surface = Box::new(curve.surface().clone());
    match curve.curve() {
        BoundaryCurve2D::Line(line) => ParameterCurve::new(Curve2D::Line(*line), surface),
        BoundaryCurve2D::BsplineCurve(bspline) => {
            ParameterCurve::new(Curve2D::BsplineCurve(bspline.clone()), surface)
        }
        BoundaryCurve2D::NurbsCurve(nurbs) => {
            ParameterCurve::new(Curve2D::NurbsCurve(nurbs.clone()), surface)
        }
    }
}

impl
    From<
        SurfaceCurve<
            BsplineCurve<Point3>,
            Surface,
            Surface,
            ParameterCurve<BoundaryCurve2D, Surface>,
            ParameterCurve<BoundaryCurve2D, Surface>,
        >,
    > for Curve
{
    fn from(
        c: SurfaceCurve<
            BsplineCurve<Point3>,
            Surface,
            Surface,
            ParameterCurve<BoundaryCurve2D, Surface>,
            ParameterCurve<BoundaryCurve2D, Surface>,
        >,
    ) -> Curve {
        let (surface0, surface1, leader, boundary0, boundary1) = c.destruct_with_boundaries();
        Curve::IntersectionCurve(SurfaceCurve::with_boundaries(
            Box::new(surface0),
            Box::new(surface1),
            Box::new(leader.into()),
            boundary0.map(boundary_curve_2d_to_model_curve),
            boundary1.map(boundary_curve_2d_to_model_curve),
        ))
    }
}

impl
    From<
        SurfaceCurve<
            NurbsCurve<Vector4>,
            Surface,
            Surface,
            ParameterCurve<BoundaryCurve2D, Surface>,
            ParameterCurve<BoundaryCurve2D, Surface>,
        >,
    > for Curve
{
    fn from(
        c: SurfaceCurve<
            NurbsCurve<Vector4>,
            Surface,
            Surface,
            ParameterCurve<BoundaryCurve2D, Surface>,
            ParameterCurve<BoundaryCurve2D, Surface>,
        >,
    ) -> Curve {
        let (surface0, surface1, leader, boundary0, boundary1) = c.destruct_with_boundaries();
        Curve::IntersectionCurve(SurfaceCurve::with_boundaries(
            Box::new(surface0),
            Box::new(surface1),
            Box::new(leader.into()),
            boundary0.map(boundary_curve_2d_to_model_curve),
            boundary1.map(boundary_curve_2d_to_model_curve),
        ))
    }
}

impl ToSameGeometry<Curve> for Line<Point3> {
    #[inline]
    fn to_same_geometry(&self) -> Curve { Curve::from(*self) }
}

impl ToSameGeometry<Curve> for Processor<TrimmedCurve<UnitCircle<Point3>>, Matrix4> {
    #[inline]
    fn to_same_geometry(&self) -> Curve { Curve::NurbsCurve(self.to_same_geometry()) }
}

impl ToSameGeometry<Curve> for BsplineCurve<Point3> {
    #[inline]
    fn to_same_geometry(&self) -> Curve { Curve::from(self.clone()) }
}

impl Curve {
    /// Into non-ratinalized 4-dimensional B-spline curve
    pub fn lift_up(&self) -> BsplineCurve<Vector4> {
        match self {
            Curve::Line(curve) => Curve::BsplineCurve((*curve).into()).lift_up(),
            Curve::BsplineCurve(curve) => BsplineCurve::new(
                curve.knot_vector().clone(),
                curve
                    .control_points()
                    .iter()
                    .map(|pt| pt.to_vec().extend(1.0))
                    .collect(),
            ),
            Curve::NurbsCurve(curve) => curve.non_rationalized().clone(),
            Curve::ParameterCurve(curve) => sample_curve_to_nurbs(curve).non_rationalized().clone(),
            Curve::IntersectionCurve(curve) => curve.leader().lift_up(),
            Curve::TrimmedCurve(curve) => self
                .try_into_homogeneous_bspline_curve()
                .unwrap_or_else(|| sample_curve_to_nurbs(curve).non_rationalized().clone()),
        }
    }
}

fn curve2d_endpoint_hints(curve: &Curve2D) -> Option<(Point2, Point2)> {
    match curve {
        Curve2D::Line(curve) => Some((curve.0, curve.1)),
        Curve2D::Polyline(curve) => Some((*curve.first()?, *curve.last()?)),
        Curve2D::BsplineCurve(curve) => Some((
            *curve.control_points().first()?,
            *curve.control_points().last()?,
        )),
        Curve2D::NurbsCurve(curve) => Some((
            curve.control_points().first()?.to_point(),
            curve.control_points().last()?.to_point(),
        )),
        Curve2D::Conic(_) | Curve2D::TrimmedCurve(_) => None,
    }
}

fn set_curve2d_endpoints(curve: &mut Curve2D, front: Point2, back: Point2) {
    match curve {
        Curve2D::Line(curve) => {
            curve.0 = front;
            curve.1 = back;
        }
        Curve2D::Polyline(curve) => {
            if let Some(point) = curve.first_mut() {
                *point = front;
            }
            if let Some(point) = curve.last_mut() {
                *point = back;
            }
        }
        Curve2D::BsplineCurve(curve) => {
            if !curve.control_points().is_empty() {
                *curve.control_point_mut(0) = front;
            }
            if curve.control_points().len() > 1 {
                let last = curve.control_points().len() - 1;
                *curve.control_point_mut(last) = back;
            }
        }
        Curve2D::NurbsCurve(curve) => {
            if !curve.control_points().is_empty() {
                let point = curve.control_point_mut(0);
                let weight = point.weight();
                *point = front.to_vec().extend(weight);
            }
            if curve.control_points().len() > 1 {
                let last = curve.control_points().len() - 1;
                let point = curve.control_point_mut(last);
                let weight = point.weight();
                *point = back.to_vec().extend(weight);
            }
        }
        Curve2D::Conic(_) | Curve2D::TrimmedCurve(_) => {}
    }
}

fn snap_parameter_curve_endpoints(
    curve: &mut ParameterCurve<Curve2D, Box<Surface>>,
    front: Point3,
    back: Point3,
) {
    let hints = curve2d_endpoint_hints(curve.curve());
    let front_uv = hints
        .and_then(|(front_hint, _)| {
            curve
                .surface()
                .search_nearest_parameter(front, Some((front_hint.x, front_hint.y)), 100)
        })
        .or_else(|| curve.surface().search_nearest_parameter(front, None, 100))
        .map(|(u, v)| Point2::new(u, v));
    let back_uv = hints
        .and_then(|(_, back_hint)| {
            curve
                .surface()
                .search_nearest_parameter(back, Some((back_hint.x, back_hint.y)), 100)
        })
        .or_else(|| curve.surface().search_nearest_parameter(back, None, 100))
        .map(|(u, v)| Point2::new(u, v));
    if let (Some(front_uv), Some(back_uv)) = (front_uv, back_uv) {
        let (mut curve2d, surface) = curve.clone().decompose();
        set_curve2d_endpoints(&mut curve2d, front_uv, back_uv);
        *curve = ParameterCurve::new(curve2d, surface);
    }
}

impl SnapCurveEndpoints for Curve {
    fn snap_endpoints(&mut self, front: Point3, back: Point3) {
        match self {
            Curve::IntersectionCurve(curve) => {
                curve.leader_mut().snap_endpoints(front, back);
                if let Some(boundary) = curve.boundary0_mut() {
                    snap_parameter_curve_endpoints(boundary, front, back);
                }
                if let Some(boundary) = curve.boundary1_mut() {
                    snap_parameter_curve_endpoints(boundary, front, back);
                }
            }
            Curve::ParameterCurve(curve) => snap_parameter_curve_endpoints(curve, front, back),
            _ => {}
        }
    }
}

impl TryIntoHomogeneousBsplineCurve for Curve {
    fn try_into_homogeneous_bspline_curve(&self) -> Option<BsplineCurve<Vector4>> {
        match self {
            Curve::Line(curve) => curve.try_into_homogeneous_bspline_curve(),
            Curve::BsplineCurve(curve) => curve.try_into_homogeneous_bspline_curve(),
            Curve::NurbsCurve(curve) => curve.try_into_homogeneous_bspline_curve(),
            Curve::ParameterCurve(_) => None,
            Curve::IntersectionCurve(_) => None,
            Curve::TrimmedCurve(_) => exact_trimmed_homogeneous(self, 0),
        }
    }

    fn try_into_homogeneous_bspline_curve_over(
        &self,
        range: (f64, f64),
    ) -> Option<BsplineCurve<Vector4>> {
        match self {
            // Only a line has an exact analytic continuation past its own range;
            // every other variant keeps the trait's refusing default. This is
            // what lets a cylinder's profile be re-spanned onto the face's trim
            // once it has been re-homed into the modeling `Curve` enum.
            Curve::Line(curve) => curve.try_into_homogeneous_bspline_curve_over(range),
            _ => None,
        }
    }
}

impl TryFrom<ParameterCurve<Line<Point2>, Surface>> for Curve {
    type Error = ();
    fn try_from(curve: ParameterCurve<Line<Point2>, Surface>) -> std::result::Result<Self, ()> {
        let (line, surface) = curve.decompose();
        Ok(Curve::ParameterCurve(ParameterCurve::new(
            Curve2D::Line(line),
            Box::new(surface),
        )))
    }
}

impl ParameterBoundary2D<Surface> for Curve {
    fn parameter_boundary_2d(&self, surface: &Surface, tolerance: f64) -> Option<Vec<Point2>> {
        match self {
            Curve::Line(curve) => exact_line_boundary(curve, surface)
                .map(|boundary| parameter_curve_points(&boundary, tolerance))
                .or_else(|| sampled_parameter_boundary(curve, surface, tolerance)),
            Curve::BsplineCurve(curve) => exact_bspline_boundary(curve, surface)
                .map(|boundary| parameter_curve_points(&boundary, tolerance))
                .or_else(|| {
                    if let Surface::BsplineSurface(bspline_surface) = surface {
                        direct_bspline_boundary_line(curve, bspline_surface)
                            .map(|line| line_points(line, tolerance))
                    } else {
                        None
                    }
                })
                .or_else(|| sampled_parameter_boundary(curve, surface, tolerance)),
            Curve::NurbsCurve(curve) => exact_nurbs_boundary(curve, surface)
                .map(|boundary| parameter_curve_points(&boundary, tolerance))
                .or_else(|| {
                    if let Surface::NurbsSurface(nurbs_surface) = surface {
                        direct_nurbs_boundary_line(curve, nurbs_surface)
                            .map(|line| line_points(line, tolerance))
                    } else {
                        None
                    }
                })
                // Fall back to a sampled polyline trim on surfaces the exact and
                // direct-line paths do not cover (notably planar caps, whose
                // circular boundary arcs are `NurbsCurve`s on a `Surface::Plane`).
                // Without this the cap arcs yield no parameter curve and drop from
                // downstream NURBS export. Mirrors the `Line`/`BsplineCurve` arms.
                .or_else(|| sampled_parameter_boundary(curve, surface, tolerance)),
            Curve::ParameterCurve(curve) => same_surface(curve.surface().as_ref(), surface)
                .then(|| parameter_curve_points(curve, tolerance)),
            Curve::TrimmedCurve(curve) => sampled_parameter_boundary(curve, surface, tolerance),
            Curve::IntersectionCurve(curve) => {
                if let Some(boundary) = curve.boundary0().filter(|boundary| {
                    boundary_matches_surface_curve(curve.leader().as_ref(), boundary, surface)
                }) {
                    Some(parameter_curve_points(boundary, tolerance))
                } else if let Some(boundary) = curve.boundary1().filter(|boundary| {
                    boundary_matches_surface_curve(curve.leader().as_ref(), boundary, surface)
                }) {
                    Some(parameter_curve_points(boundary, tolerance))
                } else if same_surface(curve.surface0().as_ref(), surface) {
                    curve
                        .boundary0()
                        .map(|boundary| parameter_curve_points(boundary, tolerance))
                        .or_else(|| curve.leader().parameter_boundary_2d(surface, tolerance))
                } else if same_surface(curve.surface1().as_ref(), surface) {
                    curve
                        .boundary1()
                        .map(|boundary| parameter_curve_points(boundary, tolerance))
                        .or_else(|| curve.leader().parameter_boundary_2d(surface, tolerance))
                } else {
                    sampled_parameter_boundary(curve.leader().as_ref(), surface, tolerance)
                }
            }
        }
    }
}

impl ExactParameterBoundary2D<Surface> for Curve {
    type BoundaryCurve = ParameterCurve<Curve2D, Box<Surface>>;

    fn exact_parameter_boundary_2d(&self, surface: &Surface) -> Option<Self::BoundaryCurve> {
        match self {
            Curve::Line(curve) => exact_line_boundary(curve, surface),
            Curve::BsplineCurve(curve) => exact_bspline_boundary(curve, surface),
            Curve::NurbsCurve(curve) => exact_nurbs_boundary(curve, surface),
            Curve::TrimmedCurve(curve) if *curve.transform() == Matrix4::identity() => {
                let boundary = curve
                    .entity()
                    .curve()
                    .exact_parameter_boundary_2d(surface)?;
                let (basis, surface) = boundary.decompose();
                let mut trim =
                    Processor::new(TrimmedCurve::new(Box::new(basis), curve.entity().range()));
                if !curve.orientation() {
                    trim.invert();
                }
                Some(ParameterCurve::new(Curve2D::TrimmedCurve(trim), surface))
            }
            Curve::ParameterCurve(curve) if same_surface(curve.surface().as_ref(), surface) => {
                Some(curve.clone())
            }
            Curve::IntersectionCurve(curve) if same_surface(curve.surface0().as_ref(), surface) => {
                curve
                    .boundary0()
                    .cloned()
                    .or_else(|| curve.leader().exact_parameter_boundary_2d(surface))
            }
            Curve::IntersectionCurve(curve) if same_surface(curve.surface1().as_ref(), surface) => {
                curve
                    .boundary1()
                    .cloned()
                    .or_else(|| curve.leader().exact_parameter_boundary_2d(surface))
            }
            _ => None,
        }
    }
}

impl BoundaryCurveFromSamples<Surface> for ParameterCurve<Curve2D, Box<Surface>> {
    fn boundary_curve_from_samples(surface: &Surface, points: Vec<Point2>) -> Option<Self> {
        curve2d_from_sampled_boundary(points)
            .map(|curve| ParameterCurve::new(curve, Box::new(surface.clone())))
    }
}

impl Curve {
    /// Converts this curve into a face-local parameter curve on `surface`.
    ///
    /// Exact trim data is preserved when available. Otherwise this falls back
    /// to a sampled polyline trim in the surface domain.
    pub fn to_parameter_curve_on(
        &self,
        surface: &Surface,
        tolerance: f64,
    ) -> Option<ParameterCurve<Curve2D, Box<Surface>>> {
        let debug_profile = env::var("MT_PROFILE_PARAMETER_CURVE_ON").is_ok();
        let started = Instant::now();
        let exact = self.exact_parameter_boundary_2d(surface);
        let exact_hit = exact.is_some();
        let result = exact.or_else(|| {
            self.parameter_boundary_2d(surface, tolerance)
                .filter(|points| points.len() >= 2)
                .and_then(curve2d_from_sampled_boundary)
                .map(|curve| ParameterCurve::new(curve, Box::new(surface.clone())))
        });
        if debug_profile {
            let kind = match self {
                Curve::Line(_) => "Line",
                Curve::BsplineCurve(_) => "BsplineCurve",
                Curve::NurbsCurve(_) => "NurbsCurve",
                Curve::ParameterCurve(_) => "ParameterCurve",
                Curve::TrimmedCurve(_) => "TrimmedCurve",
                Curve::IntersectionCurve(curve) => match curve.leader().as_ref() {
                    Curve::Line(_) => "IntersectionCurve(Line)",
                    Curve::BsplineCurve(_) => "IntersectionCurve(BsplineCurve)",
                    Curve::NurbsCurve(_) => "IntersectionCurve(NurbsCurve)",
                    Curve::ParameterCurve(_) => "IntersectionCurve(ParameterCurve)",
                    Curve::IntersectionCurve(_) => "IntersectionCurve(IntersectionCurve)",
                    Curve::TrimmedCurve(_) => "IntersectionCurve(TrimmedCurve)",
                },
            };
            eprintln!(
                "trace bool parameter_curve_on kind={} exact={} output={} elapsed_ms={:.3}",
                kind,
                exact_hit,
                result.is_some(),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        result
    }
}

impl Cut for Curve {
    fn cut(&mut self, t: f64) -> Self {
        if matches!(self, Self::TrimmedCurve(curve) if !curve.orientation()) {
            let (start, end) = self.range_tuple();
            let right = Self::TrimmedCurve(Processor::new(TrimmedCurve::new(
                Box::new(self.clone()),
                (t, end),
            )));
            let original = std::mem::replace(self, right);
            let left = Self::TrimmedCurve(Processor::new(TrimmedCurve::new(
                Box::new(original),
                (start, t),
            )));
            std::mem::replace(self, left)
        } else {
            derive_curve_self_method!(self, Cut::cut, t)
        }
    }
}

impl Cut for Curve2D {
    fn cut(&mut self, t: f64) -> Self {
        match self {
            Self::Line(curve) => Self::Line(curve.cut(t)),
            Self::Polyline(curve) => Self::Polyline(curve.cut(t)),
            Self::Conic(curve) => Self::Conic(curve.cut(t)),
            Self::BsplineCurve(curve) => Self::BsplineCurve(curve.cut(t)),
            Self::NurbsCurve(curve) => Self::NurbsCurve(curve.cut(t)),
            Self::TrimmedCurve(curve) if curve.orientation() => Self::TrimmedCurve(curve.cut(t)),
            Self::TrimmedCurve(_) => {
                let (start, end) = self.range_tuple();
                let right = Self::TrimmedCurve(Processor::new(TrimmedCurve::new(
                    Box::new(self.clone()),
                    (t, end),
                )));
                let original = std::mem::replace(self, right);
                let left = Self::TrimmedCurve(Processor::new(TrimmedCurve::new(
                    Box::new(original),
                    (start, t),
                )));
                std::mem::replace(self, left)
            }
        }
    }
}

fn exact_trimmed_homogeneous(curve: &Curve, depth: usize) -> Option<BsplineCurve<Vector4>> {
    if depth > MAX_NATIVE_GEOMETRY_NESTING {
        return None;
    }
    let Curve::TrimmedCurve(trimmed) = curve else {
        return curve.try_into_homogeneous_bspline_curve();
    };
    let range = trimmed.entity().range();
    let basis = trimmed.entity().curve().as_ref();
    let mut spline = match basis {
        Curve::Line(line) => line.try_into_homogeneous_bspline_curve_over(range)?,
        _ => exact_trimmed_homogeneous(basis, depth + 1)?,
    };
    let own = spline.range_tuple();
    if range.0 < own.0 || range.1 > own.1 || range.0 >= range.1 {
        return None;
    }
    if range.0 > own.0 {
        spline = spline.cut(range.0);
    }
    if range.1 < own.1 {
        let _ = spline.cut(range.1);
    }
    if !trimmed.orientation() {
        spline.invert();
    }
    spline
        .control_points_mut()
        .for_each(|point| *point = *trimmed.transform() * *point);
    Some(spline)
}

impl Transformed<Matrix3> for Conic2D {
    fn transform_by(&mut self, transform: Matrix3) {
        match self {
            Self::Ellipse(curve) => curve.transform_by(transform),
            Self::Hyperbola(curve) => curve.transform_by(transform),
            Self::Parabola(curve) => curve.transform_by(transform),
        }
    }

    fn transformed(&self, transform: Matrix3) -> Self {
        let mut curve = self.clone();
        curve.transform_by(transform);
        curve
    }
}

impl Transformed<Matrix3> for Curve2D {
    fn transform_by(&mut self, transform: Matrix3) {
        match self {
            Self::Line(curve) => curve.transform_by(transform),
            Self::Polyline(curve) => curve.transform_by(transform),
            Self::Conic(curve) => curve.transform_by(transform),
            Self::BsplineCurve(curve) => curve.transform_by(transform),
            Self::NurbsCurve(curve) => curve.transform_by(transform),
            Self::TrimmedCurve(curve) => curve.transform_by(transform),
        }
    }

    fn transformed(&self, transform: Matrix3) -> Self {
        let mut curve = self.clone();
        curve.transform_by(transform);
        curve
    }
}

#[cfg(test)]
mod native_trimmed_curve_tests {
    use super::*;

    #[test]
    fn nested_inverse_trim_cut_preserves_original_parameter_chart() {
        let basis = Curve::Line(Line(Point3::origin(), Point3::new(1.0, 2.0, 3.0)));
        let mut inner = Processor::new(TrimmedCurve::new(Box::new(basis), (-2.0, 7.0)));
        inner.invert();
        let original = Curve::TrimmedCurve(Processor::new(TrimmedCurve::new(
            Box::new(Curve::TrimmedCurve(inner)),
            (1.0, 4.0),
        )))
        .inverse();
        let mut first = original.clone();
        let second = first.cut(2.0);
        for t in [1.0, 1.5, 2.0] {
            assert_eq!(first.evaluate(t), original.evaluate(t));
            assert_eq!(first.derivative(t), original.derivative(t));
        }
        for t in [2.0, 3.0, 4.0] {
            assert_eq!(second.evaluate(t), original.evaluate(t));
            assert_eq!(second.derivative(t), original.derivative(t));
        }
        assert_eq!(first.range_tuple(), (1.0, 2.0));
        assert_eq!(second.range_tuple(), (2.0, 4.0));
        assert_eq!(original.inverse().evaluate(1.0), original.evaluate(4.0));
        assert!(
            first
                .search_parameter(original.evaluate(3.0), None, 100)
                .is_none()
        );
        assert_near!(
            first
                .search_nearest_parameter(original.evaluate(3.0), None, 100)
                .unwrap(),
            2.0
        );
        let third = first.cut(1.5);
        assert_eq!(third.evaluate(1.75), original.evaluate(1.75));
        assert!(matches!(&third, Curve::TrimmedCurve(outer)
            if matches!(outer.entity().curve().as_ref(), Curve::TrimmedCurve(basis) if !basis.orientation())));
    }

    #[test]
    fn nested_2d_inverse_trim_cut_preserves_original_parameter_chart() {
        let basis = Curve2D::Line(Line(Point2::origin(), Point2::new(1.0, 2.0)));
        let mut inner = Processor::new(TrimmedCurve::new(Box::new(basis), (-2.0, 7.0)));
        inner.invert();
        let original = Curve2D::TrimmedCurve(Processor::new(TrimmedCurve::new(
            Box::new(Curve2D::TrimmedCurve(inner)),
            (1.0, 4.0),
        )))
        .inverse();
        let mut first = original.clone();
        let second = first.cut(2.0);
        assert_eq!(first.evaluate(1.5), original.evaluate(1.5));
        assert_eq!(second.evaluate(3.0), original.evaluate(3.0));
        assert_eq!(second.derivative(3.0), original.derivative(3.0));
    }
}
