//! Flattening STEP geometry onto B-spline and rational (homogeneous)
//! B-spline nets, and the predicate for which surfaces carry an exact
//! patch domain.

use super::*;

pub(super) fn polyline_knots(curve: &PolylineCurve<Point3>) -> Option<KnotVector> {
    if curve.len() < 2
        || curve
            .iter()
            .any(|point| !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite())
    {
        return None;
    }
    let end = (curve.len() - 1) as f64;
    let knots = std::iter::once(0.0)
        .chain((0..curve.len()).map(|i| i as f64))
        .chain(std::iter::once(end))
        .collect::<Vec<_>>();
    Some(KnotVector::from(knots))
}

impl TryIntoHomogeneousBsplineSurface for Sphere {
    fn try_into_homogeneous_bspline_surface(&self) -> Option<BsplineSurface<Vector4>> {
        self.0.try_into_homogeneous_bspline_surface()
    }

    fn try_into_homogeneous_bspline_surface_over(
        &self,
        parameter_range: Option<SurfaceParameterRectangle>,
    ) -> Option<HomogeneousSurfaceConversion> {
        self.0
            .try_into_homogeneous_bspline_surface_over(parameter_range)
    }
}

impl TryIntoBsplineSurface for Sphere {
    fn try_into_bspline_surface(&self) -> Option<BsplineSurface<Point3>> {
        self.0.try_into_bspline_surface()
    }
}

impl TryIntoHomogeneousBsplineCurve for Curve3D {
    fn try_into_homogeneous_bspline_curve(&self) -> Option<BsplineCurve<Vector4>> {
        match self {
            Curve3D::Line(line) => line.try_into_homogeneous_bspline_curve(),
            Curve3D::Conic(Conic3D::Ellipse(curve)) => curve.try_into_homogeneous_bspline_curve(),
            Curve3D::Conic(Conic3D::Hyperbola(curve)) => curve.try_into_homogeneous_bspline_curve(),
            Curve3D::Conic(Conic3D::Parabola(curve)) => curve.try_into_homogeneous_bspline_curve(),
            Curve3D::BsplineCurve(curve) => curve.try_into_homogeneous_bspline_curve(),
            Curve3D::ParameterCurve(_) => None,
            Curve3D::Polyline(curve) => Some(BsplineCurve::new(
                polyline_knots(curve)?,
                curve.iter().map(|point| point.to_homogeneous()).collect(),
            )),
            Curve3D::SurfaceCurve(curve) => curve.leader().try_into_homogeneous_bspline_curve(),
            Curve3D::IntersectionCurve(_) => None,
            Curve3D::TrimmedCurve(_) => None,
            Curve3D::NurbsCurve(curve) => curve.try_into_homogeneous_bspline_curve(),
        }
    }

    fn try_into_homogeneous_bspline_curve_over(
        &self,
        range: (f64, f64),
    ) -> Option<BsplineCurve<Vector4>> {
        match self {
            // Only a line has an exact analytic continuation past its own range;
            // every other variant keeps the trait's refusing default.
            Curve3D::Line(line) => line.try_into_homogeneous_bspline_curve_over(range),
            _ => None,
        }
    }
}

impl TryIntoHomogeneousBsplineCurve for Box<Curve3D> {
    fn try_into_homogeneous_bspline_curve(&self) -> Option<BsplineCurve<Vector4>> {
        self.as_ref().try_into_homogeneous_bspline_curve()
    }

    fn try_into_homogeneous_bspline_curve_over(
        &self,
        range: (f64, f64),
    ) -> Option<BsplineCurve<Vector4>> {
        self.as_ref().try_into_homogeneous_bspline_curve_over(range)
    }
}

impl TryIntoBsplineSurface for Surface {
    fn try_into_bspline_surface(&self) -> Option<BsplineSurface<Point3>> {
        match self {
            Surface::ElementarySurface(ElementarySurface::Plane(surface)) => {
                surface.try_into_bspline_surface()
            }
            Surface::ElementarySurface(ElementarySurface::Sphere(surface)) => {
                surface.try_into_bspline_surface()
            }
            Surface::ElementarySurface(ElementarySurface::CylindricalSurface(surface)) => {
                surface.try_into_bspline_surface()
            }
            Surface::ElementarySurface(ElementarySurface::ToroidalSurface(surface)) => {
                surface.try_into_bspline_surface()
            }
            Surface::ElementarySurface(ElementarySurface::ConicalSurface(surface)) => {
                surface.try_into_bspline_surface()
            }
            Surface::SweepSurface(SweepSurface::ExtrusionSurface(surface)) => {
                surface.try_into_bspline_surface()
            }
            Surface::SweepSurface(SweepSurface::RevolutionSurface(surface)) => {
                surface.try_into_bspline_surface()
            }
            Surface::BsplineSurface(surface) => surface.try_into_bspline_surface(),
            Surface::NurbsSurface(surface) => surface.try_into_bspline_surface(),
        }
    }
}

impl TryIntoHomogeneousBsplineSurface for Surface {
    fn try_into_homogeneous_bspline_surface(&self) -> Option<BsplineSurface<Vector4>> {
        match self {
            Surface::ElementarySurface(ElementarySurface::Plane(surface)) => {
                surface.try_into_homogeneous_bspline_surface()
            }
            Surface::ElementarySurface(ElementarySurface::Sphere(surface)) => {
                surface.try_into_homogeneous_bspline_surface()
            }
            Surface::ElementarySurface(ElementarySurface::CylindricalSurface(surface)) => {
                surface.try_into_homogeneous_bspline_surface()
            }
            Surface::ElementarySurface(ElementarySurface::ToroidalSurface(surface)) => {
                surface.try_into_homogeneous_bspline_surface()
            }
            Surface::ElementarySurface(ElementarySurface::ConicalSurface(surface)) => {
                surface.try_into_homogeneous_bspline_surface()
            }
            Surface::SweepSurface(SweepSurface::ExtrusionSurface(surface)) => {
                surface.try_into_homogeneous_bspline_surface()
            }
            Surface::SweepSurface(SweepSurface::RevolutionSurface(surface)) => {
                surface.try_into_homogeneous_bspline_surface()
            }
            Surface::BsplineSurface(surface) => surface.try_into_homogeneous_bspline_surface(),
            Surface::NurbsSurface(surface) => surface.try_into_homogeneous_bspline_surface(),
        }
    }

    fn try_into_homogeneous_bspline_surface_over(
        &self,
        parameter_range: Option<SurfaceParameterRectangle>,
    ) -> Option<HomogeneousSurfaceConversion> {
        match self {
            Surface::ElementarySurface(ElementarySurface::Plane(surface)) => {
                surface.try_into_homogeneous_bspline_surface_over(parameter_range)
            }
            Surface::ElementarySurface(ElementarySurface::Sphere(surface)) => {
                surface.try_into_homogeneous_bspline_surface_over(parameter_range)
            }
            Surface::ElementarySurface(ElementarySurface::CylindricalSurface(surface)) => {
                surface.try_into_homogeneous_bspline_surface_over(parameter_range)
            }
            Surface::ElementarySurface(ElementarySurface::ToroidalSurface(surface)) => {
                surface.try_into_homogeneous_bspline_surface_over(parameter_range)
            }
            Surface::ElementarySurface(ElementarySurface::ConicalSurface(surface)) => {
                surface.try_into_homogeneous_bspline_surface_over(parameter_range)
            }
            Surface::SweepSurface(SweepSurface::ExtrusionSurface(surface)) => {
                surface.try_into_homogeneous_bspline_surface_over(parameter_range)
            }
            Surface::SweepSurface(SweepSurface::RevolutionSurface(surface)) => {
                surface.try_into_homogeneous_bspline_surface_over(parameter_range)
            }
            Surface::BsplineSurface(surface) => {
                surface.try_into_homogeneous_bspline_surface_over(parameter_range)
            }
            Surface::NurbsSurface(surface) => {
                surface.try_into_homogeneous_bspline_surface_over(parameter_range)
            }
        }
    }
}

impl SupportsExactPatchDomains for Surface {
    fn supports_exact_patch_domains(&self) -> bool {
        matches!(self, Surface::BsplineSurface(_) | Surface::NurbsSurface(_))
    }
}
