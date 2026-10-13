//! Classic (0.3.2) marching intersection-curve backend.
//!
//! Extracts the mesh-vs-mesh interference segments for a surface pair, chains
//! them into polylines, and re-samples each polyline onto both surfaces to
//! produce a cleaned [`IntersectionCurve`] leader. Derived from the published
//! 0.3.2 marching backend, with strict original-carrier/transversality
//! certification and bounded prepared-query work. The classic loops store
//! consumes the native 3D leader rather than parameter-space polylines.

use monstertruck_core::cgmath64::*;
use monstertruck_geometry::prelude::*;
use monstertruck_meshing::prelude::*;

use super::ClassicResult;
use crate::transversal::polyline_construction::construct_polylines;
use crate::transversal::prepared::BooleanWork;

type Polyline = PolylineCurve<Point3>;

/// Re-sample a raw interference polyline onto both surfaces, returning a
/// cleaned polyline intersection curve. Mirrors 0.3.2
/// `IntersectionCurveWithParameters::try_new`, keeping only the 3D leader.
fn build_intersection_curve<S>(
    surface0: S,
    surface1: S,
    poly: Polyline,
    work: Option<&BooleanWork<'_>>,
) -> Option<IntersectionCurve<Polyline, S, S>>
where
    S: ParametricSurface3D
        + Clone
        + SearchParameter<SurfaceParameter, Point = Point3>
        + SearchNearestParameter<SurfaceParameter, Point = Point3>,
{
    let ic = IntersectionCurve::new(surface0.clone(), surface1.clone(), poly);
    let raw = ic.leader();
    let len = raw.len();
    if len < 2 {
        return None;
    }
    let certified = |&(point, p0, p1): &(Point3, Point2, Point2)| {
        let q0 = ic.surface0().evaluate(p0.x, p0.y);
        let q1 = ic.surface1().evaluate(p1.x, p1.y);
        let tangent = ic
            .surface0()
            .normal(p0.x, p0.y)
            .cross(ic.surface1().normal(p1.x, p1.y));
        point.near(&q0)
            && point.near(&q1)
            && tangent.magnitude2().is_finite()
            && tangent.magnitude2() > 0.0
    };
    // `search_triple` already tries the native nearest-intersection solve.
    // Independently projected carrier points are only a certificate when
    // their mutual residual satisfies the unchanged kernel tolerance.
    let recover_triple = |point: Point3| {
        let p0 = ic
            .surface0()
            .search_parameter(point, None, 100)
            .or_else(|| ic.surface0().search_nearest_parameter(point, None, 100))?;
        let p1 = ic
            .surface1()
            .search_parameter(point, None, 100)
            .or_else(|| ic.surface1().search_nearest_parameter(point, None, 100))?;
        let q0 = ic.surface0().evaluate(p0.0, p0.1);
        let q1 = ic.surface1().evaluate(p1.0, p1.1);
        q0.near(&q1)
            .then_some((q0.midpoint(q1), p0.into(), p1.into()))
    };
    let mut polyline = PolylineCurve(Vec::with_capacity(len));
    for i in 0..len - 1 {
        if let Some(work) = work {
            work.check().ok()?;
        }
        let (q, _, _) = ic
            .search_triple(i as f64, 100)
            .filter(certified)
            .or_else(|| recover_triple(raw[i]).filter(certified))?;
        polyline.push(q);
    }
    let q = if raw[0].near(&raw[len - 1]) {
        polyline[0]
    } else {
        ic.search_triple((len - 1) as f64, 100)
            .filter(certified)
            .or_else(|| recover_triple(raw[len - 1]).filter(certified))?
            .0
    };
    polyline.push(q);
    Some(IntersectionCurve::new(surface0, surface1, polyline))
}

type IntersectionTuple<S> = (Polyline, IntersectionCurve<Polyline, S, S>);

/// Marching intersection curves between two trimmed surfaces, keyed by their
/// tessellated polygon meshes. Returns, per intersection branch, the raw
/// interference polyline and the cleaned polyline intersection curve.
pub(super) fn intersection_curves<S>(
    surface0: S,
    polygon0: &PolygonMesh,
    surface1: S,
    polygon1: &PolygonMesh,
    work: Option<&BooleanWork<'_>>,
) -> ClassicResult<Option<Vec<IntersectionTuple<S>>>>
where
    S: ParametricSurface3D
        + Clone
        + SearchParameter<SurfaceParameter, Point = Point3>
        + SearchNearestParameter<SurfaceParameter, Point = Point3>,
{
    if let Some(work) = work {
        work.face_pair(polygon0, polygon1)?;
    }
    let interferences = polygon0.extract_interference(polygon1);
    if let Some(work) = work {
        work.segments(interferences.len())?;
    }
    let polylines = construct_polylines(&interferences);
    let result = polylines
        .into_iter()
        .map(|polyline| {
            let curve = build_intersection_curve(
                surface0.clone(),
                surface1.clone(),
                polyline.clone(),
                work,
            )?;
            Some((polyline, curve))
        })
        .collect();
    if let Some(work) = work {
        work.check()?;
    }
    Ok(result)
}

#[cfg(test)]
#[path = "intersection_curve_regressions.rs"]
mod regressions;
