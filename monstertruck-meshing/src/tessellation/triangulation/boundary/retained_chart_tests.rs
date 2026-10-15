//! Original per-use trim charts must survive candidate boundary projection.
use super::*;
use monstertruck_geometry::prelude::{
    Line, ParameterCurve, Point2, Point3, Processor, RevolutionSurface, Vector3,
};
use monstertruck_modeling::{Curve, Curve2D, Surface};
use std::f64::consts::{PI, TAU};

#[test]
fn retained_trim_projection_keeps_original_negative_covering_chart() {
    let surface = Surface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
        Curve::Line(Line(Point3::new(2.0, 0.0, 0.0), Point3::new(1.0, 0.0, 3.0))),
        Point3::origin(),
        Vector3::unit_z(),
    )));
    let charts = [
        Curve2D::Line(Line(Point2::new(0.35, 0.0), Point2::new(0.35, -PI))),
        Curve2D::Line(Line(Point2::new(0.35, -PI), Point2::new(0.35, -TAU))),
    ];
    let originals = charts.each_ref().map(|chart| {
        Curve::ParameterCurve(ParameterCurve::new(
            chart.clone(),
            Box::new(surface.clone()),
        ))
    });
    let trims = charts
        .each_ref()
        .map(|chart| ParameterCurve::new(chart, &surface));
    // A point on the same carrier is not necessarily on this original trim.
    assert!(
        trims[0]
            .project_boundary_point(surface.subs(0.6, -0.5), None)
            .is_none()
    );
    let polylines = trims.each_ref().map(|trim| {
        (0..=16)
            .map(|index| trim.subs(index as f64 / 16.0))
            .collect::<PolylineCurve>()
    });
    let boundary = PolyBoundaryPiece::try_new_from_aligned_trimmed(
        &surface,
        trims
            .iter()
            .zip(&originals)
            .zip(polylines)
            .map(|((trim, original), polyline)| (Some(trim), original, polyline)),
        &|source: &Surface, point: Point3, hint| source.search_parameter(point, hint, 100),
        0.001,
    )
    .expect("original trim projections must retain their common covering chart");
    let (lower, upper) = boundary.0.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(lower, upper), point| (lower.min(point.y), upper.max(point.y)),
    );
    assert!(lower <= -TAU + 1.0e-10 && upper <= 1.0e-10);
    assert!(
        boundary
            .0
            .iter()
            .all(|point| { surface.subs(point.x, point.y).near(&point.point) })
    );
}

#[test]
fn direct_retained_trim_preserves_a_shifted_negative_covering_chart() {
    let surface = Surface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
        Curve::Line(Line(Point3::new(2.0, 0.0, 0.0), Point3::new(1.0, 0.0, 3.0))),
        Point3::origin(),
        Vector3::unit_z(),
    )));
    let start = -0.31;
    let charts = [
        Curve2D::Line(Line(
            Point2::new(0.35, start),
            Point2::new(0.35, start - PI),
        )),
        Curve2D::Line(Line(
            Point2::new(0.35, start - PI),
            Point2::new(0.35, start - TAU),
        )),
    ];
    let originals = charts.each_ref().map(|chart| {
        Curve::ParameterCurve(ParameterCurve::new(
            chart.clone(),
            Box::new(surface.clone()),
        ))
    });
    let trims = charts
        .each_ref()
        .map(|chart| ParameterCurve::new(chart, &surface));
    let boundary = PolyBoundaryPiece::try_new_from_trimmed(
        &surface,
        trims
            .iter()
            .zip(&originals)
            .map(|(trim, original)| (true, Some(trim), original)),
        0.001,
    )
    .expect("direct original trims must retain their declared covering chart");
    let (lower, upper) = boundary.0.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(lower, upper), point| (lower.min(point.y), upper.max(point.y)),
    );
    assert!(lower <= start - TAU + 1.0e-10 && upper <= start + 1.0e-10);
    assert!(
        boundary
            .0
            .iter()
            .all(|point| surface.subs(point.x, point.y).near(&point.point))
    );
}
