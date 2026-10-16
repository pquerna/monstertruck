//! Original per-use trim charts must survive candidate boundary projection.
use super::*;
use monstertruck_geometry::prelude::{
    Line, ParameterCurve, Point2, Point3, Processor, RevolutionSurface,
    TrimBoundaryProjectionError, Vector3,
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
    assert_eq!(
        trims[0].project_boundary_point(surface.subs(0.6, -0.5), None),
        Err(TrimBoundaryProjectionError::Refused)
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

#[test]
fn retained_trim_projection_refuses_another_curve_on_the_same_carrier() {
    let surface = Surface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
        Curve::Line(Line(Point3::new(2.0, 0.0, 0.0), Point3::new(1.0, 0.0, 3.0))),
        Point3::origin(),
        Vector3::unit_z(),
    )));
    let chart = Curve2D::Line(Line(Point2::new(0.35, -0.2), Point2::new(0.35, -1.2)));
    let trim = ParameterCurve::new(&chart, &surface);
    let original = Curve::ParameterCurve(ParameterCurve::new(
        chart.clone(),
        Box::new(surface.clone()),
    ));
    let other_curve = (0..=16)
        .map(|index| surface.subs(0.6, -0.2 - index as f64 / 16.0))
        .collect::<PolylineCurve>();
    let boundary = PolyBoundaryPiece::try_new_from_aligned_trimmed(
        &surface,
        iter::once((Some(&trim), &original, other_curve)),
        &|source: &Surface, point: Point3, hint| source.search_parameter(point, hint, 100),
        0.001,
    );
    assert!(
        boundary.is_none(),
        "carrier membership cannot replace membership on the explicit original trim"
    );
}

#[test]
fn periodic_chart_normalization_retains_original_world_boundary_points() {
    let surface = RevolutionSurface::by_revolution(
        Line(Point3::new(2.0, 0.0, 0.0), Point3::new(2.0, 0.0, 1.0)),
        Point3::origin(),
        Vector3::unit_z(),
    );
    let coordinates = [
        Point2::new(0.2, TAU - 0.1),
        Point2::new(0.2, 0.1),
        Point2::new(0.8, 0.1),
        Point2::new(0.8, TAU - 0.1),
        Point2::new(0.2, TAU - 0.1),
    ];
    let mut curve = coordinates
        .map(|uv| {
            SurfacePoint::from((
                uv,
                surface.evaluate(uv.x, uv.y) + Vector3::new(1.0e-8, -2.0e-8, 0.0),
            ))
        })
        .to_vec();
    let original = curve.iter().map(|point| point.point).collect::<Vec<_>>();
    compact_periodic_closed_loop(&mut curve, &surface);
    let lower = curve
        .iter()
        .map(|point| point.y)
        .fold(f64::INFINITY, f64::min);
    let upper = curve
        .iter()
        .map(|point| point.y)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(upper - lower < 1.0);
    assert!(curve.iter().all(|point| original.contains(&point.point)));
}

#[test]
fn carrier_connectors_retain_existing_boundary_endpoint_positions() {
    let surface = RevolutionSurface::by_revolution(
        Line(Point3::new(2.0, 0.0, 0.0), Point3::new(2.0, 0.0, 1.0)),
        Point3::origin(),
        Vector3::unit_z(),
    );
    let ends = [Point2::new(0.2, 0.0), Point2::new(0.8, 0.3)].map(|uv| {
        SurfacePoint::from((
            uv,
            surface.evaluate(uv.x, uv.y) + Vector3::new(1.0e-8, 0.0, 0.0),
        ))
    });
    let mut cache = HashMap::default();
    let curve = polyline_on_surface(surface, ends[0], ends[1], 0.005, &mut cache);
    assert_eq!(curve.first().unwrap().point, ends[0].point);
    assert_eq!(curve.last().unwrap().point, ends[1].point);
}
