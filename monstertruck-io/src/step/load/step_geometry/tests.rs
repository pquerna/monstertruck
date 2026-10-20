//! Unit tests for the parent module.
//!
//! Split out of the module file so the source stays readable; the module
//! path is unchanged.

use super::*;
use crate::step::SurfaceParameterChart;

#[test]
fn surface_curve_rejects_invalid_pcurve_on_identical_surface() {
    let surface = Surface::ElementarySurface(ElementarySurface::Plane(Plane::xy()));
    let leader = Curve3D::Line(Line(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)));
    let invalid_trim = StepParameterCurve::new(
        Box::new(Curve2D::Line(Line(
            Point2::new(0.0, 1.0),
            Point2::new(1.0, 1.0),
        ))),
        Box::new(surface.clone()),
    );
    let surface_curve = SurfaceCurve3D::new(
        SurfaceCurveKind::SurfaceCurve,
        Box::new(leader),
        vec![SurfaceCurveAssociatedGeometry::ParameterCurve(invalid_trim)],
        SurfaceCurveRepresentation::Curve3D,
    );

    assert!(
        surface_curve.parameter_curve_on(&surface).is_none(),
        "Invalid face-local pcurves must not be accepted only because their surface entity matches."
    );
}

#[test]
fn public_parameter_curve_api_uses_descriptive_names() {
    let surface = Surface::ElementarySurface(ElementarySurface::Plane(Plane::xy()));
    let trim: StepParameterCurve = StepParameterCurve::new(
        Box::new(Curve2D::Line(Line(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
        ))),
        Box::new(surface),
    );
    let curve = Curve3D::ParameterCurve(trim);

    assert!(matches!(curve, Curve3D::ParameterCurve(_)));
}

#[test]
fn declared_nonseam_pcurves_resolve_only_identical_representations() {
    let surface = Surface::ElementarySurface(ElementarySurface::Plane(Plane::xy()));
    let uv = Curve2D::Line(Line(Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)));
    let first = StepParameterCurve::new(Box::new(uv.clone()), Box::new(surface.clone()));
    let different_uv = StepParameterCurve::new(
        Box::new(Curve2D::Line(Line(
            Point2::new(0.0, 1.0),
            Point2::new(1.0, 1.0),
        ))),
        Box::new(surface.clone()),
    );
    let mut chart = SurfaceParameterChart::identity();
    chart.swapped = true;
    let different_chart =
        StepParameterCurve::try_from_parts(Box::new(uv), Box::new(surface.clone()), chart)
            .expect("the exchanged identity axes form an admitted chart");
    for kind in [
        SurfaceCurveKind::SurfaceCurve,
        SurfaceCurveKind::IntersectionCurve,
    ] {
        let curve = |last: StepParameterCurve| {
            SurfaceCurve3D::new(
                kind,
                Box::new(Curve3D::Line(Line(
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(1.0, 0.0, 0.0),
                ))),
                vec![
                    SurfaceCurveAssociatedGeometry::ParameterCurve(first.clone()),
                    SurfaceCurveAssociatedGeometry::ParameterCurve(first.clone()),
                    SurfaceCurveAssociatedGeometry::ParameterCurve(last),
                ],
                SurfaceCurveRepresentation::ParameterCurve0,
            )
        };
        let identical = curve(first.clone());
        for orientation in [true, false] {
            let selected = identical
                .declared_parameter_curve_ref_on(&surface, orientation)
                .expect("identical declarations are not different branches");
            assert_eq!(selected, Some(&first));
        }
        for different in [different_uv.clone(), different_chart.clone()] {
            assert!(matches!(
                curve(different).declared_parameter_curve_ref_on(&surface, true),
                Err(ExactConversionError::AmbiguousStoredTrim)
            ));
        }
    }
}

#[test]
fn declared_seam_pcurves_keep_oriented_two_branch_selection() {
    let surface = Surface::ElementarySurface(ElementarySurface::Plane(Plane::xy()));
    let branch = |y| {
        StepParameterCurve::new(
            Box::new(Curve2D::Line(Line(
                Point2::new(0.0, y),
                Point2::new(1.0, y),
            ))),
            Box::new(surface.clone()),
        )
    };
    let first = branch(0.0);
    let second = branch(1.0);
    let seam = |branches: Vec<StepParameterCurve>| {
        SurfaceCurve3D::new(
            SurfaceCurveKind::SeamCurve,
            Box::new(Curve3D::Line(Line(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
            ))),
            branches
                .into_iter()
                .map(SurfaceCurveAssociatedGeometry::ParameterCurve)
                .collect(),
            SurfaceCurveRepresentation::ParameterCurve0,
        )
    };
    let oriented = seam(vec![first.clone(), second.clone()]);
    assert_eq!(
        oriented
            .declared_parameter_curve_ref_on(&surface, true)
            .expect("two seam branches"),
        Some(&first)
    );
    assert_eq!(
        oriented
            .declared_parameter_curve_ref_on(&surface, false)
            .expect("two seam branches"),
        Some(&second)
    );
    assert!(matches!(
        seam(vec![first.clone()]).declared_parameter_curve_ref_on(&surface, true),
        Err(ExactConversionError::MissingSeamParameterBranch)
    ));
    let duplicates = seam(vec![first.clone(), first.clone()]);
    assert!(matches!(
        duplicates.declared_parameter_curve_ref_on(&surface, false),
        Err(ExactConversionError::MissingSeamParameterBranch)
    ));
    assert!(matches!(
        seam(vec![first.clone(), first.clone(), first])
            .declared_parameter_curve_ref_on(&surface, true),
        Err(ExactConversionError::AmbiguousStoredTrim)
    ));
}
