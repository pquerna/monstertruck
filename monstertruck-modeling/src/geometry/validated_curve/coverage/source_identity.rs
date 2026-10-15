//! Exact retained-source association under the native uniform unit transform.

use super::super::{PreparedSource, PreparedValidatedCurve};
use super::{equal_slice, exact_curve2d};
use crate::{
    BoundedCurve, Curve, Matrix4, ParameterCurve, Point3, Surface, Transform,
    ValidatedGeometryBudget, ValidatedGeometryError, Vector4,
};
use std::sync::atomic::AtomicBool;

/// Checks the original master, never a guide or a newly selected root.
///
/// A native processor absorbs a spatial transform into its pose; its entity
/// coefficients are not transformed. An extra active restriction is admissible
/// only with the exact forward native gauge and an identity or unit-scale pose.
pub(super) fn retains_original_curve(
    prepared: &PreparedValidatedCurve<'_>,
    actual: &Curve,
    active: [f64; 2],
    scale: f64,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<bool, ValidatedGeometryError> {
    charge(0, budget, cancel)?;
    if !scale.is_finite()
        || scale <= 0.0
        || !active.into_iter().all(f64::is_finite)
        || active[0] >= active[1]
    {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    if active[0] < prepared.range[0]
        || active[1] > prepared.range[1]
        || native_range(actual, 0, budget, cancel)? != (active[0], active[1])
    {
        return Ok(false);
    }
    let transform = (scale != 1.0).then(|| Matrix4::from_scale(scale));
    retained(prepared, actual, active, transform, 0, budget, cancel)
}

fn charge(
    depth: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<(), ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    if depth > crate::geometry::MAX_NATIVE_GEOMETRY_NESTING {
        Err(ValidatedGeometryError::ResourceLimit)
    } else {
        Ok(())
    }
}
fn native_range(
    actual: &Curve,
    depth: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<(f64, f64), ValidatedGeometryError> {
    charge(depth, budget, cancel)?;
    match actual {
        Curve::IntersectionCurve(curve) => native_range(curve.leader(), depth + 1, budget, cancel),
        // Every other native family stores its range without traversing a
        // recursive spatial basis, including both kinds of trimmed processor.
        _ => Ok(actual.range_tuple()),
    }
}

fn retained(
    prepared: &PreparedValidatedCurve<'_>,
    actual: &Curve,
    active: [f64; 2],
    transform: Option<Matrix4>,
    depth: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<bool, ValidatedGeometryError> {
    charge(depth, budget, cancel)?;
    let same = match &prepared.source {
        PreparedSource::Borrowed(source) => {
            curve(source, actual, transform, depth, budget, cancel)?
        }
        PreparedSource::Owned { curve: source, .. } => {
            curve(source, actual, transform, depth, budget, cancel)?
        }
        PreparedSource::Mapped {
            transform: pose,
            range,
            orientation,
        } => {
            if let (Curve::TrimmedCurve(actual), Some(child)) = (actual, &prepared.mapped) {
                *orientation == actual.orientation()
                    && *range == actual.entity().range()
                    && same_pose(*pose, *actual.transform(), transform)
                    && retained(
                        child,
                        actual.entity().curve(),
                        child.range,
                        None,
                        depth + 1,
                        budget,
                        cancel,
                    )?
            } else {
                false
            }
        }
    };
    if same {
        return Ok(true);
    }
    let Curve::TrimmedCurve(wrapper) = actual else {
        return Ok(false);
    };
    if !wrapper.orientation() || wrapper.entity().range() != (active[0], active[1]) {
        return Ok(false);
    }
    // Restriction after scaling: its original basis already carries the units.
    if *wrapper.transform() == Matrix4::from_scale(1.0)
        && retained(
            prepared,
            wrapper.entity().curve(),
            active,
            transform,
            depth + 1,
            budget,
            cancel,
        )?
    {
        return Ok(true);
    }
    // Scaling after restriction: the processor alone carries the unit map.
    if let Some(transform) = transform
        && *wrapper.transform() == transform
    {
        return retained(
            prepared,
            wrapper.entity().curve(),
            active,
            None,
            depth + 1,
            budget,
            cancel,
        );
    }
    Ok(false)
}

fn same_pose(source: Matrix4, actual: Matrix4, transform: Option<Matrix4>) -> bool {
    let expected = transform.map_or(source, |transform| transform * source);
    (0..4).all(|i| (0..4).all(|j| expected[i][j].is_finite())) && expected == actual
}

fn point(source: Point3, actual: Point3, transform: Option<Matrix4>) -> bool {
    let expected = transform.map_or(source, |transform| transform.transform_point(source));
    (0..3).all(|i| expected[i].is_finite()) && expected == actual
}

fn homogeneous(source: Vector4, actual: Vector4, transform: Option<Matrix4>) -> bool {
    let expected = transform.map_or(source, |transform| transform * source);
    (0..4).all(|i| expected[i].is_finite()) && expected == actual
}

fn values<T: Copy>(
    source: &[T],
    actual: &[T],
    same: impl Fn(T, T) -> bool,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<bool, ValidatedGeometryError> {
    if source.len() != actual.len() {
        return Ok(false);
    }
    for (&source, &actual) in source.iter().zip(actual) {
        budget.charge(1, cancel)?;
        if !same(source, actual) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn rows<T: Copy>(
    source: &[Vec<T>],
    actual: &[Vec<T>],
    same: impl Fn(T, T) -> bool,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<bool, ValidatedGeometryError> {
    if source.len() != actual.len() {
        return Ok(false);
    }
    for (source, actual) in source.iter().zip(actual) {
        budget.charge(1, cancel)?;
        if !values(source, actual, &same, budget, cancel)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn boundary(
    source: Option<&ParameterCurve<crate::Curve2D, Box<Surface>>>,
    actual: Option<&ParameterCurve<crate::Curve2D, Box<Surface>>>,
    transform: Option<Matrix4>,
    depth: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<bool, ValidatedGeometryError> {
    charge(depth, budget, cancel)?;
    match (source, actual) {
        (None, None) => Ok(true),
        (Some(source), Some(actual)) => {
            Ok(
                exact_curve2d(source.curve(), actual.curve(), depth + 1, budget, cancel)?
                    && surface(
                        source.surface(),
                        actual.surface(),
                        transform,
                        depth + 1,
                        budget,
                        cancel,
                    )?,
            )
        }
        _ => Ok(false),
    }
}

fn curve(
    source: &Curve,
    actual: &Curve,
    transform: Option<Matrix4>,
    depth: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<bool, ValidatedGeometryError> {
    charge(depth, budget, cancel)?;
    Ok(match (source, actual) {
        (Curve::Line(source), Curve::Line(actual)) => {
            point(source.0, actual.0, transform) && point(source.1, actual.1, transform)
        }
        (Curve::BsplineCurve(source), Curve::BsplineCurve(actual)) => {
            equal_slice(source.knot_vector(), actual.knot_vector(), budget, cancel)?
                && values(
                    source.control_points(),
                    actual.control_points(),
                    |a, b| point(a, b, transform),
                    budget,
                    cancel,
                )?
        }
        (Curve::NurbsCurve(source), Curve::NurbsCurve(actual)) => {
            equal_slice(source.knot_vector(), actual.knot_vector(), budget, cancel)?
                && values(
                    source.control_points(),
                    actual.control_points(),
                    |a, b| homogeneous(a, b, transform),
                    budget,
                    cancel,
                )?
        }
        (Curve::ParameterCurve(source), Curve::ParameterCurve(actual)) => {
            exact_curve2d(source.curve(), actual.curve(), depth + 1, budget, cancel)?
                && surface(
                    source.surface(),
                    actual.surface(),
                    transform,
                    depth + 1,
                    budget,
                    cancel,
                )?
        }
        (Curve::IntersectionCurve(source), Curve::IntersectionCurve(actual)) => {
            curve(
                source.leader(),
                actual.leader(),
                transform,
                depth + 1,
                budget,
                cancel,
            )? && surface(
                source.surface0(),
                actual.surface0(),
                transform,
                depth + 1,
                budget,
                cancel,
            )? && surface(
                source.surface1(),
                actual.surface1(),
                transform,
                depth + 1,
                budget,
                cancel,
            )? && boundary(
                source.boundary0(),
                actual.boundary0(),
                transform,
                depth + 1,
                budget,
                cancel,
            )? && boundary(
                source.boundary1(),
                actual.boundary1(),
                transform,
                depth + 1,
                budget,
                cancel,
            )?
        }
        (Curve::TrimmedCurve(source), Curve::TrimmedCurve(actual)) => {
            source.orientation() == actual.orientation()
                && source.entity().range() == actual.entity().range()
                && same_pose(*source.transform(), *actual.transform(), transform)
                && curve(
                    source.entity().curve(),
                    actual.entity().curve(),
                    None,
                    depth + 1,
                    budget,
                    cancel,
                )?
        }
        _ => false,
    })
}

fn surface(
    source: &Surface,
    actual: &Surface,
    transform: Option<Matrix4>,
    depth: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<bool, ValidatedGeometryError> {
    charge(depth, budget, cancel)?;
    Ok(match (source, actual) {
        (Surface::Plane(source), Surface::Plane(actual)) => {
            source
                .defining_points()
                .into_iter()
                .zip(actual.defining_points())
                .all(|(a, b)| point(a, b, transform))
                && match (source.stored_axes(), actual.stored_axes()) {
                    (None, None) => true,
                    (Some(source), Some(actual)) => source.into_iter().zip(actual).all(|(a, b)| {
                        let expected =
                            transform.map_or(a, |transform| transform.transform_vector(a));
                        (0..3).all(|i| expected[i].is_finite()) && expected == b
                    }),
                    _ => false,
                }
        }
        (Surface::BsplineSurface(source), Surface::BsplineSurface(actual)) => {
            equal_slice(
                &source.knot_vectors().0,
                &actual.knot_vectors().0,
                budget,
                cancel,
            )? && equal_slice(
                &source.knot_vectors().1,
                &actual.knot_vectors().1,
                budget,
                cancel,
            )? && rows(
                source.control_points(),
                actual.control_points(),
                |a, b| point(a, b, transform),
                budget,
                cancel,
            )?
        }
        (Surface::NurbsSurface(source), Surface::NurbsSurface(actual)) => {
            equal_slice(
                &source.knot_vectors().0,
                &actual.knot_vectors().0,
                budget,
                cancel,
            )? && equal_slice(
                &source.knot_vectors().1,
                &actual.knot_vectors().1,
                budget,
                cancel,
            )? && rows(
                source.control_points(),
                actual.control_points(),
                |a, b| homogeneous(a, b, transform),
                budget,
                cancel,
            )?
        }
        (Surface::SphericalSurface(source), Surface::SphericalSurface(actual)) => {
            source.orientation() == actual.orientation()
                && source.entity() == actual.entity()
                && same_pose(*source.transform(), *actual.transform(), transform)
        }
        (Surface::ToroidalSurface(source), Surface::ToroidalSurface(actual)) => {
            source.orientation() == actual.orientation()
                && source.entity() == actual.entity()
                && same_pose(*source.transform(), *actual.transform(), transform)
        }
        (Surface::RevolutionSurface(source), Surface::RevolutionSurface(actual)) => {
            source.orientation() == actual.orientation()
                && same_pose(*source.transform(), *actual.transform(), transform)
                && source.entity().origin() == actual.entity().origin()
                && source.entity().axis() == actual.entity().axis()
                && curve(
                    source.entity().entity_curve(),
                    actual.entity().entity_curve(),
                    None,
                    depth + 1,
                    budget,
                    cancel,
                )?
        }
        (Surface::ExtrusionSurface(source), Surface::ExtrusionSurface(actual)) => {
            let vector = source.extruding_vector();
            let expected = transform.map_or(vector, |transform| transform.transform_vector(vector));
            (0..3).all(|i| expected[i].is_finite())
                && expected == actual.extruding_vector()
                && curve(
                    source.entity_curve(),
                    actual.entity_curve(),
                    transform,
                    depth + 1,
                    budget,
                    cancel,
                )?
        }
        // Validated native geometry does not admit a T-spline snapshot.
        _ => false,
    })
}
