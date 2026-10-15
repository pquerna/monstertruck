//! Before-clone accounting of ordinary native curve capture storage.

use super::*;
use std::mem::size_of;
use std::result::Result;
use std::sync::atomic::AtomicBool;

/// Maximum carrier/profile nesting admitted before recursive native clone/evaluation.
pub const MAX_NATIVE_GEOMETRY_NESTING: usize = 32;

#[derive(Clone, Copy)]
enum Node<'a> {
    Curve(&'a Curve),
    Curve2D(&'a Curve2D),
    Surface(&'a Surface),
}

fn add(bytes: &mut usize, amount: usize, maximum: usize) -> Result<(), ValidatedGeometryError> {
    *bytes = bytes
        .checked_add(amount)
        .ok_or(ValidatedGeometryError::ResourceLimit)?;
    if *bytes > maximum {
        Err(ValidatedGeometryError::ResourceLimit)
    } else {
        Ok(())
    }
}

fn product(count: usize, width: usize) -> Result<usize, ValidatedGeometryError> {
    count
        .checked_mul(width)
        .ok_or(ValidatedGeometryError::ResourceLimit)
}

fn push<'a>(
    stack: &mut Vec<(Node<'a>, usize)>,
    node: Node<'a>,
    maximum: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
    depth: usize,
) -> Result<(), ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    if depth > MAX_NATIVE_GEOMETRY_NESTING
        || product(
            stack.len().saturating_add(1),
            size_of::<(Node<'a>, usize)>(),
        )? > maximum
    {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    if stack.len() == stack.capacity() {
        stack
            .try_reserve_exact(1)
            .map_err(|_| ValidatedGeometryError::ResourceLimit)?;
    }
    stack.push((node, depth));
    Ok(())
}

/// Conservatively counts a curve's recursive fixed payloads and owned buffers.
///
/// Source vector capacities bound a deep clone before it occurs. Embedded
/// payloads can be overcounted; allocator headers are excluded. Every recursive
/// scan and bounded scratch growth charges `budget`. T-spline mutable-cache
/// storage has no immutable accounting contract and refuses explicitly.
pub fn curve_owned_bytes(
    curve: &Curve,
    maximum: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<usize, ValidatedGeometryError> {
    owned_bytes(Node::Curve(curve), maximum, budget, cancel)
}

/// Conservatively counts a surface capture under [`curve_owned_bytes`]'s rules.
pub fn surface_owned_bytes(
    surface: &Surface,
    maximum: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<usize, ValidatedGeometryError> {
    owned_bytes(Node::Surface(surface), maximum, budget, cancel)
}

/// Conservatively counts owned ordinary curve storage before a deep clone.
///
/// Counts recursive boxed curves/carriers, fixed object payloads and actual
/// spline/polyline vector capacities. Embedded profile object payloads can be
/// counted twice; this intentionally over-admits nothing. Allocator headers and
/// unused allocator size classes are excluded. Native T-spline mutable caches
/// have no immutable snapshot/accounting contract and refuse explicitly.
///
/// Traversal is iterative, bounded and cooperatively cancellable. Every scan
/// and scratch growth charges the supplied cumulative budget, including errors.
fn owned_bytes(
    root: Node<'_>,
    maximum: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<usize, ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    let minimum = match root {
        Node::Curve(_) => size_of::<Curve>(),
        Node::Curve2D(_) => size_of::<Curve2D>(),
        Node::Surface(_) => size_of::<Surface>(),
    };
    if maximum < minimum {
        return Err(ValidatedGeometryError::ResourceLimit);
    }
    let mut stack = Vec::new();
    let child_depth = 0;
    push(&mut stack, root, maximum, budget, cancel, child_depth)?;
    let mut bytes = 0;
    while let Some((node, depth)) = stack.pop() {
        let child_depth = depth + 1;
        budget.charge(1, cancel)?;
        match node {
            Node::Curve(curve) => {
                add(&mut bytes, size_of::<Curve>(), maximum)?;
                match curve {
                    Curve::Line(_) => {}
                    Curve::TrimmedCurve(curve) => {
                        push(
                            &mut stack,
                            Node::Curve(curve.entity().curve()),
                            maximum,
                            budget,
                            cancel,
                            child_depth,
                        )?;
                    }
                    Curve::BsplineCurve(curve) => {
                        budget.charge(curve.control_points().len(), cancel)?;
                        add(
                            &mut bytes,
                            product(curve.control_points().capacity(), size_of::<Point3>())?,
                            maximum,
                        )?;
                        add(
                            &mut bytes,
                            product(curve.knot_vector().capacity(), size_of::<f64>())?,
                            maximum,
                        )?;
                    }
                    Curve::NurbsCurve(curve) => {
                        budget.charge(curve.control_points().len(), cancel)?;
                        add(
                            &mut bytes,
                            product(curve.control_points().capacity(), size_of::<Vector4>())?,
                            maximum,
                        )?;
                        add(
                            &mut bytes,
                            product(curve.knot_vector().capacity(), size_of::<f64>())?,
                            maximum,
                        )?;
                    }
                    Curve::ParameterCurve(curve) => {
                        push(
                            &mut stack,
                            Node::Curve2D(curve.curve()),
                            maximum,
                            budget,
                            cancel,
                            child_depth,
                        )?;
                        push(
                            &mut stack,
                            Node::Surface(curve.surface()),
                            maximum,
                            budget,
                            cancel,
                            child_depth,
                        )?;
                    }
                    Curve::IntersectionCurve(curve) => {
                        push(
                            &mut stack,
                            Node::Curve(curve.leader()),
                            maximum,
                            budget,
                            cancel,
                            child_depth,
                        )?;
                        push(
                            &mut stack,
                            Node::Surface(curve.surface0()),
                            maximum,
                            budget,
                            cancel,
                            child_depth,
                        )?;
                        push(
                            &mut stack,
                            Node::Surface(curve.surface1()),
                            maximum,
                            budget,
                            cancel,
                            child_depth,
                        )?;
                        for boundary in [curve.boundary0(), curve.boundary1()].into_iter().flatten()
                        {
                            push(
                                &mut stack,
                                Node::Curve2D(boundary.curve()),
                                maximum,
                                budget,
                                cancel,
                                child_depth,
                            )?;
                            push(
                                &mut stack,
                                Node::Surface(boundary.surface()),
                                maximum,
                                budget,
                                cancel,
                                child_depth,
                            )?;
                        }
                    }
                }
            }
            Node::Curve2D(curve) => {
                add(&mut bytes, size_of::<Curve2D>(), maximum)?;
                match curve {
                    Curve2D::Line(_) | Curve2D::Conic(_) => {}
                    Curve2D::TrimmedCurve(curve) => {
                        push(
                            &mut stack,
                            Node::Curve2D(curve.entity().curve()),
                            maximum,
                            budget,
                            cancel,
                            child_depth,
                        )?;
                    }
                    Curve2D::Polyline(curve) => {
                        budget.charge(curve.len(), cancel)?;
                        add(
                            &mut bytes,
                            product(curve.0.capacity(), size_of::<Point2>())?,
                            maximum,
                        )?;
                    }
                    Curve2D::BsplineCurve(curve) => {
                        budget.charge(curve.control_points().len(), cancel)?;
                        add(
                            &mut bytes,
                            product(curve.control_points().capacity(), size_of::<Point2>())?,
                            maximum,
                        )?;
                        add(
                            &mut bytes,
                            product(curve.knot_vector().capacity(), size_of::<f64>())?,
                            maximum,
                        )?;
                    }
                    Curve2D::NurbsCurve(curve) => {
                        budget.charge(curve.control_points().len(), cancel)?;
                        add(
                            &mut bytes,
                            product(curve.control_points().capacity(), size_of::<Vector3>())?,
                            maximum,
                        )?;
                        add(
                            &mut bytes,
                            product(curve.knot_vector().capacity(), size_of::<f64>())?,
                            maximum,
                        )?;
                    }
                }
            }
            Node::Surface(surface) => {
                add(&mut bytes, size_of::<Surface>(), maximum)?;
                match surface {
                    Surface::Plane(_)
                    | Surface::SphericalSurface(_)
                    | Surface::ToroidalSurface(_) => {}
                    Surface::RevolutionSurface(surface) => {
                        push(
                            &mut stack,
                            Node::Curve(surface.entity().entity_curve()),
                            maximum,
                            budget,
                            cancel,
                            child_depth,
                        )?;
                    }
                    Surface::ExtrusionSurface(surface) => {
                        push(
                            &mut stack,
                            Node::Curve(surface.entity_curve()),
                            maximum,
                            budget,
                            cancel,
                            child_depth,
                        )?;
                    }
                    Surface::BsplineSurface(surface) => {
                        let rows = surface.control_points();
                        budget.charge(rows.len(), cancel)?;
                        add(
                            &mut bytes,
                            product(rows.capacity(), size_of::<Vec<Point3>>())?,
                            maximum,
                        )?;
                        for row in rows {
                            budget.charge(row.len(), cancel)?;
                            add(
                                &mut bytes,
                                product(row.capacity(), size_of::<Point3>())?,
                                maximum,
                            )?;
                        }
                        add(
                            &mut bytes,
                            product(surface.knot_vector_u().capacity(), size_of::<f64>())?,
                            maximum,
                        )?;
                        add(
                            &mut bytes,
                            product(surface.knot_vector_v().capacity(), size_of::<f64>())?,
                            maximum,
                        )?;
                    }
                    Surface::NurbsSurface(surface) => {
                        let rows = surface.control_points();
                        budget.charge(rows.len(), cancel)?;
                        add(
                            &mut bytes,
                            product(rows.capacity(), size_of::<Vec<Vector4>>())?,
                            maximum,
                        )?;
                        for row in rows {
                            budget.charge(row.len(), cancel)?;
                            add(
                                &mut bytes,
                                product(row.capacity(), size_of::<Vector4>())?,
                                maximum,
                            )?;
                        }
                        add(
                            &mut bytes,
                            product(surface.knot_vector_u().capacity(), size_of::<f64>())?,
                            maximum,
                        )?;
                        add(
                            &mut bytes,
                            product(surface.knot_vector_v().capacity(), size_of::<f64>())?,
                            maximum,
                        )?;
                    }
                    Surface::TsplineSurface(_) => return Err(ValidatedGeometryError::Unsupported),
                }
            }
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod native_storage_depth_regressions {
    use super::*;

    #[test]
    fn before_clone_accounting_and_jets_share_structural_nesting_limit() {
        let mut curve = Curve::Line(Line(Point3::origin(), Point3::new(1.0, 0.0, 0.0)));
        for _ in 0..=MAX_NATIVE_GEOMETRY_NESTING / 2 {
            let carrier =
                Surface::ExtrusionSurface(ExtrusionSurface::by_extrusion(curve, Vector3::unit_z()));
            curve = Curve::ParameterCurve(ParameterCurve::new(
                Curve2D::Line(Line(Point2::origin(), Point2::new(1.0, 0.0))),
                Box::new(carrier),
            ));
        }
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(100_000);
        assert_eq!(
            curve_owned_bytes(&curve, usize::MAX, &mut budget, &cancel),
            Err(ValidatedGeometryError::ResourceLimit)
        );
        let mut budget = ValidatedGeometryBudget::new(100_000);
        assert!(matches!(
            validated_curve_jet(&curve, [0.0, 1.0], &mut budget, &cancel),
            Err(ValidatedGeometryError::ResourceLimit)
        ));
    }
}
