//! Complete an unchanged original face's already-owned intersection incidence.
//! A neighboring original face can install the shared coedge even when this
//! face's seed mesh has no chain. Identity, whole original carrier/trim proofs
//! and the actual transverse tangent classify its existing use, never a receipt.

use super::native_overlap::proves_original_branch;
use super::periodic_loops::{AttachmentFailure, bind_existing_edge};
use super::*;

fn opposing_occurrence<C>(
    native: &Loops<Point3, C>,
    polygon: &Loops<Point3, PolylineCurve>,
    original: &Edge<Point3, C>,
    original_polygon: &Edge<Point3, PolylineCurve>,
    work: Option<&BooleanWork<'_>>,
) -> std::result::Result<bool, AttachmentFailure> {
    if native.len() != polygon.len() {
        return Err("intersection_original_overlap_polygon_occurrence".into());
    }
    let mut present = false;
    for (wire, polygon_wire) in native.iter().zip(polygon.iter()) {
        if wire.len() != polygon_wire.len() || wire.status != polygon_wire.status {
            return Err("intersection_original_overlap_polygon_occurrence".into());
        }
        for (edge, polygon_edge) in wire.iter().zip(polygon_wire.iter()) {
            if let Some(work) = work {
                work.charge(1)?;
            }
            if edge.id() != original.id() {
                continue;
            }
            if edge.absolute_front() != original.absolute_front()
                || edge.absolute_back() != original.absolute_back()
                || polygon_edge.id() != original_polygon.id()
                || polygon_edge.orientation() != edge.orientation()
                || polygon_edge.absolute_front() != original_polygon.absolute_front()
                || polygon_edge.absolute_back() != original_polygon.absolute_back()
            {
                return Err("intersection_original_overlap_polygon_occurrence".into());
            }
            present = true;
        }
    }
    Ok(present)
}

pub(super) fn complete<C, S>(
    shells: [&Shell<Point3, C, S>; 2],
    native: [&mut LoopsStore<Point3, C>; 2],
    polygon: [&mut LoopsStore<Point3, PolylineCurve>; 2],
    work: Option<&BooleanWork<'_>>,
) -> ClassicResult<()>
where
    C: Clone + ParametricCurve3D + BoundedCurve + TryIntoHomogeneousBsplineCurve,
    S: Clone
        + ParametricSurface3D
        + SearchParameter<SurfaceParameter, Point = Point3>
        + TryIntoAnalyticSurfaceKind
        + TryIntoHomogeneousBsplineSurface,
{
    for side in 0..2 {
        let other = 1 - side;
        for (face_index, face) in shells[side].iter().enumerate() {
            if let Some(work) = work {
                work.check()?;
            }
            let boundaries = face.absolute_boundaries();
            let loops = &native[side][face_index];
            if let Some(work) = work {
                work.charge(boundaries.iter().map(|wire| wire.len()).sum())?;
            }
            // Cut, generated and partially classified faces retain the ordinary
            // division/classification path. Only complete original uses bind.
            if loops
                .iter()
                .any(|wire| wire.status != ShapesOpStatus::Unknown)
                || loops.len() != boundaries.len()
                || !loops.iter().zip(boundaries).all(|(current, original)| {
                    current.len() == original.len()
                        && current
                            .iter()
                            .zip(original.iter())
                            .all(|(current, original)| {
                                current.id() == original.id()
                                    && current.orientation() == original.orientation()
                                    && current.absolute_front() == original.absolute_front()
                                    && current.absolute_back() == original.absolute_back()
                            })
                })
            {
                continue;
            }
            if !face.with_surface(|surface| {
                surface.original_sphere().is_some() || surface.original_torus().is_some()
            }) {
                continue;
            }
            let carrier = face.surface();
            for (wire_index, wire) in boundaries.iter().enumerate() {
                for (edge_index, original) in wire.iter().enumerate() {
                    for (other_index, opposing) in shells[other].iter().enumerate() {
                        let failure = |error: AttachmentFailure| match error {
                            AttachmentFailure::Geometry(detail) => {
                                let faces = if side == 0 {
                                    [face_index, other_index]
                                } else {
                                    [other_index, face_index]
                                };
                                ShapeOpsError::IntersectionLoopsFailed {
                                    face_index0: faces[0],
                                    face_index1: faces[1],
                                    stage: "original_boundary_incidence",
                                    detail,
                                    native_parameter: None,
                                }
                            }
                            AttachmentFailure::Work(error) => error,
                        };
                        let polygon_edge = polygon[side][face_index]
                            .get(wire_index)
                            .and_then(|wire| wire.get(edge_index))
                            .ok_or_else(|| {
                                failure("intersection_original_overlap_polygon_occurrence".into())
                            })?;
                        if !opposing_occurrence(
                            &native[other][other_index],
                            &polygon[other][other_index],
                            original,
                            polygon_edge,
                            work,
                        )
                        .map_err(failure)?
                        {
                            continue;
                        }
                        // This admission concerns an installed source boundary,
                        // not a source edge originally owned by both operands.
                        // It also prevents nested locks on a shared original.
                        if opposing
                            .absolute_boundaries()
                            .iter()
                            .flat_map(|wire| wire.iter())
                            .any(|edge| edge.id() == original.id())
                        {
                            continue;
                        }
                        let other_carrier = opposing.surface();
                        let status = original
                            .with_curve(|curve| {
                                let absolute = [
                                    original.absolute_front().point(),
                                    original.absolute_back().point(),
                                ];
                                if !proves_original_branch(
                                    curve,
                                    absolute,
                                    [&carrier, &other_carrier],
                                    opposing.absolute_boundaries(),
                                    work,
                                )? {
                                    return Ok::<_, AttachmentFailure>(None);
                                }
                                let (front, back) = curve.range_tuple();
                                let parameter = front + (back - front) * 0.5;
                                let point = curve.evaluate(parameter);
                                let mut normals = [Vector3::zero(); 2];
                                for (normal, surface) in
                                    normals.iter_mut().zip([&carrier, &other_carrier])
                                {
                                    let (u, v) = surface
                                        .search_parameter(point, None, 100)
                                        .ok_or("intersection_chart_contact_carrier")?;
                                    if !surface.evaluate(u, v).near(&point) {
                                        return Err("intersection_chart_contact_carrier".into());
                                    }
                                    *normal = surface.normal(u, v);
                                }
                                let transverse = normals[0]
                                    .cross(curve.derivative(parameter))
                                    .dot(normals[1]);
                                if !transverse.is_finite() || transverse == 0.0 {
                                    return Err("intersection_chart_tangent".into());
                                }
                                let status = if transverse > 0.0 {
                                    ShapesOpStatus::Or
                                } else {
                                    ShapesOpStatus::And
                                };
                                Ok(Some(if opposing.orientation() {
                                    status
                                } else {
                                    status.not()
                                }))
                            })
                            .map_err(failure)?;
                        if let Some(status) = status {
                            let polygon_edge = polygon_edge.clone();
                            let status = if original.orientation() {
                                status
                            } else {
                                status.not()
                            };
                            bind_existing_edge(
                                &mut native[side][face_index],
                                &mut polygon[side][face_index],
                                original,
                                &polygon_edge,
                                status,
                                work,
                            )
                            .map_err(failure)?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
