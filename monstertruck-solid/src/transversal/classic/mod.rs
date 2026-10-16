//! Classic (0.3.2) boolean pipeline -- the self-contained default backend.
//!
//! This subtree grafts the proven upstream boolean assembly from the published
//! 0.3.2 crate. It is compiled only when the marching SSI is the active backend
//! (default features); a build with an external SSI backend never touches it
//! and runs the current `integrate` pipeline unchanged. The public entry points in
//! `transversal::integrate` cfg-dispatch `and`/`or`/`difference`/
//! `symmetric_difference` here.
//!
//! Graft-boundary adaptations vs. the 0.3.2 source (all documented inline):
//! - The Alternative intersection-curve arm is a raw
//!   `IntersectionCurve<PolylineCurve<Point3>, S, S>` (0.3.2 wrapped it with
//!   parameter polylines the pipeline never consumed).
//! - `altshell_to_shell` converts that arm back to `C` via
//!   `SurfaceCurve::with_boundaries(..).into()` (the conversion the current
//!   `ShapeOpsCurve` bound provides), not `IntersectionCurve::new(..).into()`.
//! - Failures retain the stage-specific `ShapeOpsError` rather than treating
//!   a failed intermediate operation as an empty geometric result.

mod divide_face;
mod faces_classification;
mod intersection_curve;
mod loops_store;

use super::edge_lineage::{
    BooleanCoedgeOrigin, BooleanCoedgeProvenance, EdgeRewrite, EdgeTrace, PairEdgeTrace,
};
use super::integrate::{
    InvalidOutputEdgeIncidenceData, InvalidOutputShellConditionData, ShapeOpsCurve, ShapeOpsError,
    ShapeOpsSurface,
};
use super::prepared::{BooleanWork, MeshShell, PreparedBooleanSource};
use super::provenance::{
    BooleanFaceOrigin, BooleanFaceProvenance, BooleanOperand, BooleanWithProvenance, FaceLineage,
};
use crate::alternative::Alternative;
use monstertruck_geometry::prelude::*;
use monstertruck_meshing::prelude::*;
use monstertruck_topology::*;
use rustc_hash::FxHashMap as HashMap;
use std::{iter, mem::size_of};

type ClassicResult<T> = std::result::Result<T, ShapeOpsError>;

type AltCurve<C, S> = Alternative<C, IntersectionCurve<PolylineCurve<Point3>, S, S>>;
type AltCurveShell<C, S> = Shell<Point3, AltCurve<C, S>, S>;

fn oriented_shell_crossings(
    shell: &Shell<Point3, PolylineCurve<Point3>, Option<PolygonMesh>>,
    point: Point3,
    direction: Vector3,
) -> Option<isize> {
    shell.iter().try_fold(0, |count, face| {
        let crossings = face.with_surface(|polygon| {
            polygon
                .as_ref()
                .map(|polygon| polygon.signed_crossing_faces(point, direction))
        })?;
        Some(
            count
                + if face.orientation() {
                    crossings
                } else {
                    -crossings
                },
        )
    })
}

/// Convert an Alternative-curve shell back to a target-curve shell, approximating
/// each intersection-curve arm with a polyline B-spline wrapped in a
/// `SurfaceCurve` (uses the `From<SurfaceCurve<BsplineCurve<Point3>, ..>>`
/// conversion the current `ShapeOpsCurve` bound guarantees).
fn altshell_to_shell<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    altshell: &AltCurveShell<C, S>,
) -> Option<Shell<Point3, C, S>> {
    altshell.try_mapped(
        |p| Some(*p),
        |c| match c {
            Alternative::FirstType(c) => Some(c.clone()),
            Alternative::SecondType(ic) => {
                let surface0 = ic.surface0().clone();
                let surface1 = ic.surface1().clone();
                let mut points = ic.leader().as_slice().to_vec();
                points.dedup_by(|lhs, rhs| (*lhs).near(&*rhs));
                if points.len() < 2 {
                    return None;
                }
                let denominator = (points.len() - 1) as f64;
                let knot_vec = KnotVector::from(
                    iter::once(0.0)
                        .chain((0..points.len()).map(|index| index as f64 / denominator))
                        .chain(iter::once(1.0))
                        .collect::<Vec<_>>(),
                );
                let bspline = BsplineCurve::new(knot_vec, points);
                let boundary0: Option<ParameterCurve<BoundaryCurve2D, S>> = None;
                let boundary1: Option<ParameterCurve<BoundaryCurve2D, S>> = None;
                Some(
                    SurfaceCurve::with_boundaries(
                        surface0, surface1, bspline, boundary0, boundary1,
                    )
                    .into(),
                )
            }
        },
        |s| Some(s.clone()),
    )
}

/// Split one pair of shells into `[and_shell, or_shell]`: build intersection
/// loops, divide faces, classify each divided face as `and`/`or`, then ray-cast
/// the remaining `unknown` faces against the other shell. Retain failed stage
/// identity instead of conflating intermediate failure with an empty result.
fn process_one_pair_of_shells<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    shell0: &Shell<Point3, C, S>,
    shell1: &Shell<Point3, C, S>,
    tol: f64,
    retain_or: bool,
) -> ClassicResult<[Shell<Point3, C, S>; 2]> {
    if tol <= 0.0 {
        return Err(ShapeOpsError::InvalidTolerance);
    }
    let poly_shell0 = shell0.triangulation(tol);
    let poly_shell1 = shell1.triangulation(tol);
    process_shell_pair(
        shell0,
        &poly_shell0,
        shell1,
        &poly_shell1,
        tol,
        retain_or,
        None,
    )
}

fn process_shell_pair<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    shell0: &Shell<Point3, C, S>,
    poly_shell0: &MeshShell,
    shell1: &Shell<Point3, C, S>,
    poly_shell1: &MeshShell,
    tol: f64,
    retain_or: bool,
    work: Option<&BooleanWork<'_>>,
) -> ClassicResult<[Shell<Point3, C, S>; 2]> {
    process_shell_pair_with_origins(
        [
            ShellInput {
                native: shell0,
                mesh: poly_shell0,
            },
            ShellInput {
                native: shell1,
                mesh: poly_shell1,
            },
        ],
        tol,
        retain_or,
        work,
        None,
        None,
    )
    .map(|(shells, _)| shells)
}

#[derive(Debug)]
struct ShellInput<'a, C, S> {
    native: &'a Shell<Point3, C, S>,
    mesh: &'a MeshShell,
}

type ShellPairWithOrigins<C, S> = ([Shell<Point3, C, S>; 2], Vec<BooleanFaceOrigin<S>>);

fn process_shell_pair_with_origins<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    inputs: [ShellInput<'_, C, S>; 2],
    tol: f64,
    retain_or: bool,
    work: Option<&BooleanWork<'_>>,
    input_origins: Option<[&[BooleanFaceOrigin<S>]; 2]>,
    mut edge_trace: Option<&mut EdgeTrace<C>>,
) -> ClassicResult<ShellPairWithOrigins<C, S>> {
    let [
        ShellInput {
            native: shell0,
            mesh: poly_shell0,
        },
        ShellInput {
            native: shell1,
            mesh: poly_shell1,
        },
    ] = inputs;
    let mut lineage = input_origins.map(|_| FaceLineage::new());
    if let Some(work) = work {
        work.check()?;
    }
    let altshell0: AltCurveShell<C, S> =
        shell0.mapped(|x| *x, |c| Alternative::FirstType(c.clone()), Clone::clone);
    let altshell1: AltCurveShell<C, S> =
        shell1.mapped(|x| *x, |c| Alternative::FirstType(c.clone()), Clone::clone);
    let mut pair_edges = PairEdgeTrace::new();
    if let (Some(trace), Some(work)) = (edge_trace.as_deref_mut(), work) {
        for (native, mapped) in [(shell0, &altshell0), (shell1, &altshell1)] {
            for (old, new) in native.iter().zip(mapped.iter()) {
                for (old, new) in old
                    .absolute_boundaries()
                    .iter()
                    .zip(new.absolute_boundaries())
                {
                    for (old, new) in old.iter().zip(new.iter()) {
                        work.check()?;
                        if let Some(index) = trace.index(old.id()) {
                            pair_edges.bind(new.id(), index, work)?;
                        }
                    }
                }
            }
        }
    }
    let loop_result = if let (Some(trace), Some(work)) = (edge_trace.as_deref_mut(), work) {
        let origins = input_origins.ok_or(ShapeOpsError::MissingFaceLineage)?;
        let mut record = |old, new: &Edge<Point3, AltCurve<C, S>>, event| match event {
            EdgeRewrite::GeneratedUse { input, face_index } => {
                let Some(origin) = origins
                    .get(input)
                    .and_then(|faces| faces.get(face_index))
                    .copied()
                else {
                    pair_edges.error = Some(ShapeOpsError::MissingFaceLineage);
                    return false;
                };
                pair_edges.generated_use(trace, old, origin, work)
            }
            _ => pair_edges.rewrite(trace, old, new, event, work),
        };
        loops_store::create_loops_stores(
            &altshell0,
            poly_shell0,
            &altshell1,
            poly_shell1,
            tol,
            Some(work),
            Some(&mut record),
        )
    } else {
        loops_store::create_loops_stores(
            &altshell0,
            poly_shell0,
            &altshell1,
            poly_shell1,
            tol,
            work,
            None,
        )
    };
    if let Some(error) = pair_edges.error.take() {
        return Err(error);
    }
    let loops_store::LoopsStoreQuadruple {
        geom_loops_store0: loops_store0,
        geom_loops_store1: loops_store1,
        ..
    } = loop_result?;
    if let Some(work) = work {
        work.check()?;
    }
    let mut curve_divisions = divide_face::CurveDivisionCache::new(tol);
    let mut cls0 = divide_face::divide_faces_recording(
        &altshell0,
        &loops_store0,
        0,
        &mut curve_divisions,
        work,
        &mut |index, face| {
            if let (Some(origins), Some(lineage), Some(work)) =
                (input_origins, lineage.as_mut(), work)
            {
                lineage.insert(face, origins[0][index], work)?;
            }
            Ok(())
        },
    )?;
    if !cls0.integrate_by_component() {
        return Err(ShapeOpsError::UnknownClassificationFailed { shell_index: 0 });
    }
    let mut cls1 = divide_face::divide_faces_recording(
        &altshell1,
        &loops_store1,
        1,
        &mut curve_divisions,
        work,
        &mut |index, face| {
            if let (Some(origins), Some(lineage), Some(work)) =
                (input_origins, lineage.as_mut(), work)
            {
                lineage.insert(face, origins[1][index], work)?;
            }
            Ok(())
        },
    )?;
    if !cls1.integrate_by_component() {
        return Err(ShapeOpsError::UnknownClassificationFailed { shell_index: 1 });
    }
    if let Some(work) = work {
        work.check()?;
    }
    let [mut and0, mut or0, unknown0] = cls0.into_and_or_unknown(retain_or);
    unknown0.into_iter().try_for_each(|face| {
        let pt = face.boundaries()[0].vertex_iter().next().unwrap().point();
        let dir = hash::take_one_unit(pt);
        if let Some(work) = work {
            work.check()?;
        }
        let count = oriented_shell_crossings(poly_shell1, pt, dir)
            .ok_or(ShapeOpsError::UnknownClassificationFailed { shell_index: 0 })?;
        if count >= 1 {
            and0.push(face);
        } else if retain_or {
            or0.push(face);
        }
        Ok::<(), ShapeOpsError>(())
    })?;
    let [mut and1, mut or1, unknown1] = cls1.into_and_or_unknown(retain_or);
    unknown1.into_iter().try_for_each(|face| {
        let pt = face.boundaries()[0].vertex_iter().next().unwrap().point();
        let dir = hash::take_one_unit(pt);
        if let Some(work) = work {
            work.check()?;
        }
        let count = oriented_shell_crossings(poly_shell0, pt, dir)
            .ok_or(ShapeOpsError::UnknownClassificationFailed { shell_index: 1 })?;
        if count >= 1 {
            and1.push(face);
        } else if retain_or {
            or1.push(face);
        }
        Ok::<(), ShapeOpsError>(())
    })?;
    if let Some(work) = work {
        work.check()?;
    }
    and0.append(&mut and1);
    or0.append(&mut or1);
    if let Some(work) = work {
        work.output(&and0)?;
    }
    let converted = [
        altshell_to_shell(&and0)
            .ok_or(ShapeOpsError::AltShellConversionFailed { operation: "and" })?,
        if retain_or {
            altshell_to_shell(&or0)
                .ok_or(ShapeOpsError::AltShellConversionFailed { operation: "or" })?
        } else {
            Shell::default()
        },
    ];
    if let (Some(trace), Some(work)) = (edge_trace, work) {
        for (old, new) in and0.iter().zip(converted[0].iter()) {
            for (old, new) in old
                .absolute_boundaries()
                .iter()
                .zip(new.absolute_boundaries())
            {
                for (old, new) in old.iter().zip(new.iter()) {
                    work.check()?;
                    if let Some(index) = pair_edges.index(old.id()) {
                        trace.bind_native(new.id(), index, work)?;
                    }
                }
            }
        }
        trace.prune_shell(&converted[0], work)?;
    }
    let origins = if let Some(lineage) = lineage {
        and0.iter()
            .map(|face| lineage.origin(face.id()))
            .collect::<ClassicResult<Vec<_>>>()?
    } else {
        Vec::new()
    };
    Ok((converted, origins))
}

fn finalize<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    operation: &'static str,
    shell: Shell<Point3, C, S>,
) -> ClassicResult<Solid<Point3, C, S>> {
    let boundaries = shell.connected_components();
    boundaries
        .iter()
        .enumerate()
        .try_for_each(|(shell_index, shell)| {
            shell.check_solid_boundary().map_err(|source| {
                let mut incidences =
                    HashMap::<EdgeId<C>, InvalidOutputEdgeIncidenceData>::default();
                shell.iter().enumerate().for_each(|(face_index, face)| {
                    face.edge_iter().for_each(|edge| {
                        let data = incidences.entry(edge.id()).or_insert_with(|| {
                            InvalidOutputEdgeIncidenceData {
                                endpoints: [
                                    edge.absolute_front().point(),
                                    edge.absolute_back().point(),
                                ],
                                counts: [0; 2],
                                uses: [None; 2],
                            }
                        });
                        let count = data.counts[0] + data.counts[1];
                        if count < data.uses.len() {
                            data.uses[count] =
                                Some((face_index, face.orientation(), edge.orientation()));
                        }
                        data.counts[usize::from(!edge.orientation())] += 1;
                    });
                });
                let (bad_edge_count, first_bad_edge) = incidences
                    .values()
                    .filter(|data| data.counts != [1, 1])
                    .fold((0, None), |(count, mut first), data| {
                        if first.is_none() {
                            first = Some(*data);
                        }
                        (count + 1, first)
                    });
                let boundary = shell.extract_boundaries();
                let singular = shell.singular_vertices();
                ShapeOpsError::InvalidOutputShellCondition(Box::new(
                    InvalidOutputShellConditionData {
                        operation,
                        shell_index,
                        source,
                        bad_edge_count,
                        first_bad_edge,
                        empty: shell.is_empty(),
                        connected: shell.is_connected(),
                        condition: shell.shell_condition(),
                        boundary_loops: boundary.len(),
                        first_boundary_len: boundary.first().map(|wire| wire.len()),
                        first_boundary_front: boundary
                            .first()
                            .and_then(Wire::front_vertex)
                            .map(Vertex::point),
                        first_boundary_back: boundary
                            .first()
                            .and_then(Wire::back_vertex)
                            .map(Vertex::point),
                        singular_vertices: singular.len(),
                        first_singular: singular.first().map(Vertex::point),
                    },
                ))
            })
        })?;
    // These are exactly `Solid::try_new`'s release-active boundary checks.
    // Certifying borrowed shells retains failure evidence without cloning
    // successful topology or validating the same boundaries twice.
    Ok(Solid::new_unchecked(boundaries))
}

/// AND operation between two solids (classic backend).
pub(crate) fn and<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tol: f64,
) -> ClassicResult<Solid<Point3, C, S>> {
    let operation = "and";
    let pair = |a: &Shell<Point3, C, S>, b: &Shell<Point3, C, S>| {
        process_one_pair_of_shells(a, b, tol, false)
    };
    let mut iter0 = solid0.boundaries().iter();
    let mut iter1 = solid1.boundaries().iter();
    let shell0 = iter0.next().unwrap();
    let shell1 = iter1.next().unwrap();
    let [mut and_shell, _] = pair(shell0, shell1)?;
    for shell in iter0 {
        let [res, _] = pair(&and_shell, shell)?;
        and_shell = res;
    }
    for shell in iter1 {
        let [res, _] = pair(&and_shell, shell)?;
        and_shell = res;
    }
    finalize(operation, and_shell)
}

/// AND using exactly the source-owned seed shells and the common pair backend.
pub(crate) fn and_prepared<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    prepared: &PreparedBooleanSource<'_, C, S>,
    other: &Solid<Point3, C, S>,
    other_meshes: &[MeshShell],
    work: &BooleanWork<'_>,
) -> ClassicResult<Solid<Point3, C, S>> {
    let tolerance = prepared.tolerance();
    let mut sources = prepared.source.boundaries().iter().zip(&prepared.meshes);
    let mut others = other.boundaries().iter().zip(other_meshes);
    let (source, source_mesh) = sources
        .next()
        .ok_or(ShapeOpsError::EmptyOutputShell { operation: "and" })?;
    let (other, other_mesh) = others
        .next()
        .ok_or(ShapeOpsError::EmptyOutputShell { operation: "and" })?;
    let [mut result, _] = process_shell_pair(
        source,
        source_mesh,
        other,
        other_mesh,
        tolerance,
        false,
        Some(work),
    )?;
    for (shell, mesh) in sources.chain(others) {
        work.check()?;
        work.charge(result.len())?;
        let result_mesh = result.triangulation(prepared.seed_tolerance());
        work.check()?;
        work.intermediate(&result_mesh)?;
        let [next, _] = process_shell_pair(
            &result,
            &result_mesh,
            shell,
            mesh,
            tolerance,
            false,
            Some(work),
        )?;
        result = next;
    }
    work.output(&result)?;
    work.check()?;
    let result = finalize("and", result)?;
    work.check()?;
    Ok(result)
}

pub(crate) fn and_prepared_with_provenance<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    prepared: &PreparedBooleanSource<'_, C, S>,
    other: &Solid<Point3, C, S>,
    other_meshes: &[MeshShell],
    work: &BooleanWork<'_>,
) -> ClassicResult<BooleanWithProvenance<C, S>> {
    let other_solid = other;
    let mut edge_trace = EdgeTrace::new();
    edge_trace.admit(prepared.source, work)?;
    edge_trace.admit(other, work)?;
    let origins = |shell: &Shell<Point3, C, S>, operand, shell_index| {
        work.limit(
            "face_lineage_bytes",
            shell
                .len()
                .saturating_mul(size_of::<BooleanFaceOrigin<S>>()),
            work.limits.max_intermediate_prepared_bytes,
        )?;
        shell
            .iter()
            .enumerate()
            .map(|(face_index, face)| {
                work.check()?;
                Ok(BooleanFaceOrigin {
                    operand,
                    face_id: face.id(),
                    shell_index,
                    face_index,
                    orientation: face.orientation(),
                })
            })
            .collect::<ClassicResult<Vec<_>>>()
    };
    let mut sources = prepared
        .source
        .boundaries()
        .iter()
        .zip(&prepared.meshes)
        .enumerate()
        .map(|(index, (shell, mesh))| (BooleanOperand::Source, index, shell, mesh));
    let mut others = other
        .boundaries()
        .iter()
        .zip(other_meshes)
        .enumerate()
        .map(|(index, (shell, mesh))| (BooleanOperand::Other, index, shell, mesh));
    let (_, source_index, source, source_mesh) = sources
        .next()
        .ok_or(ShapeOpsError::EmptyOutputShell { operation: "and" })?;
    let (_, other_index, other, other_mesh) = others
        .next()
        .ok_or(ShapeOpsError::EmptyOutputShell { operation: "and" })?;
    let source_origins = origins(source, BooleanOperand::Source, source_index)?;
    let other_origins = origins(other, BooleanOperand::Other, other_index)?;
    let ([mut result, _], mut result_origins) = process_shell_pair_with_origins(
        [
            ShellInput {
                native: source,
                mesh: source_mesh,
            },
            ShellInput {
                native: other,
                mesh: other_mesh,
            },
        ],
        prepared.tolerance(),
        false,
        Some(work),
        Some([&source_origins, &other_origins]),
        Some(&mut edge_trace),
    )?;
    for (operand, shell_index, shell, mesh) in sources.chain(others) {
        work.check()?;
        work.charge(result.len())?;
        let result_mesh = result.triangulation(prepared.seed_tolerance());
        work.check()?;
        work.intermediate(&result_mesh)?;
        let shell_origins = origins(shell, operand, shell_index)?;
        let ([next, _], next_origins) = process_shell_pair_with_origins(
            [
                ShellInput {
                    native: &result,
                    mesh: &result_mesh,
                },
                ShellInput {
                    native: shell,
                    mesh,
                },
            ],
            prepared.tolerance(),
            false,
            Some(work),
            Some([&result_origins, &shell_origins]),
            Some(&mut edge_trace),
        )?;
        result = next;
        result_origins = next_origins;
    }
    work.output(&result)?;
    let mut selected = FaceLineage::new();
    result
        .iter()
        .zip(result_origins)
        .try_for_each(|(face, origin)| selected.insert(face.id(), origin, work))?;
    work.check()?;
    let solid = finalize("and", result)?;
    work.limit(
        "face_provenance_bytes",
        solid
            .face_iter()
            .count()
            .saturating_mul(size_of::<BooleanFaceProvenance<S>>()),
        work.limits.max_intermediate_prepared_bytes,
    )?;
    let face_origins = solid
        .face_iter()
        .map(|face| {
            work.check()?;
            Ok(BooleanFaceProvenance {
                output_face: face.id(),
                source: selected.origin(face.id())?,
            })
        })
        .collect::<ClassicResult<Vec<_>>>()?;
    let coedge_count = solid
        .face_iter()
        .map(|face| {
            face.absolute_boundaries()
                .iter()
                .map(|wire| wire.len())
                .sum::<usize>()
        })
        .sum::<usize>();
    work.limit(
        "coedge_provenance_bytes",
        coedge_count.saturating_mul(size_of::<BooleanCoedgeProvenance<C, S>>()),
        work.limits.max_intermediate_prepared_bytes,
    )?;
    let mut coedge_origins = Vec::with_capacity(coedge_count);
    for face in solid.face_iter() {
        let origin = selected.origin(face.id())?;
        let original_solid = match origin.operand {
            BooleanOperand::Source => prepared.source,
            BooleanOperand::Other => other_solid,
        };
        let original_face = original_solid
            .boundaries()
            .get(origin.shell_index)
            .and_then(|shell| shell.get(origin.face_index))
            .ok_or(ShapeOpsError::MissingFaceLineage)?;
        for (wire_index, wire) in face.absolute_boundaries().iter().enumerate() {
            for (coedge_index, edge) in wire.iter().enumerate() {
                work.check()?;
                let source = if let Some(derivation) = edge_trace.index(edge.id()) {
                    let (root, generated_use) = edge_trace.root_use(derivation, origin, work)?;
                    let mut occurrence = None;
                    let mut ambiguous = false;
                    for (source_wire, wire) in
                        original_face.absolute_boundaries().iter().enumerate()
                    {
                        for (source_coedge, old) in wire.iter().enumerate() {
                            work.check()?;
                            if old.id() == root && old.orientation() == edge.orientation() {
                                if occurrence.is_some() {
                                    ambiguous = true;
                                }
                                occurrence = Some((source_wire, source_coedge));
                            }
                        }
                    }
                    if ambiguous {
                        BooleanCoedgeOrigin::AmbiguousOriginalUse {
                            face: origin,
                            edge_id: root,
                        }
                    } else if let Some((wire_index, coedge_index)) = occurrence {
                        BooleanCoedgeOrigin::Original {
                            face: origin,
                            wire_index,
                            coedge_index,
                            edge_id: root,
                            orientation: edge.orientation() == origin.orientation,
                            derivation,
                        }
                    } else if generated_use {
                        BooleanCoedgeOrigin::GeneratedIntersection { face: origin }
                    } else {
                        return Err(ShapeOpsError::MissingFaceLineage);
                    }
                } else {
                    BooleanCoedgeOrigin::GeneratedIntersection { face: origin }
                };
                coedge_origins.push(BooleanCoedgeProvenance {
                    output_face: face.id(),
                    wire_index,
                    coedge_index,
                    output_edge: edge.id(),
                    orientation: edge.orientation() == face.orientation(),
                    source,
                });
            }
        }
    }
    Ok(BooleanWithProvenance {
        solid,
        face_origins,
        coedge_origins,
        edge_derivations: edge_trace.nodes,
    })
}

/// OR operation between two solids (classic backend).
pub(crate) fn or<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tol: f64,
) -> ClassicResult<Solid<Point3, C, S>> {
    let operation = "or";
    let pair = |a: &Shell<Point3, C, S>, b: &Shell<Point3, C, S>| {
        process_one_pair_of_shells(a, b, tol, true)
    };
    let mut iter0 = solid0.boundaries().iter();
    let mut iter1 = solid1.boundaries().iter();
    let shell0 = iter0.next().unwrap();
    let shell1 = iter1.next().unwrap();
    let [_, mut or_shell] = pair(shell0, shell1)?;
    for shell in iter0 {
        let [_, res] = pair(&or_shell, shell)?;
        or_shell = res;
    }
    for shell in iter1 {
        let [_, res] = pair(&or_shell, shell)?;
        or_shell = res;
    }
    finalize(operation, or_shell)
}

/// Difference: the region inside `solid0` but outside `solid1` (classic backend).
pub(crate) fn difference<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tol: f64,
) -> ClassicResult<Solid<Point3, C, S>> {
    let mut neg = solid1.clone();
    neg.not();
    and(solid0, &neg, tol)
}

/// Symmetric difference (XOR): the region inside exactly one solid (classic backend).
pub(crate) fn symmetric_difference<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tol: f64,
) -> ClassicResult<Solid<Point3, C, S>> {
    let d0 = difference(solid0, solid1, tol)?;
    let d1 = difference(solid1, solid0, tol)?;
    or(&d0, &d1, tol)
}

#[cfg(test)]
mod oriented_ray_tests {
    use super::*;
    use monstertruck_modeling as modeling;

    #[test]
    fn material_winding_is_independent_of_carrier_and_face_orientation_representation() {
        let original: modeling::Solid = modeling::primitive::cuboid(BoundingBox::from_iter([
            Point3::origin(),
            Point3::new(1.0, 1.0, 1.0),
        ]));
        let mut alternative = original.boundaries()[0].clone();
        let bottom = alternative
            .iter()
            .position(|face| face.edge_iter().all(|edge| edge.front().point().z == 0.0))
            .unwrap();
        let face = &alternative[bottom];
        let boundaries = face.boundaries().iter().map(Wire::inverse).collect();
        alternative[bottom] = Face::new(boundaries, face.oriented_surface().inverse()).inverse();
        alternative.check_solid_boundary().unwrap();
        assert!(alternative.is_geometric_consistent());
        assert!(!alternative[bottom].orientation());
        let original = original.boundaries()[0].triangulation(0.01);
        let alternative = alternative.triangulation(0.01);
        let direction = Vector3::new(0.137, 0.223, -0.91).normalize();
        for (point, expected) in [
            (Point3::new(0.2, 0.3, 0.4), 1),
            (Point3::new(0.2, 0.3, 1.5), 0),
            (Point3::new(0.2, 0.3, -0.5), 0),
            (Point3::new(-1.0, 0.3, 0.4), 0),
        ] {
            assert_eq!(
                oriented_shell_crossings(&original, point, direction),
                Some(expected)
            );
            assert_eq!(
                oriented_shell_crossings(&alternative, point, direction),
                Some(expected)
            );
        }
    }
}
