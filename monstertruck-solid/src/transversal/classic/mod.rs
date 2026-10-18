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
//! - `altcurve_to_curve` converts that arm back to `C` via
//!   `SurfaceCurve::with_boundaries(..).into()`; the owning native topology map
//!   bypasses this conversion for unchanged original edges.
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
    InvalidOutputEdgeIncidenceData, InvalidOutputShellConditionData,
    NativeIntersectionParameterWitness, ShapeOpsCurve, ShapeOpsError, ShapeOpsSurface,
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

/// Absolute curve identity together with both actual endpoint identities.
type NativeEdgeTopologyKey<C, S> = (EdgeId<AltCurve<C, S>>, VertexId<Point3>, VertexId<Point3>);

/// Own the identity evidence for the native -> Alternative -> native round trip.
/// Geometry cloning is not topology identity: only the input mapping and the
/// face-division source receipt authorize recovery of an original object.
struct NativeTopologyMap<'a, C, S> {
    vertices: HashMap<VertexId<Point3>, Vertex<Point3>>,
    edges: HashMap<EdgeId<AltCurve<C, S>>, Edge<Point3, C>>,
    faces: HashMap<FaceId<S>, usize>,
    source_faces: Vec<&'a Face<Point3, C, S>>,
    converted_edges: HashMap<NativeEdgeTopologyKey<C, S>, Edge<Point3, C>>,
}

impl<'a, C: ShapeOpsCurve<S>, S: ShapeOpsSurface> NativeTopologyMap<'a, C, S> {
    fn new() -> Self {
        Self {
            vertices: HashMap::default(),
            edges: HashMap::default(),
            faces: HashMap::default(),
            source_faces: Vec::new(),
            converted_edges: HashMap::default(),
        }
    }

    fn admit(&self, additional: usize, work: Option<&BooleanWork<'_>>) -> ClassicResult<()> {
        if let Some(work) = work {
            work.check()?;
            let tables =
                identity_table_bytes::<VertexId<Point3>, Vertex<Point3>>(self.vertices.len())
                    .saturating_add(identity_table_bytes::<
                        EdgeId<AltCurve<C, S>>,
                        Edge<Point3, C>,
                    >(self.edges.len()))
                    .saturating_add(identity_table_bytes::<FaceId<S>, usize>(self.faces.len()))
                    .saturating_add(identity_table_bytes::<
                        NativeEdgeTopologyKey<C, S>,
                        Edge<Point3, C>,
                    >(self.converted_edges.len()));
            work.limit(
                "native_identity_map_bytes",
                tables
                    .saturating_add(
                        self.source_faces
                            .capacity()
                            .saturating_mul(size_of::<&Face<Point3, C, S>>()),
                    )
                    .saturating_add(additional),
                work.limits.max_intermediate_prepared_bytes,
            )?;
        }
        Ok(())
    }

    fn map_input(
        &mut self,
        shell: &'a Shell<Point3, C, S>,
        work: Option<&BooleanWork<'_>>,
    ) -> ClassicResult<AltCurveShell<C, S>> {
        if let Some(work) = work {
            work.check()?;
        }
        let mapped = shell.mapped(|p| *p, |c| Alternative::FirstType(c.clone()), Clone::clone);
        for (original, mapped) in shell.iter().zip(mapped.iter()) {
            original.absolute_boundaries().iter().try_for_each(|wire| {
                if let Some(work) = work {
                    work.charge(wire.len())?;
                }
                Ok::<_, ShapeOpsError>(())
            })?;
            let old_capacity = self.source_faces.capacity();
            let capacity = if self.source_faces.len() == old_capacity {
                old_capacity.max(1).saturating_mul(2)
            } else {
                old_capacity
            };
            self.admit(
                capacity
                    .saturating_sub(old_capacity)
                    .saturating_mul(size_of::<&Face<Point3, C, S>>())
                    .saturating_add(identity_table_bytes::<FaceId<S>, usize>(1)),
                work,
            )?;
            if capacity > old_capacity {
                self.source_faces
                    .reserve_exact(capacity - self.source_faces.len());
            }
            let index = self.source_faces.len();
            self.source_faces.push(original);
            self.faces.insert(mapped.id(), index);
            self.admit(0, work)?;
            for (original, mapped) in original
                .absolute_boundaries()
                .iter()
                .zip(mapped.absolute_boundaries())
            {
                for (original, mapped) in original.iter().zip(mapped.iter()) {
                    self.admit(
                        identity_table_bytes::<EdgeId<AltCurve<C, S>>, Edge<Point3, C>>(1)
                            .saturating_add(
                                identity_table_bytes::<VertexId<Point3>, Vertex<Point3>>(2),
                            ),
                        work,
                    )?;
                    self.edges
                        .entry(mapped.id())
                        .or_insert_with(|| original.absolute_clone());
                    for (original, mapped) in [
                        (original.absolute_front(), mapped.absolute_front()),
                        (original.absolute_back(), mapped.absolute_back()),
                    ] {
                        self.vertices
                            .entry(mapped.id())
                            .or_insert_with(|| original.clone());
                    }
                }
            }
        }
        Ok(mapped)
    }

    /// Division copies the exact source surface, never a fitted replacement.
    /// Keep that receipt even when provenance collection is not requested.
    fn inherit_face(
        &mut self,
        face: FaceId<S>,
        source: FaceId<S>,
        work: Option<&BooleanWork<'_>>,
    ) -> ClassicResult<()> {
        self.admit(identity_table_bytes::<FaceId<S>, usize>(1), work)?;
        let index = self.faces[&source];
        self.faces.insert(face, index);
        Ok(())
    }

    fn vertex(&self, vertex: &Vertex<Point3>) -> Vertex<Point3> {
        self.vertices.get(&vertex.id()).unwrap_or(vertex).clone()
    }

    fn edge(
        &mut self,
        edge: &Edge<Point3, AltCurve<C, S>>,
        operation: &'static str,
        work: Option<&BooleanWork<'_>>,
    ) -> ClassicResult<Edge<Point3, C>> {
        if let Some(work) = work {
            work.check()?;
        }
        let front = self.vertex(edge.absolute_front());
        let back = self.vertex(edge.absolute_back());
        // Include endpoint identities: a vertex rewrite must not inherit the
        // original edge merely because its curve still has an original ID.
        let key = (edge.id(), front.id(), back.id());
        let mut converted = if let Some(converted) = self.converted_edges.get(&key) {
            converted.clone()
        } else {
            self.admit(
                identity_table_bytes::<NativeEdgeTopologyKey<C, S>, Edge<Point3, C>>(1),
                work,
            )?;
            let original = self.edges.get(&edge.id()).filter(|original| {
                original.absolute_front() == &front && original.absolute_back() == &back
            });
            let converted = if let Some(original) = original {
                original.clone()
            } else {
                let curve = edge
                    .with_curve(altcurve_to_curve)
                    .ok_or(ShapeOpsError::AltShellConversionFailed { operation })?;
                Edge::debug_new(&front, &back, curve)
                    .map_err(|_| ShapeOpsError::AltShellConversionFailed { operation })?
            };
            self.converted_edges.insert(key, converted.clone());
            converted
        };
        if !edge.orientation() {
            converted.invert();
        }
        Ok(converted)
    }

    fn original_face(
        &self,
        face: &Face<Point3, AltCurve<C, S>, S>,
        work: Option<&BooleanWork<'_>>,
    ) -> ClassicResult<Option<&'a Face<Point3, C, S>>> {
        let Some(&index) = self.faces.get(&face.id()) else {
            return Ok(None);
        };
        let source = self.source_faces[index];
        let original = source.absolute_boundaries();
        let mapped = face.absolute_boundaries();
        if original.len() != mapped.len() {
            return Ok(None);
        }
        // Compare every directed occurrence, including wire order and repeated
        // seam uses, under the exact source-surface receipt. No temporary
        // output wires or curve copies are needed for an unchanged face.
        for (original, mapped) in original.iter().zip(mapped) {
            if original.len() != mapped.len() {
                return Ok(None);
            }
            for (original, mapped) in original.iter().zip(mapped.iter()) {
                if let Some(work) = work {
                    work.check()?;
                }
                let Some(edge) = self.edges.get(&mapped.id()) else {
                    return Ok(None);
                };
                if edge.id() != original.id()
                    || mapped.orientation() != original.orientation()
                    || self
                        .vertices
                        .get(&mapped.absolute_front().id())
                        .unwrap_or(mapped.absolute_front())
                        != original.absolute_front()
                    || self
                        .vertices
                        .get(&mapped.absolute_back().id())
                        .unwrap_or(mapped.absolute_back())
                        != original.absolute_back()
                {
                    return Ok(None);
                }
            }
        }
        Ok(Some(source))
    }

    fn shell(
        &mut self,
        shell: &AltCurveShell<C, S>,
        operation: &'static str,
        work: Option<&BooleanWork<'_>>,
    ) -> ClassicResult<Shell<Point3, C, S>> {
        self.admit(0, work)?;
        shell
            .iter()
            .map(|face| {
                if let Some(work) = work {
                    work.check()?;
                }
                let mut converted = if let Some(original) = self.original_face(face, work)? {
                    original.absolute_clone()
                } else {
                    let boundaries = face
                        .absolute_boundaries()
                        .iter()
                        .map(|wire| {
                            wire.iter()
                                .map(|edge| self.edge(edge, operation, work))
                                .collect()
                        })
                        .collect::<ClassicResult<Vec<Wire<Point3, C>>>>()?;
                    Face::debug_new(boundaries, face.surface())
                        .map_err(|_| ShapeOpsError::AltShellConversionFailed { operation })?
                };
                if !face.orientation() {
                    converted.invert();
                }
                Ok(converted)
            })
            .collect()
    }
}

/// Conservative hash-table growth bound; source boundaries remain borrowed.
fn identity_table_bytes<K, V>(count: usize) -> usize {
    count
        .saturating_add(1)
        .saturating_mul(4)
        .saturating_mul(size_of::<(K, V)>().saturating_add(1))
        .saturating_add(16)
}

/// Convert only genuinely new edges. Original edges are recovered above
/// without cloning their native curve or allocating another topology Arc.
fn altcurve_to_curve<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(curve: &AltCurve<C, S>) -> Option<C> {
    match curve {
        Alternative::FirstType(curve) => Some(curve.clone()),
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
                SurfaceCurve::with_boundaries(surface0, surface1, bspline, boundary0, boundary1)
                    .into(),
            )
        }
    }
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
    let mut native_topology = NativeTopologyMap::new();
    let altshell0 = native_topology.map_input(shell0, work)?;
    let altshell1 = native_topology.map_input(shell1, work)?;
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
        native_roots,
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
        &native_roots,
        work,
        &mut |index, face| {
            native_topology.inherit_face(face, altshell0[index].id(), work)?;
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
        &native_roots,
        work,
        &mut |index, face| {
            native_topology.inherit_face(face, altshell1[index].id(), work)?;
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
        native_topology.shell(&and0, "and", work)?,
        if retain_or {
            native_topology.shell(&or0, "or", work)?
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

#[cfg(test)]
mod native_identity_tests {
    use monstertruck_modeling as modeling;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;
    use crate::transversal::prepared::BooleanWorkLimits;

    fn work_limits(intermediate_bytes: usize) -> BooleanWorkLimits {
        BooleanWorkLimits {
            max_faces: 64,
            max_edges: 512,
            max_mesh_vertices: 50_000,
            max_mesh_triangles: 50_000,
            max_triangle_pairs: 100_000,
            max_interference_segments: 10_000,
            max_output_faces: 64,
            max_output_edges: 512,
            max_prepared_bytes: 4 * 1024 * 1024,
            max_source_prepared_bytes: 4 * 1024 * 1024,
            max_other_prepared_bytes: 4 * 1024 * 1024,
            max_intermediate_prepared_bytes: intermediate_bytes,
        }
    }

    fn cuboid_shell() -> modeling::Shell {
        let solid: modeling::Solid = modeling::primitive::cuboid(BoundingBox::from_iter([
            Point3::origin(),
            Point3::new(1.0, 1.0, 1.0),
        ]));
        solid.boundaries()[0].clone()
    }

    #[test]
    fn identity_maps_observe_retained_byte_admission_and_cancellation() {
        let original = cuboid_shell();
        let cancel = AtomicBool::new(false);
        let work = BooleanWork::new(work_limits(0), &cancel);
        let mut identities = NativeTopologyMap::new();
        assert!(matches!(
            identities.map_input(&original, Some(&work)),
            Err(ShapeOpsError::WorkLimitExceeded {
                resource: "native_identity_map_bytes",
                ..
            })
        ));
        let work = BooleanWork::new(work_limits(4 * 1024 * 1024), &cancel);
        let mut identities = NativeTopologyMap::new();
        let mapped = identities.map_input(&original, Some(&work)).unwrap();
        // A later output pass must still consult the current admission limit.
        let restricted = BooleanWork::new(work_limits(0), &cancel);
        assert!(matches!(
            identities.inherit_face(mapped[0].id(), mapped[0].id(), Some(&restricted)),
            Err(ShapeOpsError::WorkLimitExceeded { .. })
        ));
        cancel.store(true, Ordering::Relaxed);
        assert!(matches!(
            identities.shell(&mapped, "and", Some(&work)),
            Err(ShapeOpsError::Cancelled)
        ));
    }

    #[test]
    fn original_topology_round_trip_recovers_native_ids_and_stable_face_identity() {
        let mut original = cuboid_shell();
        for (index, face) in original.iter_mut().enumerate() {
            face.set_stable_id(StableId::new(index as u64 + 1));
        }
        let mut identities = NativeTopologyMap::new();
        let mut mapped = identities.map_input(&original, None).unwrap();
        for (original, mapped) in original.iter().zip(mapped.iter()) {
            assert_ne!(original.id(), mapped.id());
            for (original, mapped) in original.edge_iter().zip(mapped.edge_iter()) {
                assert_ne!(original.front().id(), mapped.front().id());
            }
        }
        // Output orientation may change without changing the underlying face.
        mapped[0].invert();
        let output = identities.shell(&mapped, "and", None).unwrap();
        assert_eq!(output.len(), original.len());
        for (index, (original, output)) in original.iter().zip(output.iter()).enumerate() {
            assert_eq!(output.id(), original.id());
            assert_eq!(output.stable_id(), original.stable_id());
            assert_eq!(output.orientation(), original.orientation() ^ (index == 0));
            for (original, output) in original
                .absolute_boundaries()
                .iter()
                .zip(output.absolute_boundaries())
            {
                assert_eq!(original.len(), output.len());
                for (original, output) in original.iter().zip(output.iter()) {
                    assert_eq!(output.id(), original.id());
                    assert_eq!(output.orientation(), original.orientation());
                    assert_eq!(output.front().id(), original.front().id());
                    assert_eq!(output.back().id(), original.back().id());
                }
            }
        }
    }

    #[test]
    fn generated_use_shares_original_boundary_without_impersonating_original_face() {
        let original = cuboid_shell();
        let source: modeling::Shell = vec![original[0].clone()].into();
        let mut identities = NativeTopologyMap::new();
        let mapped = identities.map_input(&source, None).unwrap();
        // Deliberately use equal source coefficients: geometry equality alone
        // must never authorize recovering the source face's topology ID.
        let generated = Face::new(
            mapped[0]
                .absolute_boundaries()
                .iter()
                .map(Wire::inverse)
                .collect(),
            mapped[0].surface(),
        );
        let shell = vec![mapped[0].clone(), generated].into();
        let output = identities.shell(&shell, "and", None).unwrap();
        assert_eq!(output[0].id(), source[0].id());
        assert_ne!(output[1].id(), source[0].id());
        let retained = &output[0].absolute_boundaries()[0];
        let cap = &output[1].absolute_boundaries()[0];
        for (retained, cap) in retained.iter().zip(cap.iter().rev()) {
            assert_eq!(cap.id(), retained.id());
            assert_ne!(cap.orientation(), retained.orientation());
            assert_eq!(cap.front().id(), retained.back().id());
            assert_eq!(cap.back().id(), retained.front().id());
        }
    }

    #[test]
    fn split_edges_keep_cut_vertex_identity_and_do_not_recover_original_edge_or_face() {
        let original = cuboid_shell();
        let source: modeling::Shell = vec![original[0].clone()].into();
        let mut identities = NativeTopologyMap::new();
        let mapped = identities.map_input(&source, None).unwrap();
        let wire = &mapped[0].absolute_boundaries()[0];
        let edge = &wire[0];
        let mut first = edge.oriented_curve();
        let (start, end) = first.range_tuple();
        let middle = (start + end) * 0.5;
        let cut = Vertex::new(first.evaluate(middle));
        let second = first.cut(middle);
        let first = Edge::new(edge.front(), &cut, first);
        let second = Edge::new(&cut, edge.back(), second);
        let divided: Wire<_, _> = [first, second]
            .into_iter()
            .chain(wire.iter().skip(1).cloned())
            .collect();
        let face = Face::new(vec![divided], mapped[0].surface());
        identities
            .inherit_face(face.id(), mapped[0].id(), None)
            .unwrap();
        let output = identities.shell(&vec![face].into(), "and", None).unwrap();
        assert_ne!(output[0].id(), source[0].id());
        let output = &output[0].absolute_boundaries()[0];
        let original = &source[0].absolute_boundaries()[0];
        assert_eq!(output.len(), original.len() + 1);
        assert_ne!(output[0].id(), original[0].id());
        assert_ne!(output[1].id(), original[0].id());
        assert_ne!(output[0].id(), output[1].id());
        assert_eq!(output[0].front().id(), original[0].front().id());
        assert_eq!(output[0].back().id(), cut.id());
        assert_eq!(output[1].front().id(), cut.id());
        assert_eq!(output[1].back().id(), original[0].back().id());
        for (original, output) in original.iter().skip(1).zip(output.iter().skip(2)) {
            assert_eq!(output.id(), original.id());
        }
    }

    #[test]
    fn equal_coordinate_rewritten_vertex_stays_distinct_from_original() {
        let original = cuboid_shell();
        let source: modeling::Shell = vec![original[0].clone()].into();
        let mut identities = NativeTopologyMap::new();
        let mapped = identities.map_input(&source, None).unwrap();
        let wire = &mapped[0].absolute_boundaries()[0];
        let replaced = wire[0].front();
        let replacement = Vertex::new(replaced.point());
        let rewritten = wire
            .iter()
            .map(|edge| {
                let front = if edge.front() == replaced {
                    &replacement
                } else {
                    edge.front()
                };
                let back = if edge.back() == replaced {
                    &replacement
                } else {
                    edge.back()
                };
                if edge.front() == replaced || edge.back() == replaced {
                    Edge::new(front, back, edge.oriented_curve())
                } else {
                    edge.clone()
                }
            })
            .collect();
        let face = Face::new(vec![rewritten], mapped[0].surface());
        identities
            .inherit_face(face.id(), mapped[0].id(), None)
            .unwrap();
        let output = identities.shell(&vec![face].into(), "and", None).unwrap();
        assert_ne!(output[0].id(), source[0].id());
        let output = &output[0].absolute_boundaries()[0];
        let original = &source[0].absolute_boundaries()[0];
        assert_ne!(output[0].id(), original[0].id());
        assert_eq!(output[0].front().id(), replacement.id());
        assert_ne!(output[0].front().id(), original[0].front().id());
        assert_eq!(output[0].back().id(), original[0].back().id());
    }

    #[test]
    fn source_surface_receipt_recovers_only_the_complete_directed_boundary_list() {
        let original = cuboid_shell();
        let source: modeling::Shell = vec![original[0].clone()].into();
        let mut identities = NativeTopologyMap::new();
        let mapped = identities.map_input(&source, None).unwrap();
        let recreated = Face::new(mapped[0].absolute_boundaries().clone(), mapped[0].surface());
        identities
            .inherit_face(recreated.id(), mapped[0].id(), None)
            .unwrap();
        let wire = &mapped[0].absolute_boundaries()[0];
        let rotated = wire
            .iter()
            .cycle()
            .skip(1)
            .take(wire.len())
            .cloned()
            .collect();
        let reordered = Face::new(vec![rotated], mapped[0].surface());
        identities
            .inherit_face(reordered.id(), mapped[0].id(), None)
            .unwrap();
        let shell = vec![recreated, reordered].into();
        let output = identities.shell(&shell, "and", None).unwrap();
        assert_eq!(output[0].id(), source[0].id());
        assert_ne!(output[1].id(), source[0].id());
        for (original, output) in source[0].absolute_boundaries()[0]
            .iter()
            .cycle()
            .skip(1)
            .zip(output[1].absolute_boundaries()[0].iter())
        {
            assert_eq!(output.id(), original.id());
        }
    }
}
