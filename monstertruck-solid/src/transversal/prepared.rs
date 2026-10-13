//! Borrowed native sources with reusable, source-owned marching seeds.

use super::integrate::{ShapeOpsCurve, ShapeOpsError, ShapeOpsSurface};
use monstertruck_geometry::prelude::*;
use monstertruck_meshing::prelude::*;
use monstertruck_topology::*;
use rustc_hash::FxHashSet;
use std::{
    cell::Cell,
    mem::size_of,
    sync::atomic::{AtomicBool, Ordering},
};

pub(crate) type MeshShell = Shell<Point3, PolylineCurve<Point3>, Option<PolygonMesh>>;
type BooleanResult<T> = std::result::Result<T, ShapeOpsError>;

/// Explicit admission and work limits for a prepared native Boolean source.
///
/// Counts apply to both complete inputs together for each query. Triangle pairs
/// bound the conservative Cartesian work before mesh interference extraction;
/// segment limits bound reconstruction and graph storage. Prepared bytes count
/// actual owned buffer capacities, excluding borrowed BREP and allocator/Arc
/// bookkeeping. Native face/edge caps separately bound seed topology objects.
#[derive(Clone, Copy, Debug)]
pub struct BooleanWorkLimits {
    /// Maximum native input faces, including all boundary shells.
    pub max_faces: usize,
    /// Maximum oriented native input edge occurrences.
    pub max_edges: usize,
    /// Maximum seed mesh positions and retained per-query boundary curve samples.
    pub max_mesh_vertices: usize,
    /// Maximum seed mesh triangles.
    pub max_mesh_triangles: usize,
    /// Maximum conservative triangle pairs processed per query.
    pub max_triangle_pairs: usize,
    /// Maximum interference segments reconstructed per query.
    pub max_interference_segments: usize,
    /// Maximum faces in the selected native output.
    pub max_output_faces: usize,
    /// Maximum oriented edge occurrences in the selected native output.
    pub max_output_edges: usize,
    /// Maximum owned seed buffer bytes for both inputs together.
    pub max_prepared_bytes: usize,
    /// Maximum owned source seed bytes admitted during preparation.
    pub max_source_prepared_bytes: usize,
    /// Maximum owned derived query-input seed bytes.
    pub max_other_prepared_bytes: usize,
    /// Maximum owned intermediate seeds, native certificates, candidate metadata or curve subdivisions.
    /// Each bounded scratch object is admitted separately; opaque upstream
    /// conversion and tessellation temporaries still require a host memory cap.
    pub max_intermediate_prepared_bytes: usize,
}

/// Opaque borrowed source and its reusable marching seeds at one fixed tolerance.
///
/// No outside meshes are admitted. The source must remain immutable for the
/// entire borrow, including mutations through shared vertices, edges and faces.
/// Queries allocate separate scratch and output and never modify these seeds.
/// Dropping this value releases the retained seeds without retaining the source.
#[derive(Debug)]
pub struct PreparedBooleanSource<'a, C, S> {
    pub(crate) source: &'a Solid<Point3, C, S>,
    pub(crate) meshes: Vec<MeshShell>,
    pub(crate) tolerance: f64,
    pub(crate) limits: BooleanWorkLimits,
    stats: SeedStats,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SeedStats {
    vertices: usize,
    triangles: usize,
    bytes: usize,
}

impl<C, S> PreparedBooleanSource<'_, C, S> {
    /// Returns the number of retained mesh and boundary-curve seed points.
    pub fn retained_mesh_vertices(&self) -> usize { self.stats.vertices }
    /// Returns the number of retained seed triangles.
    pub fn retained_mesh_triangles(&self) -> usize { self.stats.triangles }
    /// Returns actual retained owned buffer bytes, including vector capacities.
    ///
    /// Excludes the borrowed BREP and allocator/Arc bookkeeping. Native object
    /// counts bound the remaining fixed-size seed topology payload.
    pub fn retained_bytes(&self) -> usize { self.stats.bytes }
    /// Returns the fixed source triangulation and Boolean tolerance.
    pub fn tolerance(&self) -> f64 { self.tolerance }
}

pub(crate) struct BooleanWork<'a> {
    pub(crate) limits: BooleanWorkLimits,
    cancel: &'a AtomicBool,
    triangle_pairs: Cell<usize>,
    segments: Cell<usize>,
}

impl<'a> BooleanWork<'a> {
    pub(crate) fn new(limits: BooleanWorkLimits, cancel: &'a AtomicBool) -> Self {
        Self {
            limits,
            cancel,
            triangle_pairs: Cell::new(0),
            segments: Cell::new(0),
        }
    }
    pub(crate) fn check(&self) -> BooleanResult<()> {
        if self.cancel.load(Ordering::Relaxed) {
            Err(ShapeOpsError::Cancelled)
        } else {
            Ok(())
        }
    }
    pub(crate) fn limit(
        &self,
        resource: &'static str,
        actual: usize,
        limit: usize,
    ) -> BooleanResult<()> {
        self.check()?;
        if actual > limit {
            Err(ShapeOpsError::WorkLimitExceeded {
                resource,
                actual,
                limit,
            })
        } else {
            Ok(())
        }
    }
    pub(crate) fn face_pair(&self, mesh0: &PolygonMesh, mesh1: &PolygonMesh) -> BooleanResult<()> {
        let count0 = mesh0.faces().triangle_iter().count();
        let count1 = mesh1.faces().triangle_iter().count();
        let count = count0
            .saturating_mul(count1)
            .saturating_add(self.triangle_pairs.get());
        self.limit("triangle_pairs", count, self.limits.max_triangle_pairs)?;
        self.triangle_pairs.set(count);
        Ok(())
    }
    pub(crate) fn segments(&self, count: usize) -> BooleanResult<()> {
        let count = self.segments.get().saturating_add(count);
        self.limit(
            "interference_segments",
            count,
            self.limits.max_interference_segments,
        )?;
        self.segments.set(count);
        Ok(())
    }
    pub(crate) fn native<P, C, S>(&self, solid: &Solid<P, C, S>) -> BooleanResult<(usize, usize)> {
        let faces = solid.face_iter().count();
        let edges = solid.face_iter().map(|face| face.edge_iter().count()).sum();
        self.limit("input_faces", faces, self.limits.max_faces)?;
        self.limit("input_edges", edges, self.limits.max_edges)?;
        Ok((faces, edges))
    }
    pub(crate) fn output<C, S>(&self, shell: &Shell<Point3, C, S>) -> BooleanResult<()> {
        self.limit("output_faces", shell.len(), self.limits.max_output_faces)?;
        self.limit(
            "output_edges",
            shell.iter().map(|face| face.edge_iter().count()).sum(),
            self.limits.max_output_edges,
        )
    }
    pub(crate) fn intermediate(&self, mesh: &MeshShell) -> BooleanResult<()> {
        let mut limits = self.limits;
        limits.max_prepared_bytes = limits.max_intermediate_prepared_bytes;
        let scratch = Self::new(limits, self.cancel);
        seed_stats(mesh, &scratch).map(|_| ())
    }
}

pub(crate) fn seed_stats(mesh: &MeshShell, work: &BooleanWork<'_>) -> BooleanResult<SeedStats> {
    let mut edges = FxHashSet::default();
    let initial = SeedStats {
        bytes: mesh.capacity().saturating_mul(size_of::<
            Face<Point3, PolylineCurve<Point3>, Option<PolygonMesh>>,
        >()),
        ..SeedStats::default()
    };
    mesh.iter()
        .enumerate()
        .try_fold(initial, |mut stats, (face_index, face)| {
            work.check()?;
            stats.bytes = stats.bytes.saturating_add(
                face.absolute_boundaries()
                    .capacity()
                    .saturating_mul(size_of::<Wire<Point3, PolylineCurve<Point3>>>()),
            );
            face.with_surface(|polygon| {
                let polygon = polygon
                    .as_ref()
                    .ok_or(ShapeOpsError::InputTessellationFailed {
                        shell_index: 0,
                        face_index,
                    })?;
                stats.vertices = stats.vertices.saturating_add(polygon.positions().len());
                stats.triangles = stats
                    .triangles
                    .saturating_add(polygon.faces().triangle_iter().count());
                let vectors = polygon
                    .positions()
                    .capacity()
                    .saturating_mul(size_of::<Point3>())
                    .saturating_add(
                        polygon
                            .uv_coords()
                            .capacity()
                            .saturating_mul(size_of::<Vector2>()),
                    )
                    .saturating_add(
                        polygon
                            .normals()
                            .capacity()
                            .saturating_mul(size_of::<Vector3>()),
                    );
                let faces = polygon.faces();
                let indices = faces
                    .tri_faces()
                    .capacity()
                    .saturating_mul(size_of::<[StandardVertex; 3]>())
                    .saturating_add(
                        faces
                            .quad_faces()
                            .capacity()
                            .saturating_mul(size_of::<[StandardVertex; 4]>()),
                    )
                    .saturating_add(
                        faces
                            .other_faces()
                            .capacity()
                            .saturating_mul(size_of::<Vec<StandardVertex>>()),
                    );
                let other_indices = faces.other_faces().iter().fold(0usize, |bytes, face| {
                    bytes
                        .saturating_add(face.capacity().saturating_mul(size_of::<StandardVertex>()))
                });
                stats.bytes = stats
                    .bytes
                    .saturating_add(vectors)
                    .saturating_add(indices)
                    .saturating_add(other_indices);
                Ok::<(), ShapeOpsError>(())
            })?;
            face.absolute_boundaries().iter().for_each(|wire| {
                stats.bytes = stats.bytes.saturating_add(
                    wire.capacity()
                        .saturating_mul(size_of::<Edge<Point3, PolylineCurve<Point3>>>()),
                );
                wire.iter()
                    .filter(|edge| edges.insert(edge.id()))
                    .for_each(|edge| {
                        edge.with_curve(|curve| {
                            stats.vertices = stats.vertices.saturating_add(curve.len());
                            stats.bytes = stats.bytes.saturating_add(
                                curve.0.capacity().saturating_mul(size_of::<Point3>()),
                            );
                        });
                    });
            });
            work.limit(
                "mesh_vertices",
                stats.vertices,
                work.limits.max_mesh_vertices,
            )?;
            work.limit(
                "mesh_triangles",
                stats.triangles,
                work.limits.max_mesh_triangles,
            )?;
            work.limit(
                "prepared_bytes",
                stats.bytes,
                work.limits.max_prepared_bytes,
            )?;
            Ok(stats)
        })
}

pub(crate) fn meshes<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid: &Solid<Point3, C, S>,
    tolerance: f64,
    work: &BooleanWork<'_>,
) -> BooleanResult<(Vec<MeshShell>, SeedStats)> {
    work.native(solid)?;
    work.limit(
        "prepared_bytes",
        solid.face_iter().count().saturating_mul(size_of::<
            Face<Point3, PolylineCurve<Point3>, Option<PolygonMesh>>,
        >()),
        work.limits.max_prepared_bytes,
    )?;
    let outer_bytes = solid
        .boundaries()
        .len()
        .saturating_mul(size_of::<MeshShell>());
    work.limit(
        "prepared_bytes",
        outer_bytes,
        work.limits.max_prepared_bytes,
    )?;
    let storage = Vec::with_capacity(solid.boundaries().len());
    let outer_bytes = storage.capacity().saturating_mul(size_of::<MeshShell>());
    work.limit(
        "prepared_bytes",
        outer_bytes,
        work.limits.max_prepared_bytes,
    )?;
    let initial = (
        storage,
        SeedStats {
            bytes: outer_bytes,
            ..SeedStats::default()
        },
    );
    let (meshes, total) =
        solid
            .boundaries()
            .iter()
            .try_fold(initial, |(mut meshes, mut total), shell| {
                work.check()?;
                let mesh = shell.triangulation(tolerance);
                work.check()?;
                let stats = seed_stats(&mesh, work)?;
                total.vertices = total.vertices.saturating_add(stats.vertices);
                total.triangles = total.triangles.saturating_add(stats.triangles);
                total.bytes = total.bytes.saturating_add(stats.bytes);
                work.limit(
                    "mesh_vertices",
                    total.vertices,
                    work.limits.max_mesh_vertices,
                )?;
                work.limit(
                    "mesh_triangles",
                    total.triangles,
                    work.limits.max_mesh_triangles,
                )?;
                work.limit(
                    "prepared_bytes",
                    total.bytes,
                    work.limits.max_prepared_bytes,
                )?;
                meshes.push(mesh);
                Ok((meshes, total))
            })?;
    Ok((meshes, total))
}

/// Prepares source-owned marching seeds for repeated queries of an exact solid.
///
/// Preparation and queries cooperate with cancellation at native stages, face
/// pairs and leader points. Existing generic tessellation and parameter-search
/// trait calls are indivisible; cancellation does not preempt a call in flight.
/// Seed limits are checked immediately after tessellation, before any Boolean
/// reconstruction. They bound retained data, not an opaque mesher's transient
/// allocation. The caller must separately bound source geometry and execution.
/// Invalid tolerances, resource excess and cancellation retain typed errors.
pub fn prepare_boolean_source<'a, C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    source: &'a Solid<Point3, C, S>,
    tolerance: f64,
    limits: BooleanWorkLimits,
    cancel: &AtomicBool,
) -> BooleanResult<PreparedBooleanSource<'a, C, S>> {
    if !tolerance.is_finite() || tolerance < TOLERANCE {
        return Err(ShapeOpsError::InvalidTolerance);
    }
    let mut preparation_limits = limits;
    preparation_limits.max_prepared_bytes = limits
        .max_source_prepared_bytes
        .min(limits.max_prepared_bytes);
    let work = BooleanWork::new(preparation_limits, cancel);
    work.check()?;
    #[cfg(feature = "marching-ssi")]
    {
        let (meshes, stats) = meshes(source, tolerance, &work)?;
        Ok(PreparedBooleanSource {
            source,
            meshes,
            tolerance,
            limits,
            stats,
        })
    }
    #[cfg(not(feature = "marching-ssi"))]
    {
        let _ = source;
        Err(ShapeOpsError::NoBackend)
    }
}

/// Intersects a prepared borrowed native source through the common AND backend.
///
/// Uses the preparation tolerance and limits, retaining exact carriers, native
/// topology assembly and release-active manifold checks. Scratch is query-local;
/// total seed admissions include both inputs. No failed branch is discarded.
pub fn and_prepared<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    prepared: &PreparedBooleanSource<'_, C, S>,
    other: &Solid<Point3, C, S>,
    cancel: &AtomicBool,
) -> BooleanResult<Solid<Point3, C, S>> {
    let work = BooleanWork::new(prepared.limits, cancel);
    work.check()?;
    #[cfg(feature = "marching-ssi")]
    {
        let source_counts = work.native(prepared.source)?;
        let other_counts = work.native(other)?;
        work.limit(
            "input_faces",
            source_counts.0.saturating_add(other_counts.0),
            prepared.limits.max_faces,
        )?;
        work.limit(
            "input_edges",
            source_counts.1.saturating_add(other_counts.1),
            prepared.limits.max_edges,
        )?;
        work.limit(
            "mesh_vertices",
            prepared.stats.vertices,
            prepared.limits.max_mesh_vertices,
        )?;
        work.limit(
            "mesh_triangles",
            prepared.stats.triangles,
            prepared.limits.max_mesh_triangles,
        )?;
        work.limit(
            "prepared_bytes",
            prepared.stats.bytes,
            prepared.limits.max_prepared_bytes,
        )?;
        let mut other_limits = prepared.limits;
        other_limits.max_faces -= source_counts.0;
        other_limits.max_edges -= source_counts.1;
        other_limits.max_mesh_vertices -= prepared.stats.vertices;
        other_limits.max_mesh_triangles -= prepared.stats.triangles;
        other_limits.max_prepared_bytes = prepared
            .limits
            .max_other_prepared_bytes
            .min(prepared.limits.max_prepared_bytes - prepared.stats.bytes);
        let other_work = BooleanWork::new(other_limits, cancel);
        let (other_meshes, _) = meshes(other, prepared.tolerance, &other_work)?;
        super::classic::and_prepared(prepared, other, &other_meshes, &work)
    }
    #[cfg(not(feature = "marching-ssi"))]
    {
        let _ = (prepared, other);
        Err(ShapeOpsError::NoBackend)
    }
}
