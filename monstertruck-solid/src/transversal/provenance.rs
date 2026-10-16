//! Topological genealogy for prepared native intersections.

use super::edge_lineage::{BooleanCoedgeProvenance, BooleanEdgeDerivation};
use super::integrate::{ShapeOpsCurve, ShapeOpsError, ShapeOpsSurface};
use super::prepared::{BooleanWork, BooleanWorkBudget, PreparedBooleanSource};
use monstertruck_geometry::prelude::Point3;
use monstertruck_topology::{FaceId, Solid};
use rustc_hash::FxHashMap;
use std::{mem::size_of, sync::atomic::AtomicBool};

/// Original operand of a face retained or divided by a prepared intersection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BooleanOperand {
    /// The immutable solid borrowed by the prepared source.
    Source,
    /// The other solid supplied to this query.
    Other,
}

/// Identity of the original oriented face use, before any Boolean subdivision.
#[derive(Debug, PartialEq, Eq)]
pub struct BooleanFaceOrigin<S> {
    /// Operand containing the original face.
    pub operand: BooleanOperand,
    /// Original native carrier identity, not a geometric equality test.
    pub face_id: FaceId<S>,
    /// Boundary shell ordinal in the original operand.
    pub shell_index: usize,
    /// Face ordinal in the original boundary shell.
    pub face_index: usize,
    /// Original face orientation relative to its carrier.
    pub orientation: bool,
}

impl<S> Copy for BooleanFaceOrigin<S> {}
impl<S> Clone for BooleanFaceOrigin<S> {
    fn clone(&self) -> Self { *self }
}

/// Genealogy of one selected output face.
#[derive(Debug)]
pub struct BooleanFaceProvenance<S> {
    /// Identity of the face in the returned solid.
    pub output_face: FaceId<S>,
    /// Original oriented operand face from which this face was divided.
    pub source: BooleanFaceOrigin<S>,
}

/// Prepared intersection and its actual face-construction genealogy.
///
/// Every returned face has one entry. This report describes native construction,
/// not a proof that marching discovered every geometric intersection. An output
/// derived from a designated clipping face is identified by its `Other` origin
/// and original face identity; a coplanar `Source` face is never a cutting cap.
#[derive(Debug)]
pub struct BooleanWithProvenance<C, S> {
    /// Selected, manifold-validated native intersection.
    pub solid: Solid<Point3, C, S>,
    /// One origin entry for each face of `solid`, in its traversal order.
    pub face_origins: Vec<BooleanFaceProvenance<S>>,
    /// One source-use entry for every output oriented boundary occurrence.
    pub coedge_origins: Vec<BooleanCoedgeProvenance<C, S>>,
    /// Bounded acyclic native source/cut graph referenced by coedge origins.
    pub edge_derivations: Vec<BooleanEdgeDerivation<C>>,
}

pub(crate) struct FaceLineage<S> {
    origins: FxHashMap<FaceId<S>, BooleanFaceOrigin<S>>,
}

impl<S> FaceLineage<S> {
    pub(crate) fn new() -> Self {
        Self {
            origins: FxHashMap::default(),
        }
    }

    pub(crate) fn insert(
        &mut self,
        face: FaceId<S>,
        source: BooleanFaceOrigin<S>,
        work: &BooleanWork<'_>,
    ) -> Result<(), ShapeOpsError> {
        work.limit(
            "face_lineage_entries",
            self.origins.len().saturating_add(1),
            work.limits
                .max_faces
                .saturating_add(work.limits.max_output_faces),
        )?;
        let buckets = self
            .origins
            .capacity()
            .saturating_add(1)
            .checked_next_power_of_two()
            .unwrap_or(usize::MAX)
            .max(8);
        let admitted_buckets = if self.origins.len() == self.origins.capacity() {
            buckets.saturating_mul(2)
        } else {
            buckets
        };
        let entry_bytes = size_of::<(FaceId<S>, BooleanFaceOrigin<S>)>().saturating_add(1);
        let bytes = admitted_buckets
            .saturating_mul(entry_bytes)
            .saturating_add(32);
        work.limit(
            "face_lineage_bytes",
            bytes,
            work.limits.max_intermediate_prepared_bytes,
        )?;
        self.origins.insert(face, source);
        let retained_buckets = self
            .origins
            .capacity()
            .saturating_add(1)
            .checked_next_power_of_two()
            .unwrap_or(usize::MAX)
            .max(8);
        work.limit(
            "face_lineage_bytes",
            retained_buckets
                .saturating_mul(entry_bytes)
                .saturating_add(32),
            work.limits.max_intermediate_prepared_bytes,
        )
    }

    pub(crate) fn origin(&self, face: FaceId<S>) -> Result<BooleanFaceOrigin<S>, ShapeOpsError> {
        self.origins
            .get(&face)
            .copied()
            .ok_or(ShapeOpsError::MissingFaceLineage)
    }
}

/// Intersects a prepared source and reports original face identity through splitting.
///
/// Uses the same bounded, cancellable backend as [`super::and_prepared`]. Lineage
/// storage additionally obeys the intermediate byte and input/output face caps.
/// Missing construction genealogy is a typed error, never a geometric fallback.
/// Generic meshing and parameter-search calls remain indivisible cancellation stages.
pub fn and_prepared_with_provenance<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    prepared: &PreparedBooleanSource<'_, C, S>,
    other: &Solid<Point3, C, S>,
    cancel: &AtomicBool,
) -> Result<BooleanWithProvenance<C, S>, ShapeOpsError> {
    and_prepared_with_provenance_impl(prepared, other, cancel, None, None)
}

/// Reports prepared AND genealogy while charging cumulative native work.
///
/// Original-use scans, cut metadata and all nested query scratch share `budget`.
/// A failed branch retains its consumption and never becomes an empty report.
pub fn and_prepared_with_provenance_counted<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    prepared: &PreparedBooleanSource<'_, C, S>,
    other: &Solid<Point3, C, S>,
    cancel: &AtomicBool,
    budget: &BooleanWorkBudget,
) -> Result<BooleanWithProvenance<C, S>, ShapeOpsError> {
    and_prepared_with_provenance_impl(prepared, other, cancel, Some(budget), None)
}

/// Reports counted prepared AND genealogy with a caller's remaining query bytes.
///
/// `max_query_bytes` excludes the already retained source. Derived input seeds
/// share this allowance and the original combined/source/other seed limits.
/// Each reconstruction scratch object is separately capped by the remaining
/// allowance after retaining those derived seeds and by the original intermediate
/// limit. This is not an aggregate scratch, allocator or process RSS bound.
/// Generic meshing temporaries still require an external host memory cap.
pub fn and_prepared_with_provenance_counted_with_query_bytes<
    C: ShapeOpsCurve<S>,
    S: ShapeOpsSurface,
>(
    prepared: &PreparedBooleanSource<'_, C, S>,
    other: &Solid<Point3, C, S>,
    cancel: &AtomicBool,
    budget: &BooleanWorkBudget,
    max_query_bytes: usize,
) -> Result<BooleanWithProvenance<C, S>, ShapeOpsError> {
    and_prepared_with_provenance_impl(prepared, other, cancel, Some(budget), Some(max_query_bytes))
}

fn and_prepared_with_provenance_impl<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    prepared: &PreparedBooleanSource<'_, C, S>,
    other: &Solid<Point3, C, S>,
    cancel: &AtomicBool,
    budget: Option<&BooleanWorkBudget>,
    max_query_bytes: Option<usize>,
) -> Result<BooleanWithProvenance<C, S>, ShapeOpsError> {
    let mut limits = prepared.limits;
    if let Some(max_query_bytes) = max_query_bytes {
        limits.max_prepared_bytes = limits
            .max_prepared_bytes
            .min(prepared.retained_bytes().saturating_add(max_query_bytes));
        limits.max_other_prepared_bytes = limits.max_other_prepared_bytes.min(max_query_bytes);
        limits.max_intermediate_prepared_bytes =
            limits.max_intermediate_prepared_bytes.min(max_query_bytes);
    }
    let work = BooleanWork::with_budget(limits, cancel, budget);
    work.check()?;
    #[cfg(feature = "marching-ssi")]
    {
        let (other_meshes, other_bytes) =
            super::prepared::query_meshes(prepared, other, &work, cancel)?;
        let mut limits = work.limits;
        if let Some(max_query_bytes) = max_query_bytes {
            limits.max_intermediate_prepared_bytes = limits
                .max_intermediate_prepared_bytes
                .min(max_query_bytes.saturating_sub(other_bytes));
        }
        let work = BooleanWork::with_budget(limits, cancel, budget);
        super::classic::and_prepared_with_provenance(prepared, other, &other_meshes, &work)
    }
    #[cfg(not(feature = "marching-ssi"))]
    {
        let _ = (prepared, other);
        Err(ShapeOpsError::NoBackend)
    }
}

#[cfg(test)]
mod native_lineage_regressions {
    #[cfg(feature = "marching-ssi")]
    use super::super::edge_lineage::BooleanCoedgeOrigin;
    use super::super::prepared::{BooleanWorkLimits, prepare_boolean_source_counted};
    use super::*;
    use monstertruck_geometry::prelude::{BoundingBox, EuclideanSpace};
    #[cfg(feature = "marching-ssi")]
    use monstertruck_geometry::prelude::{
        InnerSpace, Matrix4, ParametricCurve, ParametricSurface3D, Vector3, Vector4,
    };
    use monstertruck_modeling as modeling;

    fn limits() -> BooleanWorkLimits {
        BooleanWorkLimits {
            max_faces: 128,
            max_edges: 2048,
            max_mesh_vertices: 100_000,
            max_mesh_triangles: 100_000,
            max_triangle_pairs: 1_000_000,
            max_interference_segments: 100_000,
            max_output_faces: 128,
            max_output_edges: 2048,
            max_prepared_bytes: 16_777_216,
            max_source_prepared_bytes: 8_388_608,
            max_other_prepared_bytes: 8_388_608,
            max_intermediate_prepared_bytes: 16_777_216,
        }
    }

    fn cube(front: Point3, back: Point3) -> modeling::Solid {
        modeling::primitive::cuboid(BoundingBox::from_iter([front, back]))
    }

    #[test]
    fn cumulative_allowance_retains_failed_native_admissions() {
        let source = cube(Point3::origin(), Point3::new(1.0, 1.0, 1.0));
        let budget = BooleanWorkBudget::new(2);
        let cancel = AtomicBool::new(false);
        for consumed in 1..=2 {
            assert!(matches!(
                prepare_boolean_source_counted(&source, f64::NAN, limits(), &cancel, &budget),
                Err(ShapeOpsError::InvalidTolerance)
            ));
            assert_eq!(budget.consumed(), consumed);
        }
        assert!(matches!(
            prepare_boolean_source_counted(&source, f64::NAN, limits(), &cancel, &budget),
            Err(ShapeOpsError::WorkLimitExceeded { .. })
        ));
        assert_eq!(budget.consumed(), 2);
        assert_eq!(budget.remaining(), 0);
    }

    #[test]
    fn coincident_carriers_keep_distinct_original_operand_identity() {
        let source = cube(Point3::origin(), Point3::new(1.0, 1.0, 1.0));
        let original = &source.boundaries()[0][0];
        let coincident = monstertruck_topology::Face::new(
            original.absolute_boundaries().clone(),
            original.surface(),
        );
        let cancel = AtomicBool::new(false);
        let budget = BooleanWorkBudget::new(1000);
        let work = BooleanWork::with_budget(limits(), &cancel, Some(&budget));
        let mut lineage = FaceLineage::new();
        for (face, operand) in [
            (original, BooleanOperand::Source),
            (&coincident, BooleanOperand::Other),
        ] {
            lineage
                .insert(
                    face.id(),
                    BooleanFaceOrigin {
                        operand,
                        face_id: face.id(),
                        shell_index: 0,
                        face_index: 0,
                        orientation: face.orientation(),
                    },
                    &work,
                )
                .unwrap();
        }
        assert_ne!(original.id(), coincident.id());
        assert_eq!(
            lineage.origin(original.id()).unwrap().operand,
            BooleanOperand::Source
        );
        assert_eq!(
            lineage.origin(coincident.id()).unwrap().operand,
            BooleanOperand::Other
        );
    }

    #[cfg(feature = "marching-ssi")]
    #[test]
    fn prepared_and_reports_cut_face_and_coedge_genealogy() {
        let source = cube(Point3::origin(), Point3::new(2.0, 2.0, 2.0));
        let other = cube(Point3::new(-1.0, -1.0, -1.0), Point3::new(3.0, 3.0, 1.0));
        let cancel = AtomicBool::new(false);
        let budget = BooleanWorkBudget::new(10_000_000);
        let prepared =
            prepare_boolean_source_counted(&source, 1.0e-4, limits(), &cancel, &budget).unwrap();
        let preparation_work = budget.consumed();
        let result =
            and_prepared_with_provenance_counted(&prepared, &other, &cancel, &budget).unwrap();
        assert!(budget.consumed() > preparation_work);
        assert_eq!(
            result.face_origins.len(),
            result
                .solid
                .boundaries()
                .iter()
                .map(|shell| shell.len())
                .sum::<usize>()
        );
        assert_eq!(
            result.coedge_origins.len(),
            result
                .solid
                .boundaries()
                .iter()
                .flat_map(|shell| shell.iter())
                .flat_map(|face| face.absolute_boundaries())
                .map(|wire| wire.len())
                .sum::<usize>()
        );
        assert!(
            result
                .face_origins
                .iter()
                .any(|face| face.source.operand == BooleanOperand::Other)
        );
        assert!(
            result
                .face_origins
                .iter()
                .any(|face| face.source.operand == BooleanOperand::Source)
        );
        assert!(
            result
                .face_origins
                .iter()
                .any(|face| face.output_face != face.source.face_id)
        );
        assert!(
            result
                .edge_derivations
                .iter()
                .any(|edge| matches!(edge, BooleanEdgeDerivation::Cut { .. }))
        );
        for face in &result.face_origins {
            let operand = if face.source.operand == BooleanOperand::Source {
                &source
            } else {
                &other
            };
            assert!(
                operand
                    .boundaries()
                    .iter()
                    .flat_map(|shell| shell.iter())
                    .any(|original| original.id() == face.source.face_id)
            );
        }
    }

    #[cfg(feature = "marching-ssi")]
    #[test]
    fn prepared_original_line_overlap_keeps_source_and_generated_face_use_genealogy() {
        let original = cube(Point3::origin(), Point3::new(1.0, 1.0, 1.0));
        for scale in [None, Some(2.0), Some(-2.0)] {
            let source = if let Some(scale) = scale {
                original.mapped(
                    |point| *point,
                    |curve| {
                        let modeling::Curve::Line(line) = curve else {
                            unreachable!("literal cuboid source edges")
                        };
                        modeling::Curve::NurbsCurve(modeling::NurbsCurve::new(
                            modeling::BsplineCurve::new(
                                modeling::KnotVector::from(vec![-3.0, -3.0, 5.0, 5.0]),
                                vec![
                                    line.0.to_homogeneous() * scale,
                                    line.1.to_homogeneous() * scale,
                                ],
                            ),
                        ))
                    },
                    Clone::clone,
                )
            } else {
                original.clone()
            };
            if let Some(scale) = scale {
                assert!(
                    source.boundaries()[0]
                        .edge_iter()
                        .all(|edge| edge.with_curve(|curve| {
                            let modeling::Curve::NurbsCurve(curve) = curve else {
                                return false;
                            };
                            curve
                                .non_rationalized()
                                .control_points()
                                .iter()
                                .all(|control| control.w == scale)
                        }))
                );
            }
            let snapshot = serde_json::to_vec(&source.boundaries()[0].compress()).unwrap();
            let original_edges: Vec<_> = source.boundaries()[0]
                .edge_iter()
                .map(|edge| edge.id())
                .collect();
            let other = cube(Point3::new(-2.0, -1.0, 0.0), Point3::new(2.0, 2.0, 3.0));
            let other = modeling::builder::transformed(
                &other,
                Matrix4::from_cols(
                    Vector4::new(1.0, 0.0, -1.0, 0.0),
                    Vector4::new(0.0, 1.0, 0.0, 0.0),
                    Vector4::new(0.0, 0.0, 1.0, 0.0),
                    Vector4::new(1.0, 0.0, 0.0, 1.0),
                ),
            );
            let cancel = AtomicBool::new(false);
            let budget = BooleanWorkBudget::new(10_000_000);
            let prepared =
                prepare_boolean_source_counted(&source, 1.0e-4, limits(), &cancel, &budget)
                    .expect("unchanged original source");
            let result = and_prepared_with_provenance_counted(&prepared, &other, &cancel, &budget)
                .expect("qualified original edges retain both original and generated face uses");
            assert_eq!(result.solid.boundaries().len(), 1);
            let shell = &result.solid.boundaries()[0];
            shell
                .check_solid_boundary()
                .expect("closed oriented triangular prism");
            assert_eq!(shell.len(), 5);
            assert!(result.solid.is_geometric_consistent());
            assert!(shell.iter().all(|face| {
                face.absolute_boundaries()
                    .iter()
                    .all(|wire| wire.is_closed() && wire.is_simple())
            }));
            for vertex in shell.vertex_iter() {
                let point = vertex.point();
                assert!((0..3).all(|axis| point[axis] >= -1.0e-6 && point[axis] <= 1.0 + 1.0e-6));
                assert!(point.x + point.z >= 1.0 - 1.0e-6);
            }
            let caps: Vec<_> = result
                .face_origins
                .iter()
                .filter(|origin| origin.source.operand == BooleanOperand::Other)
                .collect();
            assert_eq!(caps.len(), 1);
            let cap = shell
                .iter()
                .find(|face| face.id() == caps[0].output_face)
                .unwrap();
            assert!(
                cap.oriented_surface()
                    .normal(0.0, 0.0)
                    .dot(Vector3::new(1.0, 0.0, 1.0))
                    < -1.4
            );
            let mut reused = 0;
            for generated in &result.coedge_origins {
                if !matches!(&generated.source, BooleanCoedgeOrigin::GeneratedIntersection { face }
                if face.operand == BooleanOperand::Other)
                {
                    continue;
                }
                let original = result.coedge_origins.iter().find(|original| {
                    original.output_edge == generated.output_edge
                        && matches!(&original.source, BooleanCoedgeOrigin::Original { face, .. }
                        if face.operand == BooleanOperand::Source)
                });
                let Some(original) = original else { continue };
                assert_ne!(original.orientation, generated.orientation);
                let BooleanCoedgeOrigin::Original {
                    face,
                    wire_index,
                    coedge_index,
                    edge_id,
                    derivation,
                    ..
                } = &original.source
                else {
                    unreachable!()
                };
                let source_edge = &source.boundaries()[face.shell_index][face.face_index]
                    .absolute_boundaries()[*wire_index][*coedge_index];
                assert_eq!(source_edge.id(), *edge_id);
                let output = shell
                    .iter()
                    .find(|face| face.id() == original.output_face)
                    .unwrap();
                let output_edge =
                    &output.absolute_boundaries()[original.wire_index][original.coedge_index];
                let domain = match &result.edge_derivations[*derivation] {
                    BooleanEdgeDerivation::Source { domain, .. } => *domain,
                    BooleanEdgeDerivation::Cut { child_domain, .. } => *child_domain,
                };
                assert_eq!(
                    output_edge.with_curve(ParametricCurve::parameter_range),
                    domain
                );
                reused += 1;
            }
            assert_eq!(reused, 2);
            assert_eq!(
                serde_json::to_vec(&source.boundaries()[0].compress()).unwrap(),
                snapshot
            );
            assert_eq!(
                source.boundaries()[0]
                    .edge_iter()
                    .map(|edge| edge.id())
                    .collect::<Vec<_>>(),
                original_edges
            );
        }
    }

    #[cfg(feature = "marching-ssi")]
    #[test]
    fn prepared_query_bytes_bound_derived_seeds_and_remaining_scratch() {
        let source = cube(Point3::origin(), Point3::new(2.0, 2.0, 2.0));
        let other = cube(Point3::new(-1.0, -1.0, -1.0), Point3::new(3.0, 3.0, 1.0));
        let cancel = AtomicBool::new(false);
        let budget = BooleanWorkBudget::new(10_000_000);
        let prepared =
            prepare_boolean_source_counted(&source, 1.0e-4, limits(), &cancel, &budget).unwrap();
        let work = BooleanWork::with_budget(limits(), &cancel, Some(&budget));
        let (other_meshes, other_bytes) =
            super::super::prepared::query_meshes(&prepared, &other, &work, &cancel).unwrap();
        drop(other_meshes);
        assert!(other_bytes > 0);
        let before_failure = budget.consumed();
        assert!(matches!(
            and_prepared_with_provenance_counted_with_query_bytes(
                &prepared,
                &other,
                &cancel,
                &budget,
                other_bytes - 1,
            ),
            Err(ShapeOpsError::WorkLimitExceeded {
                resource: "prepared_bytes",
                ..
            })
        ));
        assert!(budget.consumed() > before_failure);
        assert!(matches!(
            and_prepared_with_provenance_counted_with_query_bytes(
                &prepared, &other, &cancel, &budget, other_bytes,
            ),
            Err(ShapeOpsError::WorkLimitExceeded { actual, limit: 0, .. }) if actual > 0
        ));
        let result = and_prepared_with_provenance_counted_with_query_bytes(
            &prepared,
            &other,
            &cancel,
            &budget,
            limits().max_prepared_bytes,
        )
        .unwrap();
        assert_eq!(result.solid.boundaries().len(), 1);
        result.solid.boundaries()[0].check_solid_boundary().unwrap();
        assert!(result.solid.is_geometric_consistent());
        assert!(result.solid.vertex_iter().all(|vertex| {
            let point = vertex.point();
            point.x >= -1.0e-6
                && point.x <= 2.0 + 1.0e-6
                && point.y >= -1.0e-6
                && point.y <= 2.0 + 1.0e-6
                && point.z >= -1.0e-6
                && point.z <= 1.0 + 1.0e-6
        }));
        assert!(
            result
                .solid
                .vertex_iter()
                .any(|vertex| { (vertex.point().z - 1.0).abs() <= 1.0e-6 })
        );
        assert!(
            result
                .face_origins
                .iter()
                .any(|face| { face.source.operand == BooleanOperand::Other })
        );
    }
}
