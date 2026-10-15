//! Classic (0.3.2) face division.
//!
//! Ported from the published 0.3.2 crate's `transversal::divide_face`:
//! projects each loop's edges into the face parameter domain, splits the face
//! into positively oriented pre-faces with their contained holes, and tags each
//! with its `and`/`or`/`unknown` status. Periodic coordinates are lifted
//! continuously rather than measuring polygons with inverse-map seam jumps.

#![allow(clippy::many_single_char_names)]

use super::super::integrate::DivideFaceBoundaryData;
use super::faces_classification::FacesClassification;
use super::loops_store::*;
use super::{ClassicResult, ShapeOpsError};
use monstertruck_meshing::prelude::*;
use monstertruck_topology::{errors::Error as TopologyError, *};
use rustc_hash::FxHashMap as HashMap;
use std::ops::Deref;

use crate::transversal::prepared::BooleanWork;
mod curve_divisions;
pub(super) use curve_divisions::CurveDivisionCache;

type DivisionResult<T> = std::result::Result<T, DivisionFailure>;

enum DivisionFailure {
    Geometry {
        stage: &'static str,
        source: Option<TopologyError>,
        boundary: Option<DivideFaceBoundaryData>,
    },
    Work(ShapeOpsError),
}

impl From<&'static str> for DivisionFailure {
    fn from(stage: &'static str) -> Self {
        Self::Geometry {
            stage,
            source: None,
            boundary: None,
        }
    }
}

impl From<ShapeOpsError> for DivisionFailure {
    fn from(work: ShapeOpsError) -> Self { Self::Work(work) }
}

fn create_parameter_boundary<P, C, S>(
    face: &Face<P, C, S>,
    wire: &Wire<P, C>,
    polys: &mut CurveDivisionCache<P, C>,
    work: Option<&BooleanWork<'_>>,
) -> DivisionResult<PolylineCurve<Point2>>
where
    P: Copy,
    C: BoundedCurve<Point = P> + ParameterDivision1D<Point = P>,
    S: Clone + ParametricSurface + SearchParameter<SurfaceParameter, Point = P>,
{
    if let Some(work) = work {
        work.check()?;
    }
    let surface = face.surface();
    let pt = wire.front_vertex().unwrap().point();
    let parameter = surface.search_parameter(pt, None, 100);
    let p: Point2 = parameter.ok_or("boundary_front_parameter")?.into();
    let vec = wire.edge_iter().try_fold(vec![p], |mut vec, edge| {
        let poly = polys.get(edge, work)?;
        let mut p = *vec.last().unwrap();
        let closure = |q: &P| -> DivisionResult<Point2> {
            if let Some(work) = work {
                work.check()?;
            }
            let parameter = surface.search_parameter(*q, Some(p.into()), 100);
            let next: Point2 = parameter.ok_or("boundary_sample_parameter")?.into();
            // Inverse maps can return a canonical representative even when
            // given a hint. Lift periodic coordinates to the previous sheet
            // before measuring signed area or testing loop containment.
            p = Point2::new(
                lift_periodic(next.x, p.x, surface.period_u())
                    .ok_or("boundary_periodic_coordinate")?,
                lift_periodic(next.y, p.y, surface.period_v())
                    .ok_or("boundary_periodic_coordinate")?,
            );
            Ok(p)
        };
        let add: DivisionResult<Vec<Point2>> = match edge.orientation() {
            true => poly.iter().skip(1).map(closure).collect(),
            false => poly.iter().rev().skip(1).map(closure).collect(),
        };
        vec.append(&mut add?);
        Ok::<_, DivisionFailure>(vec)
    })?;
    Ok(PolylineCurve(vec))
}

fn lift_periodic(value: f64, previous: f64, period: Option<f64>) -> Option<f64> {
    let lifted = match period {
        Some(period) if period.is_finite() && period > 0.0 => {
            value + ((previous - value) / period).round() * period
        }
        Some(_) => return None,
        None => value,
    };
    lifted.is_finite().then_some(lifted)
}

#[derive(Clone, Debug)]
struct WireChunk<'a, C> {
    poly: PolylineCurve<Point2>,
    area: f64,
    wire: &'a BoundaryWire<Point3, C>,
}

fn enclosing_pre_face<C>(
    pre_faces: &[Vec<WireChunk<'_, C>>],
    hole: &WireChunk<'_, C>,
) -> Option<usize> {
    let point = hole.poly.front();
    pre_faces
        .iter()
        .enumerate()
        .filter_map(|(index, face)| {
            let outer = face.first()?;
            // An independent intersection loop contributes both orientations of
            // the same native edges. Its positive twin is the adjacent region,
            // not a container for the negative loop.
            let inverse_twin = outer.wire.len() == hole.wire.len()
                && outer
                    .wire
                    .iter()
                    .all(|edge| hole.wire.iter().any(|other| edge.id() == other.id()));
            if inverse_twin || !outer.poly.include(point) {
                None
            } else {
                Some((index, outer.area))
            }
        })
        .min_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1))
        .map(|(index, _)| index)
}

type FaceWithShapesOpStatus<C, S> = (Face<Point3, C, S>, ShapesOpStatus);
fn divide_one_face<C, S>(
    face: &Face<Point3, C, S>,
    loops: &Loops<Point3, C>,
    tol: f64,
    polys: &mut CurveDivisionCache<Point3, C>,
    work: Option<&BooleanWork<'_>>,
) -> DivisionResult<Vec<FaceWithShapesOpStatus<C, S>>>
where
    C: BoundedCurve<Point = Point3> + ParameterDivision1D<Point = Point3>,
    S: Clone + ParametricSurface + SearchParameter<SurfaceParameter, Point = Point3>,
{
    let (mut pre_faces, mut negative_wires) = (Vec::new(), Vec::new());
    loops.iter().try_for_each(|wire| {
        let poly = create_parameter_boundary(face, wire, polys, work)?;
        let area = poly.area();
        if area.abs() < tol {
            return Ok(());
        }
        match area > 0.0 {
            true => pre_faces.push(vec![WireChunk { poly, area, wire }]),
            false => negative_wires.push(WireChunk { poly, area, wire }),
        }
        Ok::<(), DivisionFailure>(())
    })?;
    negative_wires.into_iter().for_each(|chunk| {
        if let Some(i) = enclosing_pre_face(&pre_faces, &chunk) {
            let outer_area = pre_faces[i][0].area;
            let chunk_area = chunk.area;
            // If the sum of areas is zero, the face is canceled.
            // This happens when an intersection loop exactly matches the face boundary.
            if (outer_area + chunk_area).abs() < tol {
                pre_faces[i].clear();
            } else {
                pre_faces[i].push(chunk);
            }
        }
    });
    let vec: Vec<_> = pre_faces
        .into_iter()
        .filter(|pre_face| !pre_face.is_empty())
        .map(|pre_face| {
            let surface = face.surface();
            let op = pre_face
                .iter()
                .find(|chunk| chunk.wire.status() != ShapesOpStatus::Unknown);
            let status = match op {
                Some(chunk) => chunk.wire.status(),
                None => ShapesOpStatus::Unknown,
            };
            let wires: Vec<Wire<Point3, C>> = pre_face
                .iter()
                .map(|chunk| chunk.wire.deref().clone())
                .collect();
            let mut new_face = Face::try_new(wires, surface).map_err(|source| {
                let wire_index = pre_face
                    .iter()
                    .position(|chunk| !chunk.wire.is_closed() || !chunk.wire.is_simple())
                    .unwrap_or(0);
                let wire = pre_face[wire_index].wire;
                DivisionFailure::Geometry {
                    stage: "divided_face_topology",
                    source: Some(source),
                    boundary: Some(DivideFaceBoundaryData {
                        wire_count: pre_face.len(),
                        wire_index,
                        wire_len: wire.len(),
                        closed: wire.is_closed(),
                        simple: wire.is_simple(),
                        edges: wire
                            .iter()
                            .take(12)
                            .map(|edge| {
                                (
                                    edge.front().point(),
                                    edge.back().point(),
                                    edge.orientation(),
                                )
                            })
                            .collect(),
                    }),
                }
            })?;
            if !face.orientation() {
                new_face.invert();
            }
            Ok((new_face, status))
        })
        .collect::<DivisionResult<Vec<_>>>()?;
    Ok(vec)
}

#[cfg(test)]
pub(super) fn divide_faces<C, S>(
    shell: &Shell<Point3, C, S>,
    loops_store: &LoopsStore<Point3, C>,
    tol: f64,
    shell_index: usize,
    polys: &mut CurveDivisionCache<Point3, C>,
    work: Option<&BooleanWork<'_>>,
) -> ClassicResult<FacesClassification<Point3, C, S>>
where
    C: BoundedCurve<Point = Point3> + ParameterDivision1D<Point = Point3>,
    S: Clone + ParametricSurface + SearchParameter<SurfaceParameter, Point = Point3>,
{
    divide_faces_recording(
        shell,
        loops_store,
        tol,
        shell_index,
        polys,
        work,
        &mut |_, _| Ok(()),
    )
}

pub(super) fn divide_faces_recording<C, S>(
    shell: &Shell<Point3, C, S>,
    loops_store: &LoopsStore<Point3, C>,
    tol: f64,
    shell_index: usize,
    polys: &mut CurveDivisionCache<Point3, C>,
    work: Option<&BooleanWork<'_>>,
    record: &mut impl FnMut(usize, FaceId<S>) -> ClassicResult<()>,
) -> ClassicResult<FacesClassification<Point3, C, S>>
where
    C: BoundedCurve<Point = Point3> + ParameterDivision1D<Point = Point3>,
    S: Clone + ParametricSurface + SearchParameter<SurfaceParameter, Point = Point3>,
{
    let mut res = FacesClassification::<Point3, C, S>::default();
    shell
        .iter()
        .zip(loops_store)
        .enumerate()
        .try_for_each(|(face_index, (face, loops))| {
            if let Some(work) = work {
                work.check()?;
            }
            if loops
                .iter()
                .all(|wire| wire.status() == ShapesOpStatus::Unknown)
            {
                let unchanged = face.absolute_boundaries().len() == loops.len()
                    && face
                        .absolute_boundaries()
                        .iter()
                        .zip(loops.iter())
                        .all(|(old, new)| {
                            old.len() == new.len()
                                && old.iter().zip(new.iter()).all(|(a, b)| {
                                    a.id() == b.id() && a.orientation() == b.orientation()
                                })
                        });
                let updated = if unchanged {
                    face.clone()
                } else {
                    let wires = loops.iter().map(|wire| wire.deref().clone()).collect();
                    let mut updated = Face::try_new(wires, face.surface()).map_err(|source| {
                        ShapeOpsError::DivideFacesFailed {
                            shell_index,
                            face_index,
                            stage: "propagated_face_topology",
                            source: Some(source),
                            boundary: loops
                                .iter()
                                .enumerate()
                                .find(|(_, wire)| !wire.is_closed() || !wire.is_simple())
                                .map(|(wire_index, wire)| DivideFaceBoundaryData {
                                    wire_count: loops.len(),
                                    wire_index,
                                    wire_len: wire.len(),
                                    closed: wire.is_closed(),
                                    simple: wire.is_simple(),
                                    edges: wire
                                        .iter()
                                        .take(12)
                                        .map(|edge| {
                                            (
                                                edge.front().point(),
                                                edge.back().point(),
                                                edge.orientation(),
                                            )
                                        })
                                        .collect(),
                                }),
                        }
                    })?;
                    if !face.orientation() {
                        updated.invert();
                    }
                    updated.set_stable_id(face.stable_id());
                    updated
                };
                record(face_index, updated.id())?;
                res.push(updated, ShapesOpStatus::Unknown);
            } else {
                let vec =
                    divide_one_face(face, loops, tol, polys, work).map_err(
                        |failure| match failure {
                            DivisionFailure::Work(work) => work,
                            DivisionFailure::Geometry {
                                stage,
                                source,
                                boundary,
                            } => ShapeOpsError::DivideFacesFailed {
                                shell_index,
                                face_index,
                                stage,
                                source,
                                boundary,
                            },
                        },
                    )?;
                vec.into_iter().try_for_each(|(face, status)| {
                    record(face_index, face.id())?;
                    res.push(face, status);
                    Ok::<(), ShapeOpsError>(())
                })?;
            }
            Ok::<(), ShapeOpsError>(())
        })?;
    Ok(res)
}

#[cfg(test)]
mod ownership_tests {
    use super::*;
    use monstertruck_geometry::prelude::Line;

    fn rectangle(min: [f64; 2], max: [f64; 2]) -> PolylineCurve<Point2> {
        PolylineCurve(vec![
            Point2::new(min[0], min[1]),
            Point2::new(max[0], min[1]),
            Point2::new(max[0], max[1]),
            Point2::new(min[0], max[1]),
            Point2::new(min[0], min[1]),
        ])
    }

    fn boundary(poly: &PolylineCurve<Point2>) -> BoundaryWire<Point3, Line<Point3>> {
        let vertices = poly
            .iter()
            .take(poly.len() - 1)
            .map(|point| Vertex::new(Point3::new(point.x, point.y, 0.0)))
            .collect::<Vec<_>>();
        let wire = (0..vertices.len())
            .map(|index| {
                let front = &vertices[index];
                let back = &vertices[(index + 1) % vertices.len()];
                Edge::new(front, back, Line(front.point(), back.point()))
            })
            .collect();
        BoundaryWire::new(wire, ShapesOpStatus::And)
    }

    #[test]
    fn negative_loop_belongs_to_smallest_container_not_its_positive_twin() {
        let clip = rectangle([-1.0, -1.0], [11.0, 11.0]);
        let part = rectangle([0.0, 0.0], [10.0, 10.0]);
        let hole = rectangle([4.0, 4.0], [6.0, 6.0]);
        let (clip_wire, part_wire, hole_wire) = (boundary(&clip), boundary(&part), boundary(&hole));
        let negative_poly = PolylineCurve(hole.iter().rev().copied().collect());
        let negative_wire = hole_wire.inverse();
        let negative = WireChunk {
            area: negative_poly.area(),
            poly: negative_poly,
            wire: &negative_wire,
        };
        let faces = vec![
            vec![WireChunk {
                area: clip.area(),
                poly: clip,
                wire: &clip_wire,
            }],
            vec![WireChunk {
                area: part.area(),
                poly: part,
                wire: &part_wire,
            }],
            vec![WireChunk {
                area: hole.area(),
                poly: hole,
                wire: &hole_wire,
            }],
        ];
        assert_eq!(enclosing_pre_face(&faces, &negative), Some(1));
        // With no intermediate part boundary, the negative twin belongs to
        // the clipping rectangle; the positive enclosed region stays alive.
        assert_eq!(
            enclosing_pre_face(&[faces[0].clone(), faces[2].clone()], &negative),
            Some(0),
        );
    }

    #[test]
    fn coincident_distinct_native_boundaries_are_not_mistaken_for_inverse_twins() {
        let outer = rectangle([-1.0, -1.0], [3.0, 3.0]);
        let inner = rectangle([0.0, 0.0], [2.0, 2.0]);
        let negative_poly = PolylineCurve(inner.iter().rev().copied().collect());
        let (outer_wire, inner_wire, negative_wire) =
            (boundary(&outer), boundary(&inner), boundary(&negative_poly));
        let negative = WireChunk {
            area: negative_poly.area(),
            poly: negative_poly,
            wire: &negative_wire,
        };
        let faces = vec![
            vec![WireChunk {
                area: outer.area(),
                poly: outer,
                wire: &outer_wire,
            }],
            vec![WireChunk {
                area: inner.area(),
                poly: inner,
                wire: &inner_wire,
            }],
        ];
        assert_eq!(enclosing_pre_face(&faces, &negative), Some(1));
    }

    #[test]
    fn unknown_faces_keep_propagated_subedges_carrier_orientation_and_identity() {
        let wire = boundary(&rectangle([0.0, 0.0], [2.0, 2.0])).deref().clone();
        let surface = monstertruck_geometry::prelude::Plane::new(
            Point3::origin(),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );
        let mut face = Face::new(vec![wire.clone()], surface).inverse();
        face.set_stable_id(StableId::new(42));
        let original = wire[0].absolute_clone();
        let midpoint = Vertex::new(original.with_curve(|curve| curve.evaluate(0.5)));
        let (first, second) = original.cut_with_parameter(&midpoint, 0.5).unwrap();
        let mut updated = wire;
        updated[0] = first.clone();
        updated.insert(1, second.clone());
        let mut store: LoopsStore<_, _> = std::iter::once(&face).collect();
        store[0][0] = BoundaryWire::new(updated, ShapesOpStatus::Unknown);
        let shell = vec![face.clone()].into();
        let mut divisions = CurveDivisionCache::new(1.0e-5);
        let classification = divide_faces(&shell, &store, 1.0e-5, 0, &mut divisions, None).unwrap();
        let [and, or, unknown] = classification.and_or_unknown();
        assert!(and.is_empty() && or.is_empty());
        assert_eq!(unknown.len(), 1);
        let updated = &unknown[0];
        assert_eq!(updated.stable_id(), face.stable_id());
        assert_eq!(updated.orientation(), face.orientation());
        assert_eq!(updated.surface(), face.surface());
        assert_eq!(updated.absolute_boundaries()[0].len(), 5);
        assert!(updated.edge_iter().any(|edge| edge.id() == first.id()));
        assert!(updated.edge_iter().any(|edge| edge.id() == second.id()));
        assert!(!updated.edge_iter().any(|edge| edge.id() == original.id()));
        assert!(updated.is_geometric_consistent());
        assert_eq!(
            face.absolute_boundaries()[0].len(),
            4,
            "input remains unchanged"
        );
    }
}

#[cfg(test)]
mod curve_division_regressions;
