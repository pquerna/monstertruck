//! Classic (0.3.2) boolean loops store.
//!
//! Ported from the published 0.3.2 crate's `transversal::loops_store`.
//! The marching backend yields a raw [`IntersectionCurve`] leader rather
//! than the former parameter wrapper. Numerical failures retain face-pair
//! and projection-stage identity instead of collapsing into `None`.
//! A counted query-local seed AABB cache bypasses extraction only for strictly
//! disjoint finite meshes; admitted pairs retain full reconstruction and topology.

#![allow(clippy::many_single_char_names)]

use super::super::edge_lineage::{BooleanCutSide, EdgeRewrite, RewriteObserver};
use super::intersection_curve;
use super::{ClassicResult, ShapeOpsError};
use crate::transversal::prepared::BooleanWork;
use monstertruck_core::cgmath64::*;
use monstertruck_geometry::prelude::*;
use monstertruck_meshing::prelude::*;
use monstertruck_topology::{Vertex, *};
use rustc_hash::FxHashMap as HashMap;

mod native_component;
mod native_interval;
mod native_overlap;
mod periodic_loops;
mod seed_bounds;

type PolylineCurve = monstertruck_meshing::prelude::PolylineCurve<Point3>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ShapesOpStatus {
    Unknown,
    And,
    Or,
}

impl ShapesOpStatus {
    fn not(self) -> Self {
        match self {
            Self::Unknown => Self::Unknown,
            Self::And => Self::Or,
            Self::Or => Self::And,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct BoundaryWire<P, C> {
    wire: Wire<P, C>,
    status: ShapesOpStatus,
}

impl<P, C> BoundaryWire<P, C> {
    #[inline(always)]
    pub(super) fn new(wire: Wire<P, C>, status: ShapesOpStatus) -> Self { Self { wire, status } }
    #[inline(always)]
    pub(super) fn status(&self) -> ShapesOpStatus { self.status }
    #[inline(always)]
    pub(super) fn inverse(&self) -> Self {
        Self {
            wire: self.wire.inverse(),
            status: self.status.not(),
        }
    }
}

impl ShapesOpStatus {
    fn from_is_curve<C, S0, S1>(curve: &IntersectionCurve<C, S0, S1>) -> Option<ShapesOpStatus>
    where
        C: ParametricCurve3D + BoundedCurve,
        S0: ParametricSurface3D + SearchNearestParameter<SurfaceParameter, Point = Point3>,
        S1: ParametricSurface3D + SearchNearestParameter<SurfaceParameter, Point = Point3>, {
        let (t0, t1) = curve.range_tuple();
        let t = (t0 + t1) / 2.0;
        let (_, pt0, pt1) = curve.search_triple(t, 100)?;
        let der = curve.leader().derivative(t);
        let normal0 = curve.surface0().normal(pt0[0], pt0[1]);
        let normal1 = curve.surface1().normal(pt1[0], pt1[1]);
        let orientation = normal0.cross(der).dot(normal1);
        if !orientation.is_finite() || orientation == 0.0 {
            None
        } else if orientation > 0.0 {
            Some(ShapesOpStatus::Or)
        } else {
            Some(ShapesOpStatus::And)
        }
    }
}

impl<P, C> std::ops::Deref for BoundaryWire<P, C> {
    type Target = Wire<P, C>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target { &self.wire }
}

impl<P, C> std::ops::DerefMut for BoundaryWire<P, C> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.wire }
}

#[derive(Clone, Debug)]
pub(super) struct Loops<P, C>(Vec<BoundaryWire<P, C>>);
#[derive(Clone, Debug)]
pub(super) struct LoopsStore<P, C>(Vec<Loops<P, C>>);

impl<P, C> std::ops::Deref for Loops<P, C> {
    type Target = Vec<BoundaryWire<P, C>>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<P, C> std::ops::DerefMut for Loops<P, C> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl<P, C> std::ops::Deref for LoopsStore<P, C> {
    type Target = Vec<Loops<P, C>>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<P, C> std::ops::DerefMut for LoopsStore<P, C> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl<P, C> FromIterator<BoundaryWire<P, C>> for Loops<P, C> {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = BoundaryWire<P, C>>>(iter: I) -> Self {
        Self(Vec::from_iter(iter))
    }
}

impl<'a, P, C, S> From<&'a Face<P, C, S>> for Loops<P, C> {
    #[inline(always)]
    fn from(face: &'a Face<P, C, S>) -> Loops<P, C> {
        face.absolute_boundaries()
            .iter()
            .map(|wire| BoundaryWire::new(wire.clone(), ShapesOpStatus::Unknown))
            .collect()
    }
}

impl<'a, P: 'a, C: 'a, S: 'a> FromIterator<&'a Face<P, C, S>> for LoopsStore<P, C> {
    fn from_iter<I: IntoIterator<Item = &'a Face<P, C, S>>>(iter: I) -> Self {
        Self(iter.into_iter().map(Loops::from).collect())
    }
}

impl<'a, P, C> IntoIterator for &'a LoopsStore<P, C> {
    type Item = <&'a Vec<Loops<P, C>> as IntoIterator>::Item;
    type IntoIter = <&'a Vec<Loops<P, C>> as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter { self.0.iter() }
}

#[derive(Clone, Debug, Copy, PartialEq)]
enum ParameterKind {
    Front,
    Back,
    Inner(f64),
}

impl ParameterKind {
    fn try_new(t: f64, (t0, t1): (f64, f64)) -> Option<ParameterKind> {
        if t0.near(&t) {
            Some(ParameterKind::Front)
        } else if t1.near(&t) {
            Some(ParameterKind::Back)
        } else if t0 < t && t < t1 {
            Some(ParameterKind::Inner(t))
        } else {
            None
        }
    }
}

impl<P: Copy, C: Clone> Loops<P, C> {
    fn change_vertex(
        &mut self,
        old_vertex: &Vertex<P>,
        new_vertex: &Vertex<P>,
        emap: &mut HashMap<EdgeId<C>, Edge<P, C>>,
        record: &mut Option<&mut RewriteObserver<'_, P, C>>,
    ) -> std::result::Result<(), &'static str> {
        self.iter_mut()
            .flat_map(|wire| wire.iter_mut())
            .try_for_each(|edge| {
                let mut new_edge = if edge.absolute_front() == old_vertex {
                    emap.entry(edge.id()).or_insert_with(|| {
                        Edge::new(new_vertex, edge.absolute_back(), edge.curve())
                    })
                } else if edge.absolute_back() == old_vertex {
                    emap.entry(edge.id()).or_insert_with(|| {
                        Edge::new(edge.absolute_front(), new_vertex, edge.curve())
                    })
                } else {
                    return Ok(());
                }
                .clone();
                if let Some(record) = record.as_mut()
                    && !(**record)(edge.id(), &new_edge, EdgeRewrite::Vertex)
                {
                    return Err("edge_lineage_work");
                }
                if !edge.orientation() {
                    new_edge.invert();
                }
                // Remove the edge from the HashMap when it is no longer there because Id reassignment will occur.
                if edge.count() == 1 {
                    emap.remove(&edge.id());
                }
                *edge = new_edge;
                Ok(())
            })
    }

    fn swap_edge_into_pair(&mut self, edge_id: EdgeId<C>, pair: &[Edge<P, C>; 2]) {
        self.iter_mut().for_each(|wire| {
            // Traverse original occurrence indices backwards: insertion cannot
            // shift a not-yet-visited use, including the opposite seam coedge.
            for index in (0..wire.len()).rev() {
                if wire[index].id() == edge_id {
                    let [first, second] = if wire[index].orientation() {
                        [pair[0].clone(), pair[1].clone()]
                    } else {
                        [pair[1].inverse(), pair[0].inverse()]
                    };
                    wire[index] = first;
                    wire.insert(index + 1, second);
                }
            }
        });
    }

    #[inline(always)]
    fn add_independent_loop(&mut self, r#loop: BoundaryWire<P, C>) {
        self.push(r#loop.inverse());
        self.push(r#loop);
    }

    fn add_path(
        &mut self,
        mut path: Wire<P, C>,
        status: ShapesOpStatus,
        [a, b]: [Option<(usize, usize)>; 2],
    ) {
        let path_len = path.len();
        if let Some((wire_index0, edge_index0)) = a {
            self[wire_index0].rotate_left(edge_index0);
            let mut inverse = path.inverse();
            path.append(&mut self[wire_index0]);
            path.append(&mut inverse);
            std::mem::swap(&mut self[wire_index0].wire, &mut path);
        }
        match (a, b) {
            (Some((wire_index0, edge_index0)), Some((wire_index1, edge_index1))) => {
                if wire_index0 == wire_index1 {
                    let len = self[wire_index0].len() - 2 * path_len;
                    let edge_index1 = (len + edge_index1 - edge_index0) % len + path_len;
                    let new_wire = self[wire_index0].split_off(edge_index1);
                    self[wire_index0].status = status;
                    self.push(BoundaryWire::new(new_wire, status.not()));
                } else {
                    let mut new_wire0 = self[wire_index1].clone();
                    let mut new_wire1 = new_wire0.split_off(edge_index1);
                    new_wire0.append(&mut self[wire_index0]);
                    new_wire0.append(&mut new_wire1);
                    self[wire_index0] = new_wire0;
                    self.swap_remove(wire_index1);
                }
            }
            (None, Some((wire_index1, edge_index1))) => {
                self[wire_index1].rotate_left(edge_index1);
                let mut inverse = path.inverse();
                inverse.append(&mut self[wire_index1]);
                inverse.append(&mut path);
                self[wire_index1].wire = inverse;
            }
            (None, None) => {
                let mut inverse = path.inverse();
                inverse.append(&mut path);
                self.push(BoundaryWire::new(inverse, ShapesOpStatus::Unknown));
            }
            _ => {}
        }
    }
}

impl<P: Copy + Tolerance, C: Clone> LoopsStore<P, C> {
    #[inline(always)]
    fn change_vertex(
        &mut self,
        old_vertex: &Vertex<P>,
        new_vertex: &Vertex<P>,
        emap: &mut HashMap<EdgeId<C>, Edge<P, C>>,
        record: &mut Option<&mut RewriteObserver<'_, P, C>>,
    ) -> std::result::Result<(), &'static str> {
        self.iter_mut()
            .try_for_each(|loops| loops.change_vertex(old_vertex, new_vertex, emap, record))
    }

    #[inline(always)]
    fn swap_edge_into_pair(&mut self, edge_id: EdgeId<C>, pair: &[Edge<P, C>; 2]) {
        self.iter_mut()
            .for_each(|loops| loops.swap_edge_into_pair(edge_id, pair))
    }

    fn add_polygon_vertex_at(
        &mut self,
        loops_index: usize,
        v: &Vertex<P>,
        attachment: (usize, usize, ParameterKind),
        emap: &mut HashMap<EdgeId<C>, Edge<P, C>>,
    ) -> Option<(usize, usize, ParameterKind)>
    where
        C: Cut<Point = P>,
    {
        let (wire_index, edge_index, kind) = attachment;
        match kind {
            ParameterKind::Front => {
                let old_vertex = self[loops_index][wire_index][edge_index]
                    .absolute_front()
                    .clone();
                self.change_vertex(&old_vertex, v, emap, &mut None).ok()?;
            }
            ParameterKind::Back => {
                let old_vertex = self[loops_index][wire_index][edge_index]
                    .absolute_back()
                    .clone();
                self.change_vertex(&old_vertex, v, emap, &mut None).ok()?;
            }
            ParameterKind::Inner(t) => {
                let edge = self[loops_index][wire_index][edge_index].absolute_clone();
                let edge_id = edge.id();
                let (edge0, edge1) = edge.cut_with_parameter(v, t)?;
                self.swap_edge_into_pair(edge_id, &[edge0, edge1]);
            }
        }
        Some((wire_index, edge_index, kind))
    }
}

impl<C> LoopsStore<Point3, C> {
    fn add_geom_vertex(
        &mut self,
        (loops_index, wire_index, edge_index): (usize, usize, usize),
        v: &Vertex<Point3>,
        kind: ParameterKind,
        emap: &mut HashMap<EdgeId<C>, Edge<Point3, C>>,
        record: &mut Option<&mut RewriteObserver<'_, Point3, C>>,
    ) -> std::result::Result<(), &'static str>
    where
        C: Cut<Point = Point3, Vector = Vector3>,
    {
        match kind {
            ParameterKind::Front | ParameterKind::Back => {
                let edge = &self[loops_index][wire_index][edge_index];
                let old_vertex = if kind == ParameterKind::Front {
                    edge.absolute_front().clone()
                } else {
                    edge.absolute_back().clone()
                };
                if !v.point().near(&old_vertex.point()) {
                    return Err("curve_cut_endpoint_residual");
                }
                v.set_point(old_vertex.point());
                self.change_vertex(&old_vertex, v, emap, record)?;
            }
            ParameterKind::Inner(parameter) => {
                let edge = self[loops_index][wire_index][edge_index].absolute_clone();
                if !edge.with_curve(|curve| curve.evaluate(parameter).near(&v.point())) {
                    return Err("curve_cut_residual");
                }
                let edge_id = edge.id();
                let parent_domain = edge.with_curve(ParametricCurve::parameter_range);
                let (edge0, edge1) = edge
                    .cut_with_parameter(v, parameter)
                    .ok_or("curve_cut_endpoint")?;
                if let Some(record) = record.as_mut()
                    && (!(**record)(
                        edge_id,
                        &edge0,
                        EdgeRewrite::Cut {
                            parent_domain,
                            parameter,
                            side: BooleanCutSide::Before,
                        },
                    ) || !(**record)(
                        edge_id,
                        &edge1,
                        EdgeRewrite::Cut {
                            parent_domain,
                            parameter,
                            side: BooleanCutSide::After,
                        },
                    ))
                {
                    return Err("edge_lineage_work");
                }
                self.swap_edge_into_pair(edge_id, &[edge0, edge1]);
            }
        }
        Ok(())
    }
}

struct OriginalTrimContact<'a, C> {
    point: Point3,
    edge: &'a Edge<Point3, C>,
    parameter: f64,
    input: usize,
}

impl<C> OriginalTrimContact<'_, C> {
    fn same_origin(&self, other: &Self) -> bool {
        self.input == other.input
            && self.edge.id() == other.edge.id()
            && self.point.near(&other.point)
    }
}

struct NativeBranch<'a, C, S> {
    original_index: usize,
    polyline: PolylineCurve,
    intersection_curve: IntersectionCurve<PolylineCurve, S, S>,
    coordinates: Option<native_interval::Coordinates>,
    origins: Option<ClassicResult<[OriginalTrimContact<'a, C>; 2]>>,
    endpoint_coordinates: Option<[f64; 2]>,
}

impl<C, S> NativeBranch<'_, C, S> {
    fn priority(&self) -> f64 {
        if let Some([first, last]) = self.endpoint_coordinates {
            (last - first).abs()
        } else if self.origins.is_none() {
            self.coordinates
                .as_ref()
                .map_or(0.0, native_interval::Coordinates::extent)
        } else {
            0.0
        }
    }
}

fn native_endpoint_contact<'a, C, S, I>(
    original_edges: [I; 2],
    carriers: [&S; 2],
    seed: Point3,
    work: Option<&BooleanWork<'_>>,
) -> std::result::Result<OriginalTrimContact<'a, C>, &'static str>
where
    C: 'a
        + ParametricCurve3D
        + BoundedCurve
        + SearchNearestParameter<CurveParameter, Point = Point3>,
    S: ParametricSurface3D
        + SearchParameter<SurfaceParameter, Point = Point3>
        + SearchNearestParameter<SurfaceParameter, Point = Point3>,
    I: IntoIterator<Item = &'a Edge<Point3, C>>,
{
    // Only immutable input trims own endpoint authority. Previously generated
    // intersection coedges are not original boundaries, even when they lie on
    // both carriers and provide a close Newton seed.
    let mut closest: Option<(f64, OriginalTrimContact<'a, C>)> = None;
    let mut ambiguous = false;
    for (input, edges) in original_edges.into_iter().enumerate() {
        for edge in edges {
            if let Some(work) = work {
                work.check().map_err(|_| "boundary_contact_cancelled")?;
            }
            let candidates = edge.with_curve(|curve| {
                let range = curve.range_tuple();
                let certify = |point: Point3, parameter: f64| {
                    native_parameter_kind(edge, curve, point, parameter)?;
                    let distance = point.distance2(seed);
                    if !distance.is_finite() || !curve.evaluate(parameter).near(&point) {
                        return None;
                    }
                    let parameters =
                        carriers.map(|carrier| carrier.search_parameter(point, None, 100));
                    let [Some(first), Some(second)] = parameters else {
                        return None;
                    };
                    if !carriers[0].evaluate(first.0, first.1).near(&point)
                        || !carriers[1].evaluate(second.0, second.1).near(&point)
                    {
                        return None;
                    }
                    let tangent2 = carriers[0]
                        .normal(first.0, first.1)
                        .cross(carriers[1].normal(second.0, second.1))
                        .magnitude2();
                    (tangent2.is_finite() && tangent2 > 0.0).then_some((distance, point, parameter))
                };
                let projected =
                    curve_surface_projection(curve, None, carriers[1 - input], None, seed, 100)
                        .ok()
                        .and_then(|(point, parameter, _)| certify(point, parameter));
                let cutoff = closest
                    .as_ref()
                    .map(|(distance, _)| *distance)
                    .unwrap_or(f64::INFINITY)
                    .min(
                        projected
                            .map(|(distance, _, _)| distance)
                            .unwrap_or(f64::INFINITY),
                    );
                let endpoint = |point: Point3, parameter| {
                    if point.distance2(seed) <= cutoff + TOLERANCE2 {
                        certify(point, parameter)
                    } else {
                        None
                    }
                };
                // Native shared corners must also participate: a local solve
                // can select only one of two equidistant bounded contacts.
                [
                    projected,
                    endpoint(edge.absolute_front().point(), range.0),
                    endpoint(edge.absolute_back().point(), range.1),
                ]
            });
            for (distance, point, parameter) in candidates.into_iter().flatten() {
                let contact = OriginalTrimContact {
                    point,
                    edge,
                    parameter,
                    input,
                };
                if let Some((previous_distance, previous)) = &closest {
                    if distance < *previous_distance - TOLERANCE2 {
                        closest = Some((distance, contact));
                        ambiguous = false;
                    } else if (distance - *previous_distance).abs() <= TOLERANCE2 {
                        ambiguous |= !point.near(&previous.point);
                        if distance < *previous_distance {
                            closest = Some((distance, contact));
                        }
                    }
                } else {
                    closest = Some((distance, contact));
                }
            }
        }
    }
    if ambiguous {
        Err("boundary_contact_ambiguous_original_trims")
    } else {
        closest
            .map(|(_, point)| point)
            .ok_or("boundary_contact_no_certified_original_trim")
    }
}

#[derive(Clone, Copy, Debug)]
struct BoundaryAttachment {
    wire_index: usize,
    edge_index: usize,
    native_kind: ParameterKind,
    polygon_kind: ParameterKind,
}

fn native_parameter_kind<C>(
    edge: &Edge<Point3, C>,
    curve: &C,
    point: Point3,
    parameter: f64,
) -> Option<ParameterKind>
where
    C: ParametricCurve3D + BoundedCurve,
{
    let (front, back) = curve.range_tuple();
    if point.near(&edge.absolute_front().point()) && curve.evaluate(front).near(&point) {
        Some(ParameterKind::Front)
    } else if point.near(&edge.absolute_back().point()) && curve.evaluate(back).near(&point) {
        Some(ParameterKind::Back)
    } else if front < parameter && parameter < back {
        Some(ParameterKind::Inner(parameter))
    } else {
        None
    }
}

fn boundary_attachment<C, S>(
    geom: &LoopsStore<Point3, C>,
    poly: &LoopsStore<Point3, PolylineCurve>,
    face_index: usize,
    polygon_vertex: &Vertex<Point3>,
    native_vertex: &Vertex<Point3>,
    carriers: [&S; 2],
) -> std::result::Result<Option<BoundaryAttachment>, &'static str>
where
    C: ParametricCurve3D + BoundedCurve + SearchParameter<CurveParameter, Point = Point3>,
    S: ParametricSurface3D + SearchParameter<SurfaceParameter, Point = Point3>,
{
    let point = native_vertex.point();
    let candidate = geom[face_index]
        .iter()
        .enumerate()
        .flat_map(|(wire_index, wire)| {
            wire.iter()
                .enumerate()
                .filter_map(move |(edge_index, edge)| {
                    edge.with_curve(|curve| {
                        let parameter = curve.search_parameter(point, None, 100)?;
                        if !curve.evaluate(parameter).near(&point) {
                            return None;
                        }
                        let kind = native_parameter_kind(edge, curve, point, parameter)?;
                        Some((wire_index, edge_index, kind))
                    })
                })
        })
        .next();
    let Some((wire_index, edge_index, native_kind)) = candidate else {
        return Ok(None);
    };
    for carrier in carriers {
        let parameter = carrier
            .search_parameter(point, None, 100)
            .ok_or("boundary_contact_original_carrier")?;
        if !carrier.evaluate(parameter.0, parameter.1).near(&point) {
            return Err("boundary_contact_original_carrier_residual");
        }
    }
    let endpoint = match native_kind {
        ParameterKind::Front => Some(
            geom[face_index][wire_index][edge_index]
                .absolute_front()
                .point(),
        ),
        ParameterKind::Back => Some(
            geom[face_index][wire_index][edge_index]
                .absolute_back()
                .point(),
        ),
        ParameterKind::Inner(_) => None,
    };
    if let Some(endpoint) = endpoint {
        for carrier in carriers {
            let parameter = carrier
                .search_parameter(endpoint, None, 100)
                .ok_or("boundary_contact_original_endpoint_carrier")?;
            if !carrier.evaluate(parameter.0, parameter.1).near(&endpoint) {
                return Err("boundary_contact_original_endpoint_residual");
            }
        }
    }
    let (polygon_point, polygon_kind) =
        poly[face_index][wire_index][edge_index].with_curve(|curve| match native_kind {
            ParameterKind::Front => Ok((curve.front(), ParameterKind::Front)),
            ParameterKind::Back => Ok((curve.back(), ParameterKind::Back)),
            ParameterKind::Inner(_) => {
                let parameter = curve
                    .search_nearest_parameter(point, None, 100)
                    .ok_or("boundary_contact_polygon_parameter")?;
                let kind = ParameterKind::try_new(parameter, curve.range_tuple())
                    .ok_or("boundary_contact_polygon_bounds")?;
                if !matches!(kind, ParameterKind::Inner(_)) {
                    return Err("boundary_contact_parameter_kind");
                }
                Ok((curve.evaluate(parameter), kind))
            }
        })?;
    polygon_vertex.set_point(polygon_point);
    Ok(Some(BoundaryAttachment {
        wire_index,
        edge_index,
        native_kind,
        polygon_kind,
    }))
}

fn curve_surface_projection<C, S>(
    curve: &C,
    curve_hint: Option<f64>,
    surface: &S,
    surface_hint: Option<(f64, f64)>,
    point: Point3,
    trials: usize,
) -> std::result::Result<(Point3, f64, Point2), &'static str>
where
    C: ParametricCurve3D + BoundedCurve + SearchNearestParameter<CurveParameter, Point = Point3>,
    S: ParametricSurface3D + SearchNearestParameter<SurfaceParameter, Point = Point3>,
{
    if trials == 0 {
        return Err("curve_surface_projection_trials");
    }
    let (front, back) = curve.range_tuple();
    if !front.is_finite() || !back.is_finite() || front >= back {
        return Err("curve_surface_projection_domain");
    }
    let mut t = curve
        .search_nearest_parameter(point, curve_hint, 10)
        .filter(|parameter| parameter.is_finite())
        .ok_or("curve_surface_projection_curve_parameter")?
        .clamp(front, back);
    let mut hint = surface_hint;
    let mut settled = false;
    for _ in 0..trials {
        let pt0 = curve.evaluate(t);
        let (u, v) = surface
            .search_nearest_parameter(pt0, hint, 10)
            .ok_or("curve_surface_projection_surface_parameter")?;
        let pt1 = surface.evaluate(u, v);
        let tangent = curve.derivative(t);
        let normal = surface.normal(u, v);
        let denominator = tangent.dot(normal);
        if denominator == 0.0 && pt0.near(&pt1) {
            return Ok((pt0, t, Point2::new(u, v)));
        }
        if denominator == 0.0 || !denominator.is_finite() {
            return Err("curve_surface_projection_nontransversal");
        }
        let correction = (pt1 - pt0).dot(normal) / denominator;
        let step2 = (correction * tangent).magnitude2();
        if !correction.is_finite() || !step2.is_finite() {
            return Err("curve_surface_projection_nonfinite_step");
        }
        // A carrier residual alone is ill-conditioned at shallow crossings.
        // Certify convergence in world distance and refine once after settling.
        let small = pt0.near(&pt1) && step2 <= TOLERANCE2;
        if small && (settled || correction == 0.0) {
            return Ok((pt0, t, Point2::new(u, v)));
        }
        settled = small;
        let next = t + correction;
        if !next.is_finite() {
            return Err("curve_surface_projection_nonfinite_step");
        }
        t = next.clamp(front, back);
        hint = Some((u, v));
    }
    Err("curve_surface_projection_trials")
}

fn create_independent_loop<P, C, D>(mut poly_curve0: C) -> Wire<P, D>
where
    C: Cut<Point = P>,
    D: From<C>, {
    let (t0, t1) = poly_curve0.range_tuple();
    let t = (t0 + t1) / 2.0;
    let poly_curve1 = poly_curve0.cut(t);
    let v0 = Vertex::new(poly_curve0.front());
    let v1 = Vertex::new(poly_curve1.front());
    let edge0 = Edge::new(&v0, &v1, poly_curve0.into());
    let edge1 = Edge::new(&v1, &v0, poly_curve1.into());
    wire![edge0, edge1]
}

#[allow(dead_code)]
pub(super) struct LoopsStoreQuadruple<C> {
    pub(super) geom_loops_store0: LoopsStore<Point3, C>,
    pub(super) poly_loops_store0: LoopsStore<Point3, PolylineCurve>,
    pub(super) geom_loops_store1: LoopsStore<Point3, C>,
    pub(super) poly_loops_store1: LoopsStore<Point3, PolylineCurve>,
}

pub(super) fn create_loops_stores<C, S>(
    geom_shell0: &Shell<Point3, C, S>,
    poly_shell0: &Shell<Point3, PolylineCurve, Option<PolygonMesh>>,
    geom_shell1: &Shell<Point3, C, S>,
    poly_shell1: &Shell<Point3, PolylineCurve, Option<PolygonMesh>>,
    tolerance: f64,
    work: Option<&BooleanWork<'_>>,
    mut record: Option<&mut RewriteObserver<'_, Point3, C>>,
) -> ClassicResult<LoopsStoreQuadruple<C>>
where
    C: SearchNearestParameter<CurveParameter, Point = Point3>
        + SearchParameter<CurveParameter, Point = Point3>
        + Cut<Point = Point3, Vector = Vector3>
        + From<IntersectionCurve<PolylineCurve, S, S>>
        + TryIntoHomogeneousBsplineCurve,
    S: ParametricSurface3D
        + Clone
        + SearchParameter<SurfaceParameter, Point = Point3>
        + TryIntoAnalyticSurfaceKind
        + TryIntoHomogeneousBsplineSurface
        + SearchNearestParameter<SurfaceParameter, Point = Point3>,
{
    let seed_bounds = seed_bounds::SeedBounds::new([poly_shell0, poly_shell1], work)?;
    let mut geom_loops_store0: LoopsStore<_, _> = geom_shell0.face_iter().collect();
    let mut poly_loops_store0: LoopsStore<_, _> = poly_shell0.face_iter().collect();
    let mut geom_loops_store1: LoopsStore<_, _> = geom_shell1.face_iter().collect();
    let mut poly_loops_store1: LoopsStore<_, _> = poly_shell1.face_iter().collect();
    let store0_len = geom_loops_store0.len();
    let store1_len = geom_loops_store1.len();
    (0..store0_len)
        .flat_map(move |i| (0..store1_len).map(move |j| (i, j)))
        .try_for_each(|(face_index0, face_index1)| {
            if seed_bounds.disjoint(face_index0, face_index1, work)? {
                return Ok(());
            }
            let failure = |stage, detail| ShapeOpsError::IntersectionLoopsFailed {
                face_index0,
                face_index1,
                stage,
                detail,
            };
            let ori0 = geom_shell0[face_index0].orientation();
            let ori1 = geom_shell1[face_index1].orientation();
            let surface0 = geom_shell0[face_index0].surface();
            let surface1 = geom_shell1[face_index1].surface();
            let intersections = poly_shell0[face_index0].with_surface(|polygon0| {
                let polygon0 = polygon0
                    .as_ref()
                    .ok_or(ShapeOpsError::InputTessellationFailed {
                        shell_index: 0,
                        face_index: face_index0,
                    })?;
                poly_shell1[face_index1].with_surface(|polygon1| {
                    let polygon1 =
                        polygon1
                            .as_ref()
                            .ok_or(ShapeOpsError::InputTessellationFailed {
                                shell_index: 1,
                                face_index: face_index1,
                            })?;
                    intersection_curve::intersection_curves(
                        surface0.clone(),
                        polygon0,
                        surface1.clone(),
                        polygon1,
                        work,
                    )
                })
            })?;
            let intersections = intersections
                .ok_or_else(|| failure("intersection_curve_projection", "native_projection"))?;
            let component = {
                if intersections.len() > 1 {
                    native_component::NativeComponent::new([&surface0, &surface1], work)?
                } else {
                    None
                }
            };
            let recover_original = |curve: &IntersectionCurve<PolylineCurve, S, S>| {
                let front = native_endpoint_contact(
                    [&geom_shell0[face_index0], &geom_shell1[face_index1]].map(|face| {
                        face.absolute_boundaries()
                            .iter()
                            .flat_map(|wire| wire.iter())
                    }),
                    [&surface0, &surface1],
                    curve.front(),
                    work,
                );
                if let Some(work) = work {
                    work.check()?;
                }
                let front = front.map_err(|detail| failure("boundary_contact_front", detail))?;
                let back = native_endpoint_contact(
                    [&geom_shell0[face_index0], &geom_shell1[face_index1]].map(|face| {
                        face.absolute_boundaries()
                            .iter()
                            .flat_map(|wire| wire.iter())
                    }),
                    [&surface0, &surface1],
                    curve.back(),
                    work,
                );
                if let Some(work) = work {
                    work.check()?;
                }
                let back = back.map_err(|detail| failure("boundary_contact_back", detail))?;
                Ok::<_, ShapeOpsError>([front, back])
            };
            let mut planned = Vec::new();
            let intersections = if let Some(component) = &component {
                if let Some(work) = work {
                    work.limit(
                        "native_interval_candidates_bytes",
                        intersections
                            .len()
                            .saturating_mul(size_of::<NativeBranch<'_, C, S>>()),
                        work.limits.max_intermediate_prepared_bytes,
                    )?;
                }
                planned.reserve_exact(intersections.len());
                for (original_index, (polyline, intersection_curve)) in
                    intersections.into_iter().enumerate()
                {
                    let coordinates =
                        { native_interval::coordinates(component, &intersection_curve, work)? };
                    let origins = if polyline.front().near(&polyline.back()) {
                        None
                    } else {
                        Some(recover_original(&intersection_curve))
                    };
                    let endpoint_coordinates = {
                        match (&coordinates, &origins) {
                            (Some(coordinates), Some(Ok(origins))) => {
                                native_interval::endpoint_coordinates(
                                    component,
                                    coordinates,
                                    origins,
                                    work,
                                )
                            }
                            _ => None,
                        }
                    };
                    if let Some(work) = work {
                        work.check()?;
                    }
                    planned.push(NativeBranch {
                        original_index,
                        polyline,
                        intersection_curve,
                        coordinates,
                        origins,
                        endpoint_coordinates,
                    });
                }
                // Material-supported native spans precede collapsed or
                // uncertain chains; raw graph encounter order is not authority.
                planned.sort_unstable_by(|first, second| {
                    second
                        .priority()
                        .total_cmp(&first.priority())
                        .then_with(|| first.original_index.cmp(&second.original_index))
                });
                Vec::new()
            } else {
                intersections
            };
            let mut intervals = Vec::new();
            planned
                .into_iter()
                .chain(intersections.into_iter().enumerate().map(
                    |(original_index, (polyline, intersection_curve))| NativeBranch {
                        original_index,
                        polyline,
                        intersection_curve,
                        coordinates: None,
                        origins: None,
                        endpoint_coordinates: None,
                    },
                ))
                .try_for_each(
                    |NativeBranch {
                         original_index: _,
                         polyline,
                         mut intersection_curve,
                         coordinates,
                         origins,
                         endpoint_coordinates,
                     }| {
                        let status = ShapesOpStatus::from_is_curve(&intersection_curve)
                            .ok_or_else(|| {
                                failure("intersection_curve_status", "native_projection")
                            })?;
                        let (status0, status1) = match (ori0, ori1) {
                            (true, true) => (status, status.not()),
                            (true, false) => (status.not(), status.not()),
                            (false, true) => (status, status),
                            (false, false) => (status.not(), status),
                        };
                        if polyline.front().near(&polyline.back()) {
                            if let (Some(component), Some(coordinates)) = (&component, &coordinates)
                                && native_interval::covered(
                                    component,
                                    coordinates,
                                    None,
                                    &intervals,
                                    [&surface0, &surface1],
                                    work,
                                )?
                            {
                                return Ok(());
                            }
                            let interval = coordinates.map(native_interval::closed_interval);
                            periodic_loops::add_intersection(
                                (polyline, intersection_curve),
                                None,
                                periodic_loops::IntersectionStores {
                                    native: [&mut geom_loops_store0, &mut geom_loops_store1],
                                    polygon: [&mut poly_loops_store0, &mut poly_loops_store1],
                                    sides: [
                                        (face_index0, &surface0, status0),
                                        (face_index1, &surface1, status1),
                                    ],
                                },
                                status,
                                tolerance,
                                work,
                                &mut record,
                            )
                            .map_err(|error| match error {
                                periodic_loops::AttachmentFailure::Geometry(detail) => {
                                    failure("closed_intersection_chart", detail)
                                }
                                periodic_loops::AttachmentFailure::Work(error) => error,
                            })?;
                            if let Some(interval) = interval {
                                intervals.push(interval);
                            }
                        } else {
                            let pv0 = Vertex::new(polyline.front());
                            let pv1 = Vertex::new(polyline.back());
                            let origins = match origins {
                                Some(origins) => origins?,
                                None => recover_original(&intersection_curve)?,
                            };
                            if let (Some(component), Some(coordinates)) = (&component, &coordinates)
                                && native_interval::covered(
                                    component,
                                    coordinates,
                                    Some(&origins),
                                    &intervals,
                                    [&surface0, &surface1],
                                    work,
                                )?
                            {
                                return Ok(());
                            }
                            let front_contact = origins[0].point;
                            let back_contact = origins[1].point;
                            if front_contact.near(&back_contact) {
                                return Err(failure(
                                    "native_boundary_interval",
                                    "collapsed_original_trim_contacts",
                                ));
                            }
                            let interval = endpoint_coordinates.map(|coordinates| {
                                native_interval::open_interval(coordinates, origins)
                            });
                            if let Some(work) = work {
                                work.check()?;
                            }
                            let gv0 = Vertex::new(front_contact);
                            let gv1 = Vertex::new(back_contact);
                            let mut pemap0 = HashMap::default();
                            let mut pemap1 = HashMap::default();
                            let mut gemap0 = HashMap::default();
                            let mut gemap1 = HashMap::default();
                            let idx00 = boundary_attachment(
                                &geom_loops_store0,
                                &poly_loops_store0,
                                face_index0,
                                &pv0,
                                &gv0,
                                [&surface0, &surface1],
                            )
                            .map_err(|detail| failure("boundary_attachment0_front", detail))?;
                            if let Some(BoundaryAttachment {
                                wire_index,
                                edge_index,
                                native_kind,
                                polygon_kind,
                            }) = idx00
                            {
                                poly_loops_store0
                                    .add_polygon_vertex_at(
                                        face_index0,
                                        &pv0,
                                        (wire_index, edge_index, polygon_kind),
                                        &mut pemap0,
                                    )
                                    .ok_or_else(|| {
                                        failure(
                                            "boundary_polygon0_front",
                                            "certified_contact_subdivision",
                                        )
                                    })?;
                                geom_loops_store0
                                    .add_geom_vertex(
                                        (face_index0, wire_index, edge_index),
                                        &gv0,
                                        native_kind,
                                        &mut gemap0,
                                        &mut record,
                                    )
                                    .map_err(|detail| failure("boundary_vertex0_front", detail))?;
                                let polyline = intersection_curve.leader_mut();
                                *polyline.first_mut().unwrap() = gv0.point();
                            }
                            let idx01 = boundary_attachment(
                                &geom_loops_store0,
                                &poly_loops_store0,
                                face_index0,
                                &pv1,
                                &gv1,
                                [&surface0, &surface1],
                            )
                            .map_err(|detail| failure("boundary_attachment0_back", detail))?;
                            if let Some(BoundaryAttachment {
                                wire_index,
                                edge_index,
                                native_kind,
                                polygon_kind,
                            }) = idx01
                            {
                                poly_loops_store0
                                    .add_polygon_vertex_at(
                                        face_index0,
                                        &pv1,
                                        (wire_index, edge_index, polygon_kind),
                                        &mut pemap1,
                                    )
                                    .ok_or_else(|| {
                                        failure(
                                            "boundary_polygon0_back",
                                            "certified_contact_subdivision",
                                        )
                                    })?;
                                geom_loops_store0
                                    .add_geom_vertex(
                                        (face_index0, wire_index, edge_index),
                                        &gv1,
                                        native_kind,
                                        &mut gemap1,
                                        &mut record,
                                    )
                                    .map_err(|detail| failure("boundary_vertex0_back", detail))?;
                                let polyline = intersection_curve.leader_mut();
                                *polyline.last_mut().unwrap() = gv1.point();
                            }
                            let contact0_front = idx00.map(|_| gv0.point());
                            let contact0_back = idx01.map(|_| gv1.point());
                            let idx10 = boundary_attachment(
                                &geom_loops_store1,
                                &poly_loops_store1,
                                face_index1,
                                &pv0,
                                &gv0,
                                [&surface1, &surface0],
                            )
                            .map_err(|detail| failure("boundary_attachment1_front", detail))?;
                            if let Some(BoundaryAttachment {
                                wire_index,
                                edge_index,
                                native_kind,
                                polygon_kind,
                            }) = idx10
                            {
                                poly_loops_store1
                                    .add_polygon_vertex_at(
                                        face_index1,
                                        &pv0,
                                        (wire_index, edge_index, polygon_kind),
                                        &mut pemap0,
                                    )
                                    .ok_or_else(|| {
                                        failure(
                                            "boundary_polygon1_front",
                                            "certified_contact_subdivision",
                                        )
                                    })?;
                                geom_loops_store1
                                    .add_geom_vertex(
                                        (face_index1, wire_index, edge_index),
                                        &gv0,
                                        native_kind,
                                        &mut gemap0,
                                        &mut record,
                                    )
                                    .map_err(|detail| failure("boundary_vertex1_front", detail))?;
                                let polyline = intersection_curve.leader_mut();
                                *polyline.first_mut().unwrap() = gv0.point();
                            }
                            if contact0_front.is_some_and(|point| !point.near(&gv0.point())) {
                                return Err(failure(
                                    "boundary_contact_front",
                                    "inconsistent_native_contacts",
                                ));
                            }
                            let idx11 = boundary_attachment(
                                &geom_loops_store1,
                                &poly_loops_store1,
                                face_index1,
                                &pv1,
                                &gv1,
                                [&surface1, &surface0],
                            )
                            .map_err(|detail| failure("boundary_attachment1_back", detail))?;
                            if let Some(BoundaryAttachment {
                                wire_index,
                                edge_index,
                                native_kind,
                                polygon_kind,
                            }) = idx11
                            {
                                poly_loops_store1
                                    .add_polygon_vertex_at(
                                        face_index1,
                                        &pv1,
                                        (wire_index, edge_index, polygon_kind),
                                        &mut pemap1,
                                    )
                                    .ok_or_else(|| {
                                        failure(
                                            "boundary_polygon1_back",
                                            "certified_contact_subdivision",
                                        )
                                    })?;
                                geom_loops_store1
                                    .add_geom_vertex(
                                        (face_index1, wire_index, edge_index),
                                        &gv1,
                                        native_kind,
                                        &mut gemap1,
                                        &mut record,
                                    )
                                    .map_err(|detail| failure("boundary_vertex1_back", detail))?;
                                let polyline = intersection_curve.leader_mut();
                                *polyline.last_mut().unwrap() = gv1.point();
                            }
                            if contact0_back.is_some_and(|point| !point.near(&gv1.point())) {
                                return Err(failure(
                                    "boundary_contact_back",
                                    "inconsistent_native_contacts",
                                ));
                            }
                            if (idx00.is_none() && idx10.is_none())
                                || (idx01.is_none() && idx11.is_none())
                            {
                                return Err(failure(
                                    "boundary_contact",
                                    "unattached_native_endpoint",
                                ));
                            }
                            if !front_contact.near(&gv0.point()) || !back_contact.near(&gv1.point())
                            {
                                return Err(failure(
                                    "boundary_contact",
                                    "original_trim_contact_changed",
                                ));
                            }
                            for point in [gv0.point(), gv1.point()] {
                                for carrier in [&surface0, &surface1] {
                                    let parameter = carrier
                                        .search_parameter(point, None, 100)
                                        .ok_or_else(|| {
                                            failure(
                                                "boundary_contact",
                                                "original_carrier_parameter",
                                            )
                                        })?;
                                    if !carrier.evaluate(parameter.0, parameter.1).near(&point) {
                                        return Err(failure(
                                            "boundary_contact",
                                            "original_carrier_residual",
                                        ));
                                    }
                                }
                            }
                            let mut polyline = polyline;
                            *polyline.first_mut().unwrap() = pv0.point();
                            *polyline.last_mut().unwrap() = pv1.point();
                            let leader = intersection_curve.leader_mut();
                            *leader.first_mut().unwrap() = gv0.point();
                            *leader.last_mut().unwrap() = gv1.point();
                            periodic_loops::add_intersection(
                                (polyline, intersection_curve),
                                Some([(pv0, gv0), (pv1, gv1)]),
                                periodic_loops::IntersectionStores {
                                    native: [&mut geom_loops_store0, &mut geom_loops_store1],
                                    polygon: [&mut poly_loops_store0, &mut poly_loops_store1],
                                    sides: [
                                        (face_index0, &surface0, status0),
                                        (face_index1, &surface1, status1),
                                    ],
                                },
                                status,
                                tolerance,
                                work,
                                &mut record,
                            )
                            .map_err(|error| match error {
                                periodic_loops::AttachmentFailure::Geometry(detail) => {
                                    failure("open_intersection_chart", detail)
                                }
                                periodic_loops::AttachmentFailure::Work(error) => error,
                            })?;
                            if let Some(interval) = interval {
                                intervals.push(interval);
                            }
                        }
                        Ok::<(), ShapeOpsError>(())
                    },
                )
        })?;
    Ok(LoopsStoreQuadruple {
        geom_loops_store0,
        poly_loops_store0,
        geom_loops_store1,
        poly_loops_store1,
    })
}

#[cfg(test)]
mod projection_tests {
    use super::*;

    fn xy_plane() -> Plane {
        Plane::new(
            Point3::origin(),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        )
    }

    #[test]
    fn contained_native_boundary_projects_an_off_curve_mesh_seed() {
        let curve = BsplineCurve::new(
            KnotVector::bezier_knot(1),
            vec![Point3::new(-1.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
        );
        let surface = xy_plane();
        let (point, t, uv) = curve_surface_projection(
            &curve,
            None,
            &surface,
            None,
            Point3::new(0.25, 0.01, 0.0),
            100,
        )
        .expect("native boundary contained in the clipping plane");
        assert!(point.near(&Point3::new(0.25, 0.0, 0.0)));
        assert!(point.near(&curve.evaluate(t)));
        assert!(point.near(&surface.evaluate(uv.x, uv.y)));
    }

    #[test]
    fn transversal_boundary_projects_but_disjoint_parallel_boundary_fails() {
        let surface = xy_plane();
        let crossing = BsplineCurve::new(
            KnotVector::bezier_knot(1),
            vec![Point3::new(0.0, 0.0, -1.0), Point3::new(0.0, 0.0, 1.0)],
        );
        let (point, _, _) = curve_surface_projection(
            &crossing,
            None,
            &surface,
            None,
            Point3::new(0.01, 0.0, 0.25),
            100,
        )
        .expect("transversal boundary intersects the clipping plane");
        assert!(point.near(&Point3::origin()));
        let disjoint = BsplineCurve::new(
            KnotVector::bezier_knot(1),
            vec![Point3::new(-1.0, 0.0, 1.0), Point3::new(1.0, 0.0, 1.0)],
        );
        assert_eq!(
            curve_surface_projection(
                &disjoint,
                None,
                &surface,
                None,
                Point3::new(0.25, 0.0, 0.0),
                100,
            )
            .unwrap_err(),
            "curve_surface_projection_nontransversal",
        );
    }

    #[test]
    fn shared_boundary_subdivision_replaces_every_opposite_seam_occurrence() {
        let ring = |z: f64| {
            let vertices: Vec<_> = (0..3)
                .map(|index| {
                    let angle = std::f64::consts::TAU * index as f64 / 3.0;
                    Vertex::new(Point3::new(angle.cos(), angle.sin(), z))
                })
                .collect();
            (0..3)
                .map(|index| {
                    let next = (index + 1) % 3;
                    Edge::new(
                        &vertices[index],
                        &vertices[next],
                        PolylineCurve(vec![vertices[index].point(), vertices[next].point()]),
                    )
                })
                .collect::<Wire<_, _>>()
        };
        let top = ring(3.0);
        let bottom = ring(0.0);
        let seam = Edge::new(
            bottom.front_vertex().unwrap(),
            top.front_vertex().unwrap(),
            PolylineCurve(vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 0.0, 3.0)]),
        );
        let mut lateral = top.inverse();
        lateral.push_back(seam.inverse());
        lateral.extend(bottom.iter().cloned());
        lateral.push_back(seam.clone());
        let mut store = LoopsStore(vec![
            std::iter::once(BoundaryWire::new(lateral, ShapesOpStatus::Unknown)).collect(),
            std::iter::once(BoundaryWire::new(top, ShapesOpStatus::Unknown)).collect(),
            std::iter::once(BoundaryWire::new(bottom.inverse(), ShapesOpStatus::Unknown)).collect(),
        ]);
        let mut replacements = HashMap::default();
        for (cut, length) in [(1.5, 10), (0.75, 12)] {
            let vertex = Vertex::new(Point3::new(1.0, 0.0, cut));
            let attachment = store[0]
                .iter()
                .enumerate()
                .find_map(|(wire_index, wire)| {
                    wire.iter().enumerate().find_map(|(edge_index, edge)| {
                        edge.with_curve(|curve| {
                            let parameter = curve.search_parameter(vertex.point(), None, 100)?;
                            let kind =
                                native_parameter_kind(edge, curve, vertex.point(), parameter)?;
                            (matches!(kind, ParameterKind::Inner(_))
                                && curve.evaluate(parameter).near(&vertex.point()))
                            .then_some((wire_index, edge_index, kind))
                        })
                    })
                })
                .expect("fixture vertex has an exact interior seam attachment");
            assert!(
                store
                    .add_polygon_vertex_at(0, &vertex, attachment, &mut replacements)
                    .is_some()
            );
            assert_eq!(store[0][0].len(), length);
            assert!(
                !store
                    .iter()
                    .flat_map(|loops| loops.iter())
                    .flat_map(|wire| wire.iter())
                    .any(|edge| edge.id() == seam.id())
            );
            let mut uses = HashMap::<_, [usize; 2]>::default();
            for loops in &store {
                for wire in loops.iter() {
                    assert!(wire.is_closed());
                    assert!(wire.iter().all(Edge::is_geometric_consistent));
                    for edge in wire.iter() {
                        let counts = uses.entry(edge.id()).or_default();
                        counts[usize::from(edge.orientation())] += 1;
                    }
                }
                assert!(
                    loops
                        .iter()
                        .all(|wire| wire.status() == ShapesOpStatus::Unknown)
                );
            }
            assert!(
                uses.values().all(|counts| *counts == [1, 1]),
                "each shared child retains exactly two opposite incidences"
            );
        }
        assert_eq!(store[1][0].len(), 3);
        assert_eq!(store[2][0].len(), 3);
    }
}

#[cfg(test)]
#[path = "endpoint_contact_regressions.rs"]
mod endpoint_contact_regressions;
