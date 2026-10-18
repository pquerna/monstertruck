//! Attach intersections to their actual paired chart-boundary uses.
//! Open and spatially closed paths can both cross a periodic chart seam.
//! Native and polygon stores replay the same occurrence-level partition.

use super::*;
use crate::transversal::integrate::NativeIntersectionParameterWitness;
use std::mem::size_of;

/// Whether the target belongs to an existing curve or constructs its new trim.
#[derive(Clone, Copy, Debug)]
pub(super) enum IntersectionParameterBinding {
    /// Existing endpoint representation remains part of the inverse contract.
    BoundCurve,
    /// Original contact is not bound to a generated guide endpoint yet.
    OriginalContactConstruction,
}

pub(super) fn intersection_parameter<S>(
    curve: &IntersectionCurve<PolylineCurve, S, S>,
    point: Point3,
    binding: IntersectionParameterBinding,
    work: Option<&BooleanWork<'_>>,
    diagnostic: Option<&mut Option<NativeIntersectionParameterWitness>>,
) -> std::result::Result<f64, AttachmentFailure>
where
    S: ParametricSurface3D + SearchNearestParameter<SurfaceParameter, Point = Point3>,
{
    if let Some(work) = work {
        work.charge(curve.leader().len())?;
    }
    let mut parameter = curve
        .leader()
        .search_nearest_parameter(point, None, 100)
        .ok_or("intersection_chart_native_parameter")?;
    let (front, back) = curve.range_tuple();
    for _ in 0..100 {
        if let Some(work) = work {
            work.check()?;
        }
        let (actual, uv0, uv1) = curve
            .search_triple(parameter, 100)
            .ok_or("intersection_chart_native_projection")?;
        if !actual.near(&curve.surface0().evaluate(uv0.x, uv0.y))
            || !actual.near(&curve.surface1().evaluate(uv1.x, uv1.y))
        {
            return Err("intersection_chart_native_projection_residual".into());
        }
        let endpoint_matches = match binding {
            IntersectionParameterBinding::OriginalContactConstruction => true,
            IntersectionParameterBinding::BoundCurve if parameter.near(&front) => {
                curve.leader().front().near(&point)
            }
            IntersectionParameterBinding::BoundCurve if parameter.near(&back) => {
                curve.leader().back().near(&point)
            }
            IntersectionParameterBinding::BoundCurve => true,
        };
        // Construction binds the original contact only after restricting this
        // guide. It needs the actual carrier root, not an unbound seed endpoint.
        // Existing curve consumers retain their unchanged endpoint contract.
        if actual.near(&point) && endpoint_matches {
            return Ok(parameter);
        }
        // A guide's nearest chord parameter is only a proposal. Invert the
        // projected native curve with its actual transverse carrier tangent.
        let direction = curve
            .surface0()
            .normal(uv0.x, uv0.y)
            .cross(curve.surface1().normal(uv1.x, uv1.y));
        let guide = curve.leader().derivative(parameter);
        let denominator = direction.dot(guide);
        if !denominator.is_finite() || denominator == 0.0 {
            return Err("intersection_chart_native_parameter_tangent".into());
        }
        let derivative = direction * (guide.magnitude2() / denominator);
        let correction = (actual - point).dot(derivative) / derivative.magnitude2();
        if !correction.is_finite() {
            return Err("intersection_chart_native_parameter_step".into());
        }
        let proposed_parameter = parameter - correction;
        let next = proposed_parameter.clamp(front, back);
        if next == parameter {
            if let Some(diagnostic) = diagnostic {
                let position = |point: Point3| [point.x, point.y, point.z];
                *diagnostic = Some(NativeIntersectionParameterWitness {
                    contact: position(point),
                    range: [front, back],
                    parameter,
                    correction,
                    proposed_parameter,
                    clamped_parameter: next,
                    actual: position(actual),
                    guide_ends: [position(curve.leader().front()), position(curve.leader().back())],
                    guide_tangent: [guide.x, guide.y, guide.z],
                    projected_tangent: [derivative.x, derivative.y, derivative.z],
                });
            }
            return Err("intersection_chart_native_parameter_residual".into());
        }
        parameter = next;
    }
    Err("intersection_chart_native_parameter_trials".into())
}

fn rooted_boundary_attachment<C, S>(
    stores: (&LoopsStore<Point3, C>, &LoopsStore<Point3, PolylineCurve>),
    face_index: usize,
    (root, hint): (EdgeId<C>, f64),
    ancestry: &[(EdgeId<C>, EdgeId<C>)],
    vertices: [&Vertex<Point3>; 2],
    carriers: [&S; 2],
    work: Option<&BooleanWork<'_>>,
) -> std::result::Result<BoundaryAttachment, AttachmentFailure>
where
    C: ParametricCurve3D + BoundedCurve,
    S: ParametricSurface3D + SearchParameter<SurfaceParameter, Point = Point3>,
{
    let (native, polygon) = stores;
    let [polygon_vertex, native_vertex] = vertices;
    let point = native_vertex.point();
    let mut selected = None;
    let mut selected_edge = None;
    for (wire_index, wire) in native[face_index].iter().enumerate() {
        for (edge_index, edge) in wire.iter().enumerate() {
            if let Some(work) = work {
                work.check()?;
            }
            let ancestor = ancestry
                .iter()
                .rev()
                .find_map(|&(child, parent)| (child == edge.id()).then_some(parent))
                .unwrap_or_else(|| edge.id());
            if ancestor != root || selected_edge == Some(edge.id()) {
                continue;
            }
            let kind = edge.with_curve(|curve| {
                let (front, back) = curve.range_tuple();
                let mut parameter = hint.clamp(front, back);
                for _ in 0..100 {
                    if let Some(work) = work {
                        work.check()?;
                    }
                    let actual = curve.evaluate(parameter);
                    if actual.near(&point) {
                        return Ok(native_parameter_kind(edge, curve, point, parameter));
                    }
                    let derivative = curve.derivative(parameter);
                    let correction = (actual - point).dot(derivative) / derivative.magnitude2();
                    if !correction.is_finite() {
                        break;
                    }
                    let next = (parameter - correction).clamp(front, back);
                    if next == parameter {
                        break;
                    }
                    parameter = next;
                }
                Ok::<_, AttachmentFailure>(None)
            })?;
            if let Some(kind) = kind {
                if selected.replace((wire_index, edge_index, kind)).is_some() {
                    return Err("intersection_chart_ambiguous_root_descendant".into());
                }
                selected_edge = Some(edge.id());
            }
        }
    }
    let selected = selected.ok_or("intersection_chart_missing_root_descendant")?;
    boundary_attachment_at(
        native,
        polygon,
        (face_index, selected),
        polygon_vertex,
        native_vertex,
        carriers,
    )
    .map_err(Into::into)
}

fn paired_chart_edges<C>(loops: &Loops<Point3, C>) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    let mut first = HashMap::default();
    for (wire_index, wire) in loops.iter().enumerate() {
        first.clear();
        for (index, edge) in wire.iter().enumerate() {
            if let Some(&(previous, orientation)) = first.get(&edge.id()) {
                if orientation != edge.orientation()
                    && (previous + 1) % wire.len() != index
                    && (index + 1) % wire.len() != previous
                {
                    result.push((wire_index, previous));
                }
            } else {
                first.insert(edge.id(), (index, edge.orientation()));
            }
        }
    }
    result
}

fn tangent<C: ParametricCurve3D + BoundedCurve>(edge: &Edge<Point3, C>, front: bool) -> Vector3 {
    edge.with_curve(|curve| {
        let (start, end) = curve.range_tuple();
        let parameter = if front == edge.orientation() {
            start
        } else {
            end
        };
        let derivative = curve.derivative(parameter);
        if edge.orientation() {
            derivative
        } else {
            -derivative
        }
    })
}

fn intersection_direction<S>(
    surfaces: [&S; 2],
    point: Point3,
    orientation: ShapesOpStatus,
) -> std::result::Result<Vector3, &'static str>
where
    S: ParametricSurface3D + SearchParameter<SurfaceParameter, Point = Point3>,
{
    let normal = |surface: &S| {
        let (u, v) = surface
            .search_parameter(point, None, 100)
            .ok_or("intersection_chart_contact_carrier")?;
        if !surface.evaluate(u, v).near(&point) {
            return Err("intersection_chart_contact_carrier");
        }
        Ok(surface.normal(u, v))
    };
    let direction = normal(surfaces[0])?.cross(normal(surfaces[1])?);
    if !direction.magnitude2().is_finite() || direction.magnitude2() <= 0.0 {
        return Err("intersection_chart_tangent");
    }
    // The pre-cut certificate describes curve orientation relative to these
    // original carriers, before either face's boundary orientation is applied.
    // A regular physical intersection need not have nonzero parameter speed at
    // a bounded guide endpoint; its transverse carrier tangent is authoritative.
    match orientation {
        ShapesOpStatus::And => Ok(direction),
        ShapesOpStatus::Or => Ok(-direction),
        ShapesOpStatus::Unknown => Err("intersection_chart_orientation"),
    }
}

fn chart_corner<C, S>(
    loops: &Loops<Point3, C>,
    surface: &S,
    vertex: &Vertex<Point3>,
    departing: Vector3,
) -> std::result::Result<(usize, usize), &'static str>
where
    C: ParametricCurve3D + BoundedCurve,
    S: ParametricSurface3D + SearchParameter<SurfaceParameter, Point = Point3>,
{
    let (u, v) = surface
        .search_parameter(vertex.point(), None, 100)
        .ok_or("intersection_chart_parameter")?;
    let du = surface.derivative_u(u, v);
    let dv = surface.derivative_v(u, v);
    let normal = du.cross(dv);
    let denominator = normal.magnitude2();
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err("intersection_chart_singular");
    }
    let project = |direction: Vector3| {
        Vector2::new(
            direction.cross(dv).dot(normal),
            du.cross(direction).dot(normal),
        ) / denominator
    };
    let departing = project(departing);
    if !departing.magnitude2().is_finite() || departing.magnitude2() <= 0.0 {
        return Err("intersection_chart_tangent");
    }
    let angle = |from: Vector2, to: Vector2| {
        (from.x * to.y - from.y * to.x)
            .atan2(from.dot(to))
            .rem_euclid(std::f64::consts::TAU)
    };
    let mut selected = None;
    for (wire_index, wire) in loops.iter().enumerate() {
        for (index, outgoing) in wire.iter().enumerate() {
            if outgoing.front() != vertex {
                continue;
            }
            let incoming = &wire[(index + wire.len() - 1) % wire.len()];
            let incoming = project(tangent(incoming, false));
            let outgoing = project(tangent(outgoing, true));
            if !incoming.magnitude2().is_finite()
                || incoming.magnitude2() <= 0.0
                || !outgoing.magnitude2().is_finite()
                || outgoing.magnitude2() <= 0.0
            {
                return Err("intersection_chart_boundary_tangent");
            }
            // Absolute face boundaries have material on their left. The two
            // seam corners share a vertex, but have opposite interior wedges.
            let width = angle(outgoing, -incoming);
            let offset = angle(outgoing, departing);
            if offset > 0.0 && offset < width && selected.replace((wire_index, index)).is_some() {
                return Err("intersection_chart_ambiguous_corner");
            }
        }
    }
    selected.ok_or("intersection_chart_missing_corner")
}

pub(super) fn endpoint_corners<C, S>(
    loops: &Loops<Point3, C>,
    surface: &S,
    vertices: [&Vertex<Point3>; 2],
    departing: [Vector3; 2],
) -> std::result::Result<[Option<(usize, usize)>; 2], &'static str>
where
    C: ParametricCurve3D + BoundedCurve,
    S: ParametricSurface3D + SearchParameter<SurfaceParameter, Point = Point3>,
{
    let select = |vertex: &Vertex<Point3>, direction: Vector3| {
        let mut candidates = loops.iter().enumerate().flat_map(|(wire_index, wire)| {
            wire.iter().enumerate().filter_map(move |(index, edge)| {
                (edge.front() == vertex).then_some((wire_index, index))
            })
        });
        let first = candidates.next();
        if candidates.next().is_none() {
            // No attachment, or the sole topological occurrence: retain the
            // ordinary incremental intersection-assembly case.
            Ok(first)
        } else {
            chart_corner(loops, surface, vertex, direction).map(Some)
        }
    };
    Ok([
        select(vertices[0], departing[0])?,
        select(vertices[1], departing[1])?,
    ])
}

fn splice_wire<P: Copy, C: Clone>(
    loops: &mut Loops<P, C>,
    mut path: Wire<P, C>,
    status: ShapesOpStatus,
    start: (usize, usize),
    end: (usize, usize),
) -> std::result::Result<(), &'static str> {
    let (start_wire, start_index) = start;
    let (end_wire, end_index) = end;
    if path.is_empty() || start == end {
        return Err("intersection_chart_nonseparating_attachment");
    }
    if start_wire == end_wire {
        let boundary = &mut loops[start_wire];
        let length = boundary.len();
        boundary.rotate_left(end_index);
        let offset = (start_index + length - end_index) % length;
        let mut second = boundary.split_off(offset);
        let mut inverse = path.inverse();
        inverse.append(&mut second);
        path.append(boundary);
        loops[start_wire] = BoundaryWire::new(path, status);
        loops.push(BoundaryWire::new(inverse, status.not()));
    } else {
        let mut start_boundary = loops[start_wire].clone();
        let mut end_boundary = loops[end_wire].clone();
        start_boundary.rotate_left(start_index);
        end_boundary.rotate_left(end_index);
        let inherited = start_boundary.status();
        let mut inverse = path.inverse();
        path.append(&mut end_boundary);
        path.append(&mut inverse);
        path.append(&mut start_boundary);
        loops[start_wire] = BoundaryWire::new(path, inherited);
        loops.swap_remove(end_wire);
    }
    Ok(())
}

// Closest points of finite 3D segments. A candidate is only a mesh seed;
// admission below still requires an actual common point on native carriers.
fn segment_contact(
    first: [Point3; 2],
    second: [Point3; 2],
    distance: f64,
) -> Option<(f64, Point3)> {
    for axis in 0..3 {
        if first[0][axis].min(first[1][axis]) > second[0][axis].max(second[1][axis]) + distance
            || second[0][axis].min(second[1][axis]) > first[0][axis].max(first[1][axis]) + distance
        {
            return None;
        }
    }
    let u = first[1] - first[0];
    let v = second[1] - second[0];
    let w = first[0] - second[0];
    let a = u.magnitude2();
    let b = u.dot(v);
    let c = v.magnitude2();
    let d = u.dot(w);
    let e = v.dot(w);
    if a <= 0.0 || c <= 0.0 || !a.is_finite() || !c.is_finite() {
        return None;
    }
    let determinant = a * c - b * b;
    let mut s = if determinant > 0.0 {
        ((b * e - c * d) / determinant).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut t = (b * s + e) / c;
    if t < 0.0 {
        t = 0.0;
        s = (-d / a).clamp(0.0, 1.0);
    } else if t > 1.0 {
        t = 1.0;
        s = ((b - d) / a).clamp(0.0, 1.0);
    }
    let first_point = first[0] + s * u;
    let second_point = second[0] + t * v;
    ((first_point - second_point).magnitude2() <= distance * distance).then_some((s, second_point))
}

fn polygon_head_parameter(parameter: f64, previous_cut: f64) -> f64 {
    let segment = previous_cut.floor();
    if parameter > segment && previous_cut != segment {
        segment + (parameter - segment) / (previous_cut - segment)
    } else {
        parameter
    }
}

fn cut_polygon(head: &mut PolylineCurve, parameter: f64) -> PolylineCurve {
    let index = parameter.floor() as usize;
    let point = head.evaluate(parameter);
    let mut tail = Vec::with_capacity(head.len() - index);
    tail.push(point);
    tail.extend_from_slice(&head[index + 1..]);
    head.truncate(index + 1);
    // A mesh parameter tolerance is not a physical endpoint tolerance.
    if parameter != index as f64 {
        head.push(point);
    }
    PolylineCurve(tail)
}

struct LoopPoint<C> {
    polygon_parameter: f64,
    native_parameter: f64,
    polygon: Vertex<Point3>,
    native: Vertex<Point3>,
    on_boundary: [bool; 2],
    // Only the immediately departing part carries this whole-original receipt.
    qualified_departure: bool,
    // Each parameter belongs to that exact immutable edge's native curve.
    // All contacts are collected before any of these edges are subdivided.
    roots: [Option<(EdgeId<C>, f64)>; 2],
    direction: Option<Vector3>,
}

fn contact_index<C>(
    contacts: &[LoopPoint<C>],
    side: usize,
    root: (EdgeId<C>, f64),
    native_parameter: f64,
    shared_vertex: Option<&Vertex<Point3>>,
    point: Point3,
    polygon_point: Point3,
) -> std::result::Result<Option<usize>, &'static str> {
    let existing = contacts.iter().position(|previous| {
        previous.roots[side] == Some(root)
            || (previous.native_parameter == native_parameter
                && previous.native.point() == point
                && previous.polygon.point() == polygon_point)
            || shared_vertex == Some(&previous.native)
    });
    if let Some(index) = existing {
        let previous = &contacts[index];
        if !previous.native.point().near(&point)
            || !previous.polygon.point().near(&polygon_point)
            || (previous.roots[side].is_some_and(|previous| previous != root)
                && shared_vertex != Some(&previous.native))
        {
            return Err("intersection_chart_ambiguous_contact");
        }
    } else if contacts
        .iter()
        .any(|previous| previous.native.point().near(&point))
    {
        // Spatial nearness does not identify a root on an original seam.
        return Err("intersection_chart_ambiguous_contact");
    }
    Ok(existing)
}

#[derive(Debug)]
pub(super) enum AttachmentFailure {
    Geometry(&'static str),
    Work(ShapeOpsError),
}

impl From<&'static str> for AttachmentFailure {
    fn from(detail: &'static str) -> Self { Self::Geometry(detail) }
}

impl From<ShapeOpsError> for AttachmentFailure {
    fn from(error: ShapeOpsError) -> Self { Self::Work(error) }
}

type NativeIntersection<S> = (PolylineCurve, IntersectionCurve<PolylineCurve, S, S>);

fn same_path<'a, C: 'a>(
    wire: &Wire<Point3, C>,
    mut path: impl ExactSizeIterator<Item = &'a Edge<Point3, C>> + DoubleEndedIterator,
    start: usize,
    reverse: bool,
    work: Option<&BooleanWork<'_>>,
) -> std::result::Result<bool, AttachmentFailure> {
    for offset in 0..path.len() {
        if let Some(work) = work {
            work.check()?;
        }
        let actual = &wire[(start + offset) % wire.len()];
        let expected = if reverse {
            path.next_back()
        } else {
            path.next()
        }
        .ok_or("intersection_chart_empty_path")?;
        let (front, back, orientation) = if reverse {
            (expected.back(), expected.front(), !expected.orientation())
        } else {
            (expected.front(), expected.back(), expected.orientation())
        };
        if actual.id() != expected.id()
            || actual.orientation() != orientation
            || actual.front() != front
            || actual.back() != back
        {
            return Ok(false);
        }
    }
    Ok(true)
}

/// A proved original overlap can already be an owned material boundary, or an
/// installed forward/inverse slit awaiting closure. Neither is a new corner.
/// Only actual directed native occurrences and their polygon replay bind here.
fn bind_existing_path<C>(
    native: &mut Loops<Point3, C>,
    polygon: &mut Loops<Point3, PolylineCurve>,
    path: &Wire<Point3, C>,
    polygon_path: &Wire<Point3, PolylineCurve>,
    status: ShapesOpStatus,
    work: Option<&BooleanWork<'_>>,
) -> std::result::Result<bool, AttachmentFailure> {
    bind_existing_iter(
        native,
        polygon,
        path.iter(),
        polygon_path.iter(),
        status,
        work,
    )
}

pub(super) fn bind_existing_edge<C>(
    native: &mut Loops<Point3, C>,
    polygon: &mut Loops<Point3, PolylineCurve>,
    edge: &Edge<Point3, C>,
    polygon_edge: &Edge<Point3, PolylineCurve>,
    status: ShapesOpStatus,
    work: Option<&BooleanWork<'_>>,
) -> std::result::Result<bool, AttachmentFailure> {
    bind_existing_iter(
        native,
        polygon,
        std::iter::once(edge),
        std::iter::once(polygon_edge),
        status,
        work,
    )
}

fn bind_existing_iter<'a, C: 'a>(
    native: &mut Loops<Point3, C>,
    polygon: &mut Loops<Point3, PolylineCurve>,
    path: impl ExactSizeIterator<Item = &'a Edge<Point3, C>> + DoubleEndedIterator + Clone,
    polygon_path: impl ExactSizeIterator<Item = &'a Edge<Point3, PolylineCurve>>
    + DoubleEndedIterator
    + Clone,
    status: ShapesOpStatus,
    work: Option<&BooleanWork<'_>>,
) -> std::result::Result<bool, AttachmentFailure> {
    if native.len() != polygon.len() || path.len() == 0 || path.len() != polygon_path.len() {
        return Err("intersection_original_overlap_polygon_occurrence".into());
    }
    let mut total = [0; 2];
    for (wire, polygon_wire) in native.iter_mut().zip(polygon.iter_mut()) {
        if wire.len() != polygon_wire.len() || wire.status != polygon_wire.status {
            return Err("intersection_original_overlap_polygon_occurrence".into());
        }
        if path.len() > wire.len() {
            continue;
        }
        let mut count = [0; 2];
        for start in 0..wire.len() {
            for (direction, reverse) in [false, true].into_iter().enumerate() {
                let matches = same_path(wire, path.clone(), start, reverse, work)?;
                if matches != same_path(polygon_wire, polygon_path.clone(), start, reverse, work)? {
                    return Err("intersection_original_overlap_polygon_occurrence".into());
                }
                if matches {
                    count[direction] += 1;
                    total[direction] += 1;
                    if total[direction] > 1 {
                        return Err("intersection_original_overlap_ambiguous_occurrence".into());
                    }
                }
            }
        }
        let expected = match count {
            [1, 0] => status,
            [0, 1] => status.not(),
            // A slit has both uses, not one whole-region material decision.
            [0, 0] | [1, 1] => continue,
            _ => return Err("intersection_original_overlap_ambiguous_occurrence".into()),
        };
        if wire.status != ShapesOpStatus::Unknown && wire.status != expected {
            return Err("intersection_original_overlap_material_conflict".into());
        }
        wire.status = expected;
        polygon_wire.status = expected;
    }
    Ok(total != [0; 2])
}

fn record_inserted_overlaps<C>(
    path: &Wire<Point3, C>,
    points: &[LoopPoint<C>],
    start: usize,
    (input, face_index): (usize, usize),
    record: &mut Option<&mut RewriteObserver<'_, Point3, C>>,
) -> std::result::Result<(), AttachmentFailure> {
    let Some(record) = record.as_mut() else {
        return Ok(());
    };
    for (offset, edge) in path.iter().enumerate() {
        if points[(start + offset) % points.len()].qualified_departure
            && !(**record)(
                edge.id(),
                edge,
                EdgeRewrite::GeneratedUse { input, face_index },
            )
        {
            return Err("edge_lineage_work".into());
        }
    }
    Ok(())
}

pub(super) struct IntersectionStores<'a, C, S> {
    pub(super) native: [&'a mut LoopsStore<Point3, C>; 2],
    pub(super) polygon: [&'a mut LoopsStore<Point3, PolylineCurve>; 2],
    pub(super) original_boundaries: [&'a [Wire<Point3, C>]; 2],
    pub(super) sides: [(usize, &'a S, ShapesOpStatus); 2],
}

pub(super) fn install_owned_intersection<C, S>(
    binding: native_overlap::OwnedEndpointBinding<C>,
    stores: IntersectionStores<'_, C, S>,
    orientation: ShapesOpStatus,
    work: Option<&BooleanWork<'_>>,
    record: &mut Option<&mut RewriteObserver<'_, Point3, C>>,
) -> std::result::Result<(), AttachmentFailure>
where
    C: Clone + ParametricCurve3D + BoundedCurve,
    S: ParametricSurface3D + SearchParameter<SurfaceParameter, Point = Point3>,
{
    // Only the whole-original-curve proof constructs this receipt. Preserve
    // its actual directed topology before isolated point proposals can replace
    // an already-owned shared endpoint with a new projected vertex.
    install_original_path(
        stores,
        (
            [binding.native_edge].into_iter().collect(),
            [binding.polygon_edge].into_iter().collect(),
        ),
        false,
        orientation,
        work,
        record,
    )
}

fn install_original_path<C, S>(
    stores: IntersectionStores<'_, C, S>,
    (native_path, polygon_path): (Wire<Point3, C>, Wire<Point3, PolylineCurve>),
    closed: bool,
    orientation: ShapesOpStatus,
    work: Option<&BooleanWork<'_>>,
    record: &mut Option<&mut RewriteObserver<'_, Point3, C>>,
) -> std::result::Result<(), AttachmentFailure>
where
    C: Clone + ParametricCurve3D + BoundedCurve,
    S: ParametricSurface3D + SearchParameter<SurfaceParameter, Point = Point3>,
{
    let IntersectionStores {
        native,
        polygon,
        sides,
        ..
    } = stores;
    let surfaces = [sides[0].1, sides[1].1];
    for (side, &(face, surface, status)) in sides.iter().enumerate() {
        if bind_existing_path(
            &mut native[side][face],
            &mut polygon[side][face],
            &native_path,
            &polygon_path,
            status,
            work,
        )? {
            continue;
        }
        let corners = if closed {
            [None; 2]
        } else {
            let front = native_path
                .front_vertex()
                .ok_or("intersection_chart_empty_path")?;
            let back = native_path
                .back_vertex()
                .ok_or("intersection_chart_empty_path")?;
            let departing = [
                -intersection_direction(surfaces, back.point(), orientation)?,
                intersection_direction(surfaces, front.point(), orientation)?,
            ];
            let corners = endpoint_corners(&native[side][face], surface, [back, front], departing)?;
            if corners[0].is_some() && corners[0] == corners[1] {
                return Err("intersection_chart_nonseparating_attachment".into());
            }
            corners
        };
        for edge in &native_path {
            if let Some(record) = record.as_mut()
                && !(**record)(
                    edge.id(),
                    edge,
                    EdgeRewrite::GeneratedUse {
                        input: side,
                        face_index: face,
                    },
                )
            {
                return Err("edge_lineage_work".into());
            }
        }
        if closed {
            native[side][face].add_independent_loop(BoundaryWire::new(native_path.clone(), status));
            polygon[side][face]
                .add_independent_loop(BoundaryWire::new(polygon_path.clone(), status));
        } else {
            native[side][face].add_path(native_path.clone(), status, corners);
            polygon[side][face].add_path(polygon_path.clone(), status, corners);
        }
    }
    Ok(())
}

pub(super) fn add_intersection<C, S>(
    (mut polyline, mut curve): NativeIntersection<S>,
    endpoints: Option<[(Vertex<Point3>, Vertex<Point3>); 2]>,
    stores: IntersectionStores<'_, C, S>,
    orientation: ShapesOpStatus,
    tolerance: f64,
    work: Option<&BooleanWork<'_>>,
    record: &mut Option<&mut RewriteObserver<'_, Point3, C>>,
) -> std::result::Result<(), AttachmentFailure>
where
    C: SearchNearestParameter<CurveParameter, Point = Point3>
        + SearchParameter<CurveParameter, Point = Point3>
        + Cut<Point = Point3, Vector = Vector3>
        + From<IntersectionCurve<PolylineCurve, S, S>>
        + TryIntoHomogeneousBsplineCurve,
    S: ParametricSurface3D
        + Clone
        + SearchParameter<SurfaceParameter, Point = Point3>
        + SearchNearestParameter<SurfaceParameter, Point = Point3>
        + TryIntoAnalyticSurfaceKind
        + TryIntoHomogeneousBsplineSurface,
{
    let IntersectionStores {
        native,
        polygon,
        original_boundaries,
        sides,
    } = stores;
    let [(face0, surface0, status0), (face1, surface1, status1)] = sides;
    let faces = [face0, face1];
    let surfaces = [surface0, surface1];
    let statuses = [status0, status1];
    let chart_edges = [
        paired_chart_edges(&native[0][faces[0]]),
        paired_chart_edges(&native[1][faces[1]]),
    ];
    let closed = endpoints.is_none();
    if closed {
        let mut original = None;
        for side in 0..2 {
            if let Some(candidate) = native_overlap::original_loop(
                &native[side][faces[side]],
                &polygon[side][faces[side]],
                surfaces[side],
                (surfaces[1 - side], original_boundaries[1 - side]),
                &curve,
                &chart_edges[side],
                work,
            )? {
                if original.is_some() {
                    return Err("intersection_original_chart_boundary_ambiguous".into());
                }
                original = Some(candidate);
            }
        }
        if let Some(path) = original {
            return install_original_path(
                IntersectionStores {
                    native,
                    polygon,
                    original_boundaries,
                    sides,
                },
                path,
                true,
                orientation,
                work,
                record,
            );
        }
    }
    if let Some(endpoints) = &endpoints {
        // A whole qualified original boundary owns this component before
        // isolated seam proposals are considered. Existing temporary slits on
        // the other face must not turn unrelated contained-arc midpoints into
        // contacts on this bounded original branch.
        if let Some((edge, polygon_edge, _)) = native_overlap::original_overlap(
            &curve,
            [
                (&native[0][faces[0]], &polygon[0][faces[0]]),
                (&native[1][faces[1]], &polygon[1][faces[1]]),
            ],
            original_boundaries,
            [&endpoints[0].1, &endpoints[1].1],
            [&endpoints[0].0, &endpoints[1].0],
            work,
        )? {
            return install_original_path(
                IntersectionStores {
                    native,
                    polygon,
                    original_boundaries,
                    sides,
                },
                (
                    std::iter::once(edge).collect(),
                    std::iter::once(polygon_edge).collect(),
                ),
                false,
                orientation,
                work,
                record,
            );
        }
    }
    let mut contacts: Vec<LoopPoint<C>> = endpoints
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(index, (polygon_vertex, native_vertex))| LoopPoint {
            polygon_parameter: if index == 0 {
                0.0
            } else {
                polyline.range_tuple().1
            },
            native_parameter: if index == 0 {
                0.0
            } else {
                curve.range_tuple().1
            },
            on_boundary: std::array::from_fn(|side| {
                native[side][faces[side]]
                    .iter()
                    .any(|wire| wire.iter().any(|edge| edge.front() == &native_vertex))
            }),
            polygon: polygon_vertex,
            native: native_vertex,
            roots: [None; 2],
            direction: None,
            qualified_departure: false,
        })
        .collect();
    for side in 0..2 {
        for &(wire, edge) in &chart_edges[side] {
            polygon[side][faces[side]][wire][edge].with_curve(|boundary| {
                for (index, segment) in polyline.windows(2).enumerate() {
                    for seam in boundary.windows(2) {
                        // Both meshes approximate native geometry at tolerance.
                        // This broad phase is not a native residual allowance.
                        let Some((fraction, _)) = segment_contact(
                            [segment[0], segment[1]],
                            [seam[0], seam[1]],
                            2.0 * tolerance,
                        ) else {
                            continue;
                        };
                        // Repeated proposals from one original seam segment
                        // solve from the same seed, not from leader proximity.
                        let seed = seam[0].midpoint(seam[1]);
                        let (point, mut boundary_parameter, _) =
                            native[side][faces[side]][wire][edge].with_curve(|boundary| {
                                curve_surface_projection(
                                    boundary,
                                    None,
                                    surfaces[1 - side],
                                    None,
                                    seed,
                                    100,
                                )
                            })?;
                        if let Some(parameter) = native_overlap::canonical_plane_endpoint(
                            &native[side][faces[side]][wire][edge],
                            surfaces[1 - side],
                            boundary_parameter,
                            point,
                            work,
                        )? {
                            boundary_parameter = parameter;
                        }
                        let polygon_point = boundary.evaluate(
                            boundary
                                .search_nearest_parameter(point, None, 100)
                                .ok_or("intersection_chart_polygon_parameter")?,
                        );
                        let mut polygon_parameter = index as f64 + fraction;
                        let mut native_parameter = intersection_parameter(
                            &curve, point, IntersectionParameterBinding::BoundCurve, work, None,
                        )?;
                        let boundary = &native[side][faces[side]][wire][edge];
                        let (boundary_start, boundary_end) =
                            boundary.with_curve(BoundedCurve::range_tuple);
                        let shared_vertex = if boundary_parameter == boundary_start {
                            Some(boundary.absolute_front())
                        } else if boundary_parameter == boundary_end {
                            Some(boundary.absolute_back())
                        } else {
                            None
                        };
                        let polygon_boundary = &polygon[side][faces[side]][wire][edge];
                        let shared_polygon = if boundary_parameter == boundary_start {
                            Some(polygon_boundary.absolute_front())
                        } else if boundary_parameter == boundary_end {
                            Some(polygon_boundary.absolute_back())
                        } else {
                            None
                        };
                        if point == curve.front() {
                            native_parameter = 0.0;
                            polygon_parameter = 0.0;
                        }
                        let root = (boundary.id(), boundary_parameter);
                        if let Some(index) = contact_index(
                            &contacts,
                            side,
                            root,
                            native_parameter,
                            shared_vertex,
                            point,
                            polygon_point,
                        )? {
                            let previous = &mut contacts[index];
                            previous.roots[side] = Some(root);
                            previous.on_boundary[side] = true;
                        } else {
                            let mut on_boundary = [false; 2];
                            on_boundary[side] = true;
                            let mut roots = [None; 2];
                            roots[side] = Some(root);
                            contacts.push(LoopPoint {
                                polygon_parameter,
                                native_parameter,
                                polygon: shared_polygon
                                    .cloned()
                                    .unwrap_or_else(|| Vertex::new(polygon_point)),
                                native: shared_vertex
                                    .cloned()
                                    .unwrap_or_else(|| Vertex::new(point)),
                                on_boundary,
                                roots,
                                direction: None,
                                qualified_departure: false,
                            });
                        }
                    }
                }
                Ok::<(), AttachmentFailure>(())
            })?;
        }
    }
    if contacts.is_empty() {
        let polygon_wire = create_independent_loop(polyline);
        let native_wire = create_independent_loop(curve);
        for side in 0..2 {
            polygon[side][faces[side]]
                .add_independent_loop(BoundaryWire::new(polygon_wire.clone(), statuses[side]));
            native[side][faces[side]]
                .add_independent_loop(BoundaryWire::new(native_wire.clone(), statuses[side]));
        }
        return Ok(());
    }
    contacts.sort_by(|a, b| a.polygon_parameter.total_cmp(&b.polygon_parameter));
    let mut points = contacts;
    if points[0].polygon_parameter != 0.0 {
        points.insert(
            0,
            LoopPoint {
                polygon_parameter: 0.0,
                native_parameter: 0.0,
                polygon: Vertex::new(polyline.front()),
                native: Vertex::new(curve.front()),
                on_boundary: [false; 2],
                roots: [None; 2],
                direction: None,
                qualified_departure: false,
            },
        );
    }
    if points.len() == 1 {
        let (start, end) = polyline.range_tuple();
        let parameter = (start + end) / 2.0;
        points.push(LoopPoint {
            polygon_parameter: parameter,
            native_parameter: parameter,
            polygon: Vertex::new(polyline.evaluate(parameter)),
            native: Vertex::new(curve.evaluate(parameter)),
            on_boundary: [false; 2],
            roots: [None; 2],
            direction: None,
            qualified_departure: false,
        });
    }
    if points.windows(2).any(|pair| {
        pair[0].polygon_parameter >= pair[1].polygon_parameter
            || pair[0].native_parameter >= pair[1].native_parameter
    }) {
        return Err("intersection_chart_contact_order".into());
    }
    let mut polygon_maps = [HashMap::default(), HashMap::default()];
    let mut native_maps = [HashMap::default(), HashMap::default()];
    let mut ancestry: [Vec<(EdgeId<C>, EdgeId<C>)>; 2] = Default::default();
    for point in &mut points {
        for side in 0..2 {
            if !point.on_boundary[side]
                || native[side][faces[side]]
                    .iter()
                    .any(|wire| wire.iter().any(|edge| edge.front() == &point.native))
            {
                continue;
            }
            let BoundaryAttachment {
                wire_index,
                edge_index,
                native_kind,
                polygon_kind,
            } = if let Some(root) = point.roots[side] {
                rooted_boundary_attachment(
                    (native[side], polygon[side]),
                    faces[side],
                    root,
                    &ancestry[side],
                    [&point.polygon, &point.native],
                    [surfaces[side], surfaces[1 - side]],
                    work,
                )?
            } else {
                boundary_attachment(
                    native[side],
                    polygon[side],
                    faces[side],
                    &point.polygon,
                    &point.native,
                    [surfaces[side], surfaces[1 - side]],
                )?
                .ok_or("intersection_chart_native_contact")?
            };
            polygon[side]
                .add_polygon_vertex_at(
                    faces[side],
                    &point.polygon,
                    (wire_index, edge_index, polygon_kind),
                    &mut polygon_maps[side],
                )
                .ok_or("intersection_chart_polygon_contact")?;
            let other_capacity = ancestry[1 - side].capacity();
            let ancestry = &mut ancestry[side];
            let mut ancestry_failure = None;
            let result = {
                let mut observe = |parent, child: &Edge<Point3, C>, rewrite| {
                    let root = ancestry
                        .iter()
                        .rev()
                        .find_map(|&(previous, root)| (previous == parent).then_some(root))
                        .unwrap_or(parent);
                    if ancestry.len() == ancestry.capacity() {
                        let capacity = ancestry.capacity().saturating_mul(2).max(4);
                        if let Some(work) = work
                            && let Err(error) = work.limit(
                                "intermediate_prepared_bytes",
                                capacity
                                    .saturating_add(other_capacity)
                                    .saturating_mul(size_of::<(EdgeId<C>, EdgeId<C>)>()),
                                work.limits.max_intermediate_prepared_bytes,
                            )
                        {
                            ancestry_failure = Some(error);
                            return false;
                        }
                        ancestry.reserve_exact(capacity - ancestry.len());
                        if let Some(work) = work
                            && let Err(error) = work.limit(
                                "intermediate_prepared_bytes",
                                ancestry
                                    .capacity()
                                    .saturating_add(other_capacity)
                                    .saturating_mul(size_of::<(EdgeId<C>, EdgeId<C>)>()),
                                work.limits.max_intermediate_prepared_bytes,
                            )
                        {
                            ancestry_failure = Some(error);
                            return false;
                        }
                    }
                    ancestry.push((child.id(), root));
                    record
                        .as_mut()
                        .is_none_or(|record| (**record)(parent, child, rewrite))
                };
                native[side].add_geom_vertex(
                    (faces[side], wire_index, edge_index),
                    &point.native,
                    native_kind,
                    &mut native_maps[side],
                    &mut Some(&mut observe),
                )
            };
            if let Some(error) = ancestry_failure {
                return Err(error.into());
            }
            result?;
        }
        if point.on_boundary.iter().any(|&contact| contact) {
            point.direction = Some(intersection_direction(
                surfaces,
                point.native.point(),
                orientation,
            )?);
        } else {
            for surface in surfaces {
                let (u, v) = surface
                    .search_parameter(point.native.point(), None, 100)
                    .ok_or("intersection_chart_contact_carrier")?;
                if !surface.evaluate(u, v).near(&point.native.point()) {
                    return Err("intersection_chart_contact_carrier".into());
                }
            }
        }
        if point.native_parameter == 0.0 {
            let last = curve.leader().len() - 1;
            curve.leader_mut()[0] = point.native.point();
            if closed {
                curve.leader_mut()[last] = point.native.point();
            }
        }
    }
    // Fractional cuts rescale the head's last segment. Replay raw parameters
    // in that shortened segment and locate native cuts by the certified point.
    let mut polygon_parts = Vec::with_capacity(points.len());
    let mut native_parts = Vec::with_capacity(points.len());
    let mut previous_polygon_cut = polyline.range_tuple().1;
    let cut_end = points.len() - usize::from(!closed);
    for point in points.iter().take(cut_end).skip(1).rev() {
        let parameter = polygon_head_parameter(point.polygon_parameter, previous_polygon_cut);
        polygon_parts.push(cut_polygon(&mut polyline, parameter));
        previous_polygon_cut = point.polygon_parameter;
        let parameter = intersection_parameter(
            &curve, point.native.point(), IntersectionParameterBinding::BoundCurve, work, None,
        )?;
        native_parts.push(curve.cut(parameter));
    }
    polygon_parts.push(polyline);
    native_parts.push(curve);
    polygon_parts.reverse();
    native_parts.reverse();
    let mut polygon_wire = Wire::new();
    let mut native_wire = Wire::new();
    for (index, (mut polygon_curve, native_curve)) in
        polygon_parts.into_iter().zip(native_parts).enumerate()
    {
        let next = (index + 1) % points.len();
        if !native_curve.front().near(&points[index].native.point())
            || !native_curve.back().near(&points[next].native.point())
        {
            return Err("intersection_chart_native_endpoint".into());
        }
        polygon_curve.snap_endpoints(points[index].polygon.point(), points[next].polygon.point());
        let overlap = native_overlap::original_overlap(
            &native_curve,
            [
                (&native[0][faces[0]], &polygon[0][faces[0]]),
                (&native[1][faces[1]], &polygon[1][faces[1]]),
            ],
            original_boundaries,
            [&points[index].native, &points[next].native],
            [&points[index].polygon, &points[next].polygon],
            work,
        )?;
        let (native_edge, polygon_edge) = match overlap {
            Some((native_edge, polygon_edge, present)) => {
                points[index].qualified_departure = true;
                for (side, present) in present.into_iter().enumerate() {
                    if present {
                        points[index].on_boundary[side] = true;
                        points[next].on_boundary[side] = true;
                    }
                }
                (native_edge, polygon_edge)
            }
            None => (
                Edge::new(
                    &points[index].native,
                    &points[next].native,
                    native_curve.into(),
                ),
                Edge::new(&points[index].polygon, &points[next].polygon, polygon_curve),
            ),
        };
        native_wire.push_back(native_edge);
        polygon_wire.push_back(polygon_edge);
    }
    for side in 0..2 {
        let contacts: Vec<_> = points
            .iter()
            .enumerate()
            .filter_map(|(index, point)| {
                (point.on_boundary[side] || (!closed && (index == 0 || index + 1 == points.len())))
                    .then_some(index)
            })
            .collect();
        if contacts.is_empty() {
            polygon[side][faces[side]]
                .add_independent_loop(BoundaryWire::new(polygon_wire.clone(), statuses[side]));
            native[side][faces[side]]
                .add_independent_loop(BoundaryWire::new(native_wire.clone(), statuses[side]));
            continue;
        }
        let spans = contacts.len() - usize::from(!closed);
        for (index, &start) in contacts.iter().take(spans).enumerate() {
            let end = contacts[(index + 1) % contacts.len()];
            let length = if start == end {
                points.len()
            } else {
                (end + points.len() - start) % points.len()
            };
            let native_path: Wire<_, _> = native_wire
                .iter()
                .cycle()
                .skip(start)
                .take(length)
                .cloned()
                .collect();
            let polygon_path: Wire<_, _> = polygon_wire
                .iter()
                .cycle()
                .skip(start)
                .take(length)
                .cloned()
                .collect();
            if (0..length).all(|offset| points[(start + offset) % points.len()].qualified_departure)
                && bind_existing_path(
                    &mut native[side][faces[side]],
                    &mut polygon[side][faces[side]],
                    &native_path,
                    &polygon_path,
                    statuses[side],
                    work,
                )?
            {
                continue;
            }
            let departing = [
                -points[end]
                    .direction
                    .ok_or("intersection_chart_orientation")?,
                points[start]
                    .direction
                    .ok_or("intersection_chart_orientation")?,
            ];
            let vertices = [
                native_path
                    .back_vertex()
                    .ok_or("intersection_chart_empty_path")?,
                native_path
                    .front_vertex()
                    .ok_or("intersection_chart_empty_path")?,
            ];
            if closed {
                let end_corner = chart_corner(
                    &native[side][faces[side]],
                    surfaces[side],
                    vertices[0],
                    departing[0],
                )?;
                let start_corner = chart_corner(
                    &native[side][faces[side]],
                    surfaces[side],
                    vertices[1],
                    departing[1],
                )?;
                record_inserted_overlaps(
                    &native_path,
                    &points,
                    start,
                    (side, faces[side]),
                    record,
                )?;
                splice_wire(
                    &mut native[side][faces[side]],
                    native_path,
                    statuses[side],
                    start_corner,
                    end_corner,
                )?;
                splice_wire(
                    &mut polygon[side][faces[side]],
                    polygon_path,
                    statuses[side],
                    start_corner,
                    end_corner,
                )?;
            } else {
                let corners = endpoint_corners(
                    &native[side][faces[side]],
                    surfaces[side],
                    vertices,
                    departing,
                )?;
                if corners[0].is_some() && corners[0] == corners[1] {
                    return Err("intersection_chart_nonseparating_attachment".into());
                }
                record_inserted_overlaps(
                    &native_path,
                    &points,
                    start,
                    (side, faces[side]),
                    record,
                )?;
                native[side][faces[side]].add_path(native_path, statuses[side], corners);
                polygon[side][faces[side]].add_path(polygon_path, statuses[side], corners);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use monstertruck_modeling as modeling;

    fn overlap_square() -> (modeling::Wire, Wire<Point3, PolylineCurve>) {
        let points = [
            Point3::origin(),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let native = Vertex::from_points(points);
        let polygon = Vertex::from_points(points);
        (
            (0..4)
                .map(|index| modeling::builder::line(&native[index], &native[(index + 1) % 4]))
                .collect(),
            (0..4)
                .map(|index| {
                    Edge::new(
                        &polygon[index],
                        &polygon[(index + 1) % 4],
                        PolylineCurve(vec![points[index], points[(index + 1) % 4]]),
                    )
                })
                .collect(),
        )
    }

    #[test]
    fn qualified_boundary_uses_bind_direction_and_material_without_a_new_corner() {
        let (wire, polygon_wire) = overlap_square();
        let path = std::iter::once(wire[0].clone()).collect();
        let polygon_path = std::iter::once(polygon_wire[0].clone()).collect();
        let mut native =
            std::iter::once(BoundaryWire::new(wire, ShapesOpStatus::Unknown)).collect();
        let mut polygon =
            std::iter::once(BoundaryWire::new(polygon_wire, ShapesOpStatus::Unknown)).collect();
        assert!(
            bind_existing_path(
                &mut native,
                &mut polygon,
                &path,
                &polygon_path,
                ShapesOpStatus::And,
                None
            )
            .unwrap()
        );
        assert_eq!(native[0].status, ShapesOpStatus::And);
        assert_eq!(polygon[0].status, ShapesOpStatus::And);
        assert!(
            bind_existing_path(
                &mut native,
                &mut polygon,
                &path.inverse(),
                &polygon_path.inverse(),
                ShapesOpStatus::Or,
                None
            )
            .unwrap()
        );
        assert!(matches!(
            bind_existing_path(
                &mut native,
                &mut polygon,
                &path,
                &polygon_path,
                ShapesOpStatus::Or,
                None
            ),
            Err(AttachmentFailure::Geometry(
                "intersection_original_overlap_material_conflict"
            ))
        ));
        assert_eq!(native[0].len(), 4);
        assert!(native[0].is_closed() && native[0].is_simple());
    }

    #[test]
    fn qualified_pending_slit_keeps_both_directed_uses_and_unknown_region() {
        let (wire, polygon_wire) = overlap_square();
        let path = std::iter::once(wire[0].clone()).collect();
        let polygon_path = std::iter::once(polygon_wire[0].clone()).collect();
        let mut native = std::iter::once(BoundaryWire::new(
            [wire[0].inverse(), wire[0].clone()].into_iter().collect(),
            ShapesOpStatus::Unknown,
        ))
        .collect();
        let mut polygon = std::iter::once(BoundaryWire::new(
            [polygon_wire[0].inverse(), polygon_wire[0].clone()]
                .into_iter()
                .collect(),
            ShapesOpStatus::Unknown,
        ))
        .collect();
        assert!(
            bind_existing_path(
                &mut native,
                &mut polygon,
                &path,
                &polygon_path,
                ShapesOpStatus::And,
                None
            )
            .unwrap()
        );
        assert_eq!(native[0].status, ShapesOpStatus::Unknown);
        assert_eq!(polygon[0].status, ShapesOpStatus::Unknown);
        assert_eq!(native[0].len(), 2);
        assert_eq!(native[0][0].id(), native[0][1].id());
        assert_ne!(native[0][0].orientation(), native[0][1].orientation());
        assert!(native[0].is_closed());
    }

    #[test]
    fn qualified_boundary_binding_refuses_ambiguous_or_unreplayed_occurrences() {
        let (wire, polygon_wire) = overlap_square();
        let path = std::iter::once(wire[0].clone()).collect();
        let polygon_path = std::iter::once(polygon_wire[0].clone()).collect();
        let mut native = (0..2)
            .map(|_| BoundaryWire::new(wire.clone(), ShapesOpStatus::Unknown))
            .collect();
        let mut polygon = (0..2)
            .map(|_| BoundaryWire::new(polygon_wire.clone(), ShapesOpStatus::Unknown))
            .collect();
        assert!(matches!(
            bind_existing_path(
                &mut native,
                &mut polygon,
                &path,
                &polygon_path,
                ShapesOpStatus::And,
                None
            ),
            Err(AttachmentFailure::Geometry(
                "intersection_original_overlap_ambiguous_occurrence"
            ))
        ));
        let mut native =
            std::iter::once(BoundaryWire::new(wire, ShapesOpStatus::Unknown)).collect();
        let mut polygon =
            std::iter::once(BoundaryWire::new(polygon_wire, ShapesOpStatus::Unknown)).collect();
        let unrelated: modeling::Wire =
            std::iter::once(modeling::builder::line(path[0].front(), path[0].back())).collect();
        assert!(matches!(
            bind_existing_path(
                &mut native,
                &mut polygon,
                &unrelated,
                &polygon_path,
                ShapesOpStatus::And,
                None
            ),
            Err(AttachmentFailure::Geometry(
                "intersection_original_overlap_polygon_occurrence"
            ))
        ));
        let unrelated_polygon = std::iter::once(Edge::new(
            polygon_path[0].front(),
            polygon_path[0].back(),
            polygon_path[0].curve(),
        ))
        .collect();
        assert!(
            !bind_existing_path(
                &mut native,
                &mut polygon,
                &unrelated,
                &unrelated_polygon,
                ShapesOpStatus::And,
                None
            )
            .unwrap()
        );
        assert_eq!(native[0].status, ShapesOpStatus::Unknown);
    }

    #[test]
    fn finite_contacts_find_crossings_between_leader_samples() {
        let first = [Point3::new(-1.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)];
        let second = [Point3::new(0.0, -1.0, 0.0), Point3::new(0.0, 1.0, 0.0)];
        let (parameter, point) = segment_contact(first, second, 0.0).unwrap();
        assert_eq!(parameter, 0.5);
        assert_eq!(point, Point3::origin());
        let beyond = [Point3::new(2.0, -1.0, 0.0), Point3::new(2.0, 1.0, 0.0)];
        assert!(segment_contact(first, beyond, 0.0).is_none());
        let skew = [Point3::new(0.0, -1.0, 1.0), Point3::new(0.0, 1.0, 1.0)];
        assert!(segment_contact(first, skew, 0.01).is_none());
        assert!(segment_contact([first[0]; 2], second, 0.0).is_none());
    }

    #[test]
    fn near_contacts_require_the_same_native_seam_root_not_a_spatial_weld() {
        let vertices = Vertex::from_points([Point3::origin(), Point3::new(1.0, 0.0, 0.0)]);
        let seam: modeling::Edge = modeling::builder::line(&vertices[0], &vertices[1]);
        let point = Point3::new(0.5, 0.0, 0.0);
        let contacts = [LoopPoint {
            polygon_parameter: 1.0,
            native_parameter: 2.0,
            polygon: Vertex::new(point),
            native: Vertex::new(point),
            on_boundary: [true, false],
            roots: [Some((seam.id(), 0.5)), None],
            direction: None,
            qualified_departure: false,
        }];
        assert_eq!(
            contact_index(&contacts, 0, (seam.id(), 0.5), 2.0, None, point, point),
            Ok(Some(0))
        );
        let nearby = point + Vector3::new(1.0e-8, 0.0, 0.0);
        assert!(nearby.near(&point));
        assert_eq!(
            contact_index(
                &contacts,
                0,
                (seam.id(), 0.5 + 1.0e-8),
                2.0 + 1.0e-8,
                None,
                nearby,
                nearby,
            ),
            Err("intersection_chart_ambiguous_contact")
        );
        assert_eq!(
            contact_index(&contacts, 1, (seam.id(), 0.5), 2.0, None, nearby, nearby),
            Err("intersection_chart_ambiguous_contact")
        );
        let unrelated: modeling::Edge = modeling::builder::line(&vertices[0], &vertices[1]);
        assert_eq!(
            contact_index(&contacts, 0, (unrelated.id(), 0.5), 2.0, None, point, point),
            Err("intersection_chart_ambiguous_contact")
        );
    }

    #[test]
    fn repeated_chart_corners_are_selected_by_geometry_not_wire_order() {
        let bottom: modeling::Wire = modeling::primitive::circle(
            Point3::new(1.0, 0.0, 0.0),
            Point3::origin(),
            Vector3::unit_z(),
            2,
        );
        let top = modeling::builder::translated(&bottom, Vector3::new(0.0, 0.0, 3.0));
        let seam =
            modeling::builder::line(bottom.front_vertex().unwrap(), top.front_vertex().unwrap());
        let midpoint = Vertex::new(Point3::new(1.0, 0.0, 1.5));
        let mut wire = top.inverse();
        wire.push_back(seam.inverse());
        wire.extend(bottom.iter().cloned());
        wire.push_back(seam.clone());
        let surface =
            modeling::Surface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
                modeling::Curve::Line(Line(Point3::new(1.0, 0.0, 3.0), Point3::new(1.0, 0.0, 0.0))),
                Point3::origin(),
                Vector3::unit_z(),
            )));
        let mut loops: Loops<_, _> =
            std::iter::once(BoundaryWire::new(wire, ShapesOpStatus::Unknown)).collect();
        let (first, second) = seam.cut_with_parameter(&midpoint, 0.5).unwrap();
        loops.swap_edge_into_pair(seam.id(), &[first, second]);
        let circle: modeling::Wire = modeling::primitive::circle(
            midpoint.point(),
            Point3::new(0.0, 0.0, 1.5),
            Vector3::unit_z(),
            2,
        );
        let arc = Edge::new(&midpoint, circle[0].back(), circle[0].curve());
        for _ in 0..loops[0].len() {
            loops[0].rotate_left(1);
            let start = chart_corner(&loops, &surface, &midpoint, Vector3::unit_y()).unwrap();
            let end = chart_corner(&loops, &surface, &midpoint, -Vector3::unit_y()).unwrap();
            assert_eq!(
                endpoint_corners(
                    &loops,
                    &surface,
                    [arc.back(), arc.front()],
                    [-tangent(&arc, false), tangent(&arc, true)],
                )
                .unwrap(),
                [None, Some(start)]
            );
            assert_eq!(
                endpoint_corners(
                    &loops,
                    &surface,
                    [arc.front(), arc.back()],
                    [tangent(&arc, true), -tangent(&arc, false)],
                )
                .unwrap(),
                [Some(start), None]
            );
            assert_ne!(start, end);
            assert!(!loops[start.0][start.1].orientation());
            assert!(loops[end.0][end.1].orientation());
            assert!(chart_corner(&loops, &surface, &midpoint, Vector3::unit_z()).is_err());
        }
        let unknown = Vertex::new(Point3::new(-1.0, 0.0, 1.5));
        assert_eq!(
            chart_corner(&loops, &surface, &unknown, Vector3::unit_y()),
            Err("intersection_chart_missing_corner")
        );
        let false_identity = Vertex::new(midpoint.point());
        assert!(chart_corner(&loops, &surface, &false_identity, Vector3::unit_y()).is_err());
        let ambiguous: Loops<_, _> = loops.iter().chain(loops.iter()).cloned().collect();
        assert_eq!(
            chart_corner(&ambiguous, &surface, &midpoint, Vector3::unit_y()),
            Err("intersection_chart_ambiguous_corner")
        );
    }

    #[test]
    fn physical_corner_direction_survives_zero_guide_speed_but_refuses_tangent_carriers() {
        let wall =
            modeling::Surface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
                modeling::Curve::Line(Line(Point3::new(1.0, 0.0, 3.0), Point3::new(1.0, 0.0, 0.0))),
                Point3::origin(),
                Vector3::unit_z(),
            )));
        let cap = modeling::Surface::Plane(Plane::new(
            Point3::new(0.0, 0.0, 0.1),
            Point3::new(1.0, 0.0, 0.1),
            Point3::new(0.0, 1.0, 0.1),
        ));
        let point = Point3::new(1.0, 0.0, 0.1);
        for reversed in [false, true] {
            let wall = if reversed {
                wall.inverse()
            } else {
                wall.clone()
            };
            let curve = IntersectionCurve::new(
                wall.clone(),
                cap.clone(),
                PolylineCurve(vec![
                    point,
                    point,
                    Point3::new(0.0, 1.0, 0.1),
                    Point3::new(-1.0, 0.0, 0.1),
                    Point3::new(0.0, -1.0, 0.1),
                    point,
                ]),
            );
            let derivative = curve.derivative(0.0).magnitude2();
            assert!(!derivative.is_finite() || derivative == 0.0);
            let orientation = ShapesOpStatus::from_is_curve(&curve).unwrap();
            let direction = intersection_direction([&wall, &cap], point, orientation).unwrap();
            assert!(direction.normalize().near(&Vector3::unit_y()));
            assert!(
                intersection_direction([&wall, &cap], point, orientation.not(),)
                    .unwrap()
                    .normalize()
                    .near(&-Vector3::unit_y())
            );
            assert_eq!(
                intersection_direction([&wall, &wall], point, orientation,),
                Err("intersection_chart_tangent")
            );
            assert_eq!(
                intersection_direction([&wall, &cap], point, ShapesOpStatus::Unknown,),
                Err("intersection_chart_orientation")
            );
            assert_eq!(
                intersection_direction([&wall, &cap], Point3::new(0.75, 0.0, 0.1), orientation,),
                Err("intersection_chart_contact_carrier")
            );
        }
    }

    #[test]
    fn multiple_fractional_contacts_in_one_leader_segment_conserve_its_geometry() {
        for (first, second) in [(1.75, 1.25), (1.0 + 1.0e-8, 1.0 + 5.0e-9)] {
            let mut head = PolylineCurve(
                (0..4)
                    .map(|index| Point3::new(index as f64, 0.0, 0.0))
                    .collect(),
            );
            let tail = cut_polygon(&mut head, first);
            let parameter = polygon_head_parameter(second, first);
            let middle = cut_polygon(&mut head, parameter);
            assert_eq!(head.front(), Point3::origin());
            assert!(head.back().near(&Point3::new(second, 0.0, 0.0)));
            assert_eq!(head.back(), middle.front());
            assert_eq!(middle.back(), tail.front());
            assert!(middle.back().near(&Point3::new(first, 0.0, 0.0)));
            assert_eq!(tail.back(), Point3::new(3.0, 0.0, 0.0));
            assert!(head.len() >= 2 && middle.len() >= 2 && tail.len() >= 2);
            assert!(head.back().x < middle.back().x);
        }
    }

    #[test]
    fn rooted_attachment_requires_the_actual_current_cut_ancestry() {
        let (wire, polygon_wire) = overlap_square();
        let root = wire[0].id();
        let mut native = LoopsStore(vec![
            std::iter::once(BoundaryWire::new(wire, ShapesOpStatus::Unknown)).collect(),
        ]);
        let mut polygon = LoopsStore(vec![
            std::iter::once(BoundaryWire::new(polygon_wire, ShapesOpStatus::Unknown)).collect(),
        ]);
        let midpoint = Vertex::new(Point3::new(0.5, 0.0, 0.0));
        let polygon_midpoint = Vertex::new(midpoint.point());
        polygon
            .add_polygon_vertex_at(
                0,
                &polygon_midpoint,
                (0, 0, ParameterKind::Inner(0.5)),
                &mut HashMap::default(),
            )
            .unwrap();
        let mut ancestry = Vec::new();
        native
            .add_geom_vertex(
                (0, 0, 0),
                &midpoint,
                ParameterKind::Inner(0.5),
                &mut HashMap::default(),
                &mut Some(&mut |parent, child: &modeling::Edge, _| {
                    ancestry.push((child.id(), parent));
                    true
                }),
            )
            .unwrap();
        let point = Vertex::new(Point3::new(0.25, 0.0, 0.0));
        let polygon_point = Vertex::new(point.point());
        let first = modeling::Surface::Plane(Plane::xy());
        let second = modeling::Surface::Plane(Plane::new(
            Point3::origin(),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ));
        assert!(matches!(
            rooted_boundary_attachment(
                (&native, &polygon),
                0,
                (root, 0.25),
                &[],
                [&polygon_point, &point],
                [&first, &second],
                None,
            ),
            Err(AttachmentFailure::Geometry(
                "intersection_chart_missing_root_descendant"
            ))
        ));
        let attachment = rooted_boundary_attachment(
            (&native, &polygon),
            0,
            (root, 0.25),
            &ancestry,
            [&polygon_point, &point],
            [&first, &second],
            None,
        )
        .unwrap();
        // This child's literal `Line` has its own unit gauge after the cut.
        assert_eq!(attachment.native_kind, ParameterKind::Inner(0.5));
        assert_eq!(
            native[0][attachment.wire_index][attachment.edge_index].id(),
            ancestry[0].0
        );
        assert_eq!(point.point(), Point3::new(0.25, 0.0, 0.0));
    }
}
