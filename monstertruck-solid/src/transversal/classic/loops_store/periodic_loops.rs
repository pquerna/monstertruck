//! Attach closed intersections to their actual paired chart-boundary uses.
//! A closed curve in space need not be a contractible loop in a periodic chart.
//! Native and polygon stores replay the same occurrence-level partition.

use super::*;

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
    edge: &Edge<Point3, C>,
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
        select(edge.back(), -tangent(edge, false))?,
        select(edge.front(), tangent(edge, true))?,
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

struct LoopPoint {
    polygon_parameter: f64,
    native_parameter: f64,
    polygon: Vertex<Point3>,
    native: Vertex<Point3>,
    on_boundary: [bool; 2],
    direction: Option<Vector3>,
}

type ClosedIntersection<S> = (PolylineCurve, IntersectionCurve<PolylineCurve, S, S>);

pub(super) fn add_closed_intersection<C, S>(
    (mut polyline, mut curve): ClosedIntersection<S>,
    native: [&mut LoopsStore<Point3, C>; 2],
    polygon: [&mut LoopsStore<Point3, PolylineCurve>; 2],
    sides: [(usize, &S, ShapesOpStatus); 2],
    orientation: ShapesOpStatus,
    tolerance: f64,
    record: &mut Option<&mut RewriteObserver<'_, Point3, C>>,
) -> std::result::Result<(), &'static str>
where
    C: SearchNearestParameter<CurveParameter, Point = Point3>
        + SearchParameter<CurveParameter, Point = Point3>
        + Cut<Point = Point3, Vector = Vector3>
        + From<IntersectionCurve<PolylineCurve, S, S>>,
    S: ParametricSurface3D
        + Clone
        + SearchParameter<SurfaceParameter, Point = Point3>
        + SearchNearestParameter<SurfaceParameter, Point = Point3>,
{
    let [(face0, surface0, status0), (face1, surface1, status1)] = sides;
    let faces = [face0, face1];
    let surfaces = [surface0, surface1];
    let statuses = [status0, status1];
    let chart_edges = [
        paired_chart_edges(&native[0][faces[0]]),
        paired_chart_edges(&native[1][faces[1]]),
    ];
    let mut contacts: Vec<LoopPoint> = Vec::new();
    for side in 0..2 {
        for &(wire, edge) in &chart_edges[side] {
            polygon[side][faces[side]][wire][edge].with_curve(|boundary| {
                for (index, segment) in polyline.windows(2).enumerate() {
                    for seam in boundary.windows(2) {
                        // Both meshes approximate native geometry at tolerance.
                        // This broad phase is not a native residual allowance.
                        let Some((fraction, seed)) = segment_contact(
                            [segment[0], segment[1]],
                            [seam[0], seam[1]],
                            2.0 * tolerance,
                        ) else {
                            continue;
                        };
                        let (point, _, _) =
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
                        let polygon_point = boundary.evaluate(
                            boundary
                                .search_nearest_parameter(point, None, 100)
                                .ok_or("intersection_chart_polygon_parameter")?,
                        );
                        let mut polygon_parameter = index as f64 + fraction;
                        let mut native_parameter = curve
                            .search_parameter(point, Some(polygon_parameter), 100)
                            .ok_or("intersection_chart_native_contact")?;
                        if point.near(&curve.front()) {
                            native_parameter = 0.0;
                            polygon_parameter = 0.0;
                        }
                        if let Some(previous) = contacts
                            .iter_mut()
                            .find(|previous| previous.native.point().near(&point))
                        {
                            if !previous.polygon.point().near(&polygon_point) {
                                return Err("intersection_chart_polygon_contact_mismatch");
                            }
                            previous.on_boundary[side] = true;
                        } else {
                            let mut on_boundary = [false; 2];
                            on_boundary[side] = true;
                            contacts.push(LoopPoint {
                                polygon_parameter,
                                native_parameter,
                                polygon: Vertex::new(polygon_point),
                                native: Vertex::new(point),
                                on_boundary,
                                direction: None,
                            });
                        }
                    }
                }
                Ok::<(), &'static str>(())
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
                direction: None,
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
            direction: None,
        });
    }
    if points.windows(2).any(|pair| {
        pair[0].polygon_parameter >= pair[1].polygon_parameter
            || pair[0].native_parameter >= pair[1].native_parameter
    }) {
        return Err("intersection_chart_contact_order");
    }
    let mut polygon_maps = [HashMap::default(), HashMap::default()];
    let mut native_maps = [HashMap::default(), HashMap::default()];
    for point in &mut points {
        for side in 0..2 {
            if !point.on_boundary[side] {
                continue;
            }
            let BoundaryAttachment {
                wire_index,
                edge_index,
                native_kind,
                polygon_kind,
            } = boundary_attachment(
                native[side],
                polygon[side],
                faces[side],
                &point.polygon,
                &point.native,
                [surfaces[side], surfaces[1 - side]],
            )?
            .ok_or("intersection_chart_native_contact")?;
            polygon[side]
                .add_polygon_vertex_at(
                    faces[side],
                    &point.polygon,
                    (wire_index, edge_index, polygon_kind),
                    &mut polygon_maps[side],
                )
                .ok_or("intersection_chart_polygon_contact")?;
            native[side].add_geom_vertex(
                (faces[side], wire_index, edge_index),
                &point.native,
                native_kind,
                &mut native_maps[side],
                record,
            )?;
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
                    return Err("intersection_chart_contact_carrier");
                }
            }
        }
        if point.native_parameter == 0.0 {
            let last = curve.leader().len() - 1;
            curve.leader_mut()[0] = point.native.point();
            curve.leader_mut()[last] = point.native.point();
        }
    }
    // Fractional cuts rescale the head's last segment. Replay raw parameters
    // in that shortened segment and locate native cuts by the certified point.
    let mut polygon_parts = Vec::with_capacity(points.len());
    let mut native_parts = Vec::with_capacity(points.len());
    let mut previous_polygon_cut = polyline.range_tuple().1;
    for point in points.iter().skip(1).rev() {
        let parameter = polygon_head_parameter(point.polygon_parameter, previous_polygon_cut);
        polygon_parts.push(cut_polygon(&mut polyline, parameter));
        previous_polygon_cut = point.polygon_parameter;
        let parameter = curve
            .search_parameter(point.native.point(), Some(point.native_parameter), 100)
            .ok_or("intersection_chart_native_cut_parameter")?;
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
            return Err("intersection_chart_native_endpoint");
        }
        polygon_curve.snap_endpoints(points[index].polygon.point(), points[next].polygon.point());
        polygon_wire.push_back(Edge::new(
            &points[index].polygon,
            &points[next].polygon,
            polygon_curve,
        ));
        native_wire.push_back(Edge::new(
            &points[index].native,
            &points[next].native,
            native_curve.into(),
        ));
    }
    for side in 0..2 {
        let contacts: Vec<_> = points
            .iter()
            .enumerate()
            .filter_map(|(index, point)| point.on_boundary[side].then_some(index))
            .collect();
        if contacts.is_empty() {
            polygon[side][faces[side]]
                .add_independent_loop(BoundaryWire::new(polygon_wire.clone(), statuses[side]));
            native[side][faces[side]]
                .add_independent_loop(BoundaryWire::new(native_wire.clone(), statuses[side]));
            continue;
        }
        for (index, &start) in contacts.iter().enumerate() {
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
            let start_corner = chart_corner(
                &native[side][faces[side]],
                surfaces[side],
                native_path.front_vertex().unwrap(),
                points[start]
                    .direction
                    .ok_or("intersection_chart_orientation")?,
            )?;
            let end_corner = chart_corner(
                &native[side][faces[side]],
                surfaces[side],
                native_path.back_vertex().unwrap(),
                -points[end]
                    .direction
                    .ok_or("intersection_chart_orientation")?,
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
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use monstertruck_modeling as modeling;

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
                endpoint_corners(&loops, &surface, &arc).unwrap(),
                [None, Some(start)]
            );
            assert_eq!(
                endpoint_corners(&loops, &surface, &arc.inverse()).unwrap(),
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
}
