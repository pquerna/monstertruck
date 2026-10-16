//! Literal tensor-boundary pcurves in the original, nonperiodized source chart.
//!
//! A row of copied controls is not enough: the restricted original knot basis
//! must agree too. Candidates retain native edge identity and raw parameter
//! gauge. Only a unique, closed assignment of the complete oriented source wire
//! is usable; neither inverse-map hints nor endpoint proximity choose a seam.

use super::ShapeOpsError;
use crate::transversal::prepared::BooleanWork;
use monstertruck_core::cgmath64::*;
use monstertruck_geometry::prelude::*;
use monstertruck_topology::{Edge, EdgeId, Face};
use std::{mem::size_of, ops::Bound};

type ChartResult<T> = std::result::Result<T, ShapeOpsError>;

/// One ORIGINAL coedge occurrence, before any descendant-edge transport.
/// `map` always consumes the absolute native curve parameter, not an oriented
/// fraction. The containing vectors follow `face.absolute_boundaries()` order.
pub(super) struct SourceChartMap<C> {
    edge: EdgeId<C>,
    orientation: bool,
    parameter_range: (f64, f64),
    from: Point2,
    to: Point2,
}

impl<C> Copy for SourceChartMap<C> {}
impl<C> Clone for SourceChartMap<C> {
    fn clone(&self) -> Self { *self }
}

impl<C> SourceChartMap<C> {
    pub(super) fn edge(&self) -> EdgeId<C> { self.edge }
    pub(super) fn orientation(&self) -> bool { self.orientation }
    pub(super) fn parameter_range(&self) -> (f64, f64) { self.parameter_range }

    /// Evaluate only within `parameter_range()`. Endpoint branches retain the
    /// exact source knots, including the two distinct sides of a closed seam.
    pub(super) fn map(&self, parameter: f64) -> Point2 {
        if parameter == self.parameter_range.0 {
            self.from
        } else if parameter == self.parameter_range.1 {
            self.to
        } else {
            let fraction = (parameter - self.parameter_range.0)
                / (self.parameter_range.1 - self.parameter_range.0);
            self.from + (self.to - self.from) * fraction
        }
    }

    fn front(&self) -> Point2 { if self.orientation { self.from } else { self.to } }
    fn back(&self) -> Point2 { if self.orientation { self.to } else { self.from } }
}

fn checkpoint(work: Option<&BooleanWork<'_>>) -> ChartResult<()> {
    if let Some(work) = work {
        work.check()?;
    }
    Ok(())
}

fn admit(work: Option<&BooleanWork<'_>>, bytes: usize) -> ChartResult<()> {
    if let Some(work) = work {
        work.limit(
            "original_boundary_chart_bytes",
            bytes,
            work.limits.max_intermediate_prepared_bytes,
        )?;
    }
    Ok(())
}

fn curve_data(source: OriginalCurveCoefficients<'_>) -> Option<(&KnotVector, usize)> {
    match source {
        OriginalCurveCoefficients::Polynomial(curve) => {
            Some((curve.knot_vector(), curve.control_points().len()))
        }
        OriginalCurveCoefficients::Rational(curve) => {
            let curve = curve.non_rationalized();
            Some((curve.knot_vector(), curve.control_points().len()))
        }
        OriginalCurveCoefficients::Line(_) => None,
    }
}

fn curve_control(source: OriginalCurveCoefficients<'_>, index: usize) -> Option<Vector4> {
    match source {
        OriginalCurveCoefficients::Polynomial(curve) => {
            Some(curve.control_points().get(index)?.to_homogeneous())
        }
        OriginalCurveCoefficients::Rational(curve) => curve
            .non_rationalized()
            .control_points()
            .get(index)
            .copied(),
        OriginalCurveCoefficients::Line(_) => None,
    }
}

fn tensor_data(
    source: OriginalSurfaceCoefficients<'_>,
) -> Option<(&KnotVector, &KnotVector, usize, usize)> {
    let (uk, vk, nu, nv, rectangular) = match source {
        OriginalSurfaceCoefficients::Polynomial(surface) => {
            let controls = surface.control_points();
            let (uk, vk) = surface.knot_vectors();
            let nv = controls.first()?.len();
            (
                uk,
                vk,
                controls.len(),
                nv,
                controls.iter().all(|row| row.len() == nv),
            )
        }
        OriginalSurfaceCoefficients::Rational(surface) => {
            let surface = surface.non_rationalized();
            let controls = surface.control_points();
            let (uk, vk) = surface.knot_vectors();
            let nv = controls.first()?.len();
            (
                uk,
                vk,
                controls.len(),
                nv,
                controls.iter().all(|row| row.len() == nv),
            )
        }
        OriginalSurfaceCoefficients::Extrusion { .. } => return None,
    };
    rectangular.then_some((uk, vk, nu, nv))
}

fn tensor_control(source: OriginalSurfaceCoefficients<'_>, u: usize, v: usize) -> Option<Vector4> {
    match source {
        OriginalSurfaceCoefficients::Polynomial(surface) => {
            Some(surface.control_points().get(u)?.get(v)?.to_homogeneous())
        }
        OriginalSurfaceCoefficients::Rational(surface) => surface
            .non_rationalized()
            .control_points()
            .get(u)?
            .get(v)
            .copied(),
        OriginalSurfaceCoefficients::Extrusion { .. } => None,
    }
}

fn clamped(knots: &KnotVector, count: usize) -> Option<usize> {
    let degree = knots.len().checked_sub(count.checked_add(1)?)?;
    if degree == 0 || degree >= count || !knots.iter().all(|knot| knot.is_finite()) {
        return None;
    }
    let low = knots[degree];
    let high = knots[count];
    (low < high
        && knots[..=degree].iter().all(|&knot| knot == low)
        && knots[count..].iter().all(|&knot| knot == high)
        && knots[degree + 1] > low
        && knots[count - 1] < high
        && knots.windows(2).all(|pair| pair[0] <= pair[1]))
    .then_some(degree)
}

// Error-free addition refuses an unrepresentable difference rather than
// silently proving equality using rounded knot differences.
fn exact_add(a: f64, b: f64) -> Option<f64> {
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    let sum = a + b;
    let virtual_b = sum - a;
    let recovered_a = sum - virtual_b;
    let error_a = a - recovered_a;
    let error_b = b - virtual_b;
    let error = error_a + error_b;
    ([sum, virtual_b, recovered_a, error_a, error_b, error]
        .into_iter()
        .all(f64::is_finite)
        && error == 0.0)
        .then_some(sum)
}

// Each f64 is a dyadic rational. Two 53-bit significands multiply exactly in
// u128, so projective-scale and knot-ratio comparisons need no rounded product,
// division, tolerance, big integer allocation, or assumptions about the gauge.
fn product(a: f64, b: f64) -> Option<(bool, u128, i32)> {
    fn dyadic(value: f64) -> (u128, i32) {
        let bits = value.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & ((1_u64 << 52) - 1);
        if exponent == 0 {
            (u128::from(fraction), -1074)
        } else {
            (u128::from(fraction | (1_u64 << 52)), exponent - 1075)
        }
    }
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    let (a_bits, a_exponent) = dyadic(a);
    let (b_bits, b_exponent) = dyadic(b);
    let bits = a_bits * b_bits;
    if bits == 0 {
        return Some((false, 0, 0));
    }
    let shift = bits.trailing_zeros();
    Some((
        a.is_sign_negative() != b.is_sign_negative(),
        bits >> shift,
        a_exponent + b_exponent + shift as i32,
    ))
}

fn same_product(a: f64, b: f64, c: f64, d: f64) -> bool {
    matches!((product(a, b), product(c, d)), (Some(left), Some(right)) if left == right)
}

/// A copied control slice restricts the original basis only at clamped global
/// ends or multiplicity-p interior knots. In particular a simple P2 knot, or a
/// discontinuous multiplicity-(p+1) split, is NOT a clamped Bezier subrow.
fn restricted_basis(
    along: &KnotVector,
    along_count: usize,
    offset: usize,
    count: usize,
    degree: usize,
) -> Option<(f64, f64)> {
    let low = along[offset + degree];
    let high = along[offset + count];
    if low >= high
        || (offset != 0 && (along[offset] >= low || along[offset + degree + 1] <= low))
        || (offset + count != along_count
            && (along[offset + count - 1] >= high || along[offset + count + degree] <= high))
        || (0..degree)
            .any(|index| along[offset + 1 + index] != low || along[offset + count + index] != high)
    {
        None
    } else {
        Some((low, high))
    }
}

struct Tensor<'a> {
    coefficients: OriginalSurfaceCoefficients<'a>,
    uk: &'a KnotVector,
    vk: &'a KnotVector,
    nu: usize,
    nv: usize,
    up: usize,
    vp: usize,
}

fn candidates<C>(
    edge: &Edge<Point3, C>,
    tensor: &Tensor<'_>,
    work: Option<&BooleanWork<'_>>,
    bytes: &mut usize,
) -> ChartResult<Option<Vec<SourceChartMap<C>>>>
where
    C: ParametricCurve3D + BoundedCurve + TryIntoHomogeneousBsplineCurve,
{
    checkpoint(work)?;
    edge.with_curve(|curve| {
        let Some(source) = curve.original_curve_coefficients() else {
            return Ok(None);
        };
        let Some((knots, count)) = curve_data(source) else {
            return Ok(None);
        };
        if let Some(work) = work {
            work.charge(knots.len().saturating_mul(4))?;
        }
        let Some(degree) = clamped(knots, count) else {
            return Ok(None);
        };
        let native = (knots[degree], knots[count]);
        if curve.range_tuple() != native {
            return Ok(None);
        }
        let Some(native_span) = exact_add(native.1, -native.0) else {
            return Ok(None);
        };
        let Some(first) = curve_control(source, 0) else {
            return Ok(None);
        };
        let Some(last) = curve_control(source, count - 1) else {
            return Ok(None);
        };
        if first.w == 0.0 || last.w == 0.0 {
            return Ok(None);
        }
        for (control, vertex) in [(first, edge.absolute_front()), (last, edge.absolute_back())] {
            let point = vertex.point();
            if (0..3).any(|axis| !same_product(control[axis], 1.0, point[axis], control.w)) {
                return Ok(None);
            }
        }
        let offsets = |along_count: usize, along_degree: usize| {
            if degree == along_degree && count <= along_count {
                along_count - count + 1
            } else {
                0
            }
        };
        let capacity = offsets(tensor.nu, tensor.up)
            .saturating_add(offsets(tensor.nv, tensor.vp))
            .saturating_mul(4);
        *bytes = bytes.saturating_add(capacity.saturating_mul(size_of::<SourceChartMap<C>>()));
        admit(work, *bytes)?;
        let mut result = Vec::with_capacity(capacity);
        for axis in 0..2 {
            let (along, along_count, along_degree, transverse, transverse_count) = if axis == 0 {
                (tensor.uk, tensor.nu, tensor.up, tensor.vk, tensor.nv)
            } else {
                (tensor.vk, tensor.nv, tensor.vp, tensor.uk, tensor.nu)
            };
            for offset in 0..offsets(along_count, along_degree) {
                if let Some(work) = work {
                    work.charge(degree.saturating_add(1))?;
                }
                let Some((low, high)) = restricted_basis(along, along_count, offset, count, degree)
                else {
                    continue;
                };
                let Some(source_span) = exact_add(high, -low) else {
                    continue;
                };
                for reverse in [false, true] {
                    let mut same_knots = true;
                    for index in degree + 1..count {
                        checkpoint(work)?;
                        let native_delta = exact_add(knots[index], -native.0);
                        let source_delta = if reverse {
                            exact_add(high, -along[offset + count + degree - index])
                        } else {
                            exact_add(along[offset + index], -low)
                        };
                        same_knots &= matches!((native_delta, source_delta), (Some(a), Some(b))
                            if same_product(a, source_span, b, native_span));
                    }
                    if !same_knots {
                        continue;
                    }
                    for upper in [false, true] {
                        let row = if upper { transverse_count - 1 } else { 0 };
                        let control = |index| {
                            let index = offset + if reverse { count - 1 - index } else { index };
                            if axis == 0 {
                                tensor_control(tensor.coefficients, index, row)
                            } else {
                                tensor_control(tensor.coefficients, row, index)
                            }
                        };
                        let Some(boundary_first) = control(0) else {
                            continue;
                        };
                        if boundary_first.w == 0.0 {
                            continue;
                        }
                        let mut identical = true;
                        for index in 0..count {
                            checkpoint(work)?;
                            let (Some(original), Some(boundary)) =
                                (curve_control(source, index), control(index))
                            else {
                                identical = false;
                                break;
                            };
                            if original.w == 0.0
                                || boundary.w == 0.0
                                || original.w.is_sign_positive() != first.w.is_sign_positive()
                                || boundary.w.is_sign_positive()
                                    != boundary_first.w.is_sign_positive()
                                || (0..4).any(|component| {
                                    !same_product(
                                        original[component],
                                        boundary_first.w,
                                        boundary[component],
                                        first.w,
                                    )
                                })
                            {
                                identical = false;
                                break;
                            }
                        }
                        if !identical {
                            continue;
                        }
                        let fixed = if upper {
                            transverse[transverse_count]
                        } else {
                            transverse[transverse.len() - transverse_count - 1]
                        };
                        let point = |along| {
                            if axis == 0 {
                                Point2::new(along, fixed)
                            } else {
                                Point2::new(fixed, along)
                            }
                        };
                        let (from, to) = if reverse {
                            (point(high), point(low))
                        } else {
                            (point(low), point(high))
                        };
                        result.push(SourceChartMap {
                            edge: edge.id(),
                            orientation: edge.orientation(),
                            parameter_range: native,
                            from,
                            to,
                        });
                    }
                }
            }
        }
        Ok(Some(result))
    })
}

// Count paths in a layered graph, saturating at two. Enumerating assignments
// would be exponential on coincident rows; this is polynomial in the actual
// candidate counts, and every adjacency comparison is charged.
fn unique_wire<C>(
    layers: &[Vec<SourceChartMap<C>>],
    work: Option<&BooleanWork<'_>>,
    bytes: usize,
) -> ChartResult<Option<Vec<SourceChartMap<C>>>> {
    if layers.is_empty() || layers.iter().any(Vec::is_empty) {
        return Ok(None);
    }
    let count = layers
        .iter()
        .fold(0usize, |sum, layer| sum.saturating_add(layer.len()));
    let scratch = count
        .saturating_mul(size_of::<u8>() + size_of::<usize>())
        .saturating_add(layers.len().saturating_mul(
            2 * size_of::<Vec<usize>>() + size_of::<usize>() + size_of::<SourceChartMap<C>>(),
        ));
    admit(work, bytes.saturating_add(scratch))?;
    let mut paths: Vec<Vec<u8>> = layers.iter().map(|layer| vec![0; layer.len()]).collect();
    let mut predecessors: Vec<Vec<usize>> =
        layers.iter().map(|layer| vec![0; layer.len()]).collect();
    let mut chosen = None;
    for start in 0..layers[0].len() {
        checkpoint(work)?;
        for layer in &mut paths {
            layer.fill(0);
        }
        paths[0][start] = 1;
        for index in 1..layers.len() {
            for next in 0..layers[index].len() {
                for previous in 0..layers[index - 1].len() {
                    checkpoint(work)?;
                    let ways = paths[index - 1][previous];
                    if ways != 0
                        && layers[index - 1][previous].back() == layers[index][next].front()
                    {
                        if paths[index][next] == 0 {
                            predecessors[index][next] = previous;
                        }
                        paths[index][next] = paths[index][next].saturating_add(ways).min(2);
                    }
                }
            }
        }
        let last = layers.len() - 1;
        for end in 0..layers[last].len() {
            checkpoint(work)?;
            let ways = paths[last][end];
            if ways == 0 || layers[last][end].back() != layers[0][start].front() {
                continue;
            }
            if ways > 1 || chosen.is_some() {
                return Ok(None);
            }
            let mut assignment = vec![0; layers.len()];
            assignment[last] = end;
            for index in (1..layers.len()).rev() {
                assignment[index - 1] = predecessors[index][assignment[index]];
            }
            chosen = Some(assignment);
        }
    }
    Ok(chosen.map(|assignment| {
        assignment
            .into_iter()
            .enumerate()
            .map(|(index, candidate)| layers[index][candidate])
            .collect()
    }))
}

/// Exact original tensor-boundary occurrence charts, or refusal if any original
/// curve/basis is unsupported, no closed assignment exists, or it is ambiguous.
/// No surface period is inferred: raw source knots are the entire chart domain.
pub(super) fn original_boundary_charts<C, S>(
    face: &Face<Point3, C, S>,
    work: Option<&BooleanWork<'_>>,
) -> ChartResult<Option<Vec<Vec<SourceChartMap<C>>>>>
where
    C: ParametricCurve3D + BoundedCurve + TryIntoHomogeneousBsplineCurve,
    S: ParametricSurface3D + TryIntoHomogeneousBsplineSurface,
{
    checkpoint(work)?;
    face.with_surface(|surface| {
        let Some(coefficients) = surface.original_surface_coefficients() else {
            return Ok(None);
        };
        if let Some(work) = work {
            let scan = match coefficients {
                OriginalSurfaceCoefficients::Polynomial(surface) => {
                    let (u, v) = surface.knot_vectors();
                    u.len()
                        .saturating_add(v.len())
                        .saturating_mul(4)
                        .saturating_add(surface.control_points().len())
                }
                OriginalSurfaceCoefficients::Rational(surface) => {
                    let surface = surface.non_rationalized();
                    let (u, v) = surface.knot_vectors();
                    u.len()
                        .saturating_add(v.len())
                        .saturating_mul(4)
                        .saturating_add(surface.control_points().len())
                }
                OriginalSurfaceCoefficients::Extrusion { .. } => return Ok(None),
            };
            work.charge(scan)?;
        }
        let Some((uk, vk, nu, nv)) = tensor_data(coefficients) else {
            return Ok(None);
        };
        let (Some(up), Some(vp)) = (clamped(uk, nu), clamped(vk, nv)) else {
            return Ok(None);
        };
        if surface.parameter_range()
            != (
                (Bound::Included(uk[up]), Bound::Included(uk[nu])),
                (Bound::Included(vk[vp]), Bound::Included(vk[nv])),
            )
        {
            return Ok(None);
        }
        let tensor = Tensor {
            coefficients,
            uk,
            vk,
            nu,
            nv,
            up,
            vp,
        };
        let boundaries = face.absolute_boundaries();
        let mut retained = boundaries
            .len()
            .saturating_mul(size_of::<Vec<SourceChartMap<C>>>());
        admit(work, retained)?;
        let mut result = Vec::with_capacity(boundaries.len());
        for wire in boundaries {
            checkpoint(work)?;
            let mut bytes = retained.saturating_add(
                wire.len()
                    .saturating_mul(size_of::<Vec<SourceChartMap<C>>>()),
            );
            admit(work, bytes)?;
            let mut layers = Vec::with_capacity(wire.len());
            for edge in wire.iter() {
                let Some(layer) = candidates(edge, &tensor, work, &mut bytes)? else {
                    return Ok(None);
                };
                if layer.is_empty() {
                    return Ok(None);
                }
                layers.push(layer);
            }
            let Some(charts) = unique_wire(&layers, work, bytes)? else {
                return Ok(None);
            };
            retained = retained.saturating_add(
                charts
                    .capacity()
                    .saturating_mul(size_of::<SourceChartMap<C>>()),
            );
            result.push(charts);
        }
        Ok(Some(result))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use monstertruck_modeling as modeling;
    use monstertruck_topology::Vertex;

    fn original_torus(independent_gauge: bool) -> modeling::Face {
        // The actual original tensor has independently scaled homogeneous
        // circle controls, not a periodic analytic torus or a fitted carrier.
        let circle = [
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 2.0, 2.0],
            [-2.0, 2.0, 2.0],
            [-4.0, 0.0, 4.0],
            [-4.0, -4.0, 4.0],
            [0.0, -8.0, 8.0],
            [8.0, -8.0, 8.0],
            [16.0, 0.0, 16.0],
        ];
        let knots = KnotVector::from(vec![
            0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0,
        ]);
        let tensor = |[x, y, w]: [f64; 3], [a, b, v]: [f64; 3]| {
            Vector4::new(
                (6.0 * v + 2.0 * a) * x,
                (6.0 * v + 2.0 * a) * y,
                2.0 * b * w,
                v * w,
            )
        };
        let surface = modeling::Surface::NurbsSurface(NurbsSurface::new(BsplineSurface::new(
            (knots.clone(), knots),
            circle
                .iter()
                .map(|&theta| circle.iter().map(|&phi| tensor(theta, phi)).collect())
                .collect(),
        )));
        let vertices = Vertex::from_points([
            Point3::new(8.0, 0.0, 0.0),
            Point3::new(-8.0, 0.0, 0.0),
            Point3::new(4.0, 0.0, 0.0),
        ]);
        let seam = |half: usize, meridian: bool| {
            let front = if half == 0 {
                0
            } else if meridian {
                2
            } else {
                1
            };
            let back = if half == 1 {
                0
            } else if meridian {
                2
            } else {
                1
            };
            let start = 2.0 * half as f64;
            let mut curve = NurbsCurve::new(BsplineCurve::new(
                KnotVector::from(vec![
                    start,
                    start,
                    start,
                    start + 1.0,
                    start + 1.0,
                    start + 2.0,
                    start + 2.0,
                    start + 2.0,
                ]),
                circle[4 * half..=4 * half + 4]
                    .iter()
                    .map(|&point| {
                        let control = if meridian {
                            tensor(circle[0], point)
                        } else {
                            tensor(point, circle[0])
                        };
                        // One global projective gauge per curve; source row u=4
                        // additionally differs from u=0 by a factor of sixteen.
                        control * if independent_gauge { -2.0 } else { 1.0 }
                    })
                    .collect(),
            ));
            if independent_gauge {
                curve.parameter_transform(2.0, 10.0);
            }
            Edge::new(
                &vertices[front],
                &vertices[back],
                modeling::Curve::NurbsCurve(curve),
            )
        };
        let theta = [seam(0, false), seam(1, false)];
        let phi = [seam(0, true), seam(1, true)];
        let wire = [
            theta[0].clone(),
            theta[1].clone(),
            phi[0].clone(),
            phi[1].clone(),
            theta[1].inverse(),
            theta[0].inverse(),
            phi[1].inverse(),
            phi[0].inverse(),
        ]
        .into_iter()
        .collect();
        Face::try_new(vec![wire], surface).expect("literal original closed tensor wire")
    }

    #[test]
    fn original_torus_whole_wire_separates_opposed_occurrences_and_native_gauges() {
        for independent in [false, true] {
            let face = original_torus(independent);
            assert_eq!(face.surface().period_u(), None);
            assert_eq!(face.surface().period_v(), None);
            let charts = original_boundary_charts(&face, None)
                .unwrap()
                .expect("unique source rectangle");
            let expected = [
                (0.0, 0.0),
                (2.0, 0.0),
                (4.0, 0.0),
                (4.0, 2.0),
                (4.0, 4.0),
                (2.0, 4.0),
                (0.0, 4.0),
                (0.0, 2.0),
                (0.0, 0.0),
            ];
            assert_eq!(charts.len(), 1);
            assert_eq!(charts[0].len(), 8);
            for (index, (chart, edge)) in charts[0]
                .iter()
                .zip(face.absolute_boundaries()[0].iter())
                .enumerate()
            {
                assert_eq!(chart.edge(), edge.id());
                assert_eq!(chart.orientation(), edge.orientation());
                assert_eq!(
                    chart.parameter_range(),
                    edge.with_curve(BoundedCurve::range_tuple)
                );
                assert_eq!(chart.front(), Point2::from(expected[index]));
                assert_eq!(chart.back(), Point2::from(expected[index + 1]));
                let range = chart.parameter_range();
                let middle = chart.map((range.0 + range.1) * 0.5);
                assert_eq!(middle, chart.from + (chart.to - chart.from) * 0.5);
            }
            // These are the SAME native phi edge, yet different occurrences
            // belong to different exact raw chart boundaries.
            assert_eq!(charts[0][2].edge(), charts[0][7].edge());
            assert_eq!(charts[0][2].front().x, 4.0);
            assert_eq!(charts[0][7].back().x, 0.0);
        }
    }

    fn row_candidates(
        knots: KnotVector,
        controls: &[Point3],
        offset: usize,
        reverse: bool,
    ) -> Vec<SourceChartMap<modeling::Curve>> {
        let surface = modeling::Surface::BsplineSurface(BsplineSurface::new(
            (knots, KnotVector::bezier_knot(1)),
            controls
                .iter()
                .map(|&point| vec![point, point + Vector3::unit_z()])
                .collect(),
        ));
        let coefficients = surface.original_surface_coefficients().unwrap();
        let (uk, vk, nu, nv) = tensor_data(coefficients).unwrap();
        let tensor = Tensor {
            coefficients,
            uk,
            vk,
            nu,
            nv,
            up: clamped(uk, nu).unwrap(),
            vp: clamped(vk, nv).unwrap(),
        };
        let mut row = controls[offset..offset + 3].to_vec();
        if reverse {
            row.reverse();
        }
        let vertices = Vertex::from_points([row[0], row[2]]);
        let mut curve = BsplineCurve::new(KnotVector::bezier_knot(2), row);
        curve.parameter_transform(2.0, 10.0);
        let edge = Edge::new(
            &vertices[0],
            &vertices[1],
            modeling::Curve::BsplineCurve(curve),
        );
        candidates(&edge, &tensor, None, &mut 0).unwrap().unwrap()
    }

    #[test]
    fn copied_p2_subrow_without_original_endpoint_basis_is_refused_at_either_end() {
        let controls = [
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, -1.0, 0.0),
        ];
        let simple = KnotVector::from(vec![0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0]);
        for offset in [0, 1] {
            assert!(row_candidates(simple.clone(), &controls, offset, false).is_empty());
        }
    }

    #[test]
    fn exact_c0_restrictions_preserve_original_basis_and_reversed_raw_gauge() {
        let controls = [
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, -1.0, 0.0),
            Point3::new(3.0, 0.0, 0.0),
        ];
        let knots = KnotVector::from(vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        for offset in [0, 2] {
            for reverse in [false, true] {
                let charts = row_candidates(knots.clone(), &controls, offset, reverse);
                assert_eq!(charts.len(), 1);
                let low = (offset / 2) as f64;
                let high = low + 1.0;
                assert_eq!(
                    charts[0].map(10.0),
                    Point2::new(if reverse { high } else { low }, 0.0)
                );
                assert_eq!(
                    charts[0].map(12.0),
                    Point2::new(if reverse { low } else { high }, 0.0)
                );
            }
        }
    }

    #[test]
    fn whole_wire_ambiguity_is_refused_instead_of_choosing_first_candidate() {
        let face = original_torus(false);
        let charts = original_boundary_charts(&face, None).unwrap().unwrap();
        let chart = charts[0][0];
        // Two distinct complete cycles, differing only in source chart side,
        // cannot be selected by endpoints or the first candidate's ordering.
        let first = SourceChartMap {
            from: Point2::new(0.0, 0.0),
            to: Point2::new(1.0, 0.0),
            ..chart
        };
        let second = SourceChartMap {
            from: Point2::new(0.0, 1.0),
            to: Point2::new(1.0, 1.0),
            ..chart
        };
        let reverse = |map: SourceChartMap<modeling::Curve>| SourceChartMap {
            orientation: false,
            ..map
        };
        assert!(
            unique_wire(
                &[vec![first, second], vec![reverse(first), reverse(second)]],
                None,
                0
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn exact_projective_comparison_does_not_round_overflow_or_underflow() {
        assert!(same_product(f64::MAX, 2.0, f64::MAX, 2.0));
        assert!(same_product(
            f64::MIN_POSITIVE,
            f64::MIN_POSITIVE,
            f64::MIN_POSITIVE,
            f64::MIN_POSITIVE
        ));
        assert!(!same_product(
            1.0 + f64::EPSILON,
            1.0 - f64::EPSILON,
            1.0,
            1.0
        ));
        assert!(same_product(-0.0, 2.0, 0.0, -4.0));
    }
}
