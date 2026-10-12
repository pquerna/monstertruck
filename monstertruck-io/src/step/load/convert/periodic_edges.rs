//! Exact edge subdivision for the native two-distinct-vertices representation.

use super::*;

/// Subdivide a one-period closed edge once, sharing both subedges in every use.
///
/// This changes neither its curve locus nor any face, carrier or orientation.
/// Nonperiodic, collapsed and inconsistent self-loops remain conversion errors.
fn split_edges(
    vertices: &mut Vec<Point3>,
    edges: &mut Vec<CompressedEdge<Curve3D>>,
) -> Result<Vec<Option<usize>>, StepConvertingError> {
    if !edges.iter().any(|edge| edge.vertices.0 == edge.vertices.1) {
        return Ok(Vec::new());
    }
    let mut splits = vec![None; edges.len()];
    for index in 0..splits.len() {
        let edge = &mut edges[index];
        if edge.vertices.0 != edge.vertices.1 {
            continue;
        }
        let point = vertices[edge.vertices.0];
        let (start, end) = edge.curve.range_tuple();
        let period = edge
            .curve
            .period()
            .ok_or("closed edge has no periodic curve")?;
        let middle = start + (end - start) * 0.5;
        let midpoint = edge.curve.subs(middle);
        if !start.is_finite()
            || !end.is_finite()
            || !period.is_finite()
            || period <= 0.0
            || !(end - start).near(&period)
            || !edge.curve.front().near(&point)
            || !edge.curve.back().near(&point)
            || !midpoint.x.is_finite()
            || !midpoint.y.is_finite()
            || !midpoint.z.is_finite()
            || midpoint.near(&point)
        {
            return Err("closed edge is not one consistent nondegenerate curve period".into());
        }
        let second_curve = edge.curve.cut(middle);
        let original_vertex = edge.vertices.0;
        let middle_vertex = vertices.len();
        edge.vertices.1 = middle_vertex;
        vertices.push(midpoint);
        splits[index] = Some(edges.len());
        edges.push(CompressedEdge {
            vertices: (middle_vertex, original_vertex),
            curve: second_curve,
        });
    }
    Ok(splits)
}

fn split_indices(
    edge: CompressedEdgeIndex,
    splits: &[Option<usize>],
) -> [Option<CompressedEdgeIndex>; 2] {
    match splits[edge.index] {
        None => [Some(edge), None],
        Some(index) => {
            let second = CompressedEdgeIndex { index, ..edge };
            if edge.orientation {
                [Some(edge), Some(second)]
            } else {
                [Some(second), Some(edge)]
            }
        }
    }
}

pub(super) fn plain(shell: &mut StepCompressedShell) -> Result<(), StepConvertingError> {
    let splits = split_edges(&mut shell.vertices, &mut shell.edges)?;
    if splits.iter().any(Option::is_some) {
        shell.faces.iter_mut().for_each(|face| {
            face.boundaries.iter_mut().for_each(|wire| {
                *wire = std::mem::take(wire)
                    .into_iter()
                    .flat_map(|edge| split_indices(edge, &splits).into_iter().flatten())
                    .collect();
            });
        });
    }
    Ok(())
}

pub(super) fn trimmed(shell: &mut StepCompressedTrimmedShell) -> Result<(), StepConvertingError> {
    let splits = split_edges(&mut shell.vertices, &mut shell.edges)?;
    if splits.iter().any(Option::is_some) {
        shell.faces.iter_mut().for_each(|face| {
            face.boundaries.iter_mut().for_each(|wire| {
                *wire = std::mem::take(wire)
                    .into_iter()
                    .flat_map(|edge| {
                        let Some(index) = splits[edge.index] else {
                            return [Some(edge), None];
                        };
                        let (first_trim, second_trim) = match edge.trim_curve {
                            Some(mut curve) => {
                                let (start, end) = curve.range_tuple();
                                let second = curve.cut(start + (end - start) * 0.5);
                                (Some(curve), Some(second))
                            }
                            None => (None, None),
                        };
                        let (first_index, second_index) = if edge.orientation {
                            (edge.index, index)
                        } else {
                            (index, edge.index)
                        };
                        [
                            Some(CompressedEdgeUse {
                                index: first_index,
                                orientation: edge.orientation,
                                trim_curve: first_trim,
                            }),
                            Some(CompressedEdgeUse {
                                index: second_index,
                                orientation: edge.orientation,
                                trim_curve: second_trim,
                            }),
                        ]
                    })
                    .flatten()
                    .collect();
            });
        });
    }
    Ok(())
}
