//! Exact edge subdivision for the native two-distinct-vertices representation.

use super::*;

/// Subdivide a finite closed edge once, sharing both subedges in every use.
///
/// Closure comes from the declared common vertex and actual active endpoints,
/// not periodicity. Explicit trimmed and nonperiodic spline loops remain valid.
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
        if !start.is_finite() || !end.is_finite() || start >= end {
            return Err("closed edge has no finite increasing active interval".into());
        }
        let middle = start.midpoint(end);
        let midpoint = edge.curve.subs(middle);
        if !(start < middle && middle < end)
            || !edge.curve.front().near(&point)
            || !edge.curve.back().near(&point)
            || !midpoint.x.is_finite()
            || !midpoint.y.is_finite()
            || !midpoint.z.is_finite()
            || midpoint.near(&point)
        {
            return Err(
                "closed edge has inconsistent endpoints or no noncollapsed midpoint witness".into(),
            );
        }
        let first_curve = Curve3D::TrimmedCurve(SourceTrimmedCurve::internal(Processor::new(
            TrimmedCurve::new(Box::new(edge.curve.clone()), (start, middle)),
        )));
        let original_curve = std::mem::replace(&mut edge.curve, first_curve);
        let second_curve = Curve3D::TrimmedCurve(SourceTrimmedCurve::internal(Processor::new(
            TrimmedCurve::new(Box::new(original_curve), (middle, end)),
        )));
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

trait PeriodicTrimRestriction: BoundedCurve {
    fn split_restricted(self) -> (Self, Self);
}

impl PeriodicTrimRestriction for Curve2D {
    fn split_restricted(self) -> (Self, Self) {
        let (start, end) = self.range_tuple();
        let middle = start.midpoint(end);
        let first = Curve2D::TrimmedCurve(SourceTrimmedCurve::internal(Processor::new(
            TrimmedCurve::new(Box::new(self.clone()), (start, middle)),
        )));
        let second = Curve2D::TrimmedCurve(SourceTrimmedCurve::internal(Processor::new(
            TrimmedCurve::new(Box::new(self), (middle, end)),
        )));
        (first, second)
    }
}

impl PeriodicTrimRestriction for step_geometry::StepParameterCurve {
    fn split_restricted(self) -> (Self, Self) {
        self.split_with(PeriodicTrimRestriction::split_restricted)
    }
}

fn trimmed_impl<C: PeriodicTrimRestriction>(
    shell: &mut CompressedTrimmedShell<Point3, Curve3D, Surface, C>,
    provenance: Option<&mut SourceShellProvenance>,
) -> Result<(), StepConvertingError> {
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
                            Some(curve) => {
                                let (first, second) = curve.split_restricted();
                                (Some(first), Some(second))
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
        if let Some(provenance) = provenance {
            let edges = &shell.edges;
            provenance.faces.iter_mut().for_each(|face| {
                face.wires.iter_mut().for_each(|wire| {
                    wire.coedges = std::mem::take(&mut wire.coedges)
                        .into_iter()
                        .flat_map(|source| {
                            let edge = CompressedEdgeIndex {
                                index: source.native_edge_index,
                                orientation: source.native_orientation,
                            };
                            split_indices(edge, &splits)
                                .into_iter()
                                .flatten()
                                .map(move |child| SourceCoedgeProvenance {
                                    native_edge_index: child.index,
                                    restricted_curve_range: SourceCurveInterval::new(
                                        edges[child.index].curve.range_tuple(),
                                    ),
                                    ..source
                                })
                        })
                        .collect();
                });
            });
        }
    }
    Ok(())
}

pub(super) fn trimmed(shell: &mut StepCompressedTrimmedShell) -> Result<(), StepConvertingError> {
    trimmed_impl(shell, None)
}

pub(super) fn compact_trimmed(
    shell: &mut CompactTrimmedShell,
    provenance: &mut SourceShellProvenance,
) -> Result<(), StepConvertingError> {
    trimmed_impl(shell, Some(provenance))
}

#[cfg(test)]
mod source_parameter_restriction_regressions {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn periodic_inverse_children_preserve_source_parameters_and_linear_uv_domains() {
        let range = (0.2, 0.2 + TAU);
        let mut conic = Processor::new(TrimmedCurve::new(UnitCircle::<Point3>::new(), range));
        conic.invert();
        let original = Curve3D::Conic(Conic3D::Ellipse(conic));
        let original_uv = Curve2D::Line(Line(Point2::new(4.0, 7.0), Point2::new(8.0, 11.0)));
        let mut shell = CompressedTrimmedShell {
            vertices: vec![original.front()],
            edges: vec![CompressedEdge {
                vertices: (0, 0),
                curve: original.clone(),
            }],
            faces: vec![CompressedTrimmedFace {
                surface: Surface::ElementarySurface(ElementarySurface::Plane(
                    step_geometry::Plane::xy(),
                )),
                orientation: true,
                boundaries: vec![vec![CompressedEdgeUse {
                    index: 0,
                    orientation: false,
                    trim_curve: Some(original_uv.clone()),
                }]],
            }],
        };
        let mut provenance = SourceShellProvenance::default();
        compact_trimmed(&mut shell, &mut provenance).expect("one exact inverse period splits");
        let middle = range.0 + (range.1 - range.0) * 0.5;
        assert_eq!(shell.edges[0].curve.range_tuple(), (range.0, middle));
        assert_eq!(shell.edges[1].curve.range_tuple(), (middle, range.1));
        for edge in &shell.edges {
            let (start, end) = edge.curve.range_tuple();
            for parameter in [start, start + (end - start) * 0.37, end] {
                assert!(edge.curve.subs(parameter).near(&original.subs(parameter)));
                assert!(edge.curve.der(parameter).near(&original.der(parameter)));
            }
        }
        let wire = &shell.faces[0].boundaries[0];
        assert_eq!((wire[0].index, wire[1].index), (1, 0));
        for (edge_use, expected_range) in wire.iter().zip([(0.0, 0.5), (0.5, 1.0)]) {
            let trim = edge_use
                .trim_curve
                .as_ref()
                .expect("declared UV remains present");
            assert_eq!(trim.range_tuple(), expected_range);
            for parameter in [expected_range.0, expected_range.1] {
                assert_eq!(trim.subs(parameter), original_uv.subs(parameter));
                assert_eq!(trim.der(parameter), original_uv.der(parameter));
            }
        }
    }

    #[test]
    fn nonperiodic_closed_rational_curve_preserves_finite_active_parameter_identity() {
        use monstertruck_geometry::prelude::{BsplineCurve, KnotVector, NurbsCurve, Vector4};
        let original = Curve3D::NurbsCurve(NurbsCurve::new(BsplineCurve::new(
            KnotVector::from(vec![2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0]),
            vec![
                Vector4::new(0.0, 0.0, 0.0, 1.0),
                Vector4::new(1.0, 0.0, 0.0, 1.0),
                Vector4::new(1.0, 1.0, 0.0, 1.0),
                Vector4::new(0.0, 1.0, 0.0, 1.0),
                Vector4::new(0.0, 0.0, 0.0, 1.0),
            ],
        )));
        assert!(original.period().is_none());
        let mut vertices = vec![Point3::origin()];
        let mut edges = vec![CompressedEdge {
            vertices: (0, 0),
            curve: original,
        }];
        split_edges(&mut vertices, &mut edges)
            .expect("finite closed curve needs no natural period");
        assert_eq!(edges[0].curve.range_tuple(), (2.0, 4.0));
        assert_eq!(edges[1].curve.range_tuple(), (4.0, 6.0));
        assert_eq!(vertices[1], Point3::new(1.0, 1.0, 0.0));
        for (edge, parameter, expected) in [
            (0, 2.37, Point3::new(0.37, 0.0, 0.0)),
            (0, 3.61, Point3::new(1.0, 0.61, 0.0)),
            (1, 4.27, Point3::new(0.73, 1.0, 0.0)),
            (1, 5.83, Point3::new(0.0, 0.17, 0.0)),
        ] {
            assert!(edges[edge].curve.subs(parameter).near(&expected));
        }
    }
}
