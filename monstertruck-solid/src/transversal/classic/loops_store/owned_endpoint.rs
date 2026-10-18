//! Endpoint identities granted by a complete immutable original branch proof.
//! Proximity only proposes a source edge; actual bounded native endpoints, the
//! whole original carriers and opposing original trim must all authorize reuse.

use super::*;

/// Whole original edge receipt, directed by the bounded native intersection.
pub(in crate::transversal::classic::loops_store) struct OwnedEndpointBinding<C> {
    pub(in crate::transversal::classic::loops_store) native_edge: Edge<Point3, C>,
    pub(in crate::transversal::classic::loops_store) polygon_edge: Edge<Point3, PolylineCurve>,
}

pub(super) fn projected_point<S>(
    intersection: &IntersectionCurve<PolylineCurve, S, S>,
    parameter: f64,
    work: Option<&BooleanWork<'_>>,
) -> std::result::Result<Option<Point3>, AttachmentFailure>
where
    S: ParametricSurface3D + SearchNearestParameter<SurfaceParameter, Point = Point3>,
{
    if let Some(work) = work {
        work.check()?;
    }
    let (front, back) = intersection.range_tuple();
    if !parameter.is_finite() || parameter < front || parameter > back {
        return Ok(None);
    }
    let Some((point, uv0, uv1)) = intersection.search_triple(parameter, 100) else {
        return Ok(None);
    };
    let tangent = intersection
        .surface0()
        .normal(uv0.x, uv0.y)
        .cross(intersection.surface1().normal(uv1.x, uv1.y));
    Ok(
        ([point.x, point.y, point.z, tangent.x, tangent.y, tangent.z]
            .into_iter()
            .all(f64::is_finite)
            && tangent.magnitude2() > 0.0
            && point.near(&intersection.surface0().evaluate(uv0.x, uv0.y))
            && point.near(&intersection.surface1().evaluate(uv1.x, uv1.y)))
        .then_some(point),
    )
}

/// This is a numerical membership check, not endpoint identity authority. Its
/// caller must first prove the complete original source branch and material trim.
/// In particular no nearby interior inverse parameter is rounded to an endpoint.
pub(super) fn bounded_endpoints<S>(
    intersection: &IntersectionCurve<PolylineCurve, S, S>,
    directed: [Point3; 2],
    work: Option<&BooleanWork<'_>>,
) -> std::result::Result<bool, AttachmentFailure>
where
    S: ParametricSurface3D + SearchNearestParameter<SurfaceParameter, Point = Point3>,
{
    let (front, back) = intersection.range_tuple();
    if !front.is_finite() || !back.is_finite() || front >= back {
        return Ok(false);
    }
    for (endpoint, target) in [front, back].into_iter().zip(directed) {
        // Identity comes from the caller's whole immutable owned branch and
        // trim proof, not from a nearest-point solver's returned scalar.
        // Evaluate the actual bounded endpoint at its existing native
        // consistency bound; never round an interior inverse parameter.
        let Some(actual) = projected_point(intersection, endpoint, work)? else {
            return Ok(false);
        };
        if !actual.near(&target) {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(in crate::transversal::classic::loops_store) fn early_owned_endpoints<C, S>(
    intersection: &IntersectionCurve<PolylineCurve, S, S>,
    stores: [OriginalLoopPair<'_, C>; 2],
    original_boundaries: [&[Wire<Point3, C>]; 2],
    work: Option<&BooleanWork<'_>>,
) -> std::result::Result<Option<OwnedEndpointBinding<C>>, AttachmentFailure>
where
    C: Clone
        + ParametricCurve3D
        + BoundedCurve
        + SearchParameter<CurveParameter, Point = Point3>
        + TryIntoHomogeneousBsplineCurve,
    S: ParametricSurface3D
        + SearchNearestParameter<SurfaceParameter, Point = Point3>
        + TryIntoAnalyticSurfaceKind
        + TryIntoHomogeneousBsplineSurface,
{
    if intersection.leader().len() < 2 {
        return Ok(None);
    }
    let (front, back) = intersection.range_tuple();
    let Some(first) = projected_point(intersection, front, work)? else {
        return Ok(None);
    };
    let Some(last) = projected_point(intersection, back, work)? else {
        return Ok(None);
    };
    let planes = OriginalPlane::from_surface(intersection.surface0())
        .zip(OriginalPlane::from_surface(intersection.surface1()))
        .map(|(first, second)| [first, second]);
    let mut selected: Option<OwnedEndpointBinding<C>> = None;
    for (side, (native, polygon)) in stores.into_iter().enumerate() {
        if native.len() != polygon.len() {
            return Err("intersection_original_overlap_polygon_occurrence".into());
        }
        for (wire, polygon_wire) in native.iter().zip(polygon.iter()) {
            if wire.len() != polygon_wire.len() {
                return Err("intersection_original_overlap_polygon_occurrence".into());
            }
            for (edge, polygon_edge) in wire.iter().zip(polygon_wire.iter()) {
                if let Some(work) = work {
                    work.check()?;
                }
                if edge.front() == edge.back() {
                    continue;
                }
                // A current cut or generated edge is not an immutable original
                // coedge merely because it still exposes literal coefficients.
                let mut owned = false;
                for original in original_boundaries[side]
                    .iter()
                    .flat_map(|wire| wire.iter())
                {
                    if let Some(work) = work {
                        work.check()?;
                    }
                    if original.id() == edge.id()
                        && original.absolute_front() == edge.absolute_front()
                        && original.absolute_back() == edge.absolute_back()
                    {
                        owned = true;
                        break;
                    }
                }
                if !owned {
                    continue;
                }
                let forward = first.near(&edge.front().point()) && last.near(&edge.back().point());
                let reverse = first.near(&edge.back().point()) && last.near(&edge.front().point());
                if forward == reverse {
                    continue;
                }
                let vertices = if forward {
                    [edge.front(), edge.back()]
                } else {
                    [edge.back(), edge.front()]
                };
                let directed = vertices.map(Vertex::point);
                let proved = edge.with_curve(|curve| {
                    let absolute = [edge.absolute_front().point(), edge.absolute_back().point()];
                    if let Some(planes) = &planes {
                        if !proves_overlap(curve, absolute, directed, intersection, planes, work)? {
                            return Ok(false);
                        }
                        let Some(source) = curve.original_curve_coefficients() else {
                            return Ok(false);
                        };
                        if !original_face::contains_curve(
                            source,
                            original_boundaries[1 - side],
                            &planes[1 - side],
                            work,
                        )? {
                            return Ok(false);
                        }
                        Ok(true)
                    } else {
                        sphere_boundary::proves_overlap(
                            curve,
                            absolute,
                            directed,
                            intersection,
                            (side, original_boundaries[1 - side]),
                            work,
                        )
                    }
                })?;
                if !proved {
                    continue;
                }
                let orientation = if forward {
                    edge.orientation()
                } else {
                    !edge.orientation()
                };
                let polygon_orientation = if forward {
                    polygon_edge.orientation()
                } else {
                    !polygon_edge.orientation()
                };
                if let Some(old) = selected.as_mut() {
                    if old.native_edge.id() != edge.id()
                        || old.polygon_edge.id() != polygon_edge.id()
                        || old.native_edge.orientation() != orientation
                        || old.polygon_edge.orientation() != polygon_orientation
                    {
                        return Err("intersection_original_overlap_ambiguous".into());
                    }
                } else {
                    selected = Some(OwnedEndpointBinding {
                        native_edge: if forward {
                            edge.clone()
                        } else {
                            edge.inverse()
                        },
                        polygon_edge: if forward {
                            polygon_edge.clone()
                        } else {
                            polygon_edge.inverse()
                        },
                    });
                }
            }
        }
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use monstertruck_modeling as modeling;

    struct Fixture {
        source: modeling::Wire,
        native: Loops<Point3, modeling::Curve>,
        polygon: Loops<Point3, PolylineCurve>,
        opposite: modeling::Wire,
        points: Vec<Point3>,
    }

    impl Fixture {
        fn new() -> Self {
            let weight = 0.5_f64.sqrt();
            let curve = modeling::Curve::NurbsCurve(NurbsCurve::new(BsplineCurve::new(
                KnotVector::from(vec![-2.0, -2.0, -2.0, 5.0, 5.0, 5.0]),
                vec![
                    Vector4::new(5.0, 0.0, 0.0, 1.0),
                    Vector4::new(5.0 * weight, 5.0 * weight, 0.0, weight),
                    Vector4::new(0.0, 5.0, 0.0, 1.0),
                ],
            )));
            let points = (0..=32)
                .map(|index| curve.evaluate(-2.0 + 7.0 * f64::from(index) / 32.0))
                .collect::<Vec<_>>();
            let ends = Vertex::from_points([curve.front(), curve.back()]);
            let source: modeling::Wire =
                std::iter::once(Edge::new(&ends[0], &ends[1], curve)).collect();
            let native =
                std::iter::once(BoundaryWire::new(source.clone(), ShapesOpStatus::Unknown))
                    .collect();
            let polygon_ends = Vertex::from_points([ends[0].point(), ends[1].point()]);
            let polygon_wire = std::iter::once(Edge::new(
                &polygon_ends[0],
                &polygon_ends[1],
                PolylineCurve(points.clone()),
            ))
            .collect();
            let polygon =
                std::iter::once(BoundaryWire::new(polygon_wire, ShapesOpStatus::Unknown)).collect();
            Self {
                source,
                native,
                polygon,
                opposite: Self::square(6.0),
                points,
            }
        }

        fn square(extent: f64) -> modeling::Wire {
            let vertices = Vertex::from_points([
                Point3::new(-extent, -extent, 0.0),
                Point3::new(extent, -extent, 0.0),
                Point3::new(extent, extent, 0.0),
                Point3::new(-extent, extent, 0.0),
            ]);
            (0..4)
                .map(|index| modeling::builder::line(&vertices[index], &vertices[(index + 1) % 4]))
                .collect()
        }

        fn bind(
            &self,
            points: Vec<Point3>,
        ) -> std::result::Result<Option<OwnedEndpointBinding<modeling::Curve>>, AttachmentFailure>
        {
            let branch = IntersectionCurve::new(
                modeling::Surface::SphericalSurface(Processor::new(Sphere::new(
                    Point3::origin(),
                    5.0,
                ))),
                modeling::Surface::Plane(Plane::xy()),
                PolylineCurve(points),
            );
            let empty_native = std::iter::empty().collect();
            let empty_polygon = std::iter::empty().collect();
            early_owned_endpoints(
                &branch,
                [
                    (&self.native, &self.polygon),
                    (&empty_native, &empty_polygon),
                ],
                [
                    std::slice::from_ref(&self.source),
                    std::slice::from_ref(&self.opposite),
                ],
                None,
            )
        }
    }

    #[test]
    fn owned_endpoint_identity_does_not_depend_on_nearest_inverse_returning_endpoint_scalar() {
        let fixture = Fixture::new();
        let mut points = fixture.points.clone();
        // The known bounded endpoint remains within native consistency, while
        // nearest-point ranking genuinely proposes a tiny interior parameter.
        points[0] = Point3::new(5.0, -1.0e-9, 0.0);
        let branch = IntersectionCurve::new(
            modeling::Surface::SphericalSurface(Processor::new(Sphere::new(Point3::origin(), 5.0))),
            modeling::Surface::Plane(Plane::xy()),
            PolylineCurve(points.clone()),
        );
        let proposal = periodic_loops::intersection_parameter(
            &branch,
            fixture.source[0].front().point(),
            None,
            None,
        )
        .expect("actual bounded source endpoint consistency");
        assert!(proposal > branch.range_tuple().0);
        let binding = fixture.bind(points).unwrap().unwrap();
        assert_eq!(binding.native_edge.id(), fixture.source[0].id());
        assert_eq!(binding.native_edge.front(), fixture.source[0].front());
        assert_eq!(binding.polygon_edge.id(), fixture.polygon[0][0].id());
    }

    #[test]
    fn complete_owned_sphere_arc_authorizes_declared_vertices_not_numerical_proposals() {
        let fixture = Fixture::new();
        let mut points = fixture.points.clone();
        points[0] = Point3::new(5.0, 1.0e-16, 0.0);
        *points.last_mut().unwrap() = Point3::new(1.0e-16, 5.0, 0.0);
        assert_ne!(points[0], fixture.source[0].front().point());
        let binding = fixture.bind(points.clone()).unwrap().unwrap();
        assert_eq!(binding.native_edge.id(), fixture.source[0].id());
        assert_eq!(binding.polygon_edge.id(), fixture.polygon[0][0].id());
        assert_eq!(binding.native_edge.front(), fixture.source[0].front());
        assert_eq!(binding.native_edge.back(), fixture.source[0].back());
        assert_eq!(
            binding.native_edge.with_curve(BoundedCurve::range_tuple),
            (-2.0, 5.0)
        );
        points.reverse();
        let reversed = fixture.bind(points).unwrap().unwrap();
        assert_eq!(reversed.native_edge.front(), fixture.source[0].back());
        assert_eq!(reversed.native_edge.back(), fixture.source[0].front());
        assert_eq!(reversed.polygon_edge.front(), fixture.polygon[0][0].back());
    }

    #[test]
    fn complete_carrier_identity_does_not_supply_a_missing_bounded_native_endpoint() {
        let fixture = Fixture::new();
        let points = fixture.points[1..].to_vec();
        assert!(fixture.bind(points).unwrap().is_none());
        let points = fixture.points[..32].to_vec();
        assert!(fixture.bind(points).unwrap().is_none());
    }

    #[test]
    fn owned_sphere_endpoints_require_complete_opposing_original_material_support() {
        let mut fixture = Fixture::new();
        fixture.opposite = Fixture::square(4.0);
        assert!(fixture.bind(fixture.points.clone()).unwrap().is_none());
    }

    #[test]
    fn literal_lookalike_curve_is_not_an_owned_original_coedge() {
        let mut fixture = Fixture::new();
        let edge = &fixture.source[0];
        fixture.source =
            std::iter::once(Edge::new(edge.front(), edge.back(), edge.curve())).collect();
        assert_ne!(fixture.source[0].id(), fixture.native[0][0].id());
        assert!(fixture.bind(fixture.points.clone()).unwrap().is_none());
    }
}

#[cfg(test)]
mod ambiguity_tests {
    use super::*;
    use monstertruck_modeling as modeling;

    #[test]
    fn whole_plane_branch_with_two_distinct_owned_sources_has_no_endpoint_authority() {
        let vertices = Vertex::from_points([Point3::origin(), Point3::new(1.0, 0.0, 0.0)]);
        let first = modeling::builder::line(&vertices[0], &vertices[1]);
        let second = modeling::builder::line(&vertices[0], &vertices[1]);
        let source: modeling::Wire = [first, second.inverse()].into_iter().collect();
        let native: Loops<_, _> =
            std::iter::once(BoundaryWire::new(source.clone(), ShapesOpStatus::Unknown)).collect();
        let polygon_vertices =
            Vertex::from_points(vertices.iter().map(Vertex::point).collect::<Vec<_>>());
        let polygon_curve = PolylineCurve(vec![vertices[0].point(), vertices[1].point()]);
        let polygon_wire = [
            Edge::new(
                &polygon_vertices[0],
                &polygon_vertices[1],
                polygon_curve.clone(),
            ),
            Edge::new(
                &polygon_vertices[0],
                &polygon_vertices[1],
                polygon_curve.clone(),
            )
            .inverse(),
        ]
        .into_iter()
        .collect();
        let polygon =
            std::iter::once(BoundaryWire::new(polygon_wire, ShapesOpStatus::Unknown)).collect();
        let opposite_vertices = Vertex::from_points([
            Point3::new(-2.0, 0.0, -2.0),
            Point3::new(2.0, 0.0, -2.0),
            Point3::new(2.0, 0.0, 2.0),
            Point3::new(-2.0, 0.0, 2.0),
        ]);
        let opposite: modeling::Wire = (0..4)
            .map(|index| {
                modeling::builder::line(
                    &opposite_vertices[index],
                    &opposite_vertices[(index + 1) % 4],
                )
            })
            .collect();
        let branch = IntersectionCurve::new(
            modeling::Surface::Plane(Plane::xy()),
            modeling::Surface::Plane(Plane::new(
                Point3::origin(),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 0.0, 1.0),
            )),
            polygon_curve,
        );
        let empty_native = std::iter::empty().collect();
        let empty_polygon = std::iter::empty().collect();
        assert!(matches!(
            early_owned_endpoints(
                &branch,
                [(&native, &polygon), (&empty_native, &empty_polygon)],
                [
                    std::slice::from_ref(&source),
                    std::slice::from_ref(&opposite)
                ],
                None,
            ),
            Err(AttachmentFailure::Geometry(
                "intersection_original_overlap_ambiguous"
            )),
        ));
    }
}
