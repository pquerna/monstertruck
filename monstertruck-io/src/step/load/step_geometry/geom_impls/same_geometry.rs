//! Lifting a concrete `monstertruck-geometry` carrier back INTO the STEP
//! enums, and the enum-level predicates that ride along with it.

use super::*;

impl ToSameGeometry<Curve2D> for Line<Point2> {
    #[inline]
    fn to_same_geometry(&self) -> Curve2D { Curve2D::Line(*self) }
}

impl ToSameGeometry<Curve2D> for Processor<TrimmedCurve<UnitCircle<Point2>>, Matrix3> {
    #[inline]
    fn to_same_geometry(&self) -> Curve2D { Curve2D::Conic(Conic2D::Ellipse(*self)) }
}

impl ToSameGeometry<Curve2D> for BsplineCurve<Point2> {
    #[inline]
    fn to_same_geometry(&self) -> Curve2D { Curve2D::BsplineCurve(self.clone()) }
}

impl ToSameGeometry<Curve3D> for Line<Point3> {
    #[inline]
    fn to_same_geometry(&self) -> Curve3D { Curve3D::Line(*self) }
}

impl ToSameGeometry<Curve3D> for Processor<TrimmedCurve<UnitCircle<Point3>>, Matrix4> {
    #[inline]
    fn to_same_geometry(&self) -> Curve3D { Curve3D::Conic(Conic3D::Ellipse(*self)) }
}

impl ToSameGeometry<Curve3D> for BsplineCurve<Point3> {
    #[inline]
    fn to_same_geometry(&self) -> Curve3D { Curve3D::BsplineCurve(self.clone()) }
}

impl Conic3D {
    pub fn posture(&self) -> Matrix4 {
        match self {
            Conic3D::Ellipse(processor) => *processor.transform(),
            Conic3D::Hyperbola(processor) => *processor.transform(),
            Conic3D::Parabola(processor) => *processor.transform(),
        }
    }
}

impl IncludeCurve<Curve3D> for Plane {
    fn include(&self, curve: &Curve3D) -> bool {
        match curve {
            Curve3D::Line(line) => self.include(line),
            Curve3D::BsplineCurve(bsp) => self.include(bsp),
            Curve3D::NurbsCurve(bsp) => self.include(bsp),
            Curve3D::Conic(conic) => {
                let mat = conic.posture();
                let axis = mat.z.truncate();
                axis.cross(self.normal()).so_small()
            }
            Curve3D::Polyline(poly) => poly
                .iter()
                .all(|p| self.search_parameter(*p, None, 1).is_some()),
            Curve3D::ParameterCurve(curve) => matches!(
                curve.surface(),
                Surface::ElementarySurface(ElementarySurface::Plane(surface)) if self == surface
            ),
            Curve3D::SurfaceCurve(curve) => self.include(curve.leader()),
            Curve3D::IntersectionCurve(curve) => self.include(curve.leader().as_ref()),
            Curve3D::TrimmedCurve(curve) => curve.transform().invert().is_some_and(|transform| {
                self.transformed(transform)
                    .include(curve.entity().curve().as_ref())
            }),
        }
    }
}

impl ToSameGeometry<Surface> for Plane {
    #[inline]
    fn to_same_geometry(&self) -> Surface {
        Surface::ElementarySurface(ElementarySurface::Plane(*self))
    }
}

impl ToSameGeometry<Surface> for ExtrusionSurface<Curve3D, Vector3> {
    #[inline]
    fn to_same_geometry(&self) -> Surface {
        Surface::SweepSurface(SweepSurface::ExtrusionSurface(
            ExtrusionSurface::by_extrusion(
                Box::new(self.entity_curve().clone()),
                self.extruding_vector(),
            ),
        ))
    }
}

impl ToSameGeometry<Surface> for RevolutionSurface<Curve3D> {
    #[inline]
    fn to_same_geometry(&self) -> Surface {
        let default = || {
            let (curve, origin, axis) = (self.entity_curve().clone(), self.origin(), self.axis());
            let processor = Processor::new(RevolutionSurface::by_revolution(
                Box::new(curve),
                origin,
                axis,
            ));
            Surface::SweepSurface(SweepSurface::RevolutionSurface(processor))
        };
        match self.entity_curve() {
            Curve3D::Line(line) => {
                let &Line(p, q) = line;
                let v = q - p;
                let axis = self.axis();
                if v.cross(axis).so_small() {
                    let o = self.origin();
                    let origin = o + (p - o).dot(axis) * axis;
                    let revo = RevolutionSurface::by_revolution(*line, origin, axis);
                    let processor = Processor::new(revo);
                    Surface::ElementarySurface(ElementarySurface::CylindricalSurface(processor))
                } else {
                    default()
                }
            }
            Curve3D::SurfaceCurve(_) => default(),
            Curve3D::IntersectionCurve(_) => default(),
            _ => default(),
        }
    }
}

#[test]
fn to_same_geometry_revolution_of_axis_parallel_line_is_uninverted_cylinder() {
    let axis = Vector3::unit_z();
    let center = Point3::new(0.5, -0.5, 0.0);
    let radius = 2.0;
    let p = center + radius * Vector3::unit_x();
    let line = Line(p, p + axis);
    let revolution = RevolutionSurface::by_revolution(Curve3D::Line(line), center, axis);

    let surface = revolution.to_same_geometry();

    match surface {
        Surface::ElementarySurface(ElementarySurface::CylindricalSurface(processor)) => {
            assert!(
                processor.orientation(),
                "cylinder from axis-parallel line must not be inverted.",
            );

            let entity = processor.entity();
            assert_eq!(entity.axis(), axis, "cylinder axis must match.");

            let origin = entity.origin();
            assert_near!(origin.z, 0.0);
            let in_plane = origin - center;
            assert_near!(in_plane.dot(axis), 0.0);

            let profile_distance = (line.0 - origin).magnitude();
            assert_near!(profile_distance, radius);
        }
        other => panic!("expected cylindrical surface, got {other:?}"),
    }
}

#[test]
fn to_same_geometry_revolution_preserves_profile_gauge_and_oriented_normals() {
    let mut reflected_profile = Processor::with_transform(
        TrimmedCurve::new(UnitCircle::<Point3>::new(), (0.2, 1.1)),
        Matrix4::from_cols(
            Vector4::new(-1.0, 0.0, 0.0, 0.0),
            Vector4::new(0.0, 0.0, 1.0, 0.0),
            Vector4::new(0.0, -1.0, 0.0, 0.0),
            Vector4::new(3.0, 0.0, 2.0, 1.0),
        ),
    );
    reflected_profile.invert();
    let profiles = [
        Curve3D::Line(Line(Point3::new(5.0, 0.0, 0.0), Point3::new(3.0, 0.0, 8.0))),
        Curve3D::Line(Line(Point3::new(3.0, 0.0, 8.0), Point3::new(0.0, 0.0, 8.0))),
        Curve3D::BsplineCurve(BsplineCurve::new(
            KnotVector::bezier_knot(2),
            vec![
                Point3::new(3.0, 0.0, 1.0),
                Point3::new(4.0, 0.0, 2.0),
                Point3::new(2.0, 0.0, 4.0),
            ],
        )),
        Curve3D::Conic(Conic3D::Ellipse(reflected_profile)),
    ];
    for profile in profiles {
        for profile in [profile.clone(), profile.inverse()] {
            for axis in [Vector3::unit_z(), -Vector3::unit_z()] {
                let original = RevolutionSurface::by_revolution(
                    profile.clone(),
                    Point3::new(0.5, -0.5, 0.0),
                    axis,
                );
                let lifted: Surface = original.to_same_geometry();
                assert_eq!(lifted.parameter_range(), original.parameter_range());
                assert_eq!(lifted.period_u(), original.period_u());
                assert_eq!(lifted.period_v(), original.period_v());
                let (start, end) = profile.range_tuple();
                for (fraction, v) in [(0.1, 0.2), (0.37, 1.1), (0.9, 4.7)] {
                    let u = start + fraction * (end - start);
                    assert_near!(lifted.evaluate(u, v), original.evaluate(u, v));
                    assert_near!(lifted.derivative_u(u, v), original.derivative_u(u, v));
                    assert_near!(lifted.derivative_v(u, v), original.derivative_v(u, v));
                    assert_near!(lifted.normal(u, v), original.normal(u, v));
                }
            }
        }
    }
}

#[test]
fn revolved_step_frustum_keeps_outward_caps_and_walls_through_step_roundtrip() {
    use crate::step::{
        load::Table,
        save::{CompleteStepDisplay, StepModel},
    };
    use monstertruck_modeling::builder;
    monstertruck_topology::prelude!(Point3, Curve3D, Surface);

    fn assert_outward_material(solid: &Solid) {
        assert_eq!(solid.boundaries().len(), 1);
        solid.boundaries()[0].check_solid_boundary().unwrap();
        let mut caps = [0; 2];
        let mut walls = 0;
        for face in solid.face_iter() {
            let surface = face.oriented_surface();
            if face.edge_iter().all(|edge| edge.front().point().z == 8.0) {
                assert!(surface.normal(0.31, 0.43).z > 0.999);
                caps[1] += 1;
            } else if face.edge_iter().all(|edge| edge.front().point().z == 0.0) {
                assert!(surface.normal(0.31, 0.43).z < -0.999);
                caps[0] += 1;
            } else {
                let point = surface.evaluate(0.31, 0.43);
                let radial = Vector3::new(point.x, point.y, 0.0);
                assert!(surface.normal(0.31, 0.43).dot(radial) > 0.0);
                walls += 1;
            }
        }
        assert_eq!(caps, [4, 4]);
        assert_eq!(walls, 4);
    }

    let vertices = builder::vertices([
        (0.0, 0.0, 8.0),
        (3.0, 0.0, 8.0),
        (5.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ]);
    let meridian: Wire = vertices
        .windows(2)
        .map(|pair| builder::line(&pair[0], &pair[1]))
        .collect();
    let solid = Solid::new(vec![builder::revolve_wire(
        &meridian,
        Point3::origin(),
        Vector3::unit_z(),
        builder::SweepAngle::Closed,
        4,
    )]);
    assert_outward_material(&solid);
    let compressed = solid.compress();
    let step =
        CompleteStepDisplay::new(StepModel::from(&compressed), Default::default()).to_string();
    let table = Table::from_step(&step).unwrap();
    assert_eq!(table.manifold_solid_brep.len(), 1);
    let (compressed, reports) = table
        .to_compressed_solid_reported(table.manifold_solid_brep.values().next().unwrap())
        .unwrap();
    assert!(reports.iter().all(|report| report.is_lossless()));
    let restored = Solid::extract(compressed).unwrap();
    assert_outward_material(&restored);
}

#[test]
fn to_same_geometry_2d_line_round_trip() {
    let line = Line(Point2::new(0.0, 0.0), Point2::new(2.0, 1.0));
    let curve: Curve2D = line.to_same_geometry();
    match curve {
        Curve2D::Line(rebuilt) => assert_eq!(rebuilt, line),
        other => panic!("expected Curve2D::Line, got {other:?}"),
    }
}

#[test]
fn to_same_geometry_2d_ellipse_wraps_in_conic() {
    let scale = Matrix3::from_nonuniform_scale(2.0, 3.0);
    let arc = Processor::with_transform(
        TrimmedCurve::new(UnitCircle::<Point2>::new(), (0.0, TAU)),
        scale,
    );
    let curve: Curve2D = arc.to_same_geometry();
    match curve {
        Curve2D::Conic(Conic2D::Ellipse(rebuilt)) => assert_eq!(rebuilt, arc),
        other => panic!("expected Curve2D::Conic(Conic2D::Ellipse), got {other:?}"),
    }
}

#[test]
fn to_same_geometry_2d_bspline_curve_round_trip() {
    let knots = KnotVector::uniform_knot(2, 2);
    let control = vec![
        Point2::new(0.0, 0.0),
        Point2::new(1.0, 1.0),
        Point2::new(2.0, 0.0),
        Point2::new(3.0, 1.0),
    ];
    let spline = BsplineCurve::new(knots, control);
    let curve: Curve2D = spline.to_same_geometry();
    match curve {
        Curve2D::BsplineCurve(rebuilt) => assert_eq!(rebuilt, spline),
        other => panic!("expected Curve2D::BsplineCurve, got {other:?}"),
    }
}

#[test]
fn builder() {
    use monstertruck_meshing::prelude::*;
    use monstertruck_modeling::builder;
    monstertruck_topology::prelude!(Point3, Curve3D, Surface);

    // cube
    let v = builder::vertices([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]);
    let e = builder::line(&v[0], &v[1]);
    let f = builder::extrude(&e, Vector3::unit_y());
    let cube: Solid = builder::extrude(&f, Vector3::unit_z());
    let mut poly = cube.triangulation(0.1).to_polygon();
    poly.put_together_same_attrs(1.0e-3).remove_unused_attrs();
    assert_eq!(poly.shell_condition(), ShellCondition::Closed);

    // cylinder
    let v = builder::vertices([(1.0, 0.0, 1.0), (1.0, 0.0, 0.0)]);
    let e = builder::line(&v[0], &v[1]);
    let mut shell = builder::revolve(
        &e,
        Point3::origin(),
        Vector3::unit_z(),
        builder::SweepAngle::Closed,
        2,
    );
    let boundaries = shell.extract_boundaries();
    assert_eq!(boundaries.len(), 2);
    shell.push(builder::try_attach_plane([boundaries[0].inverse()]).unwrap());
    shell.push(builder::try_attach_plane([boundaries[1].inverse()]).unwrap());
    let cylinder = Solid::new(vec![shell]);
    let mut poly = cylinder.triangulation(0.1).to_polygon();
    poly.put_together_same_attrs(1.0e-3).remove_unused_attrs();
    assert_eq!(poly.shell_condition(), ShellCondition::Closed);

    // torus
    let v = builder::vertex((1.5, 0.0, 0.0));
    let w = builder::revolve(
        &v,
        Point3::new(1.0, 0.0, 0.0),
        Vector3::unit_y(),
        builder::SweepAngle::Closed,
        2,
    );
    let f = builder::try_attach_plane([w]).unwrap();
    let torus: Solid = builder::revolve(
        &f,
        Point3::origin(),
        Vector3::unit_z(),
        builder::SweepAngle::Closed,
        2,
    );
    let mut poly = torus.triangulation(0.1).to_polygon();
    poly.put_together_same_attrs(1.0e-3).remove_unused_attrs();
    assert_eq!(poly.shell_condition(), ShellCondition::Closed);

    // cylinder hole
    let v = builder::vertex((-1.0, -1.0, -1.0));
    let e = builder::extrude(&v, 2.0 * Vector3::unit_x());
    let f = builder::extrude(&e, 2.0 * Vector3::unit_y());
    let s: Solid = builder::extrude(&f, 2.0 * Vector3::unit_z());
    let mut shell = s.into_boundaries().pop().unwrap();
    let line = builder::line(
        &builder::vertex((0.5, 0.0, 1.0)),
        &builder::vertex((0.5, 0.0, -1.0)),
    );
    let hole = builder::revolve(
        &line,
        Point3::origin(),
        -Vector3::unit_z(),
        builder::SweepAngle::Closed,
        2,
    );
    let boundary = hole.extract_boundaries();
    assert_eq!(boundary.len(), 2);
    if boundary[0][0].front().point().z < 0.0 {
        let _ = shell[0].add_boundary(boundary[0].inverse());
        let _ = shell[5].add_boundary(boundary[1].inverse());
    } else {
        let _ = shell[0].add_boundary(boundary[1].inverse());
        let _ = shell[5].add_boundary(boundary[0].inverse());
    }
    shell.extend(hole);
    let solid = Solid::new(vec![shell]);
    let mut poly = solid.triangulation(0.1).to_polygon();
    poly.put_together_same_attrs(1.0e-3).remove_unused_attrs();
    assert_eq!(poly.shell_condition(), ShellCondition::Closed);
}
