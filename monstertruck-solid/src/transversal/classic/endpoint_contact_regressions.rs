use super::*;
use monstertruck_modeling as modeling;

fn circular_stores() -> (
    LoopsStore<Point3, modeling::Curve>,
    LoopsStore<Point3, PolylineCurve>,
    modeling::Surface,
) {
    let wire: modeling::Wire = modeling::primitive::circle(
        Point3::new(5.0, 0.0, 0.0),
        Point3::origin(),
        Vector3::unit_z(),
        2,
    );
    let face = modeling::builder::try_attach_plane(vec![wire]).expect("native bounded circle");
    let polygon_face = face.mapped(
        |point| *point,
        |curve| PolylineCurve::from_curve(curve, curve.range_tuple(), 1.0e-5),
        |_| (),
    );
    (
        std::iter::once(&face).collect(),
        std::iter::once(&polygon_face).collect(),
        face.surface(),
    )
}

fn rectangle_stores(
    carrier: &modeling::Surface,
) -> (
    LoopsStore<Point3, modeling::Curve>,
    LoopsStore<Point3, PolylineCurve>,
) {
    let points = [(-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0)]
        .map(|(u, v)| carrier.evaluate(u, v));
    let vertices = Vertex::from_points(points);
    let wire: modeling::Wire = (0..4)
        .map(|index| modeling::builder::line(&vertices[index], &vertices[(index + 1) % 4]))
        .collect();
    let face = modeling::Face::new(vec![wire], carrier.clone());
    assert!(face.is_geometric_consistent());
    let polygon_face = face.mapped(
        |point| *point,
        |curve| PolylineCurve::from_curve(curve, curve.range_tuple(), 1.0e-5),
        |_| (),
    );
    (
        std::iter::once(&face).collect(),
        std::iter::once(&polygon_face).collect(),
    )
}

fn original_edges(
    stores: [&LoopsStore<Point3, modeling::Curve>; 2],
) -> [impl Iterator<Item = &Edge<Point3, modeling::Curve>>; 2] {
    stores.map(|store| store[0].iter().flat_map(|wire| wire.iter()))
}

#[test]
fn an_inscribed_triangle_seed_is_attached_only_at_the_original_native_circle_contact() {
    let (geom, poly, first) = circular_stores();
    let seed = Point3::new(-4.013745556503457, -2.981555130284838, 0.0);
    let angle = 1.0_f64.to_radians();
    let transverse = angle.sin() / 2.0_f64.sqrt();
    let normal = Vector3::new(-transverse, transverse, angle.cos());
    let origin = Point3::from_vec(normal * (0.1 - 5.0 * angle.sin()));
    let x = Vector3::unit_x().cross(normal).normalize();
    let second = modeling::Surface::Plane(Plane::new(origin, origin + x, origin + normal.cross(x)));
    let (clip_geom, clip_poly) = rectangle_stores(&second);
    let point = native_endpoint_contact(
        original_edges([&geom, &clip_geom]),
        [&first, &second],
        seed,
        None,
    )
    .expect("one common bounded original trim contact")
    .point;
    assert!(point.distance(seed) > 1.0e-5);
    let polygon_vertex = Vertex::new(seed);
    let native_vertex = Vertex::new(point);
    assert!(
        boundary_attachment(
            &geom,
            &poly,
            0,
            &polygon_vertex,
            &native_vertex,
            [&first, &second]
        )
        .expect("original circle attachment")
        .is_some()
    );
    assert!(
        boundary_attachment(
            &clip_geom,
            &clip_poly,
            0,
            &polygon_vertex,
            &native_vertex,
            [&second, &first]
        )
        .expect("clipping rectangle membership")
        .is_none()
    );
    assert!((point.to_vec().magnitude() - 5.0).abs() < TOLERANCE);
    assert!(!point.near(&seed));
    for carrier in [&first, &second] {
        let uv = carrier
            .search_parameter(point, None, 100)
            .expect("original carrier");
        assert!(carrier.evaluate(uv.0, uv.1).near(&point));
    }
}

#[test]
fn mesh_proximity_alone_cannot_attach_an_unrelated_native_carrier() {
    let (geom, _, first) = circular_stores();
    let seed = Point3::new(-4.013745556503457, -2.981555130284838, 0.0);
    let unrelated = modeling::Surface::Plane(Plane::new(
        Point3::new(0.0, 0.0, 2.0),
        Point3::new(1.0, 0.0, 2.0),
        Point3::new(0.0, 1.0, 2.0),
    ));
    let (other, _) = rectangle_stores(&unrelated);
    assert!(
        native_endpoint_contact(
            original_edges([&geom, &other]),
            [&first, &unrelated],
            seed,
            None
        )
        .is_err()
    );
}

#[test]
fn a_nonunique_native_boundary_contact_is_not_an_arbitrary_endpoint() {
    let (geom, _, first) = circular_stores();
    let second = modeling::Surface::Plane(Plane::new(
        Point3::origin(),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
    ));
    let (other, _) = rectangle_stores(&second);
    assert!(
        native_endpoint_contact(
            original_edges([&geom, &other]),
            [&first, &second],
            Point3::origin(),
            None
        )
        .is_err()
    );
}

#[test]
fn shallow_crossings_reconcile_independently_seeded_native_contacts_in_world_distance() {
    let (geom, _, first) = circular_stores();
    let angle = 1.0_f64.to_radians();
    let normal = Vector3::new(
        -angle.sin() / 2.0_f64.sqrt(),
        angle.sin() / 2.0_f64.sqrt(),
        angle.cos(),
    );
    let offset = 0.1 - 5.0 * angle.sin();
    let origin = Point3::from_vec(normal * offset);
    let x = Vector3::unit_x().cross(normal).normalize();
    let second = modeling::Surface::Plane(Plane::new(origin, origin + x, origin + normal.cross(x)));
    let (other, _) = rectangle_stores(&second);
    let in_plane_normal = Vector3::new(normal.x, normal.y, 0.0).normalize();
    let chord_center = in_plane_normal * (offset / angle.sin());
    let chord_direction = Vector3::new(in_plane_normal.y, -in_plane_normal.x, 0.0);
    let expected = Point3::from_vec(
        chord_center + chord_direction * (25.0 - chord_center.magnitude2()).sqrt(),
    );
    let seeds = [
        Point3::new(2.981569188592856, 4.013756989857972, 0.0),
        Point3::new(2.981566837664532, 4.013758736214613, 0.0),
    ];
    let contacts = seeds.map(|seed| {
        native_endpoint_contact(
            original_edges([&geom, &other]),
            [&first, &second],
            seed,
            None,
        )
        .expect("world-conditioned native contact")
        .point
    });
    assert!(contacts[0].near(&contacts[1]));
    assert!(contacts.into_iter().all(|point| point.near(&expected)));
}

#[test]
fn native_endpoint_kinds_and_replay_use_world_distance_not_parameter_nearness() {
    let points = [
        Point3::origin(),
        Point3::new(100.0, 0.0, 0.0),
        Point3::new(100.0, 1.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ];
    let vertices = Vertex::from_points(points);
    let wire: modeling::Wire = (0..4)
        .map(|index| modeling::builder::line(&vertices[index], &vertices[(index + 1) % 4]))
        .collect();
    let face: modeling::Face =
        modeling::builder::try_attach_plane(vec![wire]).expect("native rectangular face");
    let point = Point3::new(100.0 - 3.0e-6, 0.0, 0.0);
    let edge = &face.boundaries()[0][0];
    let kind = edge.with_curve(|curve| {
        let parameter = curve
            .search_parameter(point, None, 100)
            .expect("native line parameter");
        assert!(parameter.near(&curve.range_tuple().1));
        assert!(!point.near(&edge.absolute_back().point()));
        native_parameter_kind(edge, curve, point, parameter).expect("strict native trim membership")
    });
    assert!(matches!(kind, ParameterKind::Inner(_)));
    let mut store: LoopsStore<Point3, modeling::Curve> = std::iter::once(&face).collect();
    let contact = Vertex::new(point);
    store
        .add_geom_vertex((0, 0, 0), &contact, kind, &mut HashMap::default())
        .expect("replay the certified native parameter");
    assert_eq!(contact.point(), point);
    assert_eq!(store[0][0].len(), 5);
    assert!(store[0][0].wire.is_simple());
    assert!(store[0][0].wire.is_closed());
    assert_eq!(face.boundaries()[0].len(), 4);
    assert_eq!(edge.absolute_back().point(), points[1]);
}
