use monstertruck_topology::{Edge, Face, Shell, Solid, Vertex, Wire, errors::Error};
use std::collections::HashSet;

fn boundaries() -> [Wire<(), ()>; 3] {
    let vertices = Vertex::from_points([(); 4]);
    let top = [
        Edge::new(&vertices[0], &vertices[1], ()),
        Edge::new(&vertices[1], &vertices[0], ()),
    ];
    let bottom = [
        Edge::new(&vertices[2], &vertices[3], ()),
        Edge::new(&vertices[3], &vertices[2], ()),
    ];
    let seam = Edge::new(&vertices[2], &vertices[0], ());
    [
        vec![
            top[1].inverse(),
            top[0].inverse(),
            seam.inverse(),
            bottom[0].clone(),
            bottom[1].clone(),
            seam,
        ]
        .into(),
        vec![top[0].clone(), top[1].clone()].into(),
        vec![bottom[1].inverse(), bottom[0].inverse()].into(),
    ]
}

#[test]
fn opposite_seam_uses_preserve_native_manifold_and_mapped_face_identity() {
    let shell: Shell<_, _, _> = boundaries()
        .into_iter()
        .map(|wire| Face::try_new(vec![wire], ()).expect("regular chart boundary"))
        .collect();
    shell
        .check_solid_boundary()
        .expect("closed cylinder incidence");
    let mapped = shell
        .try_mapped(|_| Some(()), |_| Some(()), |_| Some(()))
        .expect("mapping preserves seam incidence");
    mapped.check_solid_boundary().expect("mapped manifold");
    assert_eq!(mapped.len(), 3);
    assert_eq!(mapped.compress().edges.len(), 5);
}

#[test]
fn missing_cap_and_wrong_seam_direction_remain_rejected() {
    let [mut lateral, _, _] = boundaries();
    lateral[5].invert();
    assert_eq!(Face::try_new(vec![lateral], ()), Err(Error::NotClosedWire));
    let [lateral, top, _] = boundaries();
    let shell: Shell<_, _, _> = vec![
        Face::try_new(vec![lateral], ()).expect("regular lateral face"),
        Face::new(vec![top], ()),
    ]
    .into();
    assert!(shell.check_solid_boundary().is_err());
}

#[test]
fn touching_loops_and_seam_spikes_are_not_regular_boundaries() {
    let v = Vertex::from_points([(); 5]);
    let touching: Wire<_, _> = vec![
        Edge::new(&v[0], &v[1], ()),
        Edge::new(&v[1], &v[2], ()),
        Edge::new(&v[2], &v[0], ()),
        Edge::new(&v[0], &v[3], ()),
        Edge::new(&v[3], &v[4], ()),
        Edge::new(&v[4], &v[0], ()),
    ]
    .into();
    assert_eq!(Face::try_new(vec![touching], ()), Err(Error::NotSimpleWire));
    let spike = Edge::new(&v[0], &v[3], ());
    let spiked: Wire<_, _> = vec![
        Edge::new(&v[0], &v[1], ()),
        Edge::new(&v[1], &v[2], ()),
        Edge::new(&v[2], &v[0], ()),
        spike.clone(),
        spike.inverse(),
    ]
    .into();
    assert_eq!(Face::try_new(vec![spiked], ()), Err(Error::NotSimpleWire));
}

#[test]
fn separating_seamed_face_cuts_return_regular_faces_and_preserve_solid_incidence() {
    let [lateral, top, bottom] = boundaries();
    let cut = Edge::new(lateral[0].back(), lateral[3].back(), ());
    let face = Face::new(vec![lateral], ());
    let mut seen = HashSet::new();
    let connectors: Vec<_> = face
        .absolute_boundaries()
        .iter()
        .flatten()
        .filter_map(|edge| (!seen.insert(edge.id())).then_some(edge.id()))
        .collect();
    let (first, second) = face.cut_by_edge(cut.clone()).expect("separating chart cut");
    for part in [&first, &second] {
        Face::try_new(part.absolute_boundaries().to_vec(), ())
            .expect("each returned face has regular boundaries");
    }
    assert_eq!(
        first
            .absolute_boundaries()
            .iter()
            .map(|wire| wire.len())
            .sum::<usize>()
            + second
                .absolute_boundaries()
                .iter()
                .map(|wire| wire.len())
                .sum::<usize>(),
        8
    );
    let joined = first
        .glue_at_boundaries(&second, &connectors)
        .expect("inverse face split");
    assert_eq!(joined.absolute_boundaries()[0].len(), 6);
    assert_eq!(joined.absolute_boundaries().len(), 1);
    assert!(
        joined
            .absolute_boundaries()
            .iter()
            .flatten()
            .all(|edge| !edge.is_same(&cut))
    );
    let canonical = first
        .glue_at_boundaries(&second, &[])
        .expect("canonical annular union");
    assert_eq!(canonical.absolute_boundaries().len(), 2);
    assert!(
        canonical
            .absolute_boundaries()
            .iter()
            .all(|wire| wire.len() == 2)
    );
    Face::try_new(canonical.absolute_boundaries().to_vec(), ())
        .expect("canonical annular boundaries are valid");
    let unknown_vertices = Vertex::from_points([(), ()]);
    let unknown = Edge::new(&unknown_vertices[0], &unknown_vertices[1], ()).id();
    let unshared = face.absolute_boundaries()[0][0].id();
    let before_first = first.absolute_boundaries().to_vec();
    let before_second = second.absolute_boundaries().to_vec();
    for selection in [
        vec![unknown],
        vec![unshared],
        vec![connectors[0], connectors[0]],
        vec![connectors[0], cut.id()],
    ] {
        assert!(first.glue_at_boundaries(&second, &selection).is_none());
        assert_eq!(first.absolute_boundaries(), &before_first);
        assert_eq!(second.absolute_boundaries(), &before_second);
    }
    for original in face.absolute_boundaries().iter().flatten() {
        assert_eq!(
            joined
                .absolute_boundaries()
                .iter()
                .flatten()
                .filter(|edge| {
                    edge.is_same(original) && edge.orientation() == original.orientation()
                })
                .count(),
            1
        );
    }
    let face_id = face.id();
    let mut solid = Solid::new(vec![
        vec![face, Face::new(vec![top], ()), Face::new(vec![bottom], ())].into(),
    ]);
    assert!(solid.cut_face_by_edge(face_id, cut));
    assert_eq!(solid.boundaries()[0].len(), 4);
    solid.boundaries()[0]
        .check_solid_boundary()
        .expect("closed split cylinder");
}

#[test]
fn pinched_or_nonseparating_face_cut_does_not_return_invalid_output_or_mutate_solid() {
    let [lateral, top, bottom] = boundaries();
    let cut = Edge::new(lateral[0].front(), lateral[3].back(), ());
    let face = Face::new(vec![lateral], ());
    assert!(face.cut_by_edge(cut.clone()).is_none());
    let face_id = face.id();
    let mut solid = Solid::new(vec![
        vec![face, Face::new(vec![top], ()), Face::new(vec![bottom], ())].into(),
    ]);
    let before = solid.boundaries()[0].compress();
    assert!(!solid.cut_face_by_edge(face_id, cut));
    assert_eq!(solid.boundaries()[0].compress(), before);
    solid.boundaries()[0]
        .check_solid_boundary()
        .expect("unchanged closed cylinder");
}

#[test]
fn gluing_preserves_all_outgoing_occurrences_at_a_seam_vertex() {
    for invert in [false, true] {
        let [lateral, _, _] = boundaries();
        let shared = lateral[0].clone();
        let fresh = Vertex::new(());
        let triangle: Wire<_, _> = vec![
            shared.inverse(),
            Edge::new(shared.front(), &fresh, ()),
            Edge::new(&fresh, shared.back(), ()),
        ]
        .into();
        let mut first = Face::new(vec![lateral], ());
        let mut second = Face::new(vec![triangle], ());
        if invert {
            first.invert();
            second.invert();
        }
        let glued = first
            .glue_at_boundaries(&second, &[])
            .expect("seamed face union");
        assert_eq!(glued.orientation(), first.orientation());
        Face::try_new(glued.absolute_boundaries().to_vec(), ())
            .expect("glued boundary remains closed and regular");
        assert_eq!(
            glued
                .absolute_boundaries()
                .iter()
                .map(|wire| wire.len())
                .sum::<usize>(),
            7
        );
        assert!(
            glued
                .absolute_boundaries()
                .iter()
                .flatten()
                .all(|edge| !edge.is_same(&shared))
        );
        for original in first
            .absolute_boundaries()
            .iter()
            .flatten()
            .chain(second.absolute_boundaries().iter().flatten())
            .filter(|edge| !edge.is_same(&shared))
        {
            assert_eq!(
                glued
                    .absolute_boundaries()
                    .iter()
                    .flatten()
                    .filter(|edge| {
                        edge.is_same(original) && edge.orientation() == original.orientation()
                    })
                    .count(),
                1,
                "every retained oriented occurrence survives"
            );
        }
    }
}

#[test]
fn gluing_rejects_internal_only_connectors_and_same_direction_shared_uses() {
    let [lateral, _, _] = boundaries();
    let shared = lateral[0].clone();
    let internal_seam = lateral[2].id();
    let fresh = Vertex::new(());
    let opposite: Wire<_, _> = vec![
        shared.inverse(),
        Edge::new(shared.front(), &fresh, ()),
        Edge::new(&fresh, shared.back(), ()),
    ]
    .into();
    let first = Face::new(vec![lateral], ());
    let opposite = Face::new(vec![opposite], ());
    let before_first = first.absolute_boundaries().to_vec();
    let before_opposite = opposite.absolute_boundaries().to_vec();
    assert!(
        first
            .glue_at_boundaries(&opposite, &[internal_seam])
            .is_none()
    );
    assert_eq!(first.absolute_boundaries(), &before_first);
    assert_eq!(opposite.absolute_boundaries(), &before_opposite);
    let same_direction: Wire<_, _> = vec![
        shared.clone(),
        Edge::new(shared.back(), &fresh, ()),
        Edge::new(&fresh, shared.front(), ()),
    ]
    .into();
    let same_direction = Face::new(vec![same_direction], ());
    let before_same_direction = same_direction.absolute_boundaries().to_vec();
    assert!(
        first
            .glue_at_boundaries(&same_direction, &[shared.id()])
            .is_none()
    );
    assert_eq!(first.absolute_boundaries(), &before_first);
    assert_eq!(same_direction.absolute_boundaries(), &before_same_direction);
}
