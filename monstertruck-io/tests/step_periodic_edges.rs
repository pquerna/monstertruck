#![cfg(feature = "step")]

use monstertruck_io::step::load::{
    Table,
    step_geometry::{Curve3D, Point3, Surface},
};
use monstertruck_topology::{Shell, Solid, Vertex, trimmed::TrimmedShell};
use monstertruck_traits::{BoundedCurve, ParametricCurve};

// Hand-authored ISO 10303 topology, independent of any CAD exporter.
// Two one-vertex circular edges and one paired seam bound exactly three faces.
fn cylinder(seam_x: f64, seam_y: f64) -> String {
    format!(
        "ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('periodic topology regression'),'2;1');
FILE_NAME('','',(''),(''),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CARTESIAN_POINT('',(0.,0.,3.));
#3=DIRECTION('',(0.,0.,1.));
#4=DIRECTION('',(1.,0.,0.));
#5=AXIS2_PLACEMENT_3D('',#1,#3,#4);
#6=AXIS2_PLACEMENT_3D('',#2,#3,#4);
#7=CIRCLE('',#5,2.);
#8=CIRCLE('',#6,2.);
#9=CARTESIAN_POINT('',({seam_x:.17},{seam_y:.17},0.));
#10=CARTESIAN_POINT('',({seam_x:.17},{seam_y:.17},3.));
#11=VERTEX_POINT('',#9);
#12=VERTEX_POINT('',#10);
#13=EDGE_CURVE('',#11,#11,#7,.T.);
#14=EDGE_CURVE('',#12,#12,#8,.T.);
#15=VECTOR('',#3,1.);
#16=LINE('',#9,#15);
#17=EDGE_CURVE('',#11,#12,#16,.T.);
#18=ORIENTED_EDGE('',*,*,#14,.F.);
#19=ORIENTED_EDGE('',*,*,#17,.F.);
#20=ORIENTED_EDGE('',*,*,#13,.T.);
#21=ORIENTED_EDGE('',*,*,#17,.T.);
#22=EDGE_LOOP('',(#18,#19,#20,#21));
#23=FACE_BOUND('',#22,.T.);
#24=CYLINDRICAL_SURFACE('',#5,2.);
#25=ADVANCED_FACE('',(#23),#24,.T.);
#26=ORIENTED_EDGE('',*,*,#14,.T.);
#27=EDGE_LOOP('',(#26));
#28=FACE_BOUND('',#27,.T.);
#29=PLANE('',#6);
#30=ADVANCED_FACE('',(#28),#29,.T.);
#31=ORIENTED_EDGE('',*,*,#13,.F.);
#32=EDGE_LOOP('',(#31));
#33=FACE_BOUND('',#32,.T.);
#34=PLANE('',#5);
#35=ADVANCED_FACE('',(#33),#34,.F.);
#36=CLOSED_SHELL('',(#25,#30,#35));
#37=MANIFOLD_SOLID_BREP('',#36);
ENDSEC;
END-ISO-10303-21;"
    )
}

fn assert_cylinder_boundary_geometry(shell: &Shell<Point3, Curve3D, Surface>) {
    assert!(shell.edge_iter().all(|edge| edge.is_geometric_consistent()));
    for (slot, face) in shell.iter().enumerate() {
        for edge in face.absolute_boundaries().iter().flatten() {
            let curve = edge.curve();
            let (start, end) = curve.range_tuple();
            for fraction in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let point = curve.subs(start + (end - start) * fraction);
                assert!((point.x.hypot(point.y) - 2.0).abs() < 1e-6);
                match slot {
                    0 => assert!((-1e-6..=3.0 + 1e-6).contains(&point.z)),
                    1 => assert!((point.z - 3.0).abs() < 1e-6),
                    2 => assert!(point.z.abs() < 1e-6),
                    _ => panic!("unexpected source face slot"),
                }
            }
        }
    }
}

#[test]
fn periodic_edges_preserve_three_source_faces_and_closed_manifold_incidence() {
    for (x, y) in [(2.0, 0.0), (0.0, 2.0), (-2.0, 0.0), (0.0, -2.0)] {
        let table = Table::from_step(&cylinder(x, y)).expect("original minimal STEP");
        let source = &table.manifold_solid_brep[&37];
        let (compressed, reports) = table
            .to_compressed_solid_reported(source)
            .expect("one-period edge representation");
        assert!(reports.iter().all(|report| report.is_lossless()));
        assert_eq!(compressed.boundaries[0].faces.len(), 3);
        assert_eq!(compressed.boundaries[0].vertices.len(), 4);
        assert_eq!(compressed.boundaries[0].edges.len(), 5);
        assert!(
            compressed.boundaries[0]
                .edges
                .iter()
                .all(|edge| edge.vertices.0 != edge.vertices.1)
        );
        let native = Solid::extract(compressed).expect("closed native cylinder");
        native.boundaries()[0]
            .check_solid_boundary()
            .expect("manifold");
        assert_cylinder_boundary_geometry(&native.boundaries()[0]);
        assert_eq!(native.boundaries()[0].compress().edges.len(), 5);
        let (trimmed, report) = table
            .to_compressed_trimmed_shell_reported(&table.shell[&36])
            .expect("same representation with exact face-local trims");
        assert!(report.is_lossless());
        let native = TrimmedShell::try_from(trimmed).expect("closed trimmed cylinder");
        native.check_solid_boundary().expect("trimmed manifold");
    }
}

#[test]
fn a_self_loop_on_a_line_is_not_a_periodic_edge() {
    let source = cylinder(2.0, 0.0).replace(
        "#13=EDGE_CURVE('',#11,#11,#7,.T.);",
        "#13=EDGE_CURVE('',#11,#11,#16,.T.);",
    );
    let table = Table::from_step(&source).expect("malformed topology parses");
    assert!(
        table
            .to_compressed_solid_reported(&table.manifold_solid_brep[&37])
            .is_err()
    );
    assert!(
        table
            .to_compressed_trimmed_shell_reported(&table.shell[&36])
            .is_err()
    );
}

#[test]
fn an_open_periodic_boundary_is_not_a_closed_solid() {
    let source = cylinder(2.0, 0.0).replace(
        "#22=EDGE_LOOP('',(#18,#19,#20,#21));",
        "#22=EDGE_LOOP('',(#18,#19,#20));",
    );
    let table = Table::from_step(&source).expect("open boundary parses");
    let compressed = table
        .to_compressed_solid(&table.manifold_solid_brep[&37])
        .expect("conversion");
    assert!(Solid::extract(compressed).is_err());
}

#[test]
fn closed_periodic_ellipses_are_not_special_cased_as_circles() {
    for sense in [".T.", ".F."] {
        let source = format!(
            "ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('periodic ellipse regression'),'2;1');
FILE_NAME('','',(''),(''),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(0.,0.,1.));
#3=DIRECTION('',(1.,0.,0.));
#4=AXIS2_PLACEMENT_3D('',#1,#2,#3);
#5=ELLIPSE('',#4,3.,2.);
#6=CARTESIAN_POINT('',(0.,2.,0.));
#7=VERTEX_POINT('',#6);
#8=EDGE_CURVE('',#7,#7,#5,{sense});
#9=ORIENTED_EDGE('',*,*,#8,.T.);
#10=EDGE_LOOP('',(#9));
#11=FACE_BOUND('',#10,.T.);
#12=PLANE('',#4);
#13=ADVANCED_FACE('',(#11),#12,.T.);
#14=OPEN_SHELL('',(#13));
ENDSEC;
END-ISO-10303-21;"
        );
        let table = Table::from_step(&source).expect("producer-independent ellipse");
        let (compressed, report) = table
            .to_compressed_shell_reported(&table.shell[&14])
            .expect("periodic ellipse representation");
        assert!(report.is_lossless());
        assert_eq!(compressed.edges.len(), 2);
        assert_eq!(compressed.vertices.len(), 2);
        assert_eq!(compressed.faces.len(), 1);
        let shell = Shell::extract(compressed).expect("closed planar ellipse boundary");
        assert!(shell.edge_iter().all(|edge| edge.is_geometric_consistent()));
        for edge in shell.edge_iter() {
            let curve = edge.curve();
            let (start, end) = curve.range_tuple();
            for fraction in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let point = curve.subs(start + (end - start) * fraction);
                assert!(point.z.abs() < 1e-6);
                assert!((point.x.powi(2) / 9.0 + point.y.powi(2) / 4.0 - 1.0).abs() < 1e-6);
            }
        }
        assert!(
            shell.check_solid_boundary().is_err(),
            "an open sheet is still not a solid"
        );
    }
}

#[test]
fn cutting_the_native_axial_seam_updates_every_oriented_use() {
    let table = Table::from_step(&cylinder(0.0, 2.0)).expect("minimal cylinder");
    let compressed = table
        .to_compressed_solid(&table.manifold_solid_brep[&37])
        .expect("periodic native conversion");
    let mut native = Solid::extract(compressed).expect("closed native solid");
    let seam = native
        .edge_iter()
        .find(|edge| (edge.front().point().z - edge.back().point().z).abs() > 1.0)
        .expect("axial seam")
        .absolute_clone();
    let curve = seam.curve();
    let (start, end) = curve.range_tuple();
    let middle = Vertex::new(curve.subs(start + (end - start) * 0.5));
    let (first, second) = native
        .cut_edge(seam.id(), &middle)
        .expect("native seam cut");
    assert!(native.edge_iter().all(|edge| edge.id() != seam.id()));
    for piece in [&first, &second] {
        let uses: Vec<_> = native
            .edge_iter()
            .filter(|edge| edge.id() == piece.id())
            .collect();
        assert_eq!(uses.len(), 2);
        assert_ne!(uses[0].orientation(), uses[1].orientation());
    }
    native.boundaries()[0]
        .check_solid_boundary()
        .expect("closed after seam cut");
    assert_cylinder_boundary_geometry(&native.boundaries()[0]);
    assert_eq!(native.boundaries()[0].len(), 3);
    assert_eq!(native.boundaries()[0].compress().edges.len(), 6);
}
