use monstertruck_core::{cgmath64::*, tolerance::*};
use monstertruck_topology::{Edge, Face, Shell, Vertex, Wire};
use monstertruck_traits::*;
use std::ops::Bound;

// Exact line geometry with its actual affine parameter range, as used by Euler tests.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Segment {
    ends: (Point3, Point3),
    range: (f64, f64),
}

impl Segment {
    fn new(front: Point3, back: Point3) -> Self {
        Self {
            ends: (front, back),
            range: (0.0, 1.0),
        }
    }
}

impl ParametricCurve for Segment {
    type Point = Point3;
    type Vector = Vector3;
    fn evaluate(&self, t: f64) -> Point3 {
        self.ends.0
            + (self.ends.1 - self.ends.0) * (t - self.range.0) / (self.range.1 - self.range.0)
    }
    fn derivative(&self, _: f64) -> Vector3 {
        (self.ends.1 - self.ends.0) / (self.range.1 - self.range.0)
    }
    fn derivative_2(&self, _: f64) -> Vector3 { Vector3::zero() }
    fn derivative_n(&self, n: usize, t: f64) -> Vector3 {
        match n {
            0 => self.evaluate(t).to_vec(),
            1 => self.derivative(t),
            _ => Vector3::zero(),
        }
    }
    fn parameter_range(&self) -> ParameterRange {
        (Bound::Included(self.range.0), Bound::Included(self.range.1))
    }
}
impl BoundedCurve for Segment {}
impl Cut for Segment {
    fn cut(&mut self, t: f64) -> Self {
        let point = self.subs(t);
        let second = Self {
            ends: (point, self.ends.1),
            range: (t, self.range.1),
        };
        self.ends.1 = point;
        self.range.1 = t;
        second
    }
}
impl SearchParameter<CurveParameter> for Segment {
    type Point = Point3;
    fn search_parameter<H: Into<SearchParameterHint1D>>(
        &self,
        point: Point3,
        _: H,
        _: usize,
    ) -> Option<f64> {
        let direction = self.ends.1 - self.ends.0;
        let fraction = (point - self.ends.0).dot(direction) / direction.magnitude2();
        let projected = self.ends.0 + direction * fraction;
        (fraction.is_finite() && (0.0..=1.0).contains(&fraction) && projected.near(&point))
            .then_some(self.range.0 + (self.range.1 - self.range.0) * fraction)
    }
}
impl Invertible for Segment {
    fn invert(&mut self) { self.ends = (self.ends.1, self.ends.0); }
}
impl ParameterTransform for Segment {
    fn parameter_transform(&mut self, scalar: f64, offset: f64) -> &mut Self {
        self.range = (
            self.range.0 * scalar + offset,
            self.range.1 * scalar + offset,
        );
        self
    }
}
impl Concat<Segment> for Segment {
    type Output = Segment;
    fn try_concat(&self, other: &Self) -> Result<Self, ConcatError<Point3>> {
        if !self.range.1.near(&other.range.0) {
            Err(ConcatError::DisconnectedParameters(
                self.range.1,
                other.range.0,
            ))
        } else if !self.ends.1.near(&other.ends.0)
            || !(self.ends.1 - self.ends.0)
                .cross(other.ends.1 - other.ends.0)
                .so_small()
        {
            Err(ConcatError::DisconnectedPoints(self.ends.1, other.ends.0))
        } else {
            Ok(Self {
                ends: (self.ends.0, other.ends.1),
                range: (self.range.0, other.range.1),
            })
        }
    }
}

fn seamed_shell() -> (Shell<Point3, Segment, ()>, Edge<Point3, Segment>) {
    let points = [
        Point3::new(1.0, 0.0, 3.0),
        Point3::new(-1.0, 1.0, 3.0),
        Point3::new(-1.0, -1.0, 3.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(-1.0, 1.0, 0.0),
        Point3::new(-1.0, -1.0, 0.0),
    ];
    let vertices = Vertex::from_points(points);
    let make = |front: usize, back: usize| {
        Edge::new(
            &vertices[front],
            &vertices[back],
            Segment::new(points[front], points[back]),
        )
    };
    let top = [make(0, 1), make(1, 2), make(2, 0)];
    let bottom = [make(3, 4), make(4, 5), make(5, 3)];
    let seam = make(3, 0);
    let lateral: Wire<_, _> = vec![
        top[2].inverse(),
        top[1].inverse(),
        top[0].inverse(),
        seam.inverse(),
        bottom[0].clone(),
        bottom[1].clone(),
        bottom[2].clone(),
        seam.clone(),
    ]
    .into();
    let top: Wire<_, _> = top.into_iter().collect();
    let bottom: Wire<_, _> = bottom
        .into_iter()
        .rev()
        .map(|edge| edge.inverse())
        .collect();
    (
        vec![
            Face::new(vec![lateral], ()),
            Face::new(vec![top], ()),
            Face::new(vec![bottom], ()),
        ]
        .into(),
        seam,
    )
}

#[test]
fn seam_edge_cut_and_vertex_concat_update_both_uses_in_one_boundary() {
    let (mut shell, seam) = seamed_shell();
    shell
        .check_solid_boundary()
        .expect("original closed incidence");
    let count = shell.compress().edges.len();
    let middle = Vertex::new(Point3::new(1.0, 0.0, 1.5));
    let (first, second) = shell.cut_edge(seam.id(), &middle).expect("exact axial cut");
    assert!(shell.edge_iter().all(|edge| edge.id() != seam.id()));
    for piece in [&first, &second] {
        assert_eq!(
            shell
                .edge_iter()
                .filter(|edge| edge.id() == piece.id())
                .count(),
            2
        );
    }
    shell
        .check_solid_boundary()
        .expect("closed after all seam uses split");
    assert!(shell.edge_iter().all(|edge| edge.is_geometric_consistent()));
    let joined = shell
        .remove_vertex_by_concat_edges(middle.id())
        .expect("inverse seam mutation");
    assert!(
        shell
            .edge_iter()
            .all(|edge| edge.id() != first.id() && edge.id() != second.id())
    );
    assert_eq!(
        shell
            .edge_iter()
            .filter(|edge| edge.id() == joined.id())
            .count(),
        2
    );
    assert_eq!(shell.compress().edges.len(), count);
    shell
        .check_solid_boundary()
        .expect("closed after paired concatenation");
    assert!(shell.edge_iter().all(|edge| edge.is_geometric_consistent()));
    let joined_curve = joined.curve();
    let (start, end) = joined_curve.range_tuple();
    assert!(
        joined_curve
            .subs(start + (end - start) * 0.5)
            .near(&middle.point())
    );
}

#[test]
fn failed_seam_geometry_cut_and_three_edge_vertex_removal_are_atomic() {
    let (mut shell, seam) = seamed_shell();
    let before = shell.compress();
    assert!(
        shell
            .cut_edge(seam.id(), &Vertex::new(Point3::new(2.0, 0.0, 1.5)))
            .is_none()
    );
    assert!(
        shell
            .remove_vertex_by_concat_edges(seam.front().id())
            .is_none()
    );
    assert_eq!(shell.compress(), before);
    shell
        .check_solid_boundary()
        .expect("unchanged source topology");
}
