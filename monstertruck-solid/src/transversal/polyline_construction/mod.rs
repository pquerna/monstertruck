//! Assemble unordered interference segments into ordered polylines.
//!
//! Truck-derived marching-SSI support: [`monstertruck_meshing`]'s
//! mesh-mesh interference extraction returns an unordered soup of
//! `(Point3, Point3)` segments; this stitches them into connected
//! [`PolylineCurve`] chains (open or closed) by walking a tolerance-keyed
//! adjacency graph. Shared triangle edges can report the same undirected
//! interference segment more than once; adjacency is a set, not a count of
//! collision events. Resurrected from the 0.3.2 published crate.

use monstertruck_core::{cgmath64::*, tolerance::*};
use monstertruck_meshing::prelude::PolylineCurve;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::collections::VecDeque;

pub fn construct_polylines(lines: &[(Point3, Point3)]) -> Vec<PolylineCurve<Point3>> {
    let mut graph: Graph = lines.iter().collect();
    let mut res = Vec::new();
    while !graph.is_empty() {
        let (mut idx, node) = graph.first_node();
        let mut wire: VecDeque<_> = vec![node.coord].into();
        while let Some((idx0, pt)) = graph.next_node(idx) {
            idx = idx0;
            wire.push_back(pt);
        }
        let mut idx = PointIndex::from(wire[0]);
        while let Some((idx0, pt)) = graph.next_node(idx) {
            idx = idx0;
            wire.push_front(pt);
        }
        res.push(PolylineCurve(wire.into()));
    }
    res
}

#[derive(Clone, Debug, Copy, Hash, PartialEq, Eq)]
struct PointIndex([i64; 3]);

impl From<Point3> for PointIndex {
    #[inline(always)]
    fn from(pt: Point3) -> PointIndex {
        let idx = pt.add_element_wise(TOLERANCE) / (2.0 * TOLERANCE);
        // SAFETY: point coordinates are finite, so the cast to `i64` always succeeds.
        PointIndex(idx.cast::<i64>().unwrap().into())
    }
}

struct Node {
    coord: Point3,
    adjacency: HashSet<PointIndex>,
}

impl Node {
    #[inline(always)]
    fn new(coord: Point3, adjacency: HashSet<PointIndex>) -> Node { Node { coord, adjacency } }

    fn pop_one_adjacency(&mut self) -> PointIndex {
        // SAFETY: nodes are removed from the graph when their adjacency set becomes empty.
        let idx = *self.adjacency.iter().next().unwrap();
        self.adjacency.remove(&idx);
        idx
    }
}

struct Graph(HashMap<PointIndex, Node>);

impl std::ops::Deref for Graph {
    type Target = HashMap<PointIndex, Node>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl std::ops::DerefMut for Graph {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl Graph {
    fn add_half_edge(&mut self, pt0: Point3, pt1: Point3) {
        let idx0 = pt0.into();
        let idx1 = pt1.into();
        if let Some(node) = self.get_mut(&idx0) {
            node.adjacency.insert(idx1);
        } else {
            self.insert(idx0, Node::new(pt0, HashSet::from_iter([idx1])));
        }
    }

    fn add_edge(&mut self, line: (Point3, Point3)) {
        if !line.0.near(&line.1) {
            self.add_half_edge(line.0, line.1);
            self.add_half_edge(line.1, line.0);
        }
    }

    #[inline(always)]
    fn first_node(&self) -> (PointIndex, &Node) {
        // SAFETY: only called inside `while !graph.is_empty()`.
        let (idx, node) = self.iter().next().unwrap();
        (*idx, node)
    }

    fn next_node(&mut self, idx: PointIndex) -> Option<(PointIndex, Point3)> {
        let node = self.get_mut(&idx)?;
        let idx0 = node.pop_one_adjacency();
        if node.adjacency.is_empty() {
            self.remove(&idx);
        }
        let node = self.get_mut(&idx0)?;
        if !node.adjacency.remove(&idx) {
            return None;
        }
        let pt = node.coord;
        if node.adjacency.is_empty() {
            self.remove(&idx0);
        }
        Some((idx0, pt))
    }
}

impl<'a> FromIterator<&'a (Point3, Point3)> for Graph {
    fn from_iter<I: IntoIterator<Item = &'a (Point3, Point3)>>(iter: I) -> Graph {
        let mut res = Graph(HashMap::default());
        iter.into_iter().for_each(|line| res.add_edge(*line));
        res
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod shared_edge_tests {
    use super::*;

    fn segment_key(a: Point3, b: Point3) -> ([i64; 3], [i64; 3]) {
        let (a, b) = (PointIndex::from(a).0, PointIndex::from(b).0);
        if a < b { (a, b) } else { (b, a) }
    }

    #[test]
    fn shared_triangle_edge_events_make_one_simple_closed_intersection() {
        let points = [
            Point3::new(0.0, 0.0, 4.0),
            Point3::new(2.0, 0.0, 4.0),
            Point3::new(2.0, 3.0, 4.0),
            Point3::new(0.0, 3.0, 4.0),
        ];
        let unique = (0..4)
            .map(|index| (points[index], points[(index + 1) % 4]))
            .collect::<Vec<_>>();
        // Model duplicate collision events from incident triangles, including
        // opposite event orientation. They are one geometric intersection.
        let events = unique
            .iter()
            .flat_map(|&(a, b)| [(a, b), (b, a), (a, b)])
            .collect::<Vec<_>>();
        let polylines = construct_polylines(&events);
        assert_eq!(polylines.len(), 1);
        let polyline = &polylines[0];
        assert_eq!(polyline.len(), 5);
        assert!(polyline[0].near(&polyline[polyline.len() - 1]));
        let actual = polyline
            .windows(2)
            .map(|edge| segment_key(edge[0], edge[1]))
            .collect::<HashSet<_>>();
        let expected = unique
            .iter()
            .map(|&(a, b)| segment_key(a, b))
            .collect::<HashSet<_>>();
        assert_eq!(actual, expected);
        let twice_area = polyline
            .windows(2)
            .map(|edge| edge[0].x * edge[1].y - edge[1].x * edge[0].y)
            .sum::<f64>();
        assert!((twice_area.abs() / 2.0 - 6.0).abs() < TOLERANCE);
    }

    #[test]
    fn deduplication_preserves_distinct_adjacent_interference_segments() {
        let points = [
            Point3::new(-5.0, 0.0, 4.0),
            Point3::new(-4.999977662529157, 0.01494570873092198, 4.0),
            Point3::new(-4.99984, 0.04, 4.0),
            Point3::new(-4.996, 0.2, 4.0),
        ];
        let events = [
            (points[0], points[1]),
            (points[1], points[0]),
            (points[1], points[2]),
            (points[2], points[1]),
            (points[2], points[3]),
        ];
        let polylines = construct_polylines(&events);
        assert_eq!(polylines.len(), 1);
        assert_eq!(polylines[0].len(), 4);
        let actual = polylines[0]
            .windows(2)
            .map(|edge| segment_key(edge[0], edge[1]))
            .collect::<HashSet<_>>();
        let expected = points
            .windows(2)
            .map(|edge| segment_key(edge[0], edge[1]))
            .collect::<HashSet<_>>();
        assert_eq!(actual, expected);
    }
}
