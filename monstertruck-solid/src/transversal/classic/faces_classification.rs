//! Classic (0.3.2) per-face and/or/unknown classification.
//!
//! Derived from the published 0.3.2 `transversal::faces_classification`.
//! Component adjacency is incidence-based: all exposed boundaries matter,
//! including the second wire of an annular cap. Contradictory neighboring
//! material classifications are refused instead of choosing input order.

use super::loops_store::ShapesOpStatus;
use monstertruck_topology::*;
use rustc_hash::FxHashMap as HashMap;

#[derive(Clone, Debug)]
pub(super) struct FacesClassification<P, C, S> {
    shell: Shell<P, C, S>,
    status: HashMap<FaceId<S>, ShapesOpStatus>,
}

impl<P, C, S> Default for FacesClassification<P, C, S> {
    fn default() -> Self {
        Self {
            shell: Default::default(),
            status: HashMap::default(),
        }
    }
}

impl<P, C, S> FacesClassification<P, C, S> {
    pub(super) fn push(&mut self, face: Face<P, C, S>, status: ShapesOpStatus) {
        self.status.insert(face.id(), status);
        self.shell.push(face);
    }

    pub(super) fn and_or_unknown(&self) -> [Shell<P, C, S>; 3] {
        let [mut and, mut or, mut unknown] = <[Shell<P, C, S>; 3]>::default();
        for face in &self.shell {
            // SAFETY: `push()` inserts every face id into `status`.
            match self.status.get(&face.id()).unwrap() {
                ShapesOpStatus::And => and.push(face.clone()),
                ShapesOpStatus::Or => or.push(face.clone()),
                ShapesOpStatus::Unknown => unknown.push(face.clone()),
            }
        }
        [and, or, unknown]
    }

    pub(super) fn integrate_by_component(&mut self) -> bool {
        let [and, or, unknown] = self.and_or_unknown();
        let and_boundary = and.extract_boundaries();
        let or_boundary = or.extract_boundaries();
        let mut adjacent = HashMap::<EdgeId<C>, [Option<ShapesOpStatus>; 2]>::default();
        for (boundary, status) in [
            (&and_boundary, ShapesOpStatus::And),
            (&or_boundary, ShapesOpStatus::Or),
        ] {
            for edge in boundary.iter().flatten() {
                let slot = &mut adjacent.entry(edge.id()).or_insert([None; 2])
                    [usize::from(edge.orientation())];
                if slot.is_some_and(|previous| previous != status) {
                    return false;
                }
                *slot = Some(status);
            }
        }
        for component in unknown.connected_components() {
            let boundary = component.extract_boundaries();
            let mut status = ShapesOpStatus::Unknown;
            for edge in boundary.iter().flatten() {
                if let Some(next) = adjacent
                    .get(&edge.id())
                    .and_then(|uses| uses[usize::from(!edge.orientation())])
                {
                    if status != ShapesOpStatus::Unknown && status != next {
                        return false;
                    }
                    status = next;
                }
            }
            if status != ShapesOpStatus::Unknown {
                for face in component {
                    // SAFETY: component faces originated in self.shell.
                    *self.status.get_mut(&face.id()).unwrap() = status;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use monstertruck_modeling as modeling;

    fn ring(points: &[[f64; 3]]) -> modeling::Wire {
        let vertices: Vec<_> = points
            .iter()
            .map(|point| {
                modeling::builder::vertex(modeling::Point3::new(point[0], point[1], point[2]))
            })
            .collect();
        (0..vertices.len())
            .map(|index| {
                modeling::builder::line(&vertices[index], &vertices[(index + 1) % vertices.len()])
            })
            .collect()
    }

    fn plane() -> modeling::Surface {
        modeling::Surface::Plane(modeling::Plane::new(
            modeling::Point3::new(0.0, 0.0, 0.0),
            modeling::Point3::new(1.0, 0.0, 0.0),
            modeling::Point3::new(0.0, 1.0, 0.0),
        ))
    }

    #[test]
    fn annular_unknown_component_uses_its_second_boundary_and_opposite_incidence() {
        let outer = ring(&[
            [-2.0, -2.0, 0.0],
            [2.0, -2.0, 0.0],
            [2.0, 2.0, 0.0],
            [-2.0, 2.0, 0.0],
        ]);
        let inner = ring(&[[-0.5, -0.5, 0.0], [0.0, 0.5, 0.0], [0.5, -0.5, 0.0]]);
        for rotation in 0..inner.len() {
            let mut inner = inner.clone();
            inner.rotate_left(rotation);
            let unknown = Face::new(vec![outer.clone(), inner.clone()], plane());
            let neighbor = Face::new(vec![inner.inverse()], plane());
            assert!(unknown.is_geometric_consistent());
            assert!(neighbor.is_geometric_consistent());
            let mut classification = FacesClassification::default();
            classification.push(unknown.clone(), ShapesOpStatus::Unknown);
            classification.push(neighbor.clone(), ShapesOpStatus::And);
            assert!(classification.integrate_by_component());
            let [and, _, unresolved] = classification.and_or_unknown();
            assert_eq!(and.len(), 2);
            assert!(unresolved.is_empty());

            let mut classification = FacesClassification::default();
            classification.push(unknown, ShapesOpStatus::Unknown);
            classification.push(neighbor.inverse(), ShapesOpStatus::And);
            assert!(classification.integrate_by_component());
            assert_eq!(
                classification.and_or_unknown()[2].len(),
                1,
                "same-direction shared use is not oriented adjacency"
            );
        }
    }

    #[test]
    fn contradictory_component_neighbors_are_refused() {
        let outer = ring(&[
            [-3.0, -2.0, 0.0],
            [3.0, -2.0, 0.0],
            [3.0, 2.0, 0.0],
            [-3.0, 2.0, 0.0],
        ]);
        let first = ring(&[[-2.0, -0.5, 0.0], [-1.5, 0.5, 0.0], [-1.0, -0.5, 0.0]]);
        let second = ring(&[[1.0, -0.5, 0.0], [1.5, 0.5, 0.0], [2.0, -0.5, 0.0]]);
        let unknown = Face::new(vec![outer, first.clone(), second.clone()], plane());
        let mut classification = FacesClassification::default();
        classification.push(unknown, ShapesOpStatus::Unknown);
        classification.push(
            Face::new(vec![first.inverse()], plane()),
            ShapesOpStatus::And,
        );
        classification.push(
            Face::new(vec![second.inverse()], plane()),
            ShapesOpStatus::Or,
        );
        assert!(!classification.integrate_by_component());
        assert_eq!(classification.and_or_unknown()[2].len(), 1);
    }

    #[test]
    fn closed_unknown_component_has_no_exposed_boundary_and_remains_unknown() {
        let shell = modeling::primitive::cuboid(modeling::BoundingBox::from_iter([
            modeling::Point3::new(0.0, 0.0, 0.0),
            modeling::Point3::new(1.0, 1.0, 1.0),
        ]))
        .into_boundaries()
        .pop()
        .unwrap();
        let known = Face::new(
            vec![ring(&[
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [4.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
            ])],
            plane(),
        );
        let mut classification = FacesClassification::default();
        for face in shell {
            classification.push(face, ShapesOpStatus::Unknown);
        }
        classification.push(known, ShapesOpStatus::And);
        assert!(classification.integrate_by_component());
        assert_eq!(classification.and_or_unknown()[2].len(), 6);
    }
}
