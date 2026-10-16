use crate::errors::Error;
use crate::*;
use monstertruck_core::entry_map::FxEntryMap as EntryMap;
use rayon::prelude::*;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

/// A face adjacent to the specified face
#[derive(Clone, Debug)]
pub struct AdjacentFace<'a, P, C, S> {
    /// face adjacent to the specified face
    pub face: &'a Face<P, C, S>,
    /// edges shared by two faces
    pub common_edges: Vec<EdgeId<C>>,
}

trait As<T> {
    fn as_(self) -> T;
}

impl<T> As<T> for T {
    fn as_(self) -> T { self }
}

type FaceAdjacencyMap<'a, P, C, S> =
    HashMap<(FaceId<S>, bool), (&'a Face<P, C, S>, Vec<AdjacentFace<'a, P, C, S>>)>;
impl<P, C, S> Shell<P, C, S> {
    /// Creates the empty shell.
    #[inline(always)]
    pub const fn new() -> Shell<P, C, S> {
        Shell {
            face_list: Vec::new(),
        }
    }

    /// Creates the empty shell with space for at least `capacity` faces.
    #[inline(always)]
    pub fn with_capacity(capacity: usize) -> Shell<P, C, S> {
        Shell {
            face_list: Vec::with_capacity(capacity),
        }
    }

    /// Returns an iterator over the faces. Practically, an alias of `iter()`.
    #[inline(always)]
    pub fn face_iter(&self) -> FaceIter<'_, P, C, S> { self.iter() }

    /// Returns a mutable iterator over the faces. Practically, an alias of `iter_mut()`.
    #[inline(always)]
    pub fn face_iter_mut(&mut self) -> FaceIterMut<'_, P, C, S> { self.iter_mut() }

    /// Creates a consuming iterator. Practically, an alias of `into_iter()`.
    #[inline(always)]
    pub fn face_into_iter(self) -> FaceIntoIter<P, C, S> { self.face_list.into_iter() }

    /// Returns an iterator over the faces. Practically, an alias of `par_iter()`.
    #[inline(always)]
    pub fn face_par_iter(&self) -> FaceParallelIter<'_, P, C, S>
    where
        P: Send,
        C: Send,
        S: Send, {
        self.par_iter()
    }

    /// Returns a mutable iterator over the faces. Practically, an alias of `par_iter_mut()`.
    #[inline(always)]
    pub fn face_par_iter_mut(&mut self) -> FaceParallelIterMut<'_, P, C, S>
    where
        P: Send,
        C: Send,
        S: Send, {
        self.par_iter_mut()
    }

    /// Creates a consuming iterator. Practically, an alias of `into_par_iter()`.
    #[inline(always)]
    pub fn face_into_par_iter(self) -> FaceParallelIntoIter<P, C, S>
    where
        P: Send,
        C: Send,
        S: Send, {
        self.into_par_iter()
    }

    /// Returns an iterator over the edges.
    #[inline(always)]
    pub fn edge_iter(&self) -> impl Iterator<Item = Edge<P, C>> + '_ {
        self.face_iter().flat_map(Face::edge_iter)
    }

    /// Returns a parallel iterator over the edges.
    #[inline(always)]
    pub fn edge_par_iter(&self) -> impl ParallelIterator<Item = Edge<P, C>> + '_
    where
        P: Send,
        C: Send,
        S: Send, {
        self.face_par_iter().flat_map(Face::boundaries).flatten()
    }

    /// Returns an iterator over the vertices.
    #[inline(always)]
    pub fn vertex_iter(&self) -> impl Iterator<Item = Vertex<P>> + '_ {
        self.edge_iter().map(|edge| edge.front().clone())
    }

    /// Returns a parallel iterator over the vertices.
    #[inline(always)]
    pub fn vertex_par_iter(&self) -> impl ParallelIterator<Item = Vertex<P>> + '_
    where
        P: Send,
        C: Send,
        S: Send, {
        self.edge_par_iter().map(|edge| edge.front().clone())
    }

    /// Moves all the faces of `other` into `self`, leaving `other` empty.
    #[inline(always)]
    pub fn append(&mut self, other: &mut Shell<P, C, S>) {
        self.face_list.append(&mut other.face_list);
    }

    /// Determines the shell conditions: non-regular, regular, oriented, or closed.  
    /// The complexity increases in proportion to the number of edges.
    ///
    /// Examples for each condition can be found on the page of
    /// [`ShellCondition`](./shell/enum.ShellCondition.html).
    pub fn shell_condition(&self) -> ShellCondition {
        self.edge_iter().collect::<Boundaries<C>>().condition()
    }

    /// Returns a vector of all boundaries as wires.
    /// # Examples
    /// ```
    /// use monstertruck_topology::*;
    /// use monstertruck_topology::shell::ShellCondition;
    /// let v = Vertex::from_points(&[(); 6]);
    /// let edge = [
    ///     Edge::new(&v[0], &v[1], ()),
    ///     Edge::new(&v[0], &v[2], ()),
    ///     Edge::new(&v[1], &v[2], ()),
    ///     Edge::new(&v[1], &v[3], ()),
    ///     Edge::new(&v[1], &v[4], ()),
    ///     Edge::new(&v[2], &v[4], ()),
    ///     Edge::new(&v[2], &v[5], ()),
    ///     Edge::new(&v[3], &v[4], ()),
    ///     Edge::new(&v[4], &v[5], ()),
    /// ];
    /// let wire = vec![
    ///     wire![&edge[0], &edge[2], &edge[1].inverse()],
    ///     wire![&edge[3], &edge[7], &edge[4].inverse()],
    ///     wire![&edge[5], &edge[8], &edge[6].inverse()],
    ///     wire![&edge[2].inverse(), &edge[4], &edge[5].inverse()],
    /// ];
    /// let shell: Shell<_, _, _> = wire.into_iter().map(|w| Face::new(vec![w], ())).collect();
    /// let boundary = shell.extract_boundaries()[0].clone();
    /// assert_eq!(
    ///     boundary,
    ///     wire![&edge[0], &edge[3], &edge[7], &edge[8], &edge[6].inverse(), &edge[1].inverse()],
    /// );
    /// ```
    /// # Remarks
    /// This method is optimized when the shell is oriented.
    /// Even if the shell is not oriented, all the edges of the boundary are extracted.
    /// However, the connected components of the boundary are split into several wires.
    pub fn extract_boundaries(&self) -> Vec<Wire<P, C>> {
        let boundaries: Boundaries<C> = self.edge_iter().collect();
        let mut vemap: HashMap<_, _> = self
            .edge_iter()
            .filter_map(|edge| {
                boundaries
                    .boundaries
                    .get(&edge.id())
                    .map(|_| (edge.front().id(), edge.clone()))
            })
            .collect();
        let mut res = Vec::new();
        while !vemap.is_empty() {
            // SAFETY: vemap is non-empty (loop guard) and its keys come from self's vertices.
            let edge = self.vertex_iter().find_map(|v| vemap.get(&v.id())).unwrap();
            if let Some(mut cursor) = vemap.remove(&edge.front().id()) {
                let mut wire = Wire::from(vec![cursor.clone()]);
                loop {
                    cursor = match vemap.remove(&cursor.back().id()) {
                        None => break,
                        Some(got) => {
                            wire.push_back(got.clone());
                            got.clone()
                        }
                    };
                }
                res.push(wire);
            }
        }
        res
    }

    /// Returns the adjacency matrix of vertices in the shell.
    ///
    /// For the returned hashmap `map` and each vertex `v`, the
    /// vector `map[&v]` consists of all vertices which is adjacent to `v`.
    /// # Examples
    /// ```
    /// use monstertruck_topology::*;
    /// use std::collections::HashSet;
    /// let v = Vertex::from_points(&[(); 4]);
    /// let edge = [
    ///     Edge::new(&v[0], &v[2], ()),
    ///     Edge::new(&v[0], &v[3], ()),
    ///     Edge::new(&v[1], &v[2], ()),
    ///     Edge::new(&v[1], &v[3], ()),
    ///     Edge::new(&v[2], &v[3], ()),
    /// ];
    /// let wire = vec![
    ///     wire![&edge[0], &edge[4], &edge[1].inverse()],
    ///     wire![&edge[2], &edge[4], &edge[3].inverse()],
    /// ];
    /// let shell: Shell<_, _, _> = wire.into_iter().map(|w| Face::new(vec![w], ())).collect();
    /// let adjacency = shell.vertex_adjacency();
    /// let v0_ads_vec = adjacency.get(&v[0].id()).unwrap();
    /// let v0_ads: HashSet<&VertexId<()>> = HashSet::from_iter(v0_ads_vec);
    /// assert_eq!(v0_ads, HashSet::from_iter(vec![&v[2].id(), &v[3].id()]));
    /// ```
    pub fn vertex_adjacency(&self) -> HashMap<VertexId<P>, Vec<VertexId<P>>> {
        let mut adjacency = EntryMap::new(|x| x, |_| Vec::new());
        let mut done_edge: HashSet<EdgeId<C>> = HashSet::default();
        self.edge_iter().for_each(|edge| {
            if done_edge.insert(edge.id()) {
                let v0 = edge.front().id();
                let v1 = edge.back().id();
                adjacency.entry_or_insert(v0).push(v1);
                adjacency.entry_or_insert(v1).push(v0);
            }
        });
        adjacency.into()
    }

    /// Returns the adjacency matrix of faces in the shell.
    ///
    /// Keys are `(face.id(), face.orientation())`, exactly the directed
    /// identity used by [`Face`]'s equality and hashing, without mutable
    /// geometry in the key. Each value retains the actual face reference and
    /// its adjacent-face vector: `map[&(face.id(), face.orientation())].1`.
    /// # Examples
    /// ```
    /// use monstertruck_topology::*;
    /// let v = Vertex::from_points(&[(); 6]);
    /// let edge = [
    ///     Edge::new(&v[0], &v[1], ()),
    ///     Edge::new(&v[0], &v[2], ()),
    ///     Edge::new(&v[1], &v[2], ()),
    ///     Edge::new(&v[1], &v[3], ()),
    ///     Edge::new(&v[1], &v[4], ()),
    ///     Edge::new(&v[2], &v[4], ()),
    ///     Edge::new(&v[2], &v[5], ()),
    ///     Edge::new(&v[3], &v[4], ()),
    ///     Edge::new(&v[4], &v[5], ()),
    /// ];
    /// let wire = vec![
    ///     wire![&edge[0], &edge[2], &edge[1].inverse()],
    ///     wire![&edge[3], &edge[7], &edge[4].inverse()],
    ///     wire![&edge[5], &edge[8], &edge[6].inverse()],
    ///     wire![&edge[2].inverse(), &edge[4], &edge[5].inverse()],
    /// ];
    /// let shell: Shell<_, _, _> = wire.into_iter().map(|w| Face::new(vec![w], ())).collect();
    /// let face_adjacency = shell.face_adjacency();
    /// assert_eq!(face_adjacency[&(shell[0].id(), shell[0].orientation())].1.len(), 1);
    /// assert_eq!(face_adjacency[&(shell[1].id(), shell[1].orientation())].1.len(), 1);
    /// assert_eq!(face_adjacency[&(shell[2].id(), shell[2].orientation())].1.len(), 1);
    /// assert_eq!(face_adjacency[&(shell[3].id(), shell[3].orientation())].1.len(), 3);
    /// ```
    /// ```
    /// use monstertruck_topology::*;
    /// use shell::AdjacentFace;
    /// let v = Vertex::from_points(&[(); 8]);
    /// let edge = [
    ///     Edge::new(&v[0], &v[1], ()),
    ///     Edge::new(&v[1], &v[2], ()),
    ///     Edge::new(&v[0], &v[3], ()),
    ///     Edge::new(&v[1], &v[6], ()),
    ///     Edge::new(&v[6], &v[7], ()),
    ///     Edge::new(&v[6], &v[7], ()),
    ///     Edge::new(&v[7], &v[4], ()),
    ///     Edge::new(&v[2], &v[5], ()),
    ///     Edge::new(&v[3], &v[4], ()),
    ///     Edge::new(&v[4], &v[5], ()),
    /// ];
    /// let wire = vec![
    ///     wire![
    ///         &edge[2],
    ///         &edge[8],
    ///         &edge[6].inverse(),
    ///         &edge[4].inverse(),
    ///         &edge[3].inverse(),
    ///         &edge[0].inverse()
    ///     ],
    ///     wire![
    ///         &edge[3],
    ///         &edge[5],
    ///         &edge[6],
    ///         &edge[9],
    ///         &edge[7].inverse(),
    ///         &edge[1].inverse(),
    ///     ],
    ///     wire![&edge[4], &edge[5].inverse()],
    /// ];
    /// let shell: Shell<_, _, _> = wire.into_iter().map(|w| Face::new(vec![w], ())).collect();
    /// let face_adjacency = shell.face_adjacency();
    /// let determine = |face0: &Face<_, _, _>, face1: &Face<_, _, _>, common_edges: usize| {
    ///     if (face0, face1) == (&shell[0], &shell[1]) || (face0, face1) == (&shell[1], &shell[0]) {
    ///         assert_eq!(common_edges, 2);
    ///     } else {
    ///         assert_eq!(common_edges, 1);
    ///     }
    /// };
    /// face_adjacency.into_iter().for_each(|(_, (face0, adjacents))| {
    ///     adjacents.into_iter().for_each(|AdjacentFace { face, common_edges }| {
    ///         determine(face, face0, common_edges.len());
    ///     })
    /// })
    /// ```
    pub fn face_adjacency(&self) -> FaceAdjacencyMap<'_, P, C, S> {
        let mut edge_face_map = EntryMap::new(|edge: &Edge<P, C>| edge.id(), |_| Vec::new());
        self.face_iter().for_each(|face| {
            let insert = |edge| edge_face_map.entry_or_insert(edge).push(face);
            face.absolute_boundaries().iter().flatten().for_each(insert)
        });
        let edge_face_map: HashMap<_, _> = edge_face_map.into();
        let mut adjacency = EntryMap::new(
            |face: &Face<P, C, S>| (face.id(), face.orientation()),
            |face| (face, Vec::new()),
        );
        // Process edges in shell order: draining `edge_face_map` in its own
        // (pointer-derived) key order would make the adjacency-list order
        // depend on the address-space layout.
        let mut done: HashSet<EdgeId<C>> = HashSet::default();
        self.face_iter()
            .flat_map(|face| face.absolute_boundaries().iter().flatten())
            .for_each(|edge| {
                let edge_id = edge.id();
                if !done.insert(edge_id) {
                    return;
                }
                // SAFETY: every shell edge was inserted into the map above.
                let vec = edge_face_map.get(&edge_id).unwrap();
                vec.iter().for_each(|face| {
                    let (_, adjacents) = adjacency.entry_or_insert(*face);
                    vec.iter().for_each(|face0| {
                        if face == face0 {
                            return;
                        }
                        let add_edge = |adjacent: &mut AdjacentFace<'_, P, C, S>| {
                            let res = &adjacent.face == face0;
                            if res {
                                adjacent.common_edges.push(edge_id);
                            }
                            res
                        };
                        let exists = adjacents.iter_mut().any(add_edge);
                        if !exists {
                            adjacents.push(AdjacentFace {
                                face: face0,
                                common_edges: vec![edge_id],
                            });
                        }
                    });
                });
            });

        adjacency.into()
    }

    /// Returns whether the shell is connected or not.
    /// # Examples
    /// ```
    /// // The empty shell is connected.
    /// use monstertruck_topology::*;
    /// assert!(Shell::<(), (), ()>::new().is_connected());
    /// ```
    /// ```
    /// // An example of a connected shell
    /// use monstertruck_topology::*;
    /// let v = Vertex::from_points(&[(); 4]);
    /// let shared_edge = Edge::new(&v[1], &v[2], ());
    /// let wire0 = wire![
    ///     &Edge::new(&v[0], &v[1], ()),
    ///     &shared_edge,
    ///     &Edge::new(&v[2], &v[0], ()),
    /// ];
    /// let face0 = Face::new(vec![wire0], ());
    /// let wire1 = wire![
    ///     &Edge::new(&v[3], &v[1], ()),
    ///     &shared_edge,
    ///     &Edge::new(&v[2], &v[3], ()),
    /// ];
    /// let face1 = Face::new(vec![wire1], ());
    /// let shell: Shell<_, _, _> = vec![face0, face1].into();
    /// assert!(shell.is_connected());
    /// ```
    /// ```
    /// // An example of a non-connected shell
    /// use monstertruck_topology::*;
    /// let v = Vertex::from_points(&[(); 6]);
    /// let wire0 = wire![
    ///     Edge::new(&v[0], &v[1], ()),
    ///     Edge::new(&v[1], &v[2], ()),
    ///     Edge::new(&v[2], &v[0], ())
    /// ];
    /// let face0 = Face::new(vec![wire0], ());
    /// let wire1 = wire![
    ///     Edge::new(&v[3], &v[4], ()),
    ///     Edge::new(&v[4], &v[5], ()),
    ///     Edge::new(&v[5], &v[3], ())
    /// ];
    /// let face1 = Face::new(vec![wire1], ());
    /// let shell: Shell<_, _, _> = vec![face0, face1].into();
    /// assert!(!shell.is_connected());
    /// ```
    pub fn is_connected(&self) -> bool {
        let mut adjacency = self.vertex_adjacency();
        // Connecting another boundary of the same face with an edge
        for face in self {
            for wire in face.boundaries.windows(2) {
                // SAFETY: face boundaries are always non-empty, so front_vertex succeeds.
                let v0 = wire[0].front_vertex().unwrap();
                let v1 = wire[1].front_vertex().unwrap();
                // SAFETY: vertex_adjacency populates all vertices from the shell.
                adjacency.get_mut(&v0.id()).unwrap().push(v1.id());
                adjacency.get_mut(&v1.id()).unwrap().push(v0.id());
            }
        }
        check_connectivity(&mut adjacency)
    }

    /// Returns a vector consisting of shells of each connected components.
    /// # Examples
    /// ```
    /// use monstertruck_topology::Shell;
    /// // The empty shell has no connected component.
    /// assert!(Shell::<(), (), ()>::new().connected_components().is_empty());
    /// ```
    /// # Remarks
    /// Since this method uses the face adjacency matrix, multiple components
    /// are perhaps generated even if the shell is connected. In that case,
    /// there is a pair of faces such that share vertices but not edges.
    /// ```
    /// use monstertruck_topology::*;
    /// let v = Vertex::from_points(&[(); 5]);
    /// let wire0 = wire![
    ///     Edge::new(&v[0], &v[1], ()),
    ///     Edge::new(&v[1], &v[2], ()),
    ///     Edge::new(&v[2], &v[0], ()),
    /// ];
    /// let wire1 = wire![
    ///     Edge::new(&v[0], &v[3], ()),
    ///     Edge::new(&v[3], &v[4], ()),
    ///     Edge::new(&v[4], &v[0], ()),
    /// ];
    /// let shell = Shell::from(vec![
    ///     Face::new(vec![wire0], ()),
    ///     Face::new(vec![wire1], ()),
    /// ]);
    /// assert!(shell.is_connected());
    /// assert_eq!(shell.connected_components().len(), 2);
    /// ```
    pub fn connected_components(&self) -> Vec<Shell<P, C, S>> {
        let mut adjacency = self.face_adjacency();
        // Seed each component in shell order: draining the adjacency map from
        // its own (pointer-derived) key order would make the component order
        // and member order depend on the address-space layout.
        self.face_iter()
            .filter_map(|face| {
                let seed = (face.id(), face.orientation());
                if !adjacency.contains_key(&seed) {
                    return None;
                }
                let mut stack = vec![seed];
                let mut component = Shell::new();
                while let Some(key) = stack.pop() {
                    if let Some((face, adjacent)) = adjacency.remove(&key) {
                        component.push(face.clone());
                        stack.extend(
                            adjacent
                                .into_iter()
                                .map(|neighbor| (neighbor.face.id(), neighbor.face.orientation())),
                        );
                    }
                }
                Some(component)
            })
            .collect()
    }

    /// Returns the vector of all singular vertices.
    ///
    /// Here, we say that a vertex is singular if, for a sufficiently small neighborhood U of
    /// the vertex, the set U - {the vertex} is not connected.
    ///
    /// A regular, oriented, or closed shell becomes a manifold if and only if the shell has
    /// no singular vertices.
    /// # Examples
    /// ```
    /// // A regular manifold: Mobius bundle
    /// use monstertruck_topology::*;
    /// use monstertruck_topology::shell::ShellCondition;
    ///
    /// let v = Vertex::from_points(&[(), (), (), ()]);
    /// let edge = [
    ///     Edge::new(&v[0], &v[1], ()),
    ///     Edge::new(&v[1], &v[2], ()),
    ///     Edge::new(&v[2], &v[0], ()),
    ///     Edge::new(&v[1], &v[3], ()),
    ///     Edge::new(&v[3], &v[2], ()),
    ///     Edge::new(&v[0], &v[3], ()),
    /// ];
    /// let wire = vec![
    ///     wire![&edge[0], &edge[3], &edge[4], &edge[2]],
    ///     wire![&edge[1], &edge[2], &edge[5], &edge[3].inverse()],
    /// ];
    /// let shell: Shell<_, _, _> = wire.into_iter().map(|w| Face::new(vec![w], ())).collect();
    /// assert_eq!(shell.shell_condition(), ShellCondition::Regular);
    /// assert!(shell.singular_vertices().is_empty());
    /// ```
    /// ```
    /// // A closed and connected shell which has a singular vertex.
    /// use monstertruck_topology::*;
    /// use monstertruck_topology::shell::*;
    ///
    /// let v = Vertex::from_points(&[(); 7]);
    /// let edge = [
    ///     Edge::new(&v[0], &v[1], ()), // 0
    ///     Edge::new(&v[0], &v[2], ()), // 1
    ///     Edge::new(&v[0], &v[3], ()), // 2
    ///     Edge::new(&v[1], &v[2], ()), // 3
    ///     Edge::new(&v[2], &v[3], ()), // 4
    ///     Edge::new(&v[3], &v[1], ()), // 5
    ///     Edge::new(&v[0], &v[4], ()), // 6
    ///     Edge::new(&v[0], &v[5], ()), // 7
    ///     Edge::new(&v[0], &v[6], ()), // 8
    ///     Edge::new(&v[4], &v[5], ()), // 9
    ///     Edge::new(&v[5], &v[6], ()), // 10
    ///     Edge::new(&v[6], &v[4], ()), // 11
    /// ];
    /// let wire = vec![
    ///     wire![&edge[0].inverse(), &edge[1], &edge[3].inverse()],
    ///     wire![&edge[1].inverse(), &edge[2], &edge[4].inverse()],
    ///     wire![&edge[2].inverse(), &edge[0], &edge[5].inverse()],
    ///     wire![&edge[3], &edge[4], &edge[5]],
    ///     wire![&edge[6].inverse(), &edge[7], &edge[9].inverse()],
    ///     wire![&edge[7].inverse(), &edge[8], &edge[10].inverse()],
    ///     wire![&edge[8].inverse(), &edge[6], &edge[11].inverse()],
    ///     wire![&edge[9], &edge[10], &edge[11]],
    /// ];
    /// let shell: Shell<_, _, _> = wire.into_iter().map(|w| Face::new(vec![w], ())).collect();
    /// assert_eq!(shell.shell_condition(), ShellCondition::Closed);
    /// assert!(shell.is_connected());
    /// assert_eq!(shell.singular_vertices(), vec![v[0].clone()]);
    /// ```
    pub fn singular_vertices(&self) -> Vec<Vertex<P>> {
        let mut vert_wise_adjacency =
            EntryMap::new(Vertex::clone, |_| EntryMap::new(Edge::id, |_| Vec::new()));
        self.face_iter()
            .flat_map(Face::absolute_boundaries)
            .for_each(|wire| {
                let first_edge = &wire[0];
                let mut edge_iter = wire.iter().peekable();
                while let Some(edge) = edge_iter.next() {
                    let adjacency = vert_wise_adjacency.entry_or_insert(edge.back());
                    let next_edge = *edge_iter.peek().unwrap_or(&first_edge);
                    adjacency.entry_or_insert(edge).push(next_edge.id());
                    adjacency.entry_or_insert(next_edge).push(edge.id());
                }
            });
        vert_wise_adjacency
            .into_iter()
            .filter_map(|(vertex, adjacency)| {
                (!check_connectivity(&mut adjacency.into())).then_some(vertex)
            })
            .collect()
    }

    /// The precondition a shell must satisfy to be a boundary of a
    /// [`Solid`](crate::Solid): non-empty, connected, closed and manifold.
    ///
    /// This is the SINGLE definition of that invariant.
    /// [`Solid::try_new`](crate::Solid::try_new) and
    /// [`TrimmedSolid::try_new`](crate::trimmed::TrimmedSolid::try_new) both
    /// call it rather than restating it, because two copies of a precondition
    /// are two preconditions: the trimmed path used to have NO copy at all and
    /// silently produced solids the plain path would have refused (spec 011,
    /// ledger class C11).
    pub fn check_solid_boundary(&self) -> Result<()> {
        if self.is_empty() {
            Err(Error::EmptyShell)
        } else if !self.is_connected() {
            Err(Error::NotConnected)
        } else if self.shell_condition() != ShellCondition::Closed {
            Err(Error::NotClosedShell)
        } else if !self.singular_vertices().is_empty() {
            Err(Error::NotManifold)
        } else {
            Ok(())
        }
    }

    /// Returns a new shell whose surfaces are mapped by `surface_mapping`,
    /// curves are mapped by `curve_mapping` and points are mapped by `point_mapping`.
    /// # Remarks
    /// Accessing geometry elements directly in the closure will result in a deadlock.
    /// So, this method does not appear to the document.
    #[doc(hidden)]
    pub fn try_mapped<Q, D, T>(
        &self,
        mut point_mapping: impl FnMut(&P) -> Option<Q>,
        mut curve_mapping: impl FnMut(&C) -> Option<D>,
        mut surface_mapping: impl FnMut(&S) -> Option<T>,
    ) -> Option<Shell<Q, D, T>> {
        let mut vertex_map = EntryMap::new(Vertex::id, move |v| v.try_mapped(&mut point_mapping));
        let mut edge_map = EntryMap::new(
            Edge::id,
            wire::edge_entry_map_try_closure(&mut vertex_map, &mut curve_mapping),
        );
        self.face_iter()
            .map(|face| {
                let wires = face
                    .absolute_boundaries()
                    .iter()
                    .map(|wire| wire.sub_try_mapped(&mut edge_map))
                    .collect::<Option<Vec<_>>>()?;
                let surface = surface_mapping(&*face.surface.lock())?;
                let mut new_face = Face::debug_new(wires, surface).ok()?;
                if !face.orientation() {
                    new_face.invert();
                }
                Some(new_face)
            })
            .collect()
    }

    /// Returns a new shell whose surfaces are mapped by `surface_mapping`,
    /// curves are mapped by `curve_mapping` and points are mapped by `point_mapping`.
    /// # Examples
    /// ```
    /// use monstertruck_topology::*;
    /// let v = Vertex::from_points(&[0, 1, 2, 3, 4, 5, 6]);
    /// let wire0 = Wire::from(vec![
    ///     Edge::new(&v[0], &v[1], 100),
    ///     Edge::new(&v[1], &v[2], 200),
    ///     Edge::new(&v[2], &v[3], 300),
    ///     Edge::new(&v[3], &v[0], 400),
    /// ]);
    /// let wire1 = Wire::from(vec![
    ///     Edge::new(&v[4], &v[5], 500),
    ///     Edge::new(&v[6], &v[5], 600).inverse(),
    ///     Edge::new(&v[6], &v[4], 700),
    /// ]);
    /// let face0 = Face::new(vec![wire0, wire1], 10000);
    /// let face1 = face0.mapped(
    ///     &move |i: &usize| *i + 7,
    ///     &move |j: &usize| *j + 700,
    ///     &move |k: &usize| *k + 10000,
    /// );
    /// let shell0 = Shell::from(vec![face0, face1.inverse()]);
    /// let shell1 = shell0.mapped(
    ///     &move |i: &usize| *i + 50,
    ///     &move |j: &usize| *j + 5000,
    ///     &move |k: &usize| *k + 500000,
    /// );
    /// # for face in shell1.face_iter() {
    /// #    for bdry in face.absolute_boundaries() {
    /// #        assert!(bdry.is_closed());
    /// #        assert!(bdry.is_simple());
    /// #    }
    /// # }
    ///
    /// for (face0, face1) in shell0.face_iter().zip(shell1.face_iter()) {
    ///     assert_eq!(
    ///         face0.surface() + 500000,
    ///         face1.surface(),
    ///     );
    ///     assert_eq!(face0.orientation(), face1.orientation());
    ///     let biters0 = face0.boundary_iters();
    ///     let biters1 = face1.boundary_iters();
    ///     for (biter0, biter1) in biters0.into_iter().zip(biters1) {
    ///         for (edge0, edge1) in biter0.zip(biter1) {
    ///             assert_eq!(
    ///                 edge0.front().point() + 50,
    ///                 edge1.front().point(),
    ///             );
    ///             assert_eq!(
    ///                 edge0.back().point() + 50,
    ///                 edge1.back().point(),
    ///             );
    ///             assert_eq!(
    ///                 edge0.curve() + 5000,
    ///                 edge1.curve(),
    ///             );
    ///         }
    ///     }
    /// }
    /// ```
    /// # Remarks
    /// Accessing geometry elements directly in the closure will result in a deadlock.
    /// So, this method does not appear to the document.
    #[doc(hidden)]
    pub fn mapped<Q, D, T>(
        &self,
        mut point_mapping: impl FnMut(&P) -> Q,
        mut curve_mapping: impl FnMut(&C) -> D,
        mut surface_mapping: impl FnMut(&S) -> T,
    ) -> Shell<Q, D, T> {
        let mut vertex_map = EntryMap::new(Vertex::id, |v| v.mapped(&mut point_mapping));
        let mut edge_map = EntryMap::new(
            Edge::id,
            wire::edge_entry_map_closure(&mut vertex_map, &mut curve_mapping),
        );
        self.face_iter()
            .map(|face| {
                let wires: Vec<Wire<_, _>> = face
                    .absolute_boundaries()
                    .iter()
                    .map(|wire| wire.sub_mapped(&mut edge_map))
                    .collect();
                let surface = surface_mapping(&*face.surface.lock());
                let mut new_face = Face::new_unchecked(wires, surface);
                if !face.orientation() {
                    new_face.invert();
                }
                new_face
            })
            .collect()
    }

    /// Returns the consistence of the geometry of end vertices
    /// and the geometry of edge.
    #[inline(always)]
    pub fn is_geometric_consistent(&self) -> bool
    where
        P: Tolerance,
        C: BoundedCurve<Point = P>,
        S: IncludeCurve<C>, {
        self.iter().all(|face| face.is_geometric_consistent())
    }

    /// Cuts one edge into two edges at vertex.
    ///
    /// # Returns
    /// Returns the tuple of new edges created by cutting the edge.
    /// Every occurrence, including both uses of a seam in one wire, shares
    /// those new edges with its original orientation.
    ///
    /// # Failures
    /// Returns `None` and not edit `self` if:
    /// - there is no edge corresponding to `edge_id` in the shell,
    /// - `vertex` is already included in the shell, or
    /// - cutting of edge fails.
    pub fn cut_edge(
        &mut self,
        edge_id: EdgeId<C>,
        vertex: &Vertex<P>,
    ) -> Option<(Edge<P, C>, Edge<P, C>)>
    where
        P: Clone,
        C: Cut<Point = P> + SearchParameter<CurveParameter, Point = P>,
    {
        if self.vertex_iter().any(|v| &v == vertex) {
            return None;
        }
        let edges = self
            .edge_iter()
            .find(|edge| edge.id() == edge_id)?
            .absolute_clone()
            .cut(vertex)?;
        self.iter_mut()
            .flat_map(|face| face.boundaries.iter_mut())
            .for_each(|wire| {
                let mut index = 0;
                while index < wire.len() {
                    if wire[index].id() == edge_id {
                        let (first, second) = if wire[index].orientation() {
                            (edges.0.clone(), edges.1.clone())
                        } else {
                            (edges.1.inverse(), edges.0.inverse())
                        };
                        wire[index] = first;
                        wire.insert(index + 1, second);
                        index += 2;
                    } else {
                        index += 1;
                    }
                }
            });
        Some(edges)
    }
    /// Removes `vertex` from `self` by concat two edges on both sides.
    ///
    /// # Returns
    /// Returns the new created edge.
    ///
    /// # Failures
    /// Returns `None` without editing if:
    /// - there is no vertex corresponding to `vertex_id` in the shell,
    /// - the vertex is included in more than two boundary uses,
    /// - the uses do not share exactly two oppositely oriented edges, or
    /// - concatenating the edges fails.
    pub fn remove_vertex_by_concat_edges(&mut self, vertex_id: VertexId<P>) -> Option<Edge<P, C>>
    where
        P: Debug,
        C: Concat<C, Point = P, Output = C> + Invertible + ParameterTransform, {
        let mut occurrences = [None, None];
        let mut count = 0;
        for (face_index, face) in self.iter().enumerate() {
            for (wire_index, wire) in face.boundaries.iter().enumerate() {
                for (index, edge) in wire.iter().enumerate() {
                    if edge.back().id() == vertex_id {
                        if count == occurrences.len() {
                            return None;
                        }
                        occurrences[count] = Some((face_index, wire_index, index));
                        count += 1;
                    }
                }
            }
        }
        let (face, wire, index) = occurrences[0]?;
        let boundary = &self[face].boundaries[wire];
        let incoming = &boundary[index];
        let outgoing = &boundary[(index + 1) % boundary.len()];
        if let Some((face, wire, index)) = occurrences[1] {
            let boundary = &self[face].boundaries[wire];
            let other_incoming = &boundary[index];
            let other_outgoing = &boundary[(index + 1) % boundary.len()];
            if !incoming.is_same(other_outgoing)
                || !outgoing.is_same(other_incoming)
                || incoming.orientation() == other_outgoing.orientation()
                || outgoing.orientation() == other_incoming.orientation()
            {
                return None;
            }
        }
        let edge = incoming.concat(outgoing).ok()?;
        let incoming_id = incoming.id();
        let incoming_orientation = incoming.orientation();
        self.iter_mut()
            .flat_map(|face| face.boundaries.iter_mut())
            .filter(|wire| wire.iter().any(|edge| edge.back().id() == vertex_id))
            .for_each(|wire| {
                *wire = wire
                    .iter()
                    .filter_map(|original| {
                        if original.back().id() == vertex_id {
                            Some(
                                if original.id() == incoming_id
                                    && original.orientation() == incoming_orientation
                                {
                                    edge.clone()
                                } else {
                                    edge.inverse()
                                },
                            )
                        } else if original.front().id() == vertex_id {
                            None
                        } else {
                            Some(original.clone())
                        }
                    })
                    .collect();
            });
        Some(edge)
    }

    /// Creates display struct for debugging the shell.
    /// # Examples
    /// ```
    /// use monstertruck_topology::*;
    /// use monstertruck_topology::shell::ShellCondition;
    /// use ShellDisplayFormat as SDF;
    ///
    /// let v = Vertex::from_points(&[0, 1, 2, 3]);
    /// let edge = [
    ///     Edge::new(&v[0], &v[1], ()), // 0
    ///     Edge::new(&v[1], &v[2], ()), // 1
    ///     Edge::new(&v[2], &v[0], ()), // 2
    ///     Edge::new(&v[1], &v[3], ()), // 3
    ///     Edge::new(&v[3], &v[2], ()), // 4
    ///     Edge::new(&v[0], &v[3], ()), // 5
    /// ];
    /// let wire = vec![
    ///     wire![&edge[0], &edge[3], &edge[4], &edge[2]],
    ///     wire![&edge[1], &edge[2], &edge[5], &edge[3].inverse()],
    /// ];
    /// let shell: Shell<_, _, _> = wire.into_iter().map(|w| Face::new(vec![w], ())).collect();
    ///
    /// let vertex_format = VertexDisplayFormat::AsPoint;
    /// let edge_format = EdgeDisplayFormat::VerticesTuple { vertex_format };
    /// let wire_format = WireDisplayFormat::EdgesList { edge_format };
    /// let face_format = FaceDisplayFormat::LoopsListTuple { wire_format };
    ///
    /// assert_eq!(
    ///     &format!("{:?}", shell.display(SDF::FacesListTuple {face_format})),
    ///     "Shell([Face([[(0, 1), (1, 3), (3, 2), (2, 0)]]), Face([[(1, 2), (2, 0), (0, 3), (3, 1)]])])",
    /// );
    /// assert_eq!(
    ///     &format!("{:?}", shell.display(SDF::FacesList {face_format})),
    ///     "[Face([[(0, 1), (1, 3), (3, 2), (2, 0)]]), Face([[(1, 2), (2, 0), (0, 3), (3, 1)]])]",
    /// );
    /// ```
    pub fn display(
        &self,
        format: ShellDisplayFormat,
    ) -> DebugDisplay<'_, Self, ShellDisplayFormat> {
        DebugDisplay {
            entity: self,
            format,
        }
    }
}

impl<P, C, S> Clone for Shell<P, C, S> {
    #[inline(always)]
    fn clone(&self) -> Shell<P, C, S> {
        Shell {
            face_list: self.face_list.clone(),
        }
    }
}

impl<P, C, S, T> From<T> for Shell<P, C, S>
where Vec<Face<P, C, S>>: From<T>
{
    #[inline(always)]
    fn from(faces: T) -> Shell<P, C, S> {
        Shell {
            face_list: faces.into(),
        }
    }
}

impl<P, C, S> FromIterator<Face<P, C, S>> for Shell<P, C, S> {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = Face<P, C, S>>>(iter: I) -> Shell<P, C, S> {
        Shell {
            face_list: Vec::from_iter(iter),
        }
    }
}

impl<P, C, S> IntoIterator for Shell<P, C, S> {
    type Item = Face<P, C, S>;
    type IntoIter = std::vec::IntoIter<Face<P, C, S>>;
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter { self.face_list.into_iter() }
}

impl<'a, P, C, S> IntoIterator for &'a Shell<P, C, S> {
    type Item = &'a Face<P, C, S>;
    type IntoIter = std::slice::Iter<'a, Face<P, C, S>>;
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter { self.face_list.iter() }
}

impl<P, C, S> AsRef<Vec<Face<P, C, S>>> for Shell<P, C, S> {
    #[inline(always)]
    fn as_ref(&self) -> &Vec<Face<P, C, S>> { &self.face_list }
}

impl<P, C, S> AsRef<[Face<P, C, S>]> for Shell<P, C, S> {
    #[inline(always)]
    fn as_ref(&self) -> &[Face<P, C, S>] { &self.face_list }
}

impl<P, C, S> std::ops::Deref for Shell<P, C, S> {
    type Target = Vec<Face<P, C, S>>;
    #[inline(always)]
    fn deref(&self) -> &Vec<Face<P, C, S>> { &self.face_list }
}

impl<P, C, S> std::ops::DerefMut for Shell<P, C, S> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Vec<Face<P, C, S>> { &mut self.face_list }
}

impl<P, C, S> std::borrow::Borrow<Vec<Face<P, C, S>>> for Shell<P, C, S> {
    #[inline(always)]
    fn borrow(&self) -> &Vec<Face<P, C, S>> { &self.face_list }
}

impl<P, C, S> std::borrow::Borrow<[Face<P, C, S>]> for Shell<P, C, S> {
    #[inline(always)]
    fn borrow(&self) -> &[Face<P, C, S>] { &self.face_list }
}

impl<P, C, S> Extend<Face<P, C, S>> for Shell<P, C, S> {
    #[inline(always)]
    fn extend<T: IntoIterator<Item = Face<P, C, S>>>(&mut self, iter: T) {
        self.face_list.extend(iter)
    }
}

impl<P, C, S> Default for Shell<P, C, S> {
    #[inline(always)]
    fn default() -> Self {
        Self {
            face_list: Vec::new(),
        }
    }
}

impl<P, C, S> PartialEq for Shell<P, C, S> {
    fn eq(&self, other: &Self) -> bool { self.face_list == other.face_list }
}

impl<P, C, S> Eq for Shell<P, C, S> {}

/// The reference iterator over all faces in shells
pub type FaceIter<'a, P, C, S> = std::slice::Iter<'a, Face<P, C, S>>;
/// The mutable reference iterator over all faces in shells
pub type FaceIterMut<'a, P, C, S> = std::slice::IterMut<'a, Face<P, C, S>>;
/// The into iterator over all faces in shells
pub type FaceIntoIter<P, C, S> = std::vec::IntoIter<Face<P, C, S>>;
/// The reference parallel iterator over all faces in shells
pub type FaceParallelIter<'a, P, C, S> = <Vec<Face<P, C, S>> as IntoParallelRefIterator<'a>>::Iter;
/// The mutable reference parallel iterator over all faces in shells
pub type FaceParallelIterMut<'a, P, C, S> =
    <Vec<Face<P, C, S>> as IntoParallelRefMutIterator<'a>>::Iter;
/// The into parallel iterator over all faces in shells
pub type FaceParallelIntoIter<P, C, S> = <Vec<Face<P, C, S>> as IntoParallelIterator>::Iter;

/// The shell conditions being determined by the half-edge model.
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum ShellCondition {
    /// This shell is not regular.
    /// # Examples
    /// ```
    /// use monstertruck_topology::*;
    /// use monstertruck_topology::shell::ShellCondition;
    /// let v = Vertex::from_points(&[(); 5]);
    /// let edge = [
    ///    Edge::new(&v[0], &v[1], ()),
    ///    Edge::new(&v[0], &v[2], ()),
    ///    Edge::new(&v[0], &v[3], ()),
    ///    Edge::new(&v[0], &v[4], ()),
    ///    Edge::new(&v[1], &v[2], ()),
    ///    Edge::new(&v[1], &v[3], ()),
    ///    Edge::new(&v[1], &v[4], ()),
    /// ];
    /// let wire = vec![
    ///    wire![&edge[0], &edge[4], &edge[1].inverse()],
    ///    wire![&edge[0], &edge[5], &edge[2].inverse()],
    ///    wire![&edge[0], &edge[6], &edge[3].inverse()],
    /// ];
    /// let shell: Shell<_, _, _> = wire.into_iter().map(|w| Face::new(vec![w], ())).collect();
    /// // The shell is irregular because three faces share edge[0].
    /// assert_eq!(shell.shell_condition(), ShellCondition::Irregular);
    /// ```
    Irregular,
    /// All edges are shared by at most two faces.
    /// # Examples
    /// ```
    /// use monstertruck_topology::*;
    /// use monstertruck_topology::shell::ShellCondition;
    /// let v = Vertex::from_points(&[(); 6]);
    /// let edge = [
    ///     Edge::new(&v[0], &v[1], ()),
    ///     Edge::new(&v[0], &v[2], ()),
    ///     Edge::new(&v[1], &v[2], ()),
    ///     Edge::new(&v[1], &v[3], ()),
    ///     Edge::new(&v[1], &v[4], ()),
    ///     Edge::new(&v[2], &v[4], ()),
    ///     Edge::new(&v[2], &v[5], ()),
    ///     Edge::new(&v[3], &v[4], ()),
    ///     Edge::new(&v[4], &v[5], ()),
    /// ];
    /// let wire = vec![
    ///     wire![&edge[0], &edge[2], &edge[1].inverse()],
    ///     wire![&edge[3], &edge[7], &edge[4].inverse()],
    ///     wire![&edge[5], &edge[8], &edge[6].inverse()],
    ///     wire![&edge[2], &edge[5], &edge[4].inverse()],
    /// ];
    /// let shell: Shell<_, _, _> = wire.into_iter().map(|w| Face::new(vec![w], ())).collect();
    /// // This shell is regular, but not oriented.
    /// // It is because the orientations of shell[0] and shell[3] are incompatible on edge[2].
    /// assert_eq!(shell.shell_condition(), ShellCondition::Regular);
    /// ```
    Regular,
    /// The orientations of faces are compatible.
    /// # Examples
    /// ```
    /// use monstertruck_topology::*;
    /// use monstertruck_topology::shell::ShellCondition;
    /// let v = Vertex::from_points(&[(); 6]);
    /// let edge = [
    ///     Edge::new(&v[0], &v[1] ,()),
    ///     Edge::new(&v[0], &v[2] ,()),
    ///     Edge::new(&v[1], &v[2] ,()),
    ///     Edge::new(&v[1], &v[3] ,()),
    ///     Edge::new(&v[1], &v[4] ,()),
    ///     Edge::new(&v[2], &v[4] ,()),
    ///     Edge::new(&v[2], &v[5] ,()),
    ///     Edge::new(&v[3], &v[4] ,()),
    ///     Edge::new(&v[4], &v[5] ,()),
    /// ];
    /// let wire = vec![
    ///     wire![&edge[0], &edge[2], &edge[1].inverse()],
    ///     wire![&edge[3], &edge[7], &edge[4].inverse()],
    ///     wire![&edge[5], &edge[8], &edge[6].inverse()],
    ///     wire![&edge[2].inverse(), &edge[4], &edge[5].inverse()],
    /// ];
    /// let shell: Shell<_, _, _> = wire.into_iter().map(|w| Face::new(vec![w], ())).collect();
    /// // The orientations of all faces in the shell are compatible on the shared edges.
    /// // This shell is not closed because edge[0] is included in only the 0th face.
    /// assert_eq!(shell.shell_condition(), ShellCondition::Oriented);
    /// ```
    Oriented,
    /// All edges are shared by two faces.
    /// # Examples
    /// ```
    /// use monstertruck_topology::*;
    /// use monstertruck_topology::shell::ShellCondition;
    /// let v = Vertex::from_points(&[(); 8]);
    /// let edge = [
    ///     Edge::new(&v[0], &v[1] ,()),
    ///     Edge::new(&v[1], &v[2] ,()),
    ///     Edge::new(&v[2], &v[3] ,()),
    ///     Edge::new(&v[3], &v[0] ,()),
    ///     Edge::new(&v[0], &v[4] ,()),
    ///     Edge::new(&v[1], &v[5] ,()),
    ///     Edge::new(&v[2], &v[6] ,()),
    ///     Edge::new(&v[3], &v[7] ,()),
    ///     Edge::new(&v[4], &v[5] ,()),
    ///     Edge::new(&v[5], &v[6] ,()),
    ///     Edge::new(&v[6], &v[7] ,()),
    ///     Edge::new(&v[7], &v[4] ,()),
    /// ];
    /// let wire = vec![
    ///     wire![&edge[0], &edge[1], &edge[2], &edge[3]],
    ///     wire![&edge[0].inverse(), &edge[4], &edge[8], &edge[5].inverse()],
    ///     wire![&edge[1].inverse(), &edge[5], &edge[9], &edge[6].inverse()],
    ///     wire![&edge[2].inverse(), &edge[6], &edge[10], &edge[7].inverse()],
    ///     wire![&edge[3].inverse(), &edge[7], &edge[11], &edge[4].inverse()],
    ///     wire![&edge[8], &edge[9], &edge[10], &edge[11]],
    /// ];
    /// let mut shell: Shell<_, _, _> = wire.into_iter().map(|w| Face::new(vec![w], ())).collect();
    /// shell[5].invert();
    /// assert_eq!(shell.shell_condition(), ShellCondition::Closed);
    /// ```
    Closed,
}

impl std::ops::BitAnd for ShellCondition {
    type Output = Self;
    fn bitand(self, other: Self) -> Self {
        match (self, other) {
            (Self::Irregular, _) => Self::Irregular,
            (_, Self::Irregular) => Self::Irregular,
            (Self::Regular, _) => Self::Regular,
            (_, Self::Regular) => Self::Regular,
            (Self::Oriented, _) => Self::Oriented,
            (_, Self::Oriented) => Self::Oriented,
            (Self::Closed, Self::Closed) => Self::Closed,
        }
    }
}

#[derive(Debug, Clone)]
struct Boundaries<C> {
    checked: HashSet<EdgeId<C>>,
    boundaries: HashMap<EdgeId<C>, bool>,
    condition: ShellCondition,
}

impl<C> Boundaries<C> {
    #[inline(always)]
    fn new() -> Self {
        Self {
            checked: Default::default(),
            boundaries: Default::default(),
            condition: ShellCondition::Oriented,
        }
    }

    #[inline(always)]
    fn insert<P>(&mut self, edge: &Edge<P, C>) {
        self.condition = self.condition
            & match (
                self.checked.insert(edge.id()),
                self.boundaries.insert(edge.id(), edge.orientation()),
            ) {
                (true, None) => ShellCondition::Oriented,
                (false, None) => ShellCondition::Irregular,
                (true, Some(_)) => panic!("unexpected case!"),
                (false, Some(ori)) => {
                    self.boundaries.remove(&edge.id());
                    match edge.orientation() == ori {
                        true => ShellCondition::Regular,
                        false => ShellCondition::Oriented,
                    }
                }
            }
    }

    #[inline(always)]
    fn condition(&self) -> ShellCondition {
        if self.condition == ShellCondition::Oriented && self.boundaries.is_empty() {
            ShellCondition::Closed
        } else {
            self.condition
        }
    }
}

impl<P, C> FromIterator<Edge<P, C>> for Boundaries<C> {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = Edge<P, C>>>(iter: I) -> Self {
        let mut boundaries = Boundaries::new();
        iter.into_iter().for_each(|edge| boundaries.insert(&edge));
        boundaries
    }
}

fn check_connectivity<T>(adjacency: &mut HashMap<T, Vec<T>>) -> bool
where T: Eq + Clone + Hash {
    create_one_component(adjacency);
    adjacency.is_empty()
}

fn create_one_component<T, U>(adjacency: &mut HashMap<T, Vec<U>>) -> Vec<T>
where
    T: Eq + Hash + Clone,
    U: As<T>, {
    let mut iter = adjacency.keys();
    let first = match iter.next() {
        Some(key) => key.clone(),
        None => return Vec::new(),
    };
    let mut stack = vec![first.as_()];
    let mut res = Vec::new();
    while let Some(i) = stack.pop() {
        if let Some(vec) = adjacency.remove(&i) {
            res.push(i);
            stack.extend(vec.into_iter().map(|x| x.as_()));
        }
    }
    res
}

impl<P: Debug, C: Debug, S: Debug> Debug for DebugDisplay<'_, Shell<P, C, S>, ShellDisplayFormat> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.format {
            ShellDisplayFormat::FacesList { face_format } => f
                .debug_list()
                .entries(
                    self.entity
                        .face_iter()
                        .map(|face| face.display(face_format)),
                )
                .finish(),
            ShellDisplayFormat::FacesListTuple { face_format } => f
                .debug_tuple("Shell")
                .field(&DebugDisplay {
                    entity: self.entity,
                    format: ShellDisplayFormat::FacesList { face_format },
                })
                .finish(),
        }
    }
}

impl<P: Send, C: Send, S: Send> FromParallelIterator<Face<P, C, S>> for Shell<P, C, S> {
    fn from_par_iter<I>(par_iter: I) -> Self
    where I: IntoParallelIterator<Item = Face<P, C, S>> {
        Self::from(Vec::from_par_iter(par_iter))
    }
}

impl<P: Send, C: Send, S: Send> IntoParallelIterator for Shell<P, C, S> {
    type Item = Face<P, C, S>;
    type Iter = FaceParallelIntoIter<P, C, S>;
    fn into_par_iter(self) -> Self::Iter { self.face_list.into_par_iter() }
}

impl<'a, P: Send + 'a, C: Send + 'a, S: Send + 'a> IntoParallelRefIterator<'a> for Shell<P, C, S> {
    type Item = &'a Face<P, C, S>;
    type Iter = FaceParallelIter<'a, P, C, S>;
    fn par_iter(&'a self) -> Self::Iter { self.face_list.par_iter() }
}

impl<'a, P: Send + 'a, C: Send + 'a, S: Send + 'a> IntoParallelRefMutIterator<'a>
    for Shell<P, C, S>
{
    type Item = &'a mut Face<P, C, S>;
    type Iter = FaceParallelIterMut<'a, P, C, S>;
    fn par_iter_mut(&'a mut self) -> Self::Iter { self.face_list.par_iter_mut() }
}

impl<P: Send, C: Send, S: Send> ParallelExtend<Face<P, C, S>> for Shell<P, C, S> {
    fn par_extend<I>(&mut self, par_iter: I)
    where I: IntoParallelIterator<Item = Face<P, C, S>> {
        self.face_list.par_extend(par_iter)
    }
}

#[cfg(test)]
mod adjacency_identity_tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn directed_face_identity_preserves_inverse_uses_and_ignores_mutable_payloads() {
        let vertices = Vertex::from_points([(); 3]);
        let wire = (0..3)
            .map(|index| {
                Edge::new(
                    &vertices[index],
                    &vertices[(index + 1) % 3],
                    Cell::new(0_u8),
                )
            })
            .collect();
        let face = Face::new(vec![wire], 0_u32);
        let inverse = face.inverse();
        assert_eq!(face.id(), inverse.id());
        assert_ne!(face, inverse);
        let shell: Shell<_, _, _> = vec![face.clone(), inverse.clone()].into();
        let adjacency = shell.face_adjacency();
        assert_eq!(adjacency.len(), 2);
        for source in [&face, &inverse] {
            let (retained, neighbors) = &adjacency[&(source.id(), source.orientation())];
            assert_eq!(*retained, source);
            assert_eq!(neighbors.len(), 1);
            assert_eq!(neighbors[0].face.id(), source.id());
            assert_ne!(neighbors[0].face.orientation(), source.orientation());
            assert_eq!(neighbors[0].common_edges.len(), 3);
        }
        face.set_surface(42);
        face.absolute_boundaries()[0][0].with_curve(|curve| curve.set(7));
        assert_eq!(adjacency[&(face.id(), face.orientation())].0.surface(), 42);
        assert_eq!(adjacency[&(face.id(), face.orientation())].1.len(), 1);
        let components = shell.connected_components();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 2);
        assert_eq!(components[0][0], face);
        assert_eq!(components[0][1], inverse);
        let duplicated: Shell<_, _, _> = vec![face.clone(), face.clone(), inverse.clone()].into();
        let duplicate_adjacency = duplicated.face_adjacency();
        assert_eq!(duplicate_adjacency.len(), 2);
        assert_eq!(
            duplicate_adjacency[&(face.id(), face.orientation())].1[0]
                .common_edges
                .len(),
            6,
        );
        let components = duplicated.connected_components();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 2);
        assert_eq!(components[0][0], face);
        assert_eq!(components[0][1], inverse);
    }
}
