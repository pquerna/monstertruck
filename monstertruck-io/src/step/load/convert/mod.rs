use super::report::{LossCategory as Cat, LossReason as Why, LossSink};
#[cfg(test)]
use super::step_geometry::SurfaceCurveKind as StepSurfaceCurveKind;
use super::*;
use monstertruck_topology::compress::*;
use std::collections::HashSet;

mod periodic_edges;

/// A compressed shell in the STEP loader's concrete geometry types.
///
/// The `*_reported` conversions (spec 011 T7) return the value paired with a
/// [`ShellLoadReport`], and spelling the four generic arguments inline at every
/// such signature is what tripped `clippy::type_complexity`. Naming the three
/// concrete instantiations once keeps the reported and unreported variants
/// visibly the same type.
pub type StepCompressedShell = CompressedShell<Point3, Curve3D, Surface>;

/// A compressed trimmed shell (exact trim curves preserved) in the STEP
/// loader's concrete geometry types. See [`StepCompressedShell`].
pub type StepCompressedTrimmedShell =
    CompressedTrimmedShell<Point3, Curve3D, Surface, step_geometry::StepParameterCurve>;

/// A compressed solid in the STEP loader's concrete geometry types.
/// See [`StepCompressedShell`].
pub type StepCompressedSolid = CompressedSolid<Point3, Curve3D, Surface>;

/// A source solid retaining declared two-dimensional trim curves per coedge.
pub type StepCompressedTrimmedSolid = CompressedTrimmedSolid<Point3, Curve3D, Surface, Curve2D>;

type CompactTrimmedShell = CompressedTrimmedShell<Point3, Curve3D, Surface, Curve2D>;
type CompactBoundary = (Vec<CompressedEdgeUse<Curve2D>>, Vec<SourceCoedgeProvenance>);

impl Table {
    fn place_holder_edge_any_to_index_and_edge_curve(
        &self,
        edge: &PlaceHolder<EdgeAnyHolder>,
    ) -> Option<(u64, EdgeCurveHolder)> {
        if let PlaceHolder::Owned(EdgeAnyHolder::OrientedEdge(edge)) = edge {
            return Some((edge.edge_element_idx()?, edge.edge_element_holder(self)?));
        }
        use PlaceHolder::Ref;
        let Ref(Name::Entity(idx)) = edge else {
            return None;
        };
        self.oriented_edge
            .get(idx)
            .and_then(|oriented_edge| {
                Some((
                    oriented_edge.edge_element_idx()?,
                    oriented_edge.edge_element_holder(self)?,
                ))
            })
            .or_else(|| {
                self.edge_curve
                    .get(idx)
                    .map(|edge_curve| (*idx, edge_curve.clone()))
            })
    }
    fn face_any_to_orientation_and_face(
        &self,
        face: Option<FaceAnyHolder>,
    ) -> Option<(bool, FaceSurfaceHolder)> {
        match face? {
            FaceAnyHolder::FaceSurface(face) => Some((true, face)),
            FaceAnyHolder::OrientedFace(oriented_face) => {
                let face_element = oriented_face.face_element_holder(self)?;
                Some((oriented_face.orientation, face_element))
            }
        }
    }

    /// The entity id behind a reference place-holder, when it has one.
    pub(super) fn ref_id<T>(place_holder: &PlaceHolder<T>) -> Option<u64> {
        if let PlaceHolder::Ref(Name::Entity(idx)) = place_holder {
            Some(*idx)
        } else {
            None
        }
    }

    /// Resolve a face bound to the `EDGE_LOOP` the loader can represent, and
    /// REPORT the ones it cannot.
    ///
    /// This is the site of the census's only genuinely silent drop: the three
    /// non-`EDGE_LOOP` outcomes used to be one `?` on an `Option`, so a wire
    /// vanished with no error, no message and no count.
    fn resolved_edge_loop(
        &self,
        bound: &FaceBoundHolder,
        sink: &LossSink,
    ) -> Option<EdgeLoopHolder> {
        match bound.resolve_bound(self) {
            BoundResolution::EdgeLoop(edge_loop) => Some(edge_loop),
            BoundResolution::VertexLoop(id) => {
                sink.lost(Cat::Wire, Why::DegenerateVertexLoop, Some(id), None);
                None
            }
            BoundResolution::Unresolved(id) => {
                // A record that IS in the file but has no arm lands in
                // `Table::dummy` with its name intact, which is a different
                // defect from an id that resolves to nothing at all.
                let unrecognised = id.and_then(|id| self.dummy.get(&id));
                let (why, detail) = match unrecognised {
                    Some(dummy) => (
                        Why::BoundNotALoopWeImplement,
                        Some(dummy.record.chars().take(80).collect::<String>()),
                    ),
                    None => (Why::BoundUnresolved, None),
                };
                sink.lost(Cat::Wire, why, id, detail);
                None
            }
        }
    }

    fn shell_vertices(
        &self,
        shell: &ShellHolder,
        sink: &LossSink,
    ) -> (Vec<Point3>, HashMap<u64, usize>) {
        use PlaceHolder::Ref;
        let mut vidx_map = HashMap::<u64, usize>::new();
        // Distinct ids we have already TRIED, as opposed to ids we succeeded on.
        //
        // The dedup guard used to be `!vidx_map.contains_key(&idx)` with the
        // insert done BEFORE the fallible resolve, so a vertex that failed still
        // consumed an index and every later vertex's `vidx_map` entry pointed one
        // slot past its own point -- silently wrong topology, not merely lossy.
        // Keeping "attempted" separate from "kept" makes the index dense over
        // successes; when nothing fails the numbering is identical to before.
        let mut attempted = HashSet::<u64>::new();
        let vertex_to_point = |v: PlaceHolder<VertexPointHolder>| {
            let Ref(Name::Entity(idx)) = v else {
                sink.listed(Cat::Vertex, 1);
                sink.lost(
                    Cat::Vertex,
                    Why::VertexUnresolved,
                    None,
                    Some("the vertex is not an entity reference".to_owned()),
                );
                return None;
            };
            if !attempted.insert(idx) {
                return None;
            }
            sink.listed(Cat::Vertex, 1);
            match EntityTable::<VertexPointHolder>::get_owned(self, idx) {
                Ok(vertex) => {
                    let index = vidx_map.len();
                    vidx_map.insert(idx, index);
                    sink.kept(Cat::Vertex, 1);
                    Some(Point3::from(&vertex.vertex_geometry))
                }
                Err(e) => {
                    eprintln!("{e}");
                    sink.lost(
                        Cat::Vertex,
                        Why::VertexUnresolved,
                        Some(idx),
                        Some(e.to_string()),
                    );
                    None
                }
            }
        };
        let vertices: Vec<Point3> = shell
            .cfs_faces_holder(self)
            .filter_map(move |face| self.face_any_to_orientation_and_face(face))
            .flat_map(move |(_, face)| face.bounds_holder(self))
            // Wires are counted ONCE, in `shell_faces`/`shell_trimmed_faces`.
            // This walk only needs the edges, so it uses the non-reporting
            // resolver and cannot double-count a lost vertex loop.
            .filter_map(move |(_, bound)| bound?.bound_holder(self))
            .flat_map(move |bound| bound.edge_list)
            .filter_map(move |edge| self.place_holder_edge_any_to_index_and_edge_curve(&edge))
            .flat_map(move |(_, edge)| [edge.edge_start, edge.edge_end])
            .filter_map(vertex_to_point)
            .collect();
        (vertices, vidx_map)
    }

    fn shell_edges(
        &self,
        shell: &ShellHolder,
        vidx_map: &HashMap<u64, usize>,
        sink: &LossSink,
    ) -> (Vec<CompressedEdge<Curve3D>>, HashMap<u64, usize>) {
        use PlaceHolder::Ref;
        let mut eidx_map = HashMap::<u64, usize>::new();
        // Same split as `shell_vertices`, for the same reason: a failed edge used
        // to consume an index and shift every later edge's `eidx_map` entry.
        let mut attempted = HashSet::<u64>::new();
        let edge_curve_to_compressed_edge = |(idx, edge): (u64, EdgeCurveHolder)| {
            if !attempted.insert(idx) {
                return None;
            }
            sink.listed(Cat::Edge, 1);
            let edge_curve = match edge.clone().into_owned(self) {
                Ok(edge_curve) => edge_curve,
                Err(e) => {
                    eprintln!("{e}");
                    sink.lost(
                        Cat::Edge,
                        Why::EdgeUnresolved,
                        Some(idx),
                        Some(e.to_string()),
                    );
                    return None;
                }
            };
            let curve = match edge_curve.parse_curve3d() {
                Ok(curve) => curve,
                Err(e) => {
                    eprintln!("{e}");
                    sink.lost(
                        Cat::Edge,
                        Why::EdgeCurveRefused,
                        Some(idx),
                        Some(e.to_string()),
                    );
                    return None;
                }
            };
            let (Ref(Name::Entity(front_idx)), Ref(Name::Entity(back_idx))) =
                (&edge.edge_start, &edge.edge_end)
            else {
                sink.lost(
                    Cat::Edge,
                    Why::EdgeEndpointUnresolved,
                    Some(idx),
                    Some("edge_start or edge_end is not an entity reference".to_owned()),
                );
                return None;
            };
            let (Some(front), Some(back)) = (vidx_map.get(front_idx), vidx_map.get(back_idx))
            else {
                sink.lost(
                    Cat::Edge,
                    Why::EdgeEndpointUnresolved,
                    Some(idx),
                    Some(format!("no kept vertex for #{front_idx} or #{back_idx}",)),
                );
                return None;
            };
            let index = eidx_map.len();
            eidx_map.insert(idx, index);
            sink.kept(Cat::Edge, 1);
            Some(CompressedEdge {
                vertices: (*front, *back),
                curve,
            })
        };
        let edges: Vec<CompressedEdge<Curve3D>> = shell
            .cfs_faces_holder(self)
            .filter_map(move |face| self.face_any_to_orientation_and_face(face))
            .flat_map(move |(_, face)| face.bounds_holder(self))
            // Wires are counted ONCE, in `shell_faces`/`shell_trimmed_faces`.
            // This walk only needs the edges, so it uses the non-reporting
            // resolver and cannot double-count a lost vertex loop.
            .filter_map(move |(_, bound)| bound?.bound_holder(self))
            .flat_map(move |bound| bound.edge_list)
            .filter_map(move |edge| self.place_holder_edge_any_to_index_and_edge_curve(&edge))
            .filter_map(edge_curve_to_compressed_edge)
            .collect();
        (edges, eidx_map)
    }
    /// Composes a `FACE_BOUND`'s own orientation with the face's `same_sense`,
    /// so a boundary loop is emitted in the SURFACE's sense and not the FACE's.
    ///
    /// # Why this composition exists (ledger C15, spec 013 V1)
    ///
    /// ISO 10303-42 orients an `ADVANCED_FACE`'s loops about the FACE normal,
    /// and the face normal is the surface normal only when `same_sense` is
    /// `.T.`. `CompressedFace`/`CompressedTrimmedFace` want the other
    /// convention: `CompressDirector::create_cface` stores `Face::boundaries`
    /// -- the ABSOLUTE boundaries, in the surface's own sense -- alongside the
    /// orientation flag, and `CompressedFace::create_face` rebuilds with
    /// `Face::try_new(stored, surface)` followed by `invert()`, which flips the
    /// EFFECTIVE traversal a second time.
    ///
    /// Passing STEP's loops through verbatim therefore traverses every
    /// `same_sense = .F.` face backwards relative to its neighbours. Measured
    /// on `occt-cube.step` (2026-08-01): `#17`, `#237` and `#331` are `.F.`
    /// against `#137`, `#284`, `#338` `.T.`, and the loaded shell came out
    /// `ShellCondition::Regular` -- all twelve edges traversed the SAME way by
    /// both their faces, none of them opposite.
    ///
    /// `normalize_trimmed_shell_orientation` then repaired that topologically,
    /// by 2-colouring from face 0. That is correct as far as it goes and its
    /// doc is explicit that it "does NOT decide global outwardness" -- but face
    /// 0 of the cube is `#17`, a `.F.` face, so the whole shell landed INWARD:
    /// six `false` flags and a divergence-theorem volume of **-1000** on a
    /// 1000-volume cube. That number is what `OperandVolume::trusted` (`> 0`,
    /// `<=` a CERTIFIED bounding box since spec 014 W2 -- the vertex hull it
    /// used before was not one) and `verify_volume_conservation` bound booleans
    /// with.
    ///
    /// The reversal machinery this reuses is the one already here for
    /// `FACE_BOUND.orientation`: flip each use, reverse the loop, invert the
    /// exact trim, and pick the other seam curve. Composing is all that was
    /// missing.
    #[inline]
    fn absolute_bound_orientation(bound: &FaceBoundHolder, face_sense: bool) -> bool {
        bound.orientation == face_sense
    }

    /// A `FACE_BOUND`'s loop, in the SURFACE's own sense.
    ///
    /// `face_sense` is the value that will be stored as the compressed face's
    /// `orientation`, and it belongs here as well as on the face: see
    /// [`Self::absolute_bound_orientation`].
    fn face_bound_to_edges(
        &self,
        bound: FaceBoundHolder,
        face_sense: bool,
        eidx_map: &HashMap<u64, usize>,
        sink: &LossSink,
    ) -> Option<Vec<CompressedEdgeIndex>> {
        use PlaceHolder::Ref;
        let ori = Self::absolute_bound_orientation(&bound, face_sense);
        let bound = self.resolved_edge_loop(&bound, sink)?;
        sink.listed(Cat::EdgeUse, bound.edge_list.len());
        let mut edges: Vec<CompressedEdgeIndex> = bound
            .edge_list
            .into_iter()
            .filter_map(|edge| {
                let Ref(Name::Entity(idx)) = edge else {
                    sink.lost(
                        Cat::EdgeUse,
                        Why::EdgeUseUnresolved,
                        None,
                        Some("the edge use is not an entity reference".to_owned()),
                    );
                    return None;
                };
                let edge_idx = if let Some(oriented_edge) = self.oriented_edge.get(&idx) {
                    let element = oriented_edge
                        .edge_element_idx()
                        .and_then(|element| eidx_map.get(&element).copied());
                    let Some(index) = element else {
                        sink.lost(Cat::EdgeUse, Why::EdgeUseUnresolved, Some(idx), None);
                        return None;
                    };
                    CompressedEdgeIndex {
                        index,
                        orientation: oriented_edge.orientation == ori,
                    }
                } else {
                    let Some(index) = eidx_map.get(&idx).copied() else {
                        sink.lost(Cat::EdgeUse, Why::EdgeUseUnresolved, Some(idx), None);
                        return None;
                    };
                    CompressedEdgeIndex {
                        index,
                        orientation: ori,
                    }
                };
                Some(edge_idx)
            })
            .collect();
        sink.kept(Cat::EdgeUse, edges.len());
        if !ori {
            edges.reverse();
        }
        Some(edges)
    }

    fn exact_trim_curve_on(
        curve: &Curve3D,
        surface: &Surface,
        orientation: bool,
    ) -> Option<step_geometry::StepParameterCurve> {
        if matches!(curve, Curve3D::SurfaceCurve(_) | Curve3D::ParameterCurve(_)) {
            return curve
                .declared_parameter_curve_on(surface, orientation)
                .ok()
                .flatten();
        }
        let mut trim_curve = step_geometry::StepParameterCurve::try_from(
            step_geometry::CurveTrimRef::new(curve, surface),
        )
        .ok()?;
        if !orientation {
            trim_curve.invert();
        }
        Some(trim_curve)
    }

    /// The trimmed sibling of [`Self::face_bound_to_edges`], likewise in the
    /// SURFACE's own sense. Reversing a use also inverts its exact trim, which
    /// `exact_trim_curve_on` already does off the composed orientation.
    fn face_bound_to_edge_uses(
        &self,
        bound: FaceBoundHolder,
        face_sense: bool,
        face_surface: &Surface,
        eidx_map: &HashMap<u64, usize>,
        sink: &LossSink,
    ) -> Option<Vec<CompressedEdgeUse<step_geometry::StepParameterCurve>>> {
        use PlaceHolder::Ref;
        let ori = Self::absolute_bound_orientation(&bound, face_sense);
        let bound = self.resolved_edge_loop(&bound, sink)?;
        sink.listed(Cat::EdgeUse, bound.edge_list.len());
        let mut edges = bound
            .edge_list
            .into_iter()
            .filter_map(|edge| {
                let Ref(Name::Entity(idx)) = edge else {
                    sink.lost(
                        Cat::EdgeUse,
                        Why::EdgeUseUnresolved,
                        None,
                        Some("the edge use is not an entity reference".to_owned()),
                    );
                    return None;
                };
                let Some((edge_entity_idx, edge_curve_holder)) =
                    self.place_holder_edge_any_to_index_and_edge_curve(&Ref(Name::Entity(idx)))
                else {
                    sink.lost(Cat::EdgeUse, Why::EdgeUseUnresolved, Some(idx), None);
                    return None;
                };
                let orientation = self
                    .oriented_edge
                    .get(&idx)
                    .map(|oriented_edge| oriented_edge.orientation == ori)
                    .unwrap_or(ori);
                let edge_curve = match edge_curve_holder.into_owned(self) {
                    Ok(edge_curve) => edge_curve,
                    Err(e) => {
                        eprintln!("{e}");
                        sink.lost(
                            Cat::EdgeUse,
                            Why::EdgeUnresolved,
                            Some(idx),
                            Some(e.to_string()),
                        );
                        return None;
                    }
                };
                let curve = match edge_curve.parse_curve3d() {
                    Ok(curve) => curve,
                    Err(e) => {
                        eprintln!("{e}");
                        sink.lost(
                            Cat::EdgeUse,
                            Why::EdgeCurveRefused,
                            Some(idx),
                            Some(e.to_string()),
                        );
                        return None;
                    }
                };
                let Some(index) = eidx_map.get(&edge_entity_idx).copied() else {
                    sink.lost(Cat::EdgeUse, Why::EdgeUseUnresolved, Some(idx), None);
                    return None;
                };
                Some(CompressedEdgeUse {
                    index,
                    orientation,
                    trim_curve: Self::exact_trim_curve_on(&curve, face_surface, orientation),
                })
            })
            .collect::<Vec<_>>();
        sink.kept(Cat::EdgeUse, edges.len());
        if !ori {
            edges.reverse();
        }
        Some(edges)
    }

    /// Resolve and convert one face's surface, REPORTING both refusal arms.
    ///
    /// Both arms used to be `.map_err(|e| eprintln!("{e}")).ok()?`, which is how a
    /// typed per-record refusal still reached the caller as `Ok(shell)` minus a
    /// face -- 253 corpus faces, plus every face of a shell whose surface class
    /// the loader refuses.
    fn face_surface_or_report(&self, face: &FaceSurfaceHolder, sink: &LossSink) -> Option<Surface> {
        let id = Self::ref_id(&face.face_geometry);
        let step_surface: SurfaceAny = match face.face_geometry.clone().into_owned(self) {
            Ok(step_surface) => step_surface,
            Err(e) => {
                eprintln!("{e}");
                sink.lost(Cat::Face, Why::SurfaceUnresolved, id, Some(e.to_string()));
                return None;
            }
        };
        match Surface::try_from(&step_surface) {
            Ok(surface) => Some(surface),
            Err(e) => {
                eprintln!("{e}");
                sink.lost(Cat::Face, Why::SurfaceRefused, id, Some(e.to_string()));
                None
            }
        }
    }

    /// The face's boundary loops, with every loop the loader cannot represent
    /// counted against [`Cat::Wire`] rather than dropped in silence.
    fn face_boundaries_or_report<T>(
        &self,
        face: &FaceSurfaceHolder,
        sink: &LossSink,
        mut to_wire: impl FnMut(FaceBoundHolder) -> Option<T>,
    ) -> Vec<T> {
        let bounds = face.bounds_holder(self);
        sink.listed(Cat::Wire, bounds.len());
        bounds
            .into_iter()
            .filter_map(|(id, bound)| {
                let Some(bound) = bound else {
                    sink.lost(Cat::Wire, Why::BoundUnresolved, id, None);
                    return None;
                };
                let wire = to_wire(bound)?;
                sink.kept(Cat::Wire, 1);
                Some(wire)
            })
            .collect()
    }

    /// Every face the shell lists, paired with the orientation, reporting the
    /// ones whose reference resolves to no face record at all.
    fn shell_listed_faces<'a>(
        &'a self,
        shell: &'a ShellHolder,
        sink: &'a LossSink,
    ) -> impl Iterator<Item = (bool, FaceSurfaceHolder)> + 'a {
        shell.cfs_faces_holder(self).filter_map(move |face| {
            sink.listed(Cat::Face, 1);
            let resolved = self.face_any_to_orientation_and_face(face);
            if resolved.is_none() {
                sink.lost(Cat::Face, Why::FaceUnresolved, None, None);
            }
            resolved
        })
    }

    fn shell_faces(
        &self,
        shell: &ShellHolder,
        eidx_map: &HashMap<u64, usize>,
        sink: &LossSink,
    ) -> Vec<CompressedFace<Surface>> {
        self.shell_listed_faces(shell, sink)
            .filter_map(|(orientation, face)| {
                let surface = self.face_surface_or_report(&face, sink)?;
                // The stored boundaries and the stored flag are ONE convention,
                // not two: see `absolute_bound_orientation`.
                let face_sense = orientation == face.same_sense;
                let boundaries = self.face_boundaries_or_report(&face, sink, |bound| {
                    self.face_bound_to_edges(bound, face_sense, eidx_map, sink)
                });
                sink.kept(Cat::Face, 1);
                Some(CompressedFace {
                    surface,
                    boundaries,
                    orientation: face_sense,
                })
            })
            .collect()
    }

    fn shell_trimmed_faces(
        &self,
        shell: &ShellHolder,
        eidx_map: &HashMap<u64, usize>,
        sink: &LossSink,
    ) -> Vec<CompressedTrimmedFace<Surface, step_geometry::StepParameterCurve>> {
        self.shell_listed_faces(shell, sink)
            .filter_map(|(orientation, face)| {
                let surface = self.face_surface_or_report(&face, sink)?;
                // The stored boundaries and the stored flag are ONE convention,
                // not two: see `absolute_bound_orientation`.
                let face_sense = orientation == face.same_sense;
                let boundaries = self.face_boundaries_or_report(&face, sink, |bound| {
                    self.face_bound_to_edge_uses(bound, face_sense, &surface, eidx_map, sink)
                });
                sink.kept(Cat::Face, 1);
                Some(CompressedTrimmedFace {
                    surface,
                    boundaries,
                    orientation: face_sense,
                })
            })
            .collect()
    }

    /// Copies only the declared UV geometry, never its carrier surface.
    fn compact_declared_trim_on(
        curve: &Curve3D,
        surface: &Surface,
        orientation: bool,
    ) -> Result<Option<Curve2D>, StepConvertingError> {
        let trim = Self::compact_declared_uv_on(curve, surface, orientation)?;
        Ok(trim.map(|mut uv| {
            if !orientation {
                uv.invert();
            }
            uv
        }))
    }

    fn compact_declared_uv_on(
        curve: &Curve3D,
        surface: &Surface,
        branch_orientation: bool,
    ) -> Result<Option<Curve2D>, StepConvertingError> {
        if let Curve3D::TrimmedCurve(curve) = curve {
            if *curve.transform() != Matrix4::identity() {
                return Err(ExactConversionError::UnestablishedCurveCorrespondence.into());
            }
            let basis = Self::compact_declared_uv_on(
                curve.entity().curve().as_ref(),
                surface,
                branch_orientation == curve.orientation(),
            )?;
            return Ok(basis.map(|basis| {
                let mut trim =
                    Processor::new(TrimmedCurve::new(Box::new(basis), curve.entity().range()));
                if !curve.orientation() {
                    trim.invert();
                }
                Curve2D::TrimmedCurve(SourceTrimmedCurve::internal(trim))
            }));
        }
        let declared = match curve {
            Curve3D::SurfaceCurve(curve) => {
                curve.declared_parameter_curve_ref_on(surface, branch_orientation)?
            }
            Curve3D::ParameterCurve(curve) if curve.surface() == surface => Some(curve),
            _ => None,
        };
        Ok(declared.map(|curve| curve.curve().clone()))
    }

    fn compact_face_bound(
        &self,
        bound: FaceBoundHolder,
        face_sense: bool,
        surface: &Surface,
        edges: &[CompressedEdge<Curve3D>],
        eidx_map: &HashMap<u64, usize>,
        sink: &LossSink,
    ) -> Result<Option<CompactBoundary>, StepConvertingError> {
        let ori = Self::absolute_bound_orientation(&bound, face_sense);
        let Some(edge_loop) = self.resolved_edge_loop(&bound, sink) else {
            return Ok(None);
        };
        sink.listed(Cat::EdgeUse, edge_loop.edge_list.len());
        let mut uses = Vec::new();
        let mut lineage = Vec::new();
        for (source_coedge_index, edge_use) in edge_loop.edge_list.iter().enumerate() {
            let source_coedge = Self::ref_id(edge_use);
            let (source_edge, source_orientation) = match edge_use {
                PlaceHolder::Owned(EdgeAnyHolder::OrientedEdge(edge)) => {
                    (edge.edge_element_idx(), edge.orientation)
                }
                PlaceHolder::Ref(Name::Entity(id)) => match self.oriented_edge.get(id) {
                    Some(edge) => (edge.edge_element_idx(), edge.orientation),
                    None => (Some(*id), true),
                },
                _ => (None, true),
            };
            let Some(index) = source_edge.and_then(|id| eidx_map.get(&id)).copied() else {
                sink.lost(Cat::EdgeUse, Why::EdgeUseUnresolved, source_coedge, None);
                continue;
            };
            let orientation = source_orientation == ori;
            let curve = &edges[index].curve;
            let trim_curve = Self::compact_declared_trim_on(curve, surface, orientation)?;
            uses.push(CompressedEdgeUse {
                index,
                orientation,
                trim_curve,
            });
            let source_curve_range = SourceCurveInterval::new(curve.range_tuple());
            lineage.push(SourceCoedgeProvenance {
                source_edge,
                source_coedge,
                source_coedge_index,
                source_orientation,
                native_edge_index: index,
                native_orientation: orientation,
                source_curve_range,
                restricted_curve_range: source_curve_range,
            });
        }
        sink.kept(Cat::EdgeUse, uses.len());
        if !ori {
            uses.reverse();
            lineage.reverse();
        }
        Ok(Some((uses, lineage)))
    }

    fn original_numeric_trim_selector(
        &self,
        selectors: &[PlaceHolder<TrimmingSelectHolder>],
    ) -> Result<f64, StepConvertingError> {
        let mut selected = None;
        for selector in selectors {
            let value = match selector {
                PlaceHolder::Owned(TrimmingSelectHolder::ParameterValue(value)) => Some(value.0),
                PlaceHolder::Ref(Name::Entity(id)) => {
                    self.parameter_value.get(id).map(|value| value.0)
                }
                _ => None,
            };
            if let Some(value) = value
                && (!value.is_finite() || selected.replace(value).is_some())
            {
                return Err(
                    step_geometry::ExactConversionError::InvalidTrimParameterSelectors.into(),
                );
            }
        }
        selected.ok_or_else(|| step_geometry::ExactConversionError::CartesianTrimSelectors.into())
    }

    fn source_edge_curve_provenance(
        &self,
        source_edge: u64,
        original_native_edge_index: usize,
        curve: &Curve3D,
    ) -> Result<SourceEdgeProvenance, StepConvertingError> {
        let edge = self
            .edge_curve
            .get(&source_edge)
            .ok_or("parsed source edge has no original EDGE_CURVE holder")?;
        let source_curve = Self::ref_id(&edge.edge_geometry);
        let source_curve_range = SourceCurveInterval::new(curve.range_tuple());
        let mut raw = &edge.edge_geometry;
        let mut parsed = curve;
        let mut trimmed_curves = Vec::new();
        loop {
            let trimmed = match raw {
                PlaceHolder::Ref(Name::Entity(id)) => self.trimmed_curve.get(id),
                PlaceHolder::Owned(CurveAnyHolder::TrimmedCurve(curve)) => Some(curve.as_ref()),
                _ => None,
            };
            if let Some(trimmed) = trimmed {
                let Curve3D::TrimmedCurve(wrapper) = parsed else {
                    return Err("parsed source trim lost its declared wrapper".into());
                };
                let basis = wrapper.entity().curve().as_ref();
                let SourceParameterGauge::Declared {
                    raw_to_internal,
                    raw_range,
                    selected_basis_range,
                    raw_parameter_bounds,
                    selected_basis_bounds,
                } = wrapper.raw_parameter_gauge()?
                else {
                    return Err(
                        step_geometry::ExactConversionError::UnrepresentableTrimParameterChart
                            .into(),
                    );
                };
                let interval = |bounds: crate::step::ParameterEnclosure| {
                    SourceCurveInterval::new((bounds.lower, bounds.upper))
                };
                trimmed_curves.push(SourceTrimmedCurveProvenance {
                    source_curve: Self::ref_id(raw),
                    source_basis_curve: Self::ref_id(&trimmed.basis_curve),
                    source_curve_range: SourceCurveInterval::new(parsed.range_tuple()),
                    source_basis_curve_range: SourceCurveInterval::new(basis.range_tuple()),
                    declared_parameter_range: SourceCurveInterval::new((
                        self.original_numeric_trim_selector(&trimmed.trim_1)?,
                        self.original_numeric_trim_selector(&trimmed.trim_2)?,
                    )),
                    selected_basis_range: interval(selected_basis_range),
                    selected_basis_bounds: selected_basis_bounds.map(interval),
                    raw_parameter_range: interval(raw_range),
                    raw_parameter_bounds: raw_parameter_bounds.map(interval),
                    raw_parameter_chart: SourceParameterChart::new(raw_to_internal),
                    sense_agreement: trimmed.sense_agreement,
                });
                raw = &trimmed.basis_curve;
                parsed = basis;
                continue;
            }
            let surface_curve = match raw {
                PlaceHolder::Ref(Name::Entity(id)) => self.surface_curve.get(id),
                PlaceHolder::Owned(CurveAnyHolder::SurfaceCurve(curve)) => Some(curve.as_ref()),
                _ => None,
            };
            if let Some(surface_curve) = surface_curve {
                raw = &surface_curve.curve_3d;
                parsed = match parsed {
                    Curve3D::SurfaceCurve(curve) => curve.leader(),
                    Curve3D::IntersectionCurve(curve) => curve.leader().as_ref(),
                    _ => parsed,
                };
                continue;
            }
            break;
        }
        Ok(SourceEdgeProvenance {
            source_edge,
            source_curve,
            source_root_curve: Self::ref_id(raw),
            original_native_edge_index,
            source_curve_range,
            source_root_curve_range: SourceCurveInterval::new(parsed.range_tuple()),
            trimmed_curves,
        })
    }

    fn compact_trimmed_shell(
        &self,
        shell: &ShellHolder,
        shell_orientation: bool,
    ) -> Result<(CompactTrimmedShell, ShellLoadReport), StepConvertingError> {
        self.validate_shell_source(shell, false)?;
        let sink = LossSink::default();
        let (vertices, vidx_map) = self.shell_vertices(shell, &sink);
        let (edges, eidx_map) = self.shell_edges(shell, &vidx_map, &sink);
        let mut faces = Vec::new();
        let mut source_edges = eidx_map
            .iter()
            .map(|(source, index)| {
                self.source_edge_curve_provenance(*source, *index, &edges[*index].curve)
            })
            .collect::<Result<Vec<_>, _>>()?;
        source_edges.sort_unstable_by_key(|edge| edge.original_native_edge_index);
        let mut provenance = SourceShellProvenance {
            faces: Vec::new(),
            edges: source_edges,
        };
        for (source_face_index, (raw, resolved)) in shell
            .cfs_faces
            .iter()
            .zip(shell.cfs_faces_holder(self))
            .enumerate()
        {
            sink.listed(Cat::Face, 1);
            let source_face_use = Self::ref_id(raw);
            let source_face = match &resolved {
                Some(FaceAnyHolder::OrientedFace(face)) => Self::ref_id(&face.face_element),
                _ => source_face_use,
            };
            let Some((orientation, face)) = self.face_any_to_orientation_and_face(resolved) else {
                sink.lost(Cat::Face, Why::FaceUnresolved, source_face_use, None);
                continue;
            };
            let Some(surface) = self.face_surface_or_report(&face, &sink) else {
                continue;
            };
            let face_sense = orientation == face.same_sense;
            let bounds = face.bounds_holder(self);
            sink.listed(Cat::Wire, bounds.len());
            let mut boundaries = Vec::new();
            let mut wires = Vec::new();
            for (source_wire_index, (source_bound, bound)) in bounds.into_iter().enumerate() {
                let Some(bound) = bound else {
                    sink.lost(Cat::Wire, Why::BoundUnresolved, source_bound, None);
                    continue;
                };
                let source_wire = Self::ref_id(&bound.bound);
                let Some((wire, coedges)) =
                    self.compact_face_bound(bound, face_sense, &surface, &edges, &eidx_map, &sink)?
                else {
                    continue;
                };
                sink.kept(Cat::Wire, 1);
                boundaries.push(wire);
                wires.push(SourceWireProvenance {
                    source_bound,
                    source_wire,
                    source_wire_index,
                    coedges,
                });
            }
            sink.kept(Cat::Face, 1);
            let native_orientation = face_sense == shell_orientation;
            faces.push(CompressedTrimmedFace {
                surface,
                boundaries,
                orientation: native_orientation,
            });
            provenance.faces.push(SourceFaceProvenance {
                source_face,
                source_face_use,
                source_face_index,
                native_orientation,
                wires,
            });
        }
        let mut shell = CompressedTrimmedShell {
            vertices,
            edges,
            faces,
        };
        periodic_edges::compact_trimmed(&mut shell, &mut provenance)?;
        let mut report = sink.into_report();
        report.set_source_shell(provenance);
        Ok((shell, report))
    }

    fn compact_trimmed_shell_reference(
        &self,
        id: u64,
    ) -> Result<(CompactTrimmedShell, ShellLoadReport), StepConvertingError> {
        if let Some(shell) = self.shell.get(&id) {
            self.compact_trimmed_shell(shell, true)
        } else if let Some(oriented) = self.oriented_shell.get(&id) {
            let shell = Self::ref_id(&oriented.shell_element)
                .and_then(|id| self.shell.get(&id))
                .ok_or("failed to reference oriented shell element")?;
            self.compact_trimmed_shell(shell, oriented.orientation)
        } else {
            Err("failed to reference shell".into())
        }
    }

    /// Constructs [`CompressedShell`] from a STEP `Shell`.
    ///
    /// # Example
    /// ```
    /// # fn main() -> anyhow::Result<()> {
    /// # use anyhow::anyhow;
    /// use monstertruck_io::step::load::{*, step_geometry::*};
    ///
    /// let step_string = include_str!(concat!(
    ///     env!("CARGO_MANIFEST_DIR"),
    ///     "/../resources/step/occt-cube.step",
    /// ));
    /// let table = Table::from_step(&step_string)?;
    /// let step_shell = table
    ///     .shell
    ///     .values()
    ///     .next()
    ///     .ok_or_else(|| anyhow!("STEP file contains no shell."))?;
    /// let cshell = table.to_compressed_shell(step_shell)?;
    /// assert_eq!(cshell.faces.len(), 6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_compressed_shell(
        &self,
        shell: &impl StepShell,
    ) -> Result<CompressedShell<Point3, Curve3D, Surface>, LoadError> {
        Ok(shell.to_compressed_shell(self)?)
    }

    /// [`Self::to_compressed_shell`] plus a truthful answer to "did I get
    /// everything?".
    ///
    /// # Why the un-suffixed method is not enough
    ///
    /// `to_compressed_shell` returns `Ok` for a shell that is short of faces,
    /// wires, edges or vertices, and always has. Measured over the corpus and the
    /// in-repo fixtures (spec 011 Phase 0): 253 faces and 15,444 shell faces lost
    /// corpus-wide, plus wires lost on four in-repo fixtures including the
    /// reference boolean fixture. The evidence was one line on stderr, or -- for a
    /// `VERTEX_LOOP` -- nothing at all.
    ///
    /// The report is ADDITIVE: this method does exactly the same conversion work
    /// and returns exactly the same shell.
    ///
    /// # Example
    /// ```
    /// # fn main() -> anyhow::Result<()> {
    /// # use anyhow::anyhow;
    /// use monstertruck_io::step::load::{*, report::*};
    ///
    /// let step_string = include_str!(concat!(
    ///     env!("CARGO_MANIFEST_DIR"),
    ///     "/../resources/step/occt-cube.step",
    /// ));
    /// let table = Table::from_step(step_string)?;
    /// let step_shell = table
    ///     .shell
    ///     .values()
    ///     .next()
    ///     .ok_or_else(|| anyhow!("STEP file contains no shell."))?;
    /// let (cshell, report) = table.to_compressed_shell_reported(step_shell)?;
    /// assert_eq!(cshell.faces.len(), 6);
    /// // The cube has no degenerate loops and no refused surfaces.
    /// assert!(report.is_lossless(), "{report}");
    /// assert_eq!(report.count(LossCategory::Face).listed, 6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_compressed_shell_reported(
        &self,
        shell: &impl StepShell,
    ) -> Result<(CompressedShell<Point3, Curve3D, Surface>, ShellLoadReport), LoadError> {
        Ok(shell.to_compressed_shell_reported(self)?)
    }

    /// Constructs `CompressedTrimmedShell` from `Shell` in STEP file while preserving
    /// exact face-local `ParameterCurve`s when they are present in the STEP data.
    pub fn to_compressed_trimmed_shell(
        &self,
        shell: &impl StepShell,
    ) -> Result<
        CompressedTrimmedShell<Point3, Curve3D, Surface, step_geometry::StepParameterCurve>,
        LoadError,
    > {
        Ok(shell.to_compressed_trimmed_shell(self)?)
    }

    /// [`Self::to_compressed_trimmed_shell`] plus the loss report. See
    /// [`Self::to_compressed_shell_reported`].
    pub fn to_compressed_trimmed_shell_reported(
        &self,
        shell: &impl StepShell,
    ) -> Result<(StepCompressedTrimmedShell, ShellLoadReport), LoadError> {
        Ok(shell.to_compressed_trimmed_shell_reported(self)?)
    }

    /// Constructs `CompressedShell`s from `ShellBasedSurfaceModel` in STEP file
    pub fn to_compressed_shells(
        &self,
        shells: &ShellBasedSurfaceModelHolder,
    ) -> Result<Vec<CompressedShell<Point3, Curve3D, Surface>>, LoadError> {
        let mut res = Vec::new();
        for place_holder in &shells.sbsm_boundary {
            let PlaceHolder::Ref(Name::Entity(idx)) = place_holder else {
                return Err("failed to reference an element of `sbsm_boundary`".into());
            };
            if let Some(shell) = self.shell.get(idx) {
                res.push(self.to_compressed_shell(shell)?);
            } else if let Some(oriented_shell) = self.oriented_shell.get(idx) {
                res.push(self.to_compressed_shell(oriented_shell)?);
            } else {
                return Err("failed to reference an element of `sbsm_boundary`".into());
            }
        }
        Ok(res)
    }

    /// Constructs `CompressedTrimmedShell`s from `ShellBasedSurfaceModel` in STEP file.
    pub fn to_compressed_trimmed_shells(
        &self,
        shells: &ShellBasedSurfaceModelHolder,
    ) -> Result<
        Vec<CompressedTrimmedShell<Point3, Curve3D, Surface, step_geometry::StepParameterCurve>>,
        LoadError,
    > {
        let mut res = Vec::new();
        for place_holder in &shells.sbsm_boundary {
            let PlaceHolder::Ref(Name::Entity(idx)) = place_holder else {
                return Err("failed to reference an element of `sbsm_boundary`".into());
            };
            if let Some(shell) = self.shell.get(idx) {
                res.push(self.to_compressed_trimmed_shell(shell)?);
            } else if let Some(oriented_shell) = self.oriented_shell.get(idx) {
                res.push(self.to_compressed_trimmed_shell(oriented_shell)?);
            } else {
                return Err("failed to reference an element of `sbsm_boundary`".into());
            }
        }
        Ok(res)
    }

    /// Constructs [`CompressedSolid`] from [`ManifoldSolidBrep`] in a STEP file.
    ///
    /// The result keeps the STEP `outer` shell as the first entry of
    /// `boundaries` and appends one entry per `void` (inner cavity).
    ///
    /// `Solid::extract` on the returned [`CompressedSolid`] requires the
    /// outer shell to satisfy [`ShellCondition::Closed`](monstertruck_topology::shell::ShellCondition::Closed). Many real-world
    /// STEP exports produce shells that are topologically *regular* but
    /// not strictly *oriented*; for those, prefer
    /// `to_compressed_trimmed_solid` and downstream meshing/healing.
    ///
    /// # Example
    /// ```
    /// # fn main() -> anyhow::Result<()> {
    /// # use anyhow::anyhow;
    /// use monstertruck_io::step::load::{*, step_geometry::*};
    ///
    /// let step_string = include_str!(concat!(
    ///     env!("CARGO_MANIFEST_DIR"),
    ///     "/../resources/step/occt-cube.step",
    /// ));
    /// let table = Table::from_step(&step_string)?;
    /// let step_solid = table
    ///     .manifold_solid_brep
    ///     .values()
    ///     .next()
    ///     .ok_or_else(|| anyhow!("STEP file contains no manifold solid B-rep."))?;
    /// let csolid = table.to_compressed_solid(step_solid)?;
    /// assert_eq!(csolid.boundaries.len(), 1);
    /// assert_eq!(csolid.boundaries[0].faces.len(), 6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_compressed_solid(
        &self,
        solid: &ManifoldSolidBrepHolder,
    ) -> Result<CompressedSolid<Point3, Curve3D, Surface>, LoadError> {
        self.to_compressed_solid_reported(solid)
            .map(|(solid, _)| solid)
    }

    /// [`Self::to_compressed_solid`] plus ONE [`ShellLoadReport`] per entry of
    /// `boundaries`, positionally aligned with it -- so a caller can say which
    /// shell lost what, not merely that the solid lost something.
    ///
    /// See [`Self::to_compressed_shell_reported`] for why this exists.
    pub fn to_compressed_solid_reported(
        &self,
        solid: &ManifoldSolidBrepHolder,
    ) -> Result<(StepCompressedSolid, Vec<ShellLoadReport>), LoadError> {
        let PlaceHolder::Ref(Name::Entity(outer_idx)) = &solid.outer else {
            return Err("failed to reference `solid.outer`".into());
        };
        let outer_shell = if let Some(step_shell) = self.shell.get(outer_idx) {
            self.to_compressed_shell_reported(step_shell)
        } else if let Some(step_shell) = self.oriented_shell.get(outer_idx) {
            self.to_compressed_shell_reported(step_shell)
        } else {
            Err("failed to reference `solid.outer`".into())
        }?;
        let mut boundaries = vec![outer_shell.0];
        let mut reports = vec![outer_shell.1];
        for shell in &solid.voids {
            let PlaceHolder::Ref(Name::Entity(outer_idx)) = shell else {
                return Err("failed to reference an element of `solid.voids`".into());
            };
            let Some(oriented_shell) = self.oriented_shell.get(outer_idx) else {
                return Err("failed to reference an element of `solid.voids`".into());
            };
            let (shell, report) = self.to_compressed_shell_reported(oriented_shell)?;
            boundaries.push(shell);
            reports.push(report);
        }
        Ok((
            CompressedSolid {
                boundaries,
                id_allocator: None,
                attributes: None,
            },
            reports,
        ))
    }

    /// Constructs a trimmed solid together with every shell's source loss report.
    ///
    /// The geometry and retained source pcurves are converted once. Declared
    /// seam branches use structural carrier correspondence; an unavailable
    /// pcurve remains absent and is not a trim-coverage certificate.
    ///
    /// Only declared UV geometry is copied into each coedge, in that use's
    /// direction; no carrier is cloned per use. Missing or ambiguous seam
    /// branches return their typed conversion error, never a recovered branch.
    /// Reports retain original STEP face, loop and edge-use references aligned
    /// with the returned native ordinals, including exact periodic children.
    pub fn to_compressed_trimmed_solid_reported(
        &self,
        solid: &ManifoldSolidBrepHolder,
    ) -> Result<(StepCompressedTrimmedSolid, Vec<ShellLoadReport>), LoadError> {
        let PlaceHolder::Ref(Name::Entity(outer_idx)) = &solid.outer else {
            return Err("failed to reference `solid.outer`".into());
        };
        let (outer, report) = self.compact_trimmed_shell_reference(*outer_idx)?;
        let mut boundaries = vec![outer];
        let mut reports = vec![report];
        for inner in &solid.voids {
            let PlaceHolder::Ref(Name::Entity(inner_idx)) = inner else {
                return Err("failed to reference a member of `solid.voids`".into());
            };
            let (shell, report) = self.compact_trimmed_shell_reference(*inner_idx)?;
            boundaries.push(shell);
            reports.push(report);
        }
        Ok((CompressedTrimmedSolid { boundaries }, reports))
    }

    /// Constructs `CompressedTrimmedSolid` from `ManifoldSolidBrep` in STEP file.
    pub fn to_compressed_trimmed_solid(
        &self,
        solid: &ManifoldSolidBrepHolder,
    ) -> Result<
        CompressedTrimmedSolid<Point3, Curve3D, Surface, step_geometry::StepParameterCurve>,
        LoadError,
    > {
        let PlaceHolder::Ref(Name::Entity(outer_idx)) = &solid.outer else {
            return Err("failed to reference `solid.outer`".into());
        };
        let outer_shell = if let Some(step_shell) = self.shell.get(outer_idx) {
            self.to_compressed_trimmed_shell(step_shell)
        } else if let Some(step_shell) = self.oriented_shell.get(outer_idx) {
            self.to_compressed_trimmed_shell(step_shell)
        } else {
            Err("failed to reference `solid.outer`".into())
        }?;
        let mut boundaries = vec![outer_shell];
        for inner in &solid.voids {
            let PlaceHolder::Ref(Name::Entity(inner_idx)) = inner else {
                return Err("failed to reference a member of `solid.voids`".into());
            };
            if let Some(shell) = self.shell.get(inner_idx) {
                boundaries.push(self.to_compressed_trimmed_shell(shell)?);
            } else if let Some(oriented_shell) = self.oriented_shell.get(inner_idx) {
                boundaries.push(self.to_compressed_trimmed_shell(oriented_shell)?);
            } else {
                return Err("failed to reference a member of `solid.voids`".into());
            }
        }
        Ok(CompressedTrimmedSolid { boundaries })
    }

    /// Extracts [`Curve3D`] curves from a [`GeometricCurveSet`].
    ///
    /// Returns `None` if the holder cannot be resolved or yields no curves.
    /// Silently skips points and curves that fail conversion.
    pub fn to_curve3d_set(&self, gcs: &GeometricCurveSetHolder) -> Option<Vec<Curve3D>> {
        let owned = gcs.clone().into_owned(self).ok()?;
        let curves: Vec<Curve3D> = owned
            .elements
            .iter()
            .filter_map(|elem| match elem {
                GeometricSetSelect::Curve(curve) => Curve3D::try_from(curve.as_ref())
                    .map_err(|e| eprintln!("skipping curve in geometric_curve_set: {e}"))
                    .ok(),
                GeometricSetSelect::Point(_) => None,
            })
            .collect();
        if curves.is_empty() {
            None
        } else {
            Some(curves)
        }
    }
}

#[derive(Clone, Debug, derive_more::From)]
pub enum NodeMatrix {
    Identity,
    Transform(Box<ItemDefinedTransformation>),
}

pub use crate::step::common::PartAttributes;

pub type ProductEntity = NodeEntity<Vec<u64>, PartAttributes>;
pub type AssembleEntity = EdgeEntity<NodeMatrix, PartAttributes>;
pub type StepAssembly = Assembly<Vec<u64>, PartAttributes, NodeMatrix, PartAttributes>;

impl TryFrom<&NodeMatrix> for Matrix3 {
    type Error = StepConvertingError;
    fn try_from(value: &NodeMatrix) -> Result<Self, Self::Error> {
        match value {
            NodeMatrix::Identity => Ok(Self::identity()),
            NodeMatrix::Transform(trans) => (&**trans).try_into(),
        }
    }
}

impl TryFrom<&NodeMatrix> for Matrix4 {
    type Error = StepConvertingError;
    fn try_from(value: &NodeMatrix) -> Result<Self, Self::Error> {
        match value {
            NodeMatrix::Identity => Ok(Self::identity()),
            NodeMatrix::Transform(trans) => (&**trans).try_into(),
        }
    }
}

impl Table {
    fn product_node_entity(
        &self,
        pds_idx: u64,
        pd: &ProductDefinitionHolder,
    ) -> Result<ProductEntity, StepConvertingError> {
        let PlaceHolder::Ref(Name::Entity(pdf_idx)) = &pd.formation else {
            return Err("failed to reference `product_definition.formation`".into());
        };
        let Some(pdf) = self.product_definition_formation.get(pdf_idx) else {
            return Err("failed to reference `prouct_definition_formation`".into());
        };
        let PlaceHolder::Ref(Name::Entity(p_idx)) = &pdf.of_product else {
            return Err("failed to reference `product_definition_formation.of_product`".into());
        };
        let Some(product) = self.product.get(p_idx) else {
            return Err("failed to reference `product`".into());
        };
        let attrs = PartAttributes {
            id: product.id.clone(),
            name: product.name.clone(),
            // `PartAttributes::description` is a display field shared with the
            // save side and is a plain `String`, so an UNSET STEP description
            // flattens to `""` here -- the one place the `Option` added for
            // spec 011 T6 loses the unset/empty distinction. Deliberate: keeping
            // it would change what `save` emits for these parts, which is a
            // separate decision from making the load stop dropping the record.
            description: product.description.clone().unwrap_or_default(),
            // Filled in below, once the representation has been RESOLVED. Set
            // here it would be a promise; set there it is a fact.
            shape_representation: None,
        };

        let Some(sdr) = self.shape_definition_representation.values().find(|sdr| {
            let &PlaceHolder::Ref(Name::Entity(idx)) = &sdr.definition else {
                return false;
            };
            pds_idx == idx
        }) else {
            return Err("failed to find `shape_definition_representation` corresp. to `product_definition_shape`".into());
        };
        let PlaceHolder::Ref(Name::Entity(sr_idx)) = &sdr.used_representation else {
            return Err(
                "failed to reference `shape_definition_representation.used_representation`".into(),
            );
        };
        let Some(sr) = self.shape_representation.get(sr_idx) else {
            return Err("failed to reference `shape_representation`".into());
        };
        // The id this whole function already had and used to discard. It is
        // recorded only on the `get` success path above, so
        // `PartAttributes::shape_representation` is `Some(id)` ONLY for an id
        // the table demonstrably holds -- there is no "carried but dangling"
        // state for a caller to have to guard against.
        let attrs = PartAttributes {
            shape_representation: Some(*sr_idx),
            ..attrs
        };
        let Some(shape) = sr
            .items
            .iter()
            .map(|place_holder| {
                if let &PlaceHolder::Ref(Name::Entity(item_idx)) = place_holder {
                    Some(item_idx)
                } else {
                    None
                }
            })
            .collect::<Option<Vec<_>>>()
        else {
            return Err("failed to reference an element of `shape_representation.items`".into());
        };

        Ok(ProductEntity { shape, attrs })
    }

    fn assy_node_entity(
        &self,
        pds_idx: u64,
        next_assy: &NextAssemblyUsageOccurrenceHolder,
    ) -> Result<(AssembleEntity, (u64, u64)), StepConvertingError> {
        let &PlaceHolder::Ref(Name::Entity(parent_idx)) = &next_assy.relating_product_definition
        else {
            return Err("failed to reference the parent node".into());
        };
        let &PlaceHolder::Ref(Name::Entity(child_idx)) = &next_assy.related_product_definition
        else {
            return Err("failed to reference the child node".into());
        };

        let attrs = PartAttributes {
            id: next_assy.id.clone(),
            name: next_assy.name.clone(),
            description: next_assy.description.clone(),
            // An EDGE is a `NEXT_ASSEMBLY_USAGE_OCCURRENCE`, not a product with
            // a shape representation of its own. This is one of the two
            // populations that legitimately carry no id; see
            // `PartAttributes::shape_representation`.
            shape_representation: None,
        };

        let Some(cdsr) = self
            .context_dependent_shape_representation
            .values()
            .find(|cdsr| {
                let &PlaceHolder::Ref(Name::Entity(idx)) = &cdsr.represented_product_relation
                else {
                    return false;
                };
                pds_idx == idx
            })
        else {
            return Err("".into());
        };

        let PlaceHolder::Ref(Name::Entity(srrwt_idx)) = &cdsr.representation_relation else {
            return Err("failed to reference `context_dependent_shape_representation.representation_relation`".into());
        };

        let Some(srrwt) = self
            .shape_representation_relationship_with_transformation
            .get(srrwt_idx)
        else {
            return Err("failed to reference `shape_representation_relationship`".into());
        };
        let idtf = srrwt.transformation_operator.clone().into_owned(self)?;

        let entity = AssembleEntity {
            matrix: NodeMatrix::Transform(idtf.into()),
            attrs,
        };

        Ok((entity, (parent_idx, child_idx)))
    }

    pub fn step_assy(&self) -> Result<StepAssembly, LoadError> {
        let mut product_entities = Vec::<ProductEntity>::new();
        let mut indices_map = HashMap::<u64, usize>::new();
        let mut assy_nodes = Vec::<(AssembleEntity, (u64, u64))>::new();
        for (&pds_idx, pds) in &self.product_definition_shape {
            let &PlaceHolder::Ref(Name::Entity(idx)) = &pds.definition else {
                return Err("failed to reference `product_definition_shape.definition`".into());
            };
            if let Some(pd) = self.product_definition.get(&idx) {
                product_entities.push(self.product_node_entity(pds_idx, pd)?);
                indices_map.insert(idx, product_entities.len() - 1);
            } else if let Some(next_assy) = self.next_assembly_usage_occurrence.get(&idx) {
                assy_nodes.push(self.assy_node_entity(pds_idx, next_assy)?);
            }
        }

        let adjacency = assy_nodes
            .into_iter()
            .map(|(entity, (from, to))| {
                let from = *indices_map.get(&from)?;
                let to = *indices_map.get(&to)?;
                Some((from, to, entity))
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| LoadError::from("failed to reference `product_definition_shape`."))?;

        StepAssembly::try_from_adjacency(product_entities, adjacency)
            .ok_or_else(|| LoadError::from("maybe the assembly graph has a cycle."))
    }

    /// The representation a loaded assembly node came from -- the id it CARRIES
    /// if it has one, and the reverse match otherwise.
    ///
    /// This is what a caller holding a [`ProductEntity`] should ask. Since spec
    /// 012 the node carries the id directly
    /// ([`PartAttributes::shape_representation`]), which is exact by
    /// construction: [`Table::step_assy`] records it on the same success path
    /// that resolved the record. [`Table::shape_representation_of_items`] stays
    /// as the fallback for the nodes that legitimately carry none -- an
    /// assembly this crate did not load, i.e. one an application or the save
    /// side built, where the field is `Default`'s `None`.
    ///
    /// **Measured on the two Scania files (`scania_assembly_graphs_now_load_and_walk`):
    /// the two routes agree node for node, and the id route reaches the same
    /// 832 of 832 and 254 of 254 solids.** A disagreement would be a finding,
    /// not a rounding error, and that pin is where it would surface.
    pub fn shape_representation_of_node(&self, node: &ProductEntity) -> Option<u64> {
        node.attrs
            .shape_representation
            .or_else(|| self.shape_representation_of_items(&node.shape))
    }

    /// The representation whose `items` are EXACTLY `items` -- the bridge from a
    /// [`ProductEntity::shape`] back to the `SHAPE_REPRESENTATION` it came from.
    ///
    /// **Since spec 012 this is the FALLBACK, not the primary route.** A node
    /// loaded by [`Table::step_assy`] carries its representation id in
    /// [`PartAttributes::shape_representation`]; prefer
    /// [`Table::shape_representation_of_node`], which reads the field and only
    /// falls back to this search when there is none. This search remains public
    /// because it answers a question the field cannot: which representation, if
    /// any, holds an item list a caller constructed itself.
    ///
    /// [`Table::step_assy`] fills a node's `shape` from
    /// `shape_definition_representation.used_representation.items`. Before spec
    /// 012 it discarded the representation's own id, so a caller holding an
    /// assembly node could not ask anything else about that representation, and
    /// this inverse was the only bridge.
    ///
    /// Matching is on the whole id list, not on containment, and that is not
    /// caution for its own sake: **measured, `coffy.step` lists `#15756` in BOTH
    /// `#15521` and `#261`**, so a containment rule would answer that item's
    /// owner ambiguously. Whole-list matching is unique on every file measured --
    /// including the two Scania ones, where no representation item is shared at
    /// all (0 of 1,571 and 0 of 1,160). `None` when nothing matches; the FIRST
    /// match in ascending id order if several somehow do.
    ///
    /// `items` that are not all entity references cannot match anything, because
    /// `ProductEntity::shape` only exists when every item was one.
    pub fn shape_representation_of_items(&self, items: &[u64]) -> Option<u64> {
        let mut best: Option<u64> = None;
        for (&id, representation) in &self.shape_representation {
            if representation.items.len() != items.len() {
                continue;
            }
            let same = representation.items.iter().zip(items).all(
                |(held, want)| matches!(held, PlaceHolder::Ref(Name::Entity(held)) if held == want),
            );
            if same && best.is_none_or(|previous| id < previous) {
                best = Some(id);
            }
        }
        best
    }

    /// The solids a part's representation reaches through the **non-transforming**
    /// `SHAPE_REPRESENTATION_RELATIONSHIP` hop, plus what the hop lost.
    ///
    /// # Why this is not [`ProductEntity::shape`]
    ///
    /// `shape` means "the items directly in this node's representation", and on
    /// real assemblies that is honestly what it holds: **measured on
    /// `Scania-8x4.stp` it is `{AXIS2_PLACEMENT_3D: 695, SHELL_BASED_SURFACE_MODEL:
    /// 44}` and on `Scania-Engine-V8-XT-Turbo.step` `{AXIS2_PLACEMENT_3D: 906}` --
    /// zero solids.** The solids live in a second representation, an
    /// `ADVANCED_BREP_SHAPE_REPRESENTATION`, and this hop is what reaches them:
    /// **832 of 832 and 254 of 254**, exactly the 1,086 solids spec 011 T6 found
    /// with no placement.
    ///
    /// Making `shape` chase the relationship silently would make an existing
    /// method's answer depend on a hop the caller cannot see -- and `shape` is
    /// also what the save side round-trips, where the direct meaning is the
    /// right one. So the hop is a separate, visible call. Promoting it to the
    /// default later is easy; un-conflating two meanings after the fact is not.
    ///
    /// # Non-transforming only
    ///
    /// A transforming relationship is a complex record carrying
    /// `REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION`, and it lands in
    /// [`Table::shape_representation_relationship_with_transformation`], a
    /// different map that [`Table::step_assy`] already walks to build the
    /// assembly's edge matrices. Following it here too would apply the same
    /// placement twice. This method reads ONLY
    /// [`Table::shape_representation_relationship`], which by construction holds
    /// simple records exclusively -- a transformation cannot be attached to one.
    ///
    /// Only the `rep_1 == shape_representation -> rep_2` direction is followed.
    /// **Measured: the reverse direction carries zero solids on both Scania
    /// files** (all 188 and 158 relationships have the part's own representation
    /// as `rep_1`), so following it would add no reach and would let one part
    /// claim another's geometry.
    ///
    /// # What the report says
    ///
    /// [`LossCategory::Representation`] counts relationships found versus
    /// relationships whose `rep_2` the table actually holds;
    /// [`LossCategory::Solid`] counts the items of those representations versus
    /// the ones that are solids. Read
    /// [`LossReason::RepresentationItemNotASolid`] before treating a lossy report
    /// as a defect: a representation listing its own `AXIS2_PLACEMENT_3D` beside
    /// its solid is normal, and is the usual reason a file reports items lost
    /// while losing no solid at all.
    ///
    /// Ids come back in ascending relationship id, then representation item
    /// order, and are NOT deduplicated -- `kept` and the returned length always
    /// agree.
    ///
    /// # Example -- how a viewer opts in
    ///
    /// The hop only has work to do when a part's own representation RELATES to
    /// the representation holding the solid; on a file whose parts carry their
    /// solids directly it correctly reaches nothing and reports no loss, which
    /// is what the fixture below shows.
    ///
    /// ```
    /// # fn main() -> anyhow::Result<()> {
    /// use monstertruck_io::step::load::{report::*, Table};
    ///
    /// let bytes = include_bytes!(concat!(
    ///     env!("CARGO_MANIFEST_DIR"),
    ///     "/../resources/step/occt-assy.step",
    /// ));
    /// let table = Table::from_step_bytes(bytes)?;
    /// let assembly = table.step_assy()?;
    ///
    /// let mut solids = Vec::new();
    /// for node in assembly.all_nodes() {
    ///     // `node.shape()` is the items DIRECTLY in this node's representation
    ///     // -- on a real assembly, placements. The hop is the extra call.
    ///     let Some(representation) = table.shape_representation_of_items(node.shape()) else {
    ///         continue;
    ///     };
    ///     let (reached, report) = table.solids_via_shape_relationship(representation);
    ///     // The report says what the hop could NOT deliver, and why.
    ///     assert_eq!(report.lost_for(LossReason::RelatedRepresentationUnresolved), 0);
    ///     solids.extend(reached);
    /// }
    /// // Every relationship this file does have resolved, so nothing was lost.
    /// assert!(solids.iter().all(|id| *id > 0));
    /// # Ok(())
    /// # }
    /// ```
    pub fn solids_via_shape_relationship(
        &self,
        shape_representation: u64,
    ) -> (Vec<u64>, ShellLoadReport) {
        let mut report = ShellLoadReport::default();
        let mut solids = Vec::new();

        let mut relationships: Vec<_> = self
            .shape_representation_relationship
            .iter()
            .filter(|(_, srr)| {
                matches!(&srr.rep_1, PlaceHolder::Ref(Name::Entity(rep_1)) if *rep_1 == shape_representation)
            })
            .collect();
        relationships.sort_unstable_by_key(|(id, _)| **id);

        for (&srr_id, srr) in relationships {
            report.note_listed(Cat::Representation, 1);
            let PlaceHolder::Ref(Name::Entity(rep_2)) = &srr.rep_2 else {
                report.note_lost(
                    Cat::Representation,
                    Why::RelatedRepresentationUnresolved,
                    Some(srr_id),
                    Some("`rep_2` is not an entity reference".to_owned()),
                );
                continue;
            };
            let Some(representation) = self.shape_representation.get(rep_2) else {
                report.note_lost(
                    Cat::Representation,
                    Why::RelatedRepresentationUnresolved,
                    Some(*rep_2),
                    Some(self.describe_unheld(*rep_2)),
                );
                continue;
            };
            report.note_kept(Cat::Representation, 1);

            for item in &representation.items {
                report.note_listed(Cat::Solid, 1);
                let PlaceHolder::Ref(Name::Entity(item_id)) = item else {
                    report.note_lost(
                        Cat::Solid,
                        Why::RepresentationItemNotASolid,
                        Some(*rep_2),
                        Some("the item is not an entity reference".to_owned()),
                    );
                    continue;
                };
                if self.manifold_solid_brep.contains_key(item_id) {
                    report.note_kept(Cat::Solid, 1);
                    solids.push(*item_id);
                } else {
                    report.note_lost(
                        Cat::Solid,
                        Why::RepresentationItemNotASolid,
                        Some(*item_id),
                        Some(self.describe_unheld(*item_id)),
                    );
                }
            }
        }

        (solids, report)
    }

    /// One line naming what `id` actually is, for a loss detail. The entity NAME
    /// is the whole point -- "not a solid" without it sends a reader back to the
    /// file with nothing but an id.
    fn describe_unheld(&self, id: u64) -> String {
        if let Some(dummy) = self.dummy.get(&id) {
            // `Table::dummy` stores `format!("{record:?}")`, whose derived
            // `Debug` puts the entity name in the first quoted field.
            // Diagnostic convenience, not a contract: an unparseable shape
            // degrades to the record text itself.
            let name = dummy.record.split('"').nth(1).unwrap_or(&dummy.record);
            format!("#{id} is `{name}`, an entity type the loader has no arm for")
        } else if self.axis2_placement_3d.contains_key(&id) {
            format!("#{id} is an AXIS2_PLACEMENT_3D")
        } else if self.shell_based_surface_model.contains_key(&id) {
            format!("#{id} is a SHELL_BASED_SURFACE_MODEL")
        } else if self.geometric_curve_set.contains_key(&id) {
            format!("#{id} is a GEOMETRIC_CURVE_SET")
        } else {
            format!("#{id} is in no table map")
        }
    }
}

/// A STEP entity a compressed shell can be built from.
///
/// The `*_reported` methods are the PRIMARY ones: they return the report of what
/// the conversion kept and lost. The un-suffixed methods are provided defaults
/// that discard it, so every existing caller keeps working unchanged.
pub trait StepShell {
    /// Convert, and report what was kept and lost.
    fn to_compressed_shell_reported(
        &self,
        table: &Table,
    ) -> Result<(CompressedShell<Point3, Curve3D, Surface>, ShellLoadReport), StepConvertingError>;

    /// Convert preserving exact trim curves, and report what was kept and lost.
    fn to_compressed_trimmed_shell_reported(
        &self,
        table: &Table,
    ) -> Result<(StepCompressedTrimmedShell, ShellLoadReport), StepConvertingError>;

    /// Convert, DISCARDING the loss report.
    ///
    /// A lossy conversion is indistinguishable from a clean one through this
    /// method -- that is exactly the defect spec 011 T7 exists to end. Prefer
    /// [`Self::to_compressed_shell_reported`] in new code.
    fn to_compressed_shell(
        &self,
        table: &Table,
    ) -> Result<CompressedShell<Point3, Curve3D, Surface>, StepConvertingError> {
        self.to_compressed_shell_reported(table)
            .map(|(shell, _)| shell)
    }

    /// Convert preserving exact trim curves, DISCARDING the loss report.
    /// Prefer [`Self::to_compressed_trimmed_shell_reported`] in new code.
    fn to_compressed_trimmed_shell(
        &self,
        table: &Table,
    ) -> Result<
        CompressedTrimmedShell<Point3, Curve3D, Surface, step_geometry::StepParameterCurve>,
        StepConvertingError,
    > {
        self.to_compressed_trimmed_shell_reported(table)
            .map(|(shell, _)| shell)
    }
}

impl StepShell for ShellHolder {
    fn to_compressed_shell_reported(
        &self,
        table: &Table,
    ) -> Result<(CompressedShell<Point3, Curve3D, Surface>, ShellLoadReport), StepConvertingError>
    {
        table.validate_shell_source(self, false)?;
        let sink = LossSink::default();
        let (vertices, vidx_map) = table.shell_vertices(self, &sink);
        let (edges, eidx_map) = table.shell_edges(self, &vidx_map, &sink);
        let faces = table.shell_faces(self, &eidx_map, &sink);
        let mut shell = CompressedShell {
            vertices,
            edges,
            faces,
            vertex_stable_ids: None,
            edge_stable_ids: None,
            face_stable_ids: None,
        };
        periodic_edges::plain(&mut shell)?;
        Ok((shell, sink.into_report()))
    }

    fn to_compressed_trimmed_shell_reported(
        &self,
        table: &Table,
    ) -> Result<
        (
            CompressedTrimmedShell<Point3, Curve3D, Surface, step_geometry::StepParameterCurve>,
            ShellLoadReport,
        ),
        StepConvertingError,
    > {
        table.validate_shell_source(self, true)?;
        let sink = LossSink::default();
        let (vertices, vidx_map) = table.shell_vertices(self, &sink);
        let (edges, eidx_map) = table.shell_edges(self, &vidx_map, &sink);
        let faces = table.shell_trimmed_faces(self, &eidx_map, &sink);
        let mut shell = CompressedTrimmedShell {
            vertices,
            edges,
            faces,
        };
        periodic_edges::trimmed(&mut shell)?;
        Ok((shell, sink.into_report()))
    }
}

impl StepShell for OrientedShellHolder {
    fn to_compressed_shell_reported(
        &self,
        table: &Table,
    ) -> Result<(CompressedShell<Point3, Curve3D, Surface>, ShellLoadReport), StepConvertingError>
    {
        let PlaceHolder::Ref(Name::Entity(idx)) = &self.shell_element else {
            return Err("failed to reference shell".into());
        };
        let Some(shell) = table.shell.get(idx) else {
            return Err("failed to reference shell".into());
        };
        let (mut res, report) = shell.to_compressed_shell_reported(table)?;
        if !self.orientation {
            for face in &mut res.faces {
                face.orientation = !face.orientation;
            }
        }
        Ok((res, report))
    }

    fn to_compressed_trimmed_shell_reported(
        &self,
        table: &Table,
    ) -> Result<
        (
            CompressedTrimmedShell<Point3, Curve3D, Surface, step_geometry::StepParameterCurve>,
            ShellLoadReport,
        ),
        StepConvertingError,
    > {
        let PlaceHolder::Ref(Name::Entity(idx)) = &self.shell_element else {
            return Err("failed to reference shell".into());
        };
        let Some(shell) = table.shell.get(idx) else {
            return Err("failed to reference shell".into());
        };
        let (mut res, report) = shell.to_compressed_trimmed_shell_reported(table)?;
        if !self.orientation {
            res.faces.iter_mut().for_each(|face| {
                face.orientation = !face.orientation;
            });
        }
        Ok((res, report))
    }
}

impl StepShell for ShellAnyHolder {
    fn to_compressed_shell_reported(
        &self,
        table: &Table,
    ) -> Result<(CompressedShell<Point3, Curve3D, Surface>, ShellLoadReport), StepConvertingError>
    {
        match self {
            ShellAnyHolder::OrientedShell(shell) => shell.to_compressed_shell_reported(table),
            ShellAnyHolder::Shell(shell) => shell.to_compressed_shell_reported(table),
        }
    }

    fn to_compressed_trimmed_shell_reported(
        &self,
        table: &Table,
    ) -> Result<
        (
            CompressedTrimmedShell<Point3, Curve3D, Surface, step_geometry::StepParameterCurve>,
            ShellLoadReport,
        ),
        StepConvertingError,
    > {
        match self {
            ShellAnyHolder::OrientedShell(shell) => {
                shell.to_compressed_trimmed_shell_reported(table)
            }
            ShellAnyHolder::Shell(shell) => shell.to_compressed_trimmed_shell_reported(table),
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod compact_source_lineage_regressions {
    use super::*;
    use crate::step::load::step_geometry::SurfaceCurveRepresentation as StepSurfaceCurveRepresentation;
    use std::f64::consts::TAU;

    fn cylinder_table() -> Table {
        Table::from_step(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../resources/step/occt-cylinder.step",
        )))
        .expect("the fixed cylinder source parses")
    }

    #[test]
    fn compact_report_preserves_original_occurrences_across_periodic_children() {
        let table = cylinder_table();
        let (solid, reports) = table
            .to_compressed_trimmed_solid_reported(&table.manifold_solid_brep[&15])
            .expect("the compact cylinder converts");
        let shell = &solid.boundaries[0];
        let source = &reports[0].source_shells()[0];
        assert_eq!(source.faces.len(), shell.faces.len());
        assert_eq!(source.faces[0].source_face, Some(17));
        assert_eq!(source.faces[0].source_face_use, Some(17));
        let wire = &source.faces[0].wires[0];
        assert_eq!(wire.source_bound, Some(18));
        assert_eq!(wire.source_wire, Some(19));
        let children: Vec<_> = wire
            .coedges
            .iter()
            .filter(|source| source.source_coedge == Some(20))
            .collect();
        assert_eq!(children.len(), 2);
        assert!(children.iter().all(|source| {
            source.source_edge == Some(21)
                && source.source_coedge_index == 0
                && !source.source_orientation
                && !source.native_orientation
        }));
        assert_eq!(
            children[0].source_curve_range,
            children[1].source_curve_range
        );
        assert_ne!(children[0].native_edge_index, children[1].native_edge_index);
        assert_eq!(
            children[0].restricted_curve_range.range_tuple().0,
            children[1].restricted_curve_range.range_tuple().1
        );
        for (face, source_face) in shell.faces.iter().zip(&source.faces) {
            assert_eq!(face.orientation, source_face.native_orientation);
            for (wire, source_wire) in face.boundaries.iter().zip(&source_face.wires) {
                assert_eq!(wire.len(), source_wire.coedges.len());
                for (edge_use, source_use) in wire.iter().zip(&source_wire.coedges) {
                    assert_eq!(edge_use.index, source_use.native_edge_index);
                    assert_eq!(edge_use.orientation, source_use.native_orientation);
                    assert_eq!(
                        shell.edges[edge_use.index].curve.range_tuple(),
                        source_use.restricted_curve_range.range_tuple()
                    );
                }
            }
        }
        assert!(
            shell
                .faces
                .iter()
                .flat_map(|face| &face.boundaries)
                .flatten()
                .any(|edge| edge.trim_curve.is_some())
        );
    }

    #[test]
    fn closed_source_declarations_retain_unbounded_uv_basis_and_actual_edge_domain() {
        let table = cylinder_table();
        let source = EntityTable::<EdgeCurveHolder>::get_owned(&table, 21)
            .expect("the original closed source edge resolves");
        let parsed = source
            .parse_curve3d()
            .expect("the closed curve parses once");
        let Curve3D::SurfaceCurve(declared) = &parsed else {
            panic!("a closed edge must retain its declared pcurve metadata");
        };
        let surface: Surface = (&EntityTable::<SurfaceAnyHolder>::get_owned(&table, 31)
            .expect("the original cylinder resolves"))
            .try_into()
            .expect("analytic cylinder");
        let pcurve = declared
            .declared_parameter_curve_ref_on(&surface, true)
            .expect("one structurally declared cylinder branch")
            .expect("closed declarations are not discarded");
        let Curve2D::TrimmedCurve(active) = pcurve.curve() else {
            panic!("the unbounded STEP LINE receives the actual active source interval");
        };
        assert_eq!(active.range_tuple(), parsed.range_tuple());
        assert!(active.orientation());
        assert_eq!(
            active.entity().curve().as_ref(),
            &Curve2D::Line(Line(Point2::new(0.0, 10.0), Point2::new(1.0, 10.0)))
        );
        let (solid, reports) = table
            .to_compressed_trimmed_solid_reported(&table.manifold_solid_brep[&15])
            .expect("compact declared UV survives periodic splitting");
        let shell = &solid.boundaries[0];
        let source = &reports[0].source_shells()[0];
        for (face, provenance) in shell.faces.iter().zip(&source.faces) {
            for (wire, source_wire) in face.boundaries.iter().zip(&provenance.wires) {
                for (edge_use, original) in wire.iter().zip(&source_wire.coedges) {
                    if original.source_edge == Some(21) {
                        assert!(edge_use.trim_curve.is_some());
                        assert_eq!(
                            shell.edges[edge_use.index].curve.range_tuple(),
                            original.restricted_curve_range.range_tuple()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn owned_inline_coedge_keeps_none_identity_without_losing_original_edge_occurrence() {
        let mut table = cylinder_table();
        let source_use = table.oriented_edge[&20].clone();
        table
            .edge_loop
            .get_mut(&19)
            .expect("original loop")
            .edge_list[0] = PlaceHolder::Owned(EdgeAnyHolder::OrientedEdge(source_use));
        let (_, reports) = table
            .to_compressed_trimmed_solid_reported(&table.manifold_solid_brep[&15])
            .expect("an inline occurrence still references its original declared source edge");
        let children: Vec<_> = reports[0].source_shells()[0].faces[0].wires[0]
            .coedges
            .iter()
            .filter(|source| source.source_coedge_index == 0)
            .collect();
        assert_eq!(children.len(), 2);
        assert!(children.iter().all(|source| source.source_coedge.is_none()
            && source.source_edge == Some(21)
            && !source.source_orientation));
    }

    #[test]
    fn compact_oriented_shell_changes_only_recorded_face_sense() {
        let mut table = cylinder_table();
        table.oriented_shell.insert(
            10000,
            OrientedShellHolder {
                label: String::new(),
                shell_element: PlaceHolder::Ref(Name::Entity(16)),
                orientation: false,
            },
        );
        let (ordinary, ordinary_report) = table
            .compact_trimmed_shell_reference(16)
            .expect("ordinary shell converts");
        let (reversed, reversed_report) = table
            .compact_trimmed_shell_reference(10000)
            .expect("oriented shell converts");
        for (face, reversed_face) in ordinary.faces.iter().zip(&reversed.faces) {
            assert_eq!(face.orientation, !reversed_face.orientation);
            assert_eq!(face.boundaries, reversed_face.boundaries);
        }
        for (face, reversed_face) in ordinary_report.source_shells()[0]
            .faces
            .iter()
            .zip(&reversed_report.source_shells()[0].faces)
        {
            assert_eq!(face.native_orientation, !reversed_face.native_orientation);
            assert_eq!(face.wires, reversed_face.wires);
        }
    }

    #[test]
    fn compact_declared_seam_selects_and_inverts_uv_without_recovery() {
        let surface =
            Surface::ElementarySurface(ElementarySurface::Plane(step_geometry::Plane::new(
                Point3::origin(),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            )));
        let parameter_curve = |u| {
            SurfaceCurveAssociatedGeometry::ParameterCurve(step_geometry::StepParameterCurve::new(
                Box::new(Curve2D::Line(Line(
                    Point2::new(u, 0.0),
                    Point2::new(u, 1.0),
                ))),
                Box::new(surface.clone()),
            ))
        };
        let seam = |branches| {
            Curve3D::SurfaceCurve(SurfaceCurve3D::new(
                StepSurfaceCurveKind::SeamCurve,
                Box::new(Curve3D::Line(Line(
                    Point3::origin(),
                    Point3::new(0.0, 1.0, 0.0),
                ))),
                branches,
                StepSurfaceCurveRepresentation::ParameterCurve1,
            ))
        };
        let curve = seam(vec![parameter_curve(0.0), parameter_curve(TAU)]);
        let forward = Table::compact_declared_trim_on(&curve, &surface, true)
            .expect("declared forward branch resolves")
            .expect("forward UV exists");
        let backward = Table::compact_declared_trim_on(&curve, &surface, false)
            .expect("declared backward branch resolves")
            .expect("backward UV exists");
        assert_eq!(forward.front(), Point2::new(0.0, 0.0));
        assert_eq!(backward.front(), Point2::new(TAU, 1.0));
        let missing =
            Table::compact_declared_trim_on(&seam(vec![parameter_curve(0.0)]), &surface, true)
                .expect_err("one seam branch must refuse");
        assert!(matches!(
            missing.downcast_ref::<ExactConversionError>(),
            Some(ExactConversionError::MissingSeamParameterBranch)
        ));
        let ambiguous = Table::compact_declared_trim_on(
            &seam(vec![
                parameter_curve(0.0),
                parameter_curve(TAU),
                parameter_curve(2.0 * TAU),
            ]),
            &surface,
            true,
        )
        .expect_err("three seam branches must refuse");
        assert!(matches!(
            ambiguous.downcast_ref::<ExactConversionError>(),
            Some(ExactConversionError::AmbiguousStoredTrim)
        ));
    }
}

#[cfg(test)]
mod trimmed_source_lineage_regressions {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn original_trim_wrapper_basis_and_root_ranges_survive_periodic_source_uses() {
        let mut table = Table::from_step(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../resources/step/occt-cylinder.step",
        )))
        .expect("fixed cylinder source parses");
        let selector = |parameter| {
            PlaceHolder::Owned(TrimmingSelectHolder::ParameterValue(Box::new(
                ParameterValueHolder(parameter),
            )))
        };
        table.trimmed_curve.insert(
            10000,
            StepTrimmedCurveHolder {
                label: String::new(),
                basis_curve: PlaceHolder::Ref(Name::Entity(25)),
                trim_1: vec![selector(TAU)],
                trim_2: vec![selector(0.0)],
                sense_agreement: false,
                master_representation: TrimmingPreference::Parameter,
            },
        );
        table
            .edge_curve
            .get_mut(&21)
            .expect("original source edge exists")
            .edge_geometry = PlaceHolder::Ref(Name::Entity(10000));
        let owned = EntityTable::<CurveAnyHolder>::get_owned(&table, 10000)
            .expect("original inverse source wrapper resolves");
        let original = Curve3D::try_from(&owned).expect("source wrapper preserves its parameter");
        let (solid, reports) = table
            .to_compressed_trimmed_solid_reported(&table.manifold_solid_brep[&15])
            .expect("compact source wrapper converts");
        let shell = &solid.boundaries[0];
        let provenance = &reports[0].source_shells()[0];
        let edge = provenance
            .edges
            .iter()
            .find(|edge| edge.source_edge == 21)
            .expect("original source edge lineage exists");
        assert_eq!(edge.source_curve, Some(10000));
        assert_eq!(edge.source_root_curve, Some(25));
        assert_eq!(edge.source_curve_range.range_tuple(), (0.0, TAU));
        assert_eq!(
            edge.source_root_curve_range.range_tuple(),
            original.range_tuple()
        );
        assert_eq!(edge.trimmed_curves.len(), 1);
        assert_eq!(edge.trimmed_curves[0].source_curve, Some(10000));
        assert_eq!(edge.trimmed_curves[0].source_basis_curve, Some(25));
        assert!(!edge.trimmed_curves[0].sense_agreement);
        let declared = &edge.trimmed_curves[0];
        assert_eq!(declared.declared_parameter_range.range_tuple(), (TAU, 0.0));
        assert_eq!(declared.selected_basis_range.range_tuple(), (0.0, TAU));
        assert_eq!(declared.raw_parameter_range.range_tuple(), (0.0, TAU));
        let chart = declared.raw_parameter_chart.parameter_chart();
        assert_eq!(chart.nominal(0.0), Some(0.0));
        assert!(chart.offset_enclosure.contains(0.0));
        assert!(chart.offset_enclosure.lower < chart.offset_enclosure.upper);
        let children: Vec<_> = provenance.faces[0].wires[0]
            .coedges
            .iter()
            .filter(|source| source.source_edge == Some(21))
            .collect();
        assert_eq!(children.len(), 2);
        for child in children {
            let native = &shell.edges[child.native_edge_index].curve;
            let (start, end) = child.restricted_curve_range.range_tuple();
            assert_eq!(native.range_tuple(), (start, end));
            assert_eq!(child.source_curve_range, edge.source_curve_range);
            for parameter in [start, start + (end - start) * 0.37, end] {
                assert!(native.subs(parameter).near(&original.subs(parameter)));
                assert!(native.der(parameter).near(&original.der(parameter)));
            }
        }
    }
}
