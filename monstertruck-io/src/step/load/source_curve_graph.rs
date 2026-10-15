//! Bounded, borrowed preflight of recursive source geometry references.

use super::*;
use monstertruck_modeling::geometry::MAX_NATIVE_GEOMETRY_NESTING;
use std::collections::HashSet;
use std::mem::size_of;

mod expansion;
use expansion::{ExpansionWalk, SourceExpansionBudget};
pub use expansion::{MAX_SOURCE_GEOMETRY_EXPANSION_BYTES, MAX_SOURCE_GEOMETRY_EXPANSION_VISITS};

struct SourceGeometryWalk<'a, 'budget> {
    table: &'a Table,
    budget: &'budget mut SourceExpansionBudget,
    ancestors: [Option<u64>; MAX_NATIVE_GEOMETRY_NESTING + 1],
    references: usize,
}

impl<'a, 'budget> SourceGeometryWalk<'a, 'budget> {
    fn new(table: &'a Table, budget: &'budget mut SourceExpansionBudget) -> Self {
        Self {
            table,
            budget,
            ancestors: [None; MAX_NATIVE_GEOMETRY_NESTING + 1],
            references: 0,
        }
    }

    fn depth(depth: usize) -> Result<(), ExactConversionError> {
        if depth > MAX_NATIVE_GEOMETRY_NESTING {
            Err(ExactConversionError::SourceGeometryNestingExceeded)
        } else {
            Ok(())
        }
    }

    fn placeholder<T>(
        &mut self,
        placeholder: &PlaceHolder<T>,
        depth: usize,
        owned: fn(&mut Self, &T, usize) -> Result<(), ExactConversionError>,
    ) -> Result<(), ExactConversionError> {
        self.budget.charge(0)?;
        Self::depth(depth)?;
        match placeholder {
            PlaceHolder::Owned(holder) => owned(self, holder, depth),
            PlaceHolder::Ref(Name::Entity(id)) => self.reference(*id, depth),
            _ => Ok(()),
        }
    }

    fn reference(&mut self, id: u64, depth: usize) -> Result<(), ExactConversionError> {
        Self::depth(depth)?;
        if self.ancestors[..self.references].contains(&Some(id)) {
            return Err(ExactConversionError::CyclicSourceGeometry { entity: id });
        }
        if self.references == self.ancestors.len() {
            return Err(ExactConversionError::SourceGeometryNestingExceeded);
        }
        self.ancestors[self.references] = Some(id);
        self.references += 1;
        let table = self.table;
        let result = if let Some(curve) = table.trimmed_curve.get(&id) {
            self.trimmed(curve, depth)
        } else if let Some(curve) = table.surface_curve.get(&id) {
            self.surface_curve(curve, depth)
        } else if let Some(curve) = table.pcurve.get(&id) {
            self.pcurve(curve, depth)
        } else if let Some(surface) = table.surface_of_linear_extrusion.get(&id) {
            self.placeholder(&surface.swept_curve, depth + 1, Self::curve)
        } else if let Some(surface) = table.surface_of_revolution.get(&id) {
            self.placeholder(&surface.swept_curve, depth + 1, Self::curve)
        } else if let Some(representation) = table.definitional_representation.get(&id) {
            self.representation(representation, depth)
        } else {
            Ok(())
        };
        self.references -= 1;
        self.ancestors[self.references] = None;
        result
    }

    fn curve(&mut self, curve: &CurveAnyHolder, depth: usize) -> Result<(), ExactConversionError> {
        Self::depth(depth)?;
        match curve {
            CurveAnyHolder::TrimmedCurve(curve) => self.trimmed(curve, depth),
            CurveAnyHolder::SurfaceCurve(curve) => self.surface_curve(curve, depth),
            CurveAnyHolder::Pcurve(curve) => self.pcurve(curve, depth),
            CurveAnyHolder::Line(_)
            | CurveAnyHolder::BoundedCurve(_)
            | CurveAnyHolder::Conic(_) => Ok(()),
        }
    }

    fn trimmed(
        &mut self,
        curve: &StepTrimmedCurveHolder,
        depth: usize,
    ) -> Result<(), ExactConversionError> {
        Self::depth(depth)?;
        self.placeholder(&curve.basis_curve, depth + 1, Self::curve)
    }

    fn surface_curve(
        &mut self,
        curve: &SurfaceCurveHolder,
        depth: usize,
    ) -> Result<(), ExactConversionError> {
        Self::depth(depth)?;
        self.placeholder(&curve.curve_3d, depth + 1, Self::curve)?;
        curve
            .associated_geometry
            .iter()
            .try_for_each(|associated| self.placeholder(associated, depth + 1, Self::associated))
    }

    fn associated(
        &mut self,
        associated: &PcurveOrSurfaceHolder,
        depth: usize,
    ) -> Result<(), ExactConversionError> {
        match associated {
            PcurveOrSurfaceHolder::Pcurve(curve) => self.pcurve(curve, depth),
            PcurveOrSurfaceHolder::Surface(surface) => self.surface(surface, depth),
        }
    }

    fn pcurve(&mut self, curve: &PcurveHolder, depth: usize) -> Result<(), ExactConversionError> {
        Self::depth(depth)?;
        self.placeholder(&curve.basis_surface, depth + 1, Self::surface)?;
        self.placeholder(&curve.reference_to_curve, depth + 1, Self::representation)
    }

    fn representation(
        &mut self,
        representation: &DefinitionalRepresentationHolder,
        depth: usize,
    ) -> Result<(), ExactConversionError> {
        Self::depth(depth)?;
        representation
            .representation_item
            .iter()
            .try_for_each(|curve| self.placeholder(curve, depth + 1, Self::curve))
    }

    fn surface(
        &mut self,
        surface: &SurfaceAnyHolder,
        depth: usize,
    ) -> Result<(), ExactConversionError> {
        Self::depth(depth)?;
        match surface {
            SurfaceAnyHolder::SweptSurface(surface) => match surface.as_ref() {
                SweptSurfaceAnyHolder::SurfaceOfLinearExtrusion(surface) => {
                    self.placeholder(&surface.swept_curve, depth + 1, Self::curve)
                }
                SweptSurfaceAnyHolder::SurfaceOfRevolution(surface) => {
                    self.placeholder(&surface.swept_curve, depth + 1, Self::curve)
                }
            },
            SurfaceAnyHolder::ElementarySurface(_) | SurfaceAnyHolder::BsplineSurface(_) => Ok(()),
        }
    }
}

impl Table {
    pub(crate) fn validate_curve_holder_source(
        &self,
        curve: &CurveAnyHolder,
    ) -> Result<(), ExactConversionError> {
        let mut budget = SourceExpansionBudget::default();
        SourceGeometryWalk::new(self, &mut budget).curve(curve, 0)?;
        ExpansionWalk::new(self, &mut budget).expand(curve)
    }

    pub(crate) fn validate_trimmed_curve_source(
        &self,
        curve: &StepTrimmedCurveHolder,
    ) -> Result<(), ExactConversionError> {
        let mut budget = SourceExpansionBudget::default();
        SourceGeometryWalk::new(self, &mut budget).trimmed(curve, 0)?;
        ExpansionWalk::new(self, &mut budget).expand(curve)
    }

    pub(crate) fn validate_curve_source(
        &self,
        curve: &PlaceHolder<CurveAnyHolder>,
    ) -> Result<(), ExactConversionError> {
        let mut budget = SourceExpansionBudget::default();
        SourceGeometryWalk::new(self, &mut budget).placeholder(
            curve,
            0,
            SourceGeometryWalk::curve,
        )?;
        ExpansionWalk::new(self, &mut budget).expand(curve)
    }

    fn validate_surface_source_with_budget(
        &self,
        surface: &PlaceHolder<SurfaceAnyHolder>,
        budget: &mut SourceExpansionBudget,
    ) -> Result<(), ExactConversionError> {
        SourceGeometryWalk::new(self, budget).placeholder(
            surface,
            0,
            SourceGeometryWalk::surface,
        )?;
        ExpansionWalk::new(self, budget).expand(surface)
    }

    fn validate_edge_source_with_budget(
        &self,
        edge: &EdgeCurveHolder,
        budget: &mut SourceExpansionBudget,
    ) -> Result<(), ExactConversionError> {
        SourceGeometryWalk::new(self, budget).placeholder(
            &edge.edge_geometry,
            0,
            SourceGeometryWalk::curve,
        )?;
        ExpansionWalk::new(self, budget).expand(edge)
    }
}

fn referenced_holder<'a, T>(
    placeholder: &'a PlaceHolder<T>,
    table: &'a HashMap<u64, T>,
) -> Option<&'a T> {
    match placeholder {
        PlaceHolder::Owned(holder) => Some(holder),
        PlaceHolder::Ref(Name::Entity(id)) => table.get(id),
        _ => None,
    }
}

impl Table {
    pub(crate) fn validate_shell_source(
        &self,
        shell: &ShellHolder,
        reparse_coedges: bool,
    ) -> Result<(), ExactConversionError> {
        let mut budget = SourceExpansionBudget::default();
        let mut edges = HashSet::new();
        let mut vertices = HashSet::new();
        budget.charge(
            size_of::<ShellHolder>()
                .saturating_add(shell.label.capacity())
                .saturating_add(
                    shell
                        .cfs_faces
                        .capacity()
                        .saturating_mul(size_of::<FaceAnyHolder>()),
                ),
        )?;
        shell.cfs_faces.iter().try_for_each(|face| {
            let face = match face {
                PlaceHolder::Owned(FaceAnyHolder::FaceSurface(face)) => Some(face),
                PlaceHolder::Owned(FaceAnyHolder::OrientedFace(face)) => {
                    budget.charge(
                        size_of::<OrientedFaceHolder>().saturating_add(face.label.capacity()),
                    )?;
                    referenced_holder(&face.face_element, &self.face_surface)
                }
                PlaceHolder::Ref(Name::Entity(id)) => {
                    if let Some(face) = self.oriented_face.get(id) {
                        budget.charge(
                            size_of::<OrientedFaceHolder>().saturating_add(face.label.capacity()),
                        )?;
                        referenced_holder(&face.face_element, &self.face_surface)
                    } else {
                        self.face_surface.get(id)
                    }
                }
                _ => None,
            };
            let Some(face) = face else {
                return Ok(());
            };
            budget.charge(
                size_of::<FaceSurfaceHolder>()
                    .saturating_add(face.label.capacity())
                    .saturating_add(
                        face.bounds
                            .capacity()
                            .saturating_mul(size_of::<FaceBoundHolder>()),
                    ),
            )?;
            self.validate_surface_source_with_budget(&face.face_geometry, &mut budget)?;
            face.bounds.iter().try_for_each(|bound| {
                let Some(bound) = referenced_holder(bound, &self.face_bound) else {
                    return Ok(());
                };
                budget
                    .charge(size_of::<FaceBoundHolder>().saturating_add(bound.label.capacity()))?;
                let Some(edge_loop) = referenced_holder(&bound.bound, &self.edge_loop) else {
                    return Ok(());
                };
                budget.charge(
                    size_of::<EdgeLoopHolder>()
                        .saturating_add(edge_loop.label.capacity())
                        .saturating_add(
                            edge_loop
                                .edge_list
                                .capacity()
                                .saturating_mul(size_of::<EdgeAnyHolder>()),
                        ),
                )?;
                edge_loop.edge_list.iter().try_for_each(|edge_use| {
                    let (source_edge, edge) = match edge_use {
                        PlaceHolder::Owned(EdgeAnyHolder::EdgeCurve(edge)) => (None, Some(edge)),
                        PlaceHolder::Owned(EdgeAnyHolder::OrientedEdge(edge)) => {
                            budget.charge(
                                size_of::<OrientedEdgeHolder>()
                                    .saturating_add(edge.label.capacity()),
                            )?;
                            (
                                Self::ref_id(&edge.edge_element),
                                referenced_holder(&edge.edge_element, &self.edge_curve),
                            )
                        }
                        PlaceHolder::Ref(Name::Entity(id)) => {
                            if let Some(edge) = self.oriented_edge.get(id) {
                                budget.charge(
                                    size_of::<OrientedEdgeHolder>()
                                        .saturating_add(edge.label.capacity()),
                                )?;
                                (
                                    Self::ref_id(&edge.edge_element),
                                    referenced_holder(&edge.edge_element, &self.edge_curve),
                                )
                            } else {
                                (Some(*id), self.edge_curve.get(id))
                            }
                        }
                        _ => (None, None),
                    };
                    let Some(edge) = edge else {
                        return Ok(());
                    };
                    if source_edge.is_none_or(|id| edges.insert(id)) {
                        self.validate_edge_source_with_budget(edge, &mut budget)?;
                        for vertex in [&edge.edge_start, &edge.edge_end] {
                            if Self::ref_id(vertex).is_some_and(|id| vertices.insert(id)) {
                                ExpansionWalk::new(self, &mut budget).expand(vertex)?;
                            }
                        }
                    }
                    if reparse_coedges {
                        self.validate_edge_source_with_budget(edge, &mut budget)?;
                    }
                    Ok(())
                })
            })
        })?;
        edges.clear();
        self.validate_shell_raw_clones(shell, reparse_coedges, &mut budget, &mut edges)
    }

    fn validate_shell_raw_clones(
        &self,
        shell: &ShellHolder,
        reparse_coedges: bool,
        budget: &mut SourceExpansionBudget,
        edges: &mut HashSet<u64>,
    ) -> Result<(), ExactConversionError> {
        // All recursive geometry has passed depth/cycle checks above. Count
        // actual raw-holder clone sites too, stopping at references: inline
        // payloads can otherwise be copied repeatedly by the topology walks.
        let mut raw = ExpansionWalk::raw_clone(self, budget);
        shell.cfs_faces.iter().try_for_each(|face| {
            let (face, oriented) = match face {
                PlaceHolder::Owned(FaceAnyHolder::FaceSurface(face)) => (Some(face), None),
                PlaceHolder::Owned(FaceAnyHolder::OrientedFace(face)) => (
                    referenced_holder(&face.face_element, &self.face_surface),
                    Some(face),
                ),
                PlaceHolder::Ref(Name::Entity(id)) => match self.oriented_face.get(id) {
                    Some(face) => (
                        referenced_holder(&face.face_element, &self.face_surface),
                        Some(face),
                    ),
                    None => (self.face_surface.get(id), None),
                },
                _ => (None, None),
            };
            if let Some(oriented) = oriented {
                raw.raw_clones(oriented, 3)?;
            }
            let Some(face) = face else {
                return Ok(());
            };
            // Vertex census, edge census and face assembly each clone these.
            raw.raw_clones(face, 3)?;
            face.bounds.iter().try_for_each(|bound| {
                let Some(bound) = referenced_holder(bound, &self.face_bound) else {
                    return Ok(());
                };
                raw.raw_clones(bound, 3)?;
                let Some(edge_loop) = referenced_holder(&bound.bound, &self.edge_loop) else {
                    return Ok(());
                };
                raw.raw_clones(edge_loop, 3)?;
                edge_loop.edge_list.iter().try_for_each(|edge_use| {
                    let (source_edge, edge) = match edge_use {
                        PlaceHolder::Owned(EdgeAnyHolder::EdgeCurve(edge)) => (None, Some(edge)),
                        PlaceHolder::Owned(EdgeAnyHolder::OrientedEdge(edge)) => (
                            Self::ref_id(&edge.edge_element),
                            referenced_holder(&edge.edge_element, &self.edge_curve),
                        ),
                        PlaceHolder::Ref(Name::Entity(id)) => match self.oriented_edge.get(id) {
                            Some(edge) => (
                                Self::ref_id(&edge.edge_element),
                                referenced_holder(&edge.edge_element, &self.edge_curve),
                            ),
                            None => (Some(*id), self.edge_curve.get(id)),
                        },
                        _ => (None, None),
                    };
                    let Some(edge) = edge else {
                        return Ok(());
                    };
                    raw.raw_clones(edge, 2 + usize::from(reparse_coedges))?;
                    if source_edge.is_none_or(|id| edges.insert(id)) {
                        raw.raw_clones(edge, 1)?;
                    }
                    Ok(())
                })
            })
        })
    }
}

#[cfg(test)]
mod source_expansion_regressions {
    use super::*;

    fn source_table(levels: usize, topology: &str) -> (Table, u64) {
        let mut data = String::from(
            "#1=CARTESIAN_POINT('',(0.,0.,0.));\n\
             #2=CARTESIAN_POINT('',(1.,1.,0.));\n\
             #3=DIRECTION('',(0.,0.,1.));\n\
             #4=(BOUNDED_CURVE() B_SPLINE_CURVE(1,(#1,#2),.UNSPECIFIED.,.F.,.F.) \
             B_SPLINE_CURVE_WITH_KNOTS((2,2),(0.,1.),.UNSPECIFIED.) CURVE() GEOMETRIC_REPRESENTATION_ITEM() \
             RATIONAL_B_SPLINE_CURVE((1.,2.)) REPRESENTATION_ITEM(''));\n\
             #5=VECTOR('',#3,1.);\n",
        );
        let mut root = 4_u64;
        for level in 0..levels {
            let surface = 6 + 2 * level as u64;
            let curve = surface + 1;
            data.push_str(&format!(
                "#{surface}=SURFACE_OF_LINEAR_EXTRUSION('',#{root},#5);\n\
                 #{curve}=SURFACE_CURVE('',#{root},(#{surface},#{surface}),.CURVE_3D.);\n"
            ));
            root = curve;
        }
        data.push_str(topology);
        let table = Table::from_step(&format!(
            "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('source resource regression'),'2;1');\n\
             FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n{data}\nENDSEC;\nEND-ISO-10303-21;"
        )).expect("source DAG parses without recursively owning its references");
        (table, root)
    }

    fn add_large_rational_leaf_payload(table: &mut Table) {
        let rational = table
            .rational_b_spline_curve
            .get_mut(&4)
            .expect("rational leaf");
        let PlaceHolder::Owned(NonRationalBsplineCurveHolder::BsplineCurveWithKnots(base)) =
            &mut rational.non_rational_b_spline_curve
        else {
            panic!("complex rational source retains its owned knot-bearing base");
        };
        base.label = "n".repeat(1024 * 1024);
    }

    #[test]
    fn repeated_rational_leaf_payload_is_charged_before_recursive_holder_cloning() {
        let (mut table, root) = source_table(4, "");
        add_large_rational_leaf_payload(&mut table);
        let error = table
            .validate_curve_source(&PlaceHolder::Ref(Name::Entity(root)))
            .expect_err(
                "81 actual expansions of the shared 1 MiB rational source exceed the byte profile",
            );
        assert!(
            matches!(error, ExactConversionError::SourceGeometryExpansionExceeded {
            expanded_bytes, visits
        } if expanded_bytes > MAX_SOURCE_GEOMETRY_EXPANSION_BYTES
            && visits < MAX_SOURCE_GEOMETRY_EXPANSION_VISITS)
        );
        assert!(
            EntityTable::<CurveAnyHolder>::get_owned(&table, root)
                .expect_err("the public holder resolver refuses before cloning the expansion")
                .to_string()
                .contains("resource profile")
        );
    }

    #[test]
    fn source_expansion_bytes_are_cumulative_across_distinct_shell_edges() {
        let (mut table, root) = source_table(
            3,
            "#20=AXIS2_PLACEMENT_3D('',#1,$,$);\n#21=PLANE('',#20);\n\
             #100=VERTEX_POINT('',#1);\n#101=VERTEX_POINT('',#2);\n\
             #110=EDGE_CURVE('',#100,#101,#11,.T.);\n\
             #111=EDGE_CURVE('',#100,#101,#11,.T.);\n\
             #112=EDGE_CURVE('',#100,#101,#11,.T.);\n\
             #120=EDGE_LOOP('',(#110,#111,#112));\n\
             #121=FACE_BOUND('',#120,.T.);\n\
             #122=FACE_SURFACE('',(#121),#21,.T.);\n\
             #123=CLOSED_SHELL('',(#122));",
        );
        assert_eq!(root, 11);
        add_large_rational_leaf_payload(&mut table);
        table
            .validate_curve_source(&PlaceHolder::Ref(Name::Entity(root)))
            .expect("one edge's expanded source is within the byte profile");
        for reparse_coedges in [false, true] {
            let error = table
                .validate_shell_source(&table.shell[&123], reparse_coedges)
                .expect_err("the shell shares one cumulative budget across all source edges");
            assert!(
                matches!(error, ExactConversionError::SourceGeometryExpansionExceeded {
                expanded_bytes, visits
            } if expanded_bytes > MAX_SOURCE_GEOMETRY_EXPANSION_BYTES
                && visits < MAX_SOURCE_GEOMETRY_EXPANSION_VISITS)
            );
        }
    }

    #[test]
    fn wide_acyclic_source_work_refuses_without_misreporting_nesting_or_cycles() {
        let (table, root) = source_table(12, "");
        let error = table
            .validate_curve_source(&PlaceHolder::Ref(Name::Entity(root)))
            .expect_err("a shallow exponentially expanded DAG exceeds the work profile");
        assert!(
            matches!(error, ExactConversionError::SourceGeometryExpansionExceeded { visits, .. }
            if visits > MAX_SOURCE_GEOMETRY_EXPANSION_VISITS)
        );
    }

    #[test]
    fn repeated_raw_topology_name_clones_share_the_same_source_byte_budget() {
        let (mut table, _) = source_table(
            0,
            "#20=AXIS2_PLACEMENT_3D('',#1,$,$);\n#21=PLANE('',#20);\n\
             #100=VERTEX_POINT('',#1);\n#101=VERTEX_POINT('',#2);\n\
             #110=EDGE_CURVE('',#100,#101,#4,.T.);\n\
             #120=EDGE_LOOP('',(#110));\n#121=FACE_BOUND('',#120,.T.);\n\
             #122=FACE_SURFACE('',(#121),#21,.T.);\n#123=CLOSED_SHELL('',(#122));",
        );
        table.face_surface.get_mut(&122).expect("source face").label = "f".repeat(24 * 1024 * 1024);
        let error = table
            .validate_shell_source(&table.shell[&123], false)
            .expect_err("three real topology passes would clone 72 MiB of face name alone");
        assert!(
            matches!(error, ExactConversionError::SourceGeometryExpansionExceeded {
            expanded_bytes, visits
        } if expanded_bytes > MAX_SOURCE_GEOMETRY_EXPANSION_BYTES
            && visits < MAX_SOURCE_GEOMETRY_EXPANSION_VISITS)
        );
    }
}
