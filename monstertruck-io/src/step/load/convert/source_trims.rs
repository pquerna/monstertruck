use super::super::step_geometry::SurfaceCurveKind;
use super::*;
use monstertruck_modeling::geometry::MAX_NATIVE_GEOMETRY_NESTING;

#[derive(Clone, Copy)]
enum SourceCurve<'a> {
    Surface(&'a SurfaceCurveHolder),
    Parameter(&'a PcurveHolder),
    Other,
}

/// Original occurrences in ONE declared face, before periodic subdivision.
#[derive(Clone, Copy)]
pub(super) enum OriginalFaceEdgeUse {
    Single(bool),
    OpposedPair,
    Repeated,
}

/// A borrowed association in one original table's document namespace.
/// No entity identifier or converted geometry can construct this binding alone.
pub(super) struct TrimSource<'a> {
    table: &'a Table,
    curve: SourceCurve<'a>,
    carrier: Option<u64>,
    edge_use: OriginalFaceEdgeUse,
}

impl Table {
    pub(super) fn original_edge_use(
        &self,
        edge_use: &PlaceHolder<EdgeAnyHolder>,
    ) -> (Option<u64>, bool) {
        match edge_use {
            PlaceHolder::Owned(EdgeAnyHolder::OrientedEdge(edge)) => {
                (edge.edge_element_idx(), edge.orientation)
            }
            PlaceHolder::Ref(Name::Entity(id)) => match self.oriented_edge.get(id) {
                Some(edge) => (edge.edge_element_idx(), edge.orientation),
                None => (Some(*id), true),
            },
            _ => (None, true),
        }
    }

    pub(super) fn original_face_edge_uses(
        &self,
        bounds: &[(Option<u64>, Option<FaceBoundHolder>)],
        face_sense: bool,
    ) -> HashMap<u64, OriginalFaceEdgeUse> {
        let mut uses = HashMap::new();
        for bound in bounds.iter().filter_map(|(_, bound)| bound.as_ref()) {
            let edge_loop = match &bound.bound {
                PlaceHolder::Owned(edge_loop) => Some(edge_loop),
                PlaceHolder::Ref(Name::Entity(id)) => self.edge_loop.get(id),
                _ => None,
            };
            let Some(edge_loop) = edge_loop else { continue };
            let absolute = Self::absolute_bound_orientation(bound, face_sense);
            for edge in &edge_loop.edge_list {
                let (Some(edge), orientation) = self.original_edge_use(edge) else {
                    continue;
                };
                let orientation = orientation == absolute;
                uses.entry(edge)
                    .and_modify(|prior| {
                        *prior = match *prior {
                            OriginalFaceEdgeUse::Single(first) if first != orientation => {
                                OriginalFaceEdgeUse::OpposedPair
                            }
                            _ => OriginalFaceEdgeUse::Repeated,
                        };
                    })
                    .or_insert(OriginalFaceEdgeUse::Single(orientation));
            }
        }
        uses
    }

    pub(super) fn original_trim_source<'a>(
        &'a self,
        edge: u64,
        face_geometry: &PlaceHolder<SurfaceAnyHolder>,
        edge_use: OriginalFaceEdgeUse,
    ) -> Result<TrimSource<'a>, StepConvertingError> {
        let edge = self
            .edge_curve
            .get(&edge)
            .ok_or(ExactConversionError::UnestablishedCurveCorrespondence)?;
        Ok(TrimSource {
            table: self,
            curve: self.original_trim_curve(&edge.edge_geometry, 0)?,
            carrier: Self::ref_id(face_geometry),
            edge_use,
        })
    }

    fn original_trim_curve<'a>(
        &'a self,
        curve: &'a PlaceHolder<CurveAnyHolder>,
        depth: usize,
    ) -> Result<SourceCurve<'a>, StepConvertingError> {
        if depth > MAX_NATIVE_GEOMETRY_NESTING {
            return Err(ExactConversionError::SourceGeometryNestingExceeded.into());
        }
        match curve {
            PlaceHolder::Owned(CurveAnyHolder::TrimmedCurve(curve)) => {
                self.original_trim_curve(&curve.basis_curve, depth + 1)
            }
            PlaceHolder::Owned(CurveAnyHolder::SurfaceCurve(curve)) => {
                Ok(SourceCurve::Surface(curve))
            }
            PlaceHolder::Owned(CurveAnyHolder::Pcurve(curve)) => Ok(SourceCurve::Parameter(curve)),
            PlaceHolder::Ref(Name::Entity(id)) => {
                if let Some(curve) = self.trimmed_curve.get(id) {
                    self.original_trim_curve(&curve.basis_curve, depth + 1)
                } else if let Some(curve) = self.surface_curve.get(id) {
                    Ok(SourceCurve::Surface(curve))
                } else if let Some(curve) = self.pcurve.get(id) {
                    Ok(SourceCurve::Parameter(curve))
                } else {
                    Ok(SourceCurve::Other)
                }
            }
            _ => Ok(SourceCurve::Other),
        }
    }
}

impl TrimSource<'_> {
    fn pcurve<'a>(
        &'a self,
        source: &'a PlaceHolder<PcurveOrSurfaceHolder>,
    ) -> Result<Option<&'a PcurveHolder>, StepConvertingError> {
        match source {
            PlaceHolder::Owned(PcurveOrSurfaceHolder::Pcurve(curve)) => Ok(Some(curve)),
            PlaceHolder::Owned(PcurveOrSurfaceHolder::Surface(_)) => Ok(None),
            PlaceHolder::Ref(Name::Entity(id)) => {
                if let Some(curve) = self.table.pcurve.get(id) {
                    Ok(Some(curve))
                } else {
                    // Bare associated surfaces have no declared parameter curve.
                    Ok(None)
                }
            }
            _ => Err(ExactConversionError::UnestablishedCurveCorrespondence.into()),
        }
    }

    pub(super) fn select<'a>(
        &self,
        curve: &'a Curve3D,
        surface: &Surface,
        orientation: bool,
    ) -> Result<Option<&'a step_geometry::StepParameterCurve>, StepConvertingError> {
        let failure = || ExactConversionError::UnestablishedCurveCorrespondence;
        match curve {
            Curve3D::SurfaceCurve(converted) => {
                let SourceCurve::Surface(source) = self.curve else {
                    return Err(failure().into());
                };
                if source.associated_geometry.len() != converted.associated_geometry().len() {
                    return Err(failure().into());
                }
                let Some(carrier) = self.carrier else {
                    // Geometry may exclude an association, never establish one.
                    return if converted.associated_geometry().iter().any(|geometry| {
                        matches!(geometry, SurfaceCurveAssociatedGeometry::ParameterCurve(candidate)
                            if candidate.surface() == surface)
                    }) {
                        Err(failure().into())
                    } else {
                        Ok(None)
                    };
                };
                let mut first = None;
                let mut second = None;
                let mut count = 0;
                let seam = converted.kind() == SurfaceCurveKind::SeamCurve
                    || matches!(self.edge_use, OriginalFaceEdgeUse::OpposedPair);
                // Conversion preserves the original association list verbatim.
                // Pair declarations with their conversions, then select by the
                // original reference, never by list position or carrier proximity.
                for (original, converted_geometry) in source
                    .associated_geometry
                    .iter()
                    .zip(converted.associated_geometry())
                {
                    let Some(original) = self.pcurve(original)? else {
                        if matches!(
                            converted_geometry,
                            SurfaceCurveAssociatedGeometry::ParameterCurve(_)
                        ) {
                            return Err(failure().into());
                        }
                        continue;
                    };
                    let original_carrier = Table::ref_id(&original.basis_surface);
                    let SurfaceCurveAssociatedGeometry::ParameterCurve(candidate) =
                        converted_geometry
                    else {
                        return Err(failure().into());
                    };
                    match original_carrier {
                        Some(original_carrier) if original_carrier != carrier => continue,
                        None if candidate.surface() != surface => continue,
                        None => return Err(failure().into()),
                        _ => {}
                    }
                    if candidate.surface() != surface {
                        return Err(failure().into());
                    }
                    count += 1;
                    if count == 1 {
                        first = Some(candidate);
                    } else if seam {
                        second = Some(candidate);
                    } else if first != Some(candidate) {
                        return Err(ExactConversionError::AmbiguousStoredTrim.into());
                    }
                }
                // No declared association to this ORIGINAL face is availability
                // failure, not invalid 3D geometry. Keep inspection possible.
                if count == 0 {
                    return Ok(None);
                }
                if matches!(self.edge_use, OriginalFaceEdgeUse::Repeated) {
                    return Err(ExactConversionError::AmbiguousStoredTrim.into());
                }
                if seam {
                    match count {
                        2 if first != second => Ok(if orientation { first } else { second }),
                        1 | 2 => Err(ExactConversionError::MissingSeamParameterBranch.into()),
                        _ => Err(ExactConversionError::AmbiguousStoredTrim.into()),
                    }
                } else {
                    Ok(first)
                }
            }
            Curve3D::ParameterCurve(converted) => {
                let SourceCurve::Parameter(source) = self.curve else {
                    return Err(failure().into());
                };
                match (self.carrier, Table::ref_id(&source.basis_surface)) {
                    (Some(carrier), Some(original)) if carrier != original => Ok(None),
                    (Some(_), Some(_)) if converted.surface() == surface => {
                        if matches!(self.edge_use, OriginalFaceEdgeUse::Single(_)) {
                            Ok(Some(converted))
                        } else {
                            Err(ExactConversionError::MissingSeamParameterBranch.into())
                        }
                    }
                    (Some(_), Some(_)) => Err(failure().into()),
                    _ if converted.surface() != surface => Ok(None),
                    _ => Err(failure().into()),
                }
            }
            _ => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn cylinder_source(second_basis: u64, second_face: u64, kind: &str) -> Table {
        // Two half-cylinder faces share both vertical edges and four cap arcs.
        // The equal cylindrical declarations have distinct original identities.
        let source = format!(
            "ISO-10303-21;\nHEADER;\n\
             FILE_DESCRIPTION(('source-qualified cylinder'),'2;1');\n\
             FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n\
             #1=CARTESIAN_POINT('',(0.,0.,0.));\n\
             #3=DIRECTION('',(0.,0.,1.));\n#4=DIRECTION('',(1.,0.,0.));\n\
             #5=AXIS2_PLACEMENT_3D('',#1,#3,#4);\n\
             #10=CYLINDRICAL_SURFACE('',#5,2.5);\n\
             #11=CYLINDRICAL_SURFACE('',#5,2.5);\n\
             #12=PLANE('',#5);\n\
             #14=CYLINDRICAL_SURFACE('',#5,2.5);\n\
             #20=CARTESIAN_POINT('',(2.5,0.,0.));\n\
             #21=CARTESIAN_POINT('',(2.5,0.,1.));\n\
             #22=CARTESIAN_POINT('',(-2.5,0.,0.));\n\
             #23=CARTESIAN_POINT('',(-2.5,0.,1.));\n\
             #24=VERTEX_POINT('',#20);\n#25=VERTEX_POINT('',#21);\n\
             #26=VERTEX_POINT('',#22);\n#27=VERTEX_POINT('',#23);\n\
             #28=VECTOR('',#3,1.);\n#29=LINE('',#20,#28);\n#30=LINE('',#22,#28);\n\
             #31=CIRCLE('',#5,2.5);\n\
             #32=CARTESIAN_POINT('',(0.,0.,1.));\n\
             #33=AXIS2_PLACEMENT_3D('',#32,#3,#4);\n#34=CIRCLE('',#33,2.5);\n\
             #13=PLANE('',#33);\n\
             #35=CARTESIAN_POINT('',(0.,0.));\n\
             #36=CARTESIAN_POINT('',(6.283185307179586,0.));\n\
             #37=DIRECTION('',(0.,1.));\n#38=VECTOR('',#37,1.);\n\
             #39=LINE('',#35,#38);\n#40=LINE('',#36,#38);\n\
             #41=(GEOMETRIC_REPRESENTATION_CONTEXT(2)PARAMETRIC_REPRESENTATION_CONTEXT()REPRESENTATION_CONTEXT('',''));\n\
             #42=DEFINITIONAL_REPRESENTATION('',(#39),#41);\n\
             #43=DEFINITIONAL_REPRESENTATION('',(#40),#41);\n\
             #44=PCURVE('',#10,#42);\n#45=PCURVE('',#{second_basis},#43);\n\
             #46={kind}('',#29,(#44,#45),.PCURVE_S1.);\n\
             #47=EDGE_CURVE('',#24,#25,#46,.T.);\n\
             #48=EDGE_CURVE('',#26,#27,#30,.T.);\n\
             #49=EDGE_CURVE('',#24,#26,#31,.T.);\n\
             #50=EDGE_CURVE('',#26,#24,#31,.T.);\n\
             #51=EDGE_CURVE('',#25,#27,#34,.T.);\n\
             #52=EDGE_CURVE('',#27,#25,#34,.T.);\n\
             #60=ORIENTED_EDGE('',*,*,#49,.T.);\n\
             #61=ORIENTED_EDGE('',*,*,#48,.T.);\n\
             #62=ORIENTED_EDGE('',*,*,#51,.F.);\n\
             #63=ORIENTED_EDGE('',*,*,#47,.F.);\n\
             #64=ORIENTED_EDGE('',*,*,#50,.T.);\n\
             #65=ORIENTED_EDGE('',*,*,#47,.T.);\n\
             #66=ORIENTED_EDGE('',*,*,#52,.F.);\n\
             #67=ORIENTED_EDGE('',*,*,#48,.F.);\n\
             #68=ORIENTED_EDGE('',*,*,#49,.T.);\n\
             #69=ORIENTED_EDGE('',*,*,#50,.T.);\n\
             #70=ORIENTED_EDGE('',*,*,#51,.T.);\n\
             #71=ORIENTED_EDGE('',*,*,#52,.T.);\n\
             #72=EDGE_LOOP('',(#60,#61,#62,#63));\n\
             #73=EDGE_LOOP('',(#64,#65,#66,#67));\n\
             #74=EDGE_LOOP('',(#68,#69));\n#75=EDGE_LOOP('',(#70,#71));\n\
             #76=FACE_OUTER_BOUND('',#72,.T.);\n#77=FACE_OUTER_BOUND('',#73,.T.);\n\
             #78=FACE_OUTER_BOUND('',#74,.T.);\n#79=FACE_OUTER_BOUND('',#75,.T.);\n\
             #80=ADVANCED_FACE('',(#76),#10,.T.);\n\
             #81=ADVANCED_FACE('',(#77),#{second_face},.T.);\n\
             #82=ADVANCED_FACE('',(#78),#12,.F.);\n\
             #83=ADVANCED_FACE('',(#79),#13,.T.);\n\
             #84=CLOSED_SHELL('',(#80,#81,#82,#83));\n\
             #85=MANIFOLD_SOLID_BREP('',#84);\nENDSEC;\nEND-ISO-10303-21;"
        );
        Table::from_step(&source).expect("independent source cylinder parses")
    }

    #[test]
    fn original_face_ids_select_equal_carrier_uv_branches_without_splitting_edges() {
        let table = cylinder_source(11, 11, "SURFACE_CURVE");
        assert!(table.entity_report.is_empty());
        let (solid, reports) = table
            .to_compressed_trimmed_solid_reported(&table.manifold_solid_brep[&85])
            .expect("original face references disambiguate the declared branches");
        assert!(reports[0].is_lossless());
        let shell = &solid.boundaries[0];
        assert_eq!(shell.vertices.len(), 4);
        assert_eq!(shell.edges.len(), 6);
        assert_eq!(shell.faces.len(), 4);
        assert_eq!(shell.faces[0].surface, shell.faces[1].surface);
        let lineage = &reports[0].source_shells()[0];
        assert_eq!(lineage.faces[0].source_face, Some(80));
        assert_eq!(lineage.faces[1].source_face, Some(81));
        let mut shared = None;
        for (face_index, expected_u) in [(0, 0.0), (1, TAU)] {
            let coedge_index = lineage.faces[face_index].wires[0]
                .coedges
                .iter()
                .position(|coedge| coedge.source_edge == Some(47))
                .expect("both source faces use the same original edge");
            let coedge = &shell.faces[face_index].boundaries[0][coedge_index];
            if let Some(index) = shared {
                assert_eq!(coedge.index, index);
            } else {
                shared = Some(coedge.index);
            }
            let trim = coedge.trim_curve.as_ref().expect("declared UV is retained");
            assert_eq!(trim.front().x, expected_u);
            assert_eq!(trim.back().x, expected_u);
            assert_eq!(trim.front().y, if coedge.orientation { 0.0 } else { 1.0 });
            assert_eq!(trim.back().y, if coedge.orientation { 1.0 } else { 0.0 });
            let Curve3D::SurfaceCurve(original) = &shell.edges[coedge.index].curve else {
                panic!("the shared edge must retain its complete surface curve");
            };
            assert_eq!(original.associated_geometry().len(), 2);
            assert!(matches!(
                original.declared_parameter_curve_ref_on(&shell.faces[face_index].surface, true),
                Err(ExactConversionError::AmbiguousStoredTrim)
            ));
        }
        // Unlike the translated bare LINE elsewhere in this same solid, a
        // spatial transform cannot transport a retained source UV declaration.
        let index = shared.expect("both original faces retained the shared edge");
        let moved = Curve3D::TrimmedCurve(SourceTrimmedCurve::internal(Processor::with_transform(
            TrimmedCurve::new(Box::new(shell.edges[index].curve.clone()), (0.0, 1.0)),
            Matrix4::from_translation(Vector3::new(0.0, 0.0, 1.0)),
        )));
        let binding = table
            .original_trim_source(
                47,
                &table.face_surface[&80].face_geometry,
                OriginalFaceEdgeUse::Single(true),
            )
            .expect("the original face binding resolves");
        let error = Table::compact_declared_trim_with(&moved, true, &|curve, orientation| {
            binding
                .select(curve, &shell.faces[0].surface, orientation)
                .map(|trim| trim.map(|trim| trim.curve().clone()))
        })
        .expect_err("a transformed retained declaration still requires correspondence proof");
        assert!(matches!(
            error.downcast_ref::<ExactConversionError>(),
            Some(ExactConversionError::UnestablishedCurveCorrespondence)
        ));
    }

    #[test]
    fn original_face_binding_refuses_mismatched_missing_and_ambiguous_identity() {
        let mismatched = cylinder_source(11, 14, "SURFACE_CURVE");
        let (solid, reports) = mismatched
            .to_compressed_trimmed_solid_reported(&mismatched.manifold_solid_brep[&85])
            .expect("a face without an original matching pcurve retains unavailable UV");
        let source_use = &reports[0].source_shells()[0].faces[1].wires[0].coedges;
        let slot = source_use
            .iter()
            .position(|coedge| coedge.source_edge == Some(47))
            .expect("the unrelated face still shares the original spatial edge");
        let coedge = &solid.boundaries[0].faces[1].boundaries[0][slot];
        assert!(
            coedge.trim_curve.is_none(),
            "equal carrier coefficients must not authorize another source face's pcurve"
        );
        let Curve3D::SurfaceCurve(original) = &solid.boundaries[0].edges[coedge.index].curve else {
            panic!("unavailable face UV must not discard the original spatial edge");
        };
        assert_eq!(original.associated_geometry().len(), 2);

        let ambiguous = cylinder_source(10, 11, "SURFACE_CURVE");
        let error = ambiguous
            .to_compressed_trimmed_solid_reported(&ambiguous.manifold_solid_brep[&85])
            .expect_err("distinct branches on the same original carrier remain ambiguous");
        let LoadError::Conversion(error) = error else {
            panic!("source ambiguity must be refused by conversion");
        };
        assert!(
            matches!(
                error.downcast_ref::<ExactConversionError>(),
                Some(ExactConversionError::AmbiguousStoredTrim)
            ),
            "unexpected same-source ambiguity refusal: {error:?}"
        );

        let mut missing = cylinder_source(11, 11, "SURFACE_CURVE");
        let carrier = missing.cylindrical_surface[&10].clone();
        // An inline face carrier has coefficients but no original entity ID.
        missing
            .face_surface
            .get_mut(&80)
            .expect("the original face exists")
            .face_geometry = PlaceHolder::Owned(SurfaceAnyHolder::ElementarySurface(Box::new(
            ElementarySurfaceAnyHolder::CylindricalSurface(Box::new(carrier)),
        )));
        let error = missing
            .to_compressed_trimmed_solid_reported(&missing.manifold_solid_brep[&85])
            .expect_err("missing original identity cannot choose between equal carriers");
        let LoadError::Conversion(error) = error else {
            panic!("missing source identity must be refused by conversion");
        };
        assert!(
            matches!(
                error.downcast_ref::<ExactConversionError>(),
                Some(ExactConversionError::UnestablishedCurveCorrespondence)
            ),
            "unexpected missing-source refusal: {error:?}"
        );
    }

    #[test]
    fn seam_without_original_face_association_keeps_shared_inspection_geometry() {
        let mut table = cylinder_source(11, 14, "SEAM_CURVE");
        table
            .face_surface
            .get_mut(&80)
            .expect("first original face")
            .face_geometry = PlaceHolder::Ref(Name::Entity(14));
        let (solid, reports) = table
            .to_compressed_trimmed_solid_reported(&table.manifold_solid_brep[&85])
            .expect("unbound UV must not invalidate the original spatial solid");
        assert!(reports[0].is_lossless());
        assert_eq!(solid.boundaries[0].faces.len(), 4);
        assert_eq!(solid.boundaries[0].edges.len(), 6);
        let mut shared = None;
        for face in 0..2 {
            let uses = &reports[0].source_shells()[0].faces[face].wires[0].coedges;
            let index = uses
                .iter()
                .position(|use_| use_.source_edge == Some(47))
                .expect("the original seam edge remains in each source face");
            let use_ = &solid.boundaries[0].faces[face].boundaries[0][index];
            assert!(use_.trim_curve.is_none());
            if let Some(prior) = shared {
                assert_eq!(use_.index, prior);
            }
            shared = Some(use_.index);
        }
    }

    #[test]
    fn bound_seam_requires_two_distinct_original_parameter_branches() {
        let table = cylinder_source(10, 10, "SEAM_CURVE");
        let (solid, reports) = table
            .to_compressed_trimmed_solid_reported(&table.manifold_solid_brep[&85])
            .expect("two distinct source-bound seam branches resolve by use orientation");
        for face in 0..2 {
            let uses = &reports[0].source_shells()[0].faces[face].wires[0].coedges;
            let index = uses
                .iter()
                .position(|use_| use_.source_edge == Some(47))
                .expect("the source seam is retained");
            let use_ = &solid.boundaries[0].faces[face].boundaries[0][index];
            let uv = use_
                .trim_curve
                .as_ref()
                .expect("the actual oriented branch is bound");
            assert_eq!(uv.front().x, if use_.orientation { 0.0 } else { TAU });
            assert_eq!(uv.back().x, uv.front().x);
        }
        let mut duplicate = cylinder_source(10, 10, "SEAM_CURVE");
        let source = duplicate.surface_curve.get_mut(&46).expect("original seam");
        source.associated_geometry[1] = source.associated_geometry[0].clone();
        for malformed in [cylinder_source(11, 11, "SEAM_CURVE"), duplicate] {
            let error = malformed
                .to_compressed_trimmed_solid_reported(&malformed.manifold_solid_brep[&85])
                .expect_err("one distinct owned branch cannot provide a seam pair");
            let LoadError::Conversion(error) = error else {
                panic!("typed conversion refusal");
            };
            assert!(matches!(
                error.downcast_ref::<ExactConversionError>(),
                Some(ExactConversionError::MissingSeamParameterBranch)
            ));
        }
    }

    #[test]
    fn generic_surface_curve_duplicates_cannot_supply_an_actual_face_seam() {
        let mut table = cylinder_source(10, 10, "SURFACE_CURVE");
        let source = table
            .surface_curve
            .get_mut(&46)
            .expect("original surface curve");
        source.associated_geometry[1] = source.associated_geometry[0].clone();
        table
            .to_compressed_trimmed_solid_reported(&table.manifold_solid_brep[&85])
            .expect("ordinary identical non-seam declarations still resolve");
        // The same source edge now occurs oppositely in TWO original bounds
        // of this actual face. Equality of two entries is not a seam pair.
        table
            .face_surface
            .get_mut(&80)
            .expect("original face")
            .bounds
            .push(PlaceHolder::Ref(Name::Entity(77)));
        let mut paired = cylinder_source(10, 10, "SURFACE_CURVE");
        // Author one complete cylinder side, rather than two half-face uses.
        // Its two seam occurrences belong to this SAME original face.
        paired
            .edge_loop
            .get_mut(&72)
            .expect("original side loop")
            .edge_list = [60, 64, 65, 66, 62, 63]
            .map(|id| PlaceHolder::Ref(Name::Entity(id)))
            .into();
        paired.shell.get_mut(&84).expect("original shell").cfs_faces = [80, 82, 83]
            .map(|id| PlaceHolder::Ref(Name::Entity(id)))
            .into();
        paired
            .surface_curve
            .get_mut(&46)
            .expect("original surface curve")
            .associated_geometry
            .swap(0, 1);
        let (solid, reports) = paired
            .to_compressed_trimmed_solid_reported(&paired.manifold_solid_brep[&85])
            .expect("actual paired occurrences select two distinct source-bound branches");
        assert_eq!(solid.boundaries[0].faces.len(), 3);
        assert_eq!(solid.boundaries[0].edges.len(), 5);
        let mut count = 0;
        for (index, source) in reports[0].source_shells()[0].faces[0].wires[0]
            .coedges
            .iter()
            .enumerate()
        {
            if source.source_edge != Some(47) {
                continue;
            }
            let use_ = &solid.boundaries[0].faces[0].boundaries[0][index];
            let uv = use_
                .trim_curve
                .as_ref()
                .expect("the actual seam occurrence has UV");
            assert_eq!(uv.front().x, if use_.orientation { TAU } else { 0.0 });
            count += 1;
        }
        assert_eq!(count, 2);
        let error = table
            .to_compressed_trimmed_solid_reported(&table.manifold_solid_brep[&85])
            .expect_err("a generic SURFACE_CURVE cannot hide a one-branch real seam");
        let LoadError::Conversion(error) = error else {
            panic!("typed conversion refusal");
        };
        assert!(matches!(
            error.downcast_ref::<ExactConversionError>(),
            Some(ExactConversionError::MissingSeamParameterBranch)
        ));
    }
}
