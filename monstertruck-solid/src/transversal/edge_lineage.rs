use super::integrate::ShapeOpsError;
use super::prepared::BooleanWork;
use super::provenance::BooleanFaceOrigin;
use monstertruck_geometry::prelude::*;
use monstertruck_topology::{Edge, EdgeId, FaceId, Shell, Solid};
use rustc_hash::{FxHashMap, FxHashSet};
use std::{mem::size_of, result::Result};

/// Which native child of an absolute-parameter curve cut was selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BooleanCutSide {
    /// Child preceding the cut in the parent's absolute curve parameter.
    Before,
    /// Child following the cut in the parent's absolute curve parameter.
    After,
}

/// Actual native source-edge and subdivision construction history.
///
/// A cut records real native parameter domains and the actual cut parameter. It
/// does not assert that a generic curve implementation preserves or affinely
/// reparameterizes its parent's domain. Concrete native consumers can interpret
/// their known curve's cut contract without nearest-point reconstruction.
#[derive(Debug)]
pub enum BooleanEdgeDerivation<C> {
    /// An edge in the original borrowed operands.
    Source {
        /// Original shared native edge identity.
        edge_id: EdgeId<C>,
        /// Complete original native domain, including endpoint inclusion.
        domain: ParameterRange,
    },
    /// A native child produced by cutting an earlier edge derivation.
    Cut {
        /// Earlier entry in the returned derivation graph.
        parent: usize,
        /// Exact native parent domain immediately before the cut.
        parent_domain: ParameterRange,
        /// Parameter actually passed to the native cut operation.
        parameter: f64,
        /// Selected child of that operation.
        side: BooleanCutSide,
        /// Actual bounded domain of the native child, not a guessed source map.
        child_domain: ParameterRange,
    },
}

/// Original oriented occurrence associated with an output Boolean coedge.
#[derive(Debug)]
pub enum BooleanCoedgeOrigin<C, S> {
    /// A retained or subdivided original occurrence, with exact construction history.
    Original {
        /// Original operand face identity and occurrence.
        face: BooleanFaceOrigin<S>,
        /// Absolute original boundary wire ordinal.
        wire_index: usize,
        /// Absolute original coedge ordinal, including opposite seam uses.
        coedge_index: usize,
        /// Original shared edge identity for retained source pcurve lookup.
        edge_id: EdgeId<C>,
        /// Original material direction relative to that edge.
        orientation: bool,
        /// Entry in the returned edge derivation graph.
        derivation: usize,
    },
    /// A new intersection boundary has no original stored coedge pcurve.
    GeneratedIntersection {
        /// Original carrier face from which the output face was divided.
        face: BooleanFaceOrigin<S>,
    },
    /// Repeated same-direction original uses have no authoritative branch binding.
    AmbiguousOriginalUse {
        /// Original carrier face.
        face: BooleanFaceOrigin<S>,
        /// Shared original edge whose occurrence cannot be distinguished.
        edge_id: EdgeId<C>,
    },
}

/// Genealogy of one complete oriented boundary occurrence in the selected output.
#[derive(Debug)]
pub struct BooleanCoedgeProvenance<C, S> {
    /// Selected output face identity.
    pub output_face: FaceId<S>,
    /// Absolute output boundary wire ordinal.
    pub wire_index: usize,
    /// Absolute output coedge ordinal.
    pub coedge_index: usize,
    /// Shared selected output edge identity.
    pub output_edge: EdgeId<C>,
    /// Output material direction relative to that edge.
    pub orientation: bool,
    /// Original-use association or typed absence of original exact pcurve lineage.
    pub source: BooleanCoedgeOrigin<C, S>,
}

impl<C, S> BooleanCoedgeProvenance<C, S> {
    /// Returns the original carrier-face use for this selected output occurrence.
    ///
    /// For a cutting-cap edge, another report entry with the identical
    /// `output_edge` identifies its opposite original source carrier. No
    /// coefficient equality, proximity or sampled surface search is involved.
    pub fn source_face(&self) -> &BooleanFaceOrigin<S> {
        match &self.source {
            BooleanCoedgeOrigin::Original { face, .. }
            | BooleanCoedgeOrigin::GeneratedIntersection { face }
            | BooleanCoedgeOrigin::AmbiguousOriginalUse { face, .. } => face,
        }
    }
}

pub(crate) type RewriteObserver<'a, P, C> =
    dyn FnMut(EdgeId<C>, &Edge<P, C>, Option<(ParameterRange, f64, BooleanCutSide)>) -> bool + 'a;

fn map_bytes<K, V>(map: &FxHashMap<K, V>, adding: bool) -> usize {
    let buckets = map
        .capacity()
        .saturating_add(1)
        .checked_next_power_of_two()
        .unwrap_or(usize::MAX)
        .max(8);
    let buckets = if adding && map.len() == map.capacity() {
        buckets.saturating_mul(2)
    } else {
        buckets
    };
    buckets
        .saturating_mul(size_of::<(K, V)>().saturating_add(1))
        .saturating_add(32)
}

pub(crate) struct EdgeTrace<C> {
    pub(crate) nodes: Vec<BooleanEdgeDerivation<C>>,
    native: FxHashMap<EdgeId<C>, usize>,
}

impl<C> EdgeTrace<C> {
    pub(crate) fn new() -> Self {
        Self {
            nodes: Vec::new(),
            native: FxHashMap::default(),
        }
    }

    fn node(
        &mut self,
        node: BooleanEdgeDerivation<C>,
        work: &BooleanWork<'_>,
    ) -> Result<usize, ShapeOpsError> {
        let capacity = if self.nodes.len() == self.nodes.capacity() {
            self.nodes.capacity().saturating_mul(2).max(4)
        } else {
            self.nodes.capacity()
        };
        work.limit(
            "edge_lineage_bytes",
            capacity.saturating_mul(size_of::<BooleanEdgeDerivation<C>>()),
            work.limits.max_intermediate_prepared_bytes,
        )?;
        let index = self.nodes.len();
        self.nodes.push(node);
        work.limit(
            "edge_lineage_bytes",
            self.nodes
                .capacity()
                .saturating_mul(size_of::<BooleanEdgeDerivation<C>>()),
            work.limits.max_intermediate_prepared_bytes,
        )?;
        Ok(index)
    }

    pub(crate) fn admit<S>(
        &mut self,
        solid: &Solid<Point3, C, S>,
        work: &BooleanWork<'_>,
    ) -> Result<(), ShapeOpsError>
    where
        C: BoundedCurve,
    {
        solid.face_iter().try_for_each(|face| {
            face.absolute_boundaries()
                .iter()
                .flatten()
                .try_for_each(|edge| {
                    work.check()?;
                    if !self.native.contains_key(&edge.id()) {
                        let domain = edge.with_curve(ParametricCurve::parameter_range);
                        let index = self.node(
                            BooleanEdgeDerivation::Source {
                                edge_id: edge.id(),
                                domain,
                            },
                            work,
                        )?;
                        self.bind_native(edge.id(), index, work)?;
                    }
                    Ok(())
                })
        })
    }

    pub(crate) fn index(&self, id: EdgeId<C>) -> Option<usize> { self.native.get(&id).copied() }

    pub(crate) fn bind_native(
        &mut self,
        id: EdgeId<C>,
        index: usize,
        work: &BooleanWork<'_>,
    ) -> Result<(), ShapeOpsError> {
        work.limit(
            "native_edge_lineage_bytes",
            map_bytes(&self.native, true),
            work.limits.max_intermediate_prepared_bytes,
        )?;
        self.native.insert(id, index);
        Ok(())
    }

    pub(crate) fn prune_shell<S>(
        &mut self,
        shell: &Shell<Point3, C, S>,
        work: &BooleanWork<'_>,
    ) -> Result<(), ShapeOpsError> {
        let count = shell
            .iter()
            .map(|face| {
                face.absolute_boundaries()
                    .iter()
                    .map(|wire| wire.len())
                    .sum::<usize>()
            })
            .sum::<usize>();
        let buckets = count
            .saturating_mul(2)
            .checked_next_power_of_two()
            .unwrap_or(usize::MAX)
            .max(8);
        work.limit(
            "selected_edge_lineage_bytes",
            buckets
                .saturating_mul(size_of::<EdgeId<C>>().saturating_add(1))
                .saturating_add(32),
            work.limits.max_intermediate_prepared_bytes,
        )?;
        let mut selected = FxHashSet::with_capacity_and_hasher(count, Default::default());
        for edge in shell
            .iter()
            .flat_map(|face| face.absolute_boundaries().iter().flatten())
        {
            work.check()?;
            selected.insert(edge.id());
        }
        work.charge(self.native.len())?;
        self.native.retain(|id, index| {
            matches!(&self.nodes[*index], BooleanEdgeDerivation::Source { edge_id, .. } if edge_id == id)
                || selected.contains(id)
        });
        work.check()
    }

    pub(crate) fn root(
        &self,
        mut index: usize,
        work: &BooleanWork<'_>,
    ) -> Result<EdgeId<C>, ShapeOpsError> {
        loop {
            work.check()?;
            match &self.nodes[index] {
                BooleanEdgeDerivation::Source { edge_id, .. } => return Ok(*edge_id),
                BooleanEdgeDerivation::Cut { parent, .. } => index = *parent,
            }
        }
    }
}

pub(crate) struct PairEdgeTrace<D> {
    mapped: FxHashMap<EdgeId<D>, usize>,
    pub(crate) error: Option<ShapeOpsError>,
}

impl<D> PairEdgeTrace<D> {
    pub(crate) fn new() -> Self {
        Self {
            mapped: FxHashMap::default(),
            error: None,
        }
    }
    pub(crate) fn bind(
        &mut self,
        id: EdgeId<D>,
        index: usize,
        work: &BooleanWork<'_>,
    ) -> Result<(), ShapeOpsError> {
        work.limit(
            "mapped_edge_lineage_bytes",
            map_bytes(&self.mapped, true),
            work.limits.max_intermediate_prepared_bytes,
        )?;
        self.mapped.insert(id, index);
        Ok(())
    }
    pub(crate) fn index(&self, id: EdgeId<D>) -> Option<usize> { self.mapped.get(&id).copied() }
    pub(crate) fn rewrite<C>(
        &mut self,
        trace: &mut EdgeTrace<C>,
        old: EdgeId<D>,
        new: &Edge<Point3, D>,
        cut: Option<(ParameterRange, f64, BooleanCutSide)>,
        work: &BooleanWork<'_>,
    ) -> bool
    where
        D: BoundedCurve,
    {
        let Some(parent) = self.index(old) else {
            return true;
        };
        let result = (|| {
            work.check()?;
            let index = if let Some((parent_domain, parameter, side)) = cut {
                trace.node(
                    BooleanEdgeDerivation::Cut {
                        parent,
                        parent_domain,
                        parameter,
                        side,
                        child_domain: new.with_curve(ParametricCurve::parameter_range),
                    },
                    work,
                )?
            } else {
                parent
            };
            self.bind(new.id(), index, work)?;
            if cut.is_none_or(|(_, _, side)| side == BooleanCutSide::After) {
                self.mapped.remove(&old);
            }
            Ok(())
        })();
        match result {
            Ok(()) => true,
            Err(error) => {
                self.error = Some(error);
                false
            }
        }
    }
}
