//! Borrowed oriented-use access to stored face-local chart data.

use super::TrimmedFace;
use crate::{Edge, EdgeId, Face, FaceId};
use monstertruck_traits::{BoundedCurve, ParameterRange};
use std::sync::atomic::{AtomicBool, Ordering};

/// Identity of one occurrence, rather than merely its shared edge.
#[derive(Debug)]
pub struct ChartCoedgeIdentity<C, S> {
    /// Native face carrier identity.
    pub face_id: FaceId<S>,
    /// Absolute boundary wire ordinal.
    pub wire_index: usize,
    /// Absolute coedge ordinal in that wire, including repeated seam uses.
    pub coedge_index: usize,
    /// Shared native edge identity.
    pub edge_id: EdgeId<C>,
    /// Direction relative to the absolute edge parameter, including face inversion.
    pub orientation: bool,
}

impl<C, S> Copy for ChartCoedgeIdentity<C, S> {}
impl<C, S> Clone for ChartCoedgeIdentity<C, S> {
    fn clone(&self) -> Self { *self }
}

/// Reason exact chart information cannot be supplied from native storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChartInformationUnavailable {
    /// This occurrence has no stored face-local pcurve.
    NoStoredPcurve,
    /// The stored pcurve is not bounded by finite native parameters.
    UnboundedPcurve,
    /// No explicit seam or pole chart correspondence is retained.
    NoStoredChartCorrespondence,
}

/// Stored pcurve access, without synthesizing an inverse-map approximation.
#[derive(Clone, Copy, Debug)]
pub enum StoredPcurve<'a, T> {
    /// Borrowed native payload and its exact parameter bounds, including endpoint inclusion.
    Available {
        /// Stored face-local curve, interpreted in the absolute edge direction.
        curve: &'a T,
        /// Native curve bounds; endpoint inclusion is not discarded.
        domain: ParameterRange,
    },
    /// Native storage lacks the requested bounded chart information.
    Unavailable(ChartInformationUnavailable),
}

/// A borrowed native coedge and its stored face-local chart payload.
///
/// Stored data is not a proof that an arbitrary caller-supplied payload matches
/// the 3D edge. Importers must preserve their exact source correspondence. Shared
/// opposite edge uses alone do not establish a periodic UV seam or a pole map.
#[derive(Debug)]
pub struct ChartCoedge<'a, P, C, S, T> {
    /// Complete oriented occurrence identity.
    pub identity: ChartCoedgeIdentity<C, S>,
    /// Shared native edge, stored in the absolute boundary wire orientation.
    pub edge: &'a Edge<P, C>,
    /// Stored bounded face-local pcurve, or a typed lack of chart information.
    pub pcurve: StoredPcurve<'a, T>,
}

/// Explicit bounded traversal failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChartTraversalError {
    /// Cooperative cancellation was requested.
    Cancelled,
    /// The oriented occurrence cap was exceeded.
    CoedgeLimit,
}

/// Lazy, allocation-free traversal of every absolute boundary occurrence.
pub struct ChartCoedges<'a, P, C, S, T> {
    face: &'a TrimmedFace<P, C, S, T>,
    wire_index: usize,
    coedge_index: usize,
}

impl<P, C, S, T> std::fmt::Debug for ChartCoedges<'_, P, C, S, T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ChartCoedges")
            .field("wire_index", &self.wire_index)
            .field("coedge_index", &self.coedge_index)
            .finish()
    }
}

impl<'a, P, C, S, T: BoundedCurve> Iterator for ChartCoedges<'a, P, C, S, T> {
    type Item = ChartCoedge<'a, P, C, S, T>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(wire) = self.face.face().absolute_boundaries().get(self.wire_index) {
            if let Some(edge) = wire.get(self.coedge_index) {
                let identity = ChartCoedgeIdentity {
                    face_id: self.face.face().id(),
                    wire_index: self.wire_index,
                    coedge_index: self.coedge_index,
                    edge_id: edge.id(),
                    orientation: edge.orientation() == self.face.face().orientation(),
                };
                let pcurve = match &self.face.trims()[self.wire_index][self.coedge_index] {
                    Some(curve) => {
                        let domain = curve.parameter_range();
                        if curve
                            .try_range_tuple()
                            .is_some_and(|(a, b)| a.is_finite() && b.is_finite() && a <= b)
                        {
                            StoredPcurve::Available { curve, domain }
                        } else {
                            StoredPcurve::Unavailable(ChartInformationUnavailable::UnboundedPcurve)
                        }
                    }
                    None => StoredPcurve::Unavailable(ChartInformationUnavailable::NoStoredPcurve),
                };
                self.coedge_index += 1;
                return Some(ChartCoedge {
                    identity,
                    edge,
                    pcurve,
                });
            }
            self.wire_index += 1;
            self.coedge_index = 0;
        }
        None
    }
}

impl<P, C, S, T: BoundedCurve> TrimmedFace<P, C, S, T> {
    /// Borrows all coedge occurrences and native bounded pcurve payloads.
    ///
    /// No geometry is copied and no inverse mapping is performed. Absolute wire
    /// ordinals remain stable when the face is inverted; occurrence orientation
    /// explicitly includes that inversion. No unstored seam/pole map is inferred.
    pub fn chart_coedges(&self) -> ChartCoedges<'_, P, C, S, T> {
        ChartCoedges {
            face: self,
            wire_index: 0,
            coedge_index: 0,
        }
    }

    /// Visits borrowed chart occurrences with bounded cooperative cancellation.
    ///
    /// Checks cancellation before each callback. A callback is indivisible and
    /// must impose its own geometric work bound. Missing pcurves remain typed
    /// payloads, not errors that silently omit boundary occurrences.
    pub fn visit_chart_coedges(
        &self,
        max_coedges: usize,
        cancel: &AtomicBool,
        mut inspect: impl FnMut(ChartCoedge<'_, P, C, S, T>),
    ) -> Result<(), ChartTraversalError> {
        for (index, coedge) in self.chart_coedges().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                return Err(ChartTraversalError::Cancelled);
            }
            if index >= max_coedges {
                return Err(ChartTraversalError::CoedgeLimit);
            }
            inspect(coedge);
        }
        if cancel.load(Ordering::Relaxed) {
            Err(ChartTraversalError::Cancelled)
        } else {
            Ok(())
        }
    }
}

impl<P, C, S> Face<P, C, S> {
    /// Reports the absence of stored face-local chart correspondence on a plain face.
    ///
    /// A 3D-only face does not retain pcurve or periodic seam/pole correspondence.
    /// Exact correspondence carried by its curve payload may instead be inspected
    /// under the existing borrowed edge and surface callbacks.
    pub const fn stored_chart_information(&self) -> ChartInformationUnavailable {
        ChartInformationUnavailable::NoStoredPcurve
    }
}
