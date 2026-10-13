//! Reuse one immutable native edge's exact subdivision across a shell pair.

use super::*;
use crate::transversal::prepared::BooleanWork;
use std::collections::hash_map::Entry;
use std::mem::size_of;

pub(in crate::transversal::classic) struct CurveDivisionCache<P, C> {
    divisions: HashMap<EdgeId<C>, PolylineCurve<P>>,
    buffer_bytes: usize,
    samples: usize,
    tolerance: f64,
}

impl<P, C> CurveDivisionCache<P, C> {
    pub(in crate::transversal::classic) fn new(tolerance: f64) -> Self {
        Self {
            divisions: HashMap::default(),
            buffer_bytes: 0,
            samples: 0,
            tolerance,
        }
    }

    pub(super) fn get(
        &mut self,
        edge: &Edge<P, C>,
        work: Option<&BooleanWork<'_>>,
    ) -> ClassicResult<&PolylineCurve<P>>
    where
        C: BoundedCurve<Point = P> + ParameterDivision1D<Point = P>,
    {
        if let Some(work) = work {
            work.check()?;
        }
        let count = self.divisions.len().saturating_add(1);
        match self.divisions.entry(edge.id()) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                // Generic subdivision remains indivisible. Borrow the actual
                // immutable curve; do not clone geometry or change its sampler.
                let points = edge.with_curve(|curve| {
                    curve
                        .parameter_division(curve.range_tuple(), self.tolerance)
                        .1
                });
                let buffers = self
                    .buffer_bytes
                    .saturating_add(points.capacity().saturating_mul(size_of::<P>()));
                let samples = self.samples.saturating_add(points.len());
                if let Some(work) = work {
                    work.check()?;
                    work.limit(
                        "curve_division_samples",
                        samples,
                        work.limits.max_mesh_vertices,
                    )?;
                    // A conservative table-growth admission bound, separate
                    // from the exact owned point-buffer capacities. The host
                    // cap still covers opaque sampler/allocator temporaries.
                    let table = count
                        .saturating_add(1)
                        .saturating_mul(4)
                        .saturating_mul(
                            size_of::<(EdgeId<C>, PolylineCurve<P>)>().saturating_add(1),
                        )
                        .saturating_add(16);
                    work.limit(
                        "curve_division_admission_bytes",
                        buffers.saturating_add(table),
                        work.limits.max_intermediate_prepared_bytes,
                    )?;
                }
                self.buffer_bytes = buffers;
                self.samples = samples;
                Ok(entry.insert(PolylineCurve(points)))
            }
        }
    }
}
