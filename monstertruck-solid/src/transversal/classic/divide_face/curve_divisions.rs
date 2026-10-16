//! Reuse one immutable native edge's exact subdivision across a shell pair.

use super::*;
use crate::transversal::prepared::BooleanWork;
use std::collections::hash_map::Entry;
use std::mem::size_of;

type CurveSamples<P, C> = (EdgeIdentityLease<C>, Vec<f64>, Vec<P>);

pub(in crate::transversal::classic) struct CurveDivisionCache<P, C> {
    divisions: HashMap<EdgeId<C>, CurveSamples<P, C>>,
    owned_bytes: usize,
    samples: usize,
    tolerance: f64,
}

impl<P, C> CurveDivisionCache<P, C> {
    pub(in crate::transversal::classic) fn new(tolerance: f64) -> Self {
        Self {
            divisions: HashMap::default(),
            owned_bytes: 0,
            samples: 0,
            tolerance,
        }
    }

    pub(super) fn get(
        &mut self,
        edge: &Edge<P, C>,
        work: Option<&BooleanWork<'_>>,
    ) -> ClassicResult<(&[f64], &[P])>
    where
        C: BoundedCurve<Point = P> + ParameterDivision1D<Point = P>,
    {
        if let Some(work) = work {
            work.check()?;
        }
        let count = self.divisions.len().saturating_add(1);
        match self.divisions.entry(edge.id()) {
            Entry::Occupied(entry) => {
                let (_, parameters, points) = entry.into_mut();
                Ok((parameters.as_slice(), points.as_slice()))
            }
            Entry::Vacant(entry) => {
                // Generic subdivision remains indivisible. Borrow the actual
                // immutable curve; do not clone geometry or change its sampler.
                let (parameters, points) = edge.with_curve(|curve| {
                    curve.parameter_division(curve.range_tuple(), self.tolerance)
                });
                let lease = edge.identity_lease();
                let buffers = self
                    .owned_bytes
                    .saturating_add(points.capacity().saturating_mul(size_of::<P>()))
                    .saturating_add(parameters.capacity().saturating_mul(size_of::<f64>()))
                    .saturating_add(lease.retained_allocation_bytes());
                let samples = self.samples.saturating_add(points.len());
                if let Some(work) = work {
                    work.check()?;
                    work.limit(
                        "curve_division_samples",
                        samples,
                        work.limits.max_mesh_vertices,
                    )?;
                    // A conservative table-growth admission bound, separate
                    // from actual parameter/point capacities and identity
                    // allocation lifetimes. No original curve buffers are retained.
                    let table = count
                        .saturating_add(1)
                        .saturating_mul(4)
                        .saturating_mul(
                            size_of::<(EdgeId<C>, CurveSamples<P, C>)>().saturating_add(1),
                        )
                        .saturating_add(16);
                    work.limit(
                        "curve_division_admission_bytes",
                        buffers.saturating_add(table),
                        work.limits.max_intermediate_prepared_bytes,
                    )?;
                }
                self.owned_bytes = buffers;
                self.samples = samples;
                let (_, parameters, points) = entry.insert((lease, parameters, points));
                Ok((parameters.as_slice(), points.as_slice()))
            }
        }
    }
}
