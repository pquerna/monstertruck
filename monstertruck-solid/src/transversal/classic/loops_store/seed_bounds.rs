//! Query-local rejection using the exact immutable polygon surfaces, not native
//! carrier bounds. Strict separation implies every triangle pair would fail the
//! extractor's own coordinate-axis AABB test; touching bounds remain admitted.
//! The single cache buffer is separately admitted as intermediate scratch and is
//! dropped with this loop-store invocation, so remeshing never reuses its bounds.

use super::*;
use crate::transversal::prepared::MeshShell;
use std::mem::size_of;

#[derive(Clone, Copy)]
struct Bounds {
    min: [f64; 3],
    max: [f64; 3],
}

pub(super) struct SeedBounds {
    entries: Vec<Option<Bounds>>,
    second: usize,
}

impl SeedBounds {
    pub(super) fn new(
        shells: [&MeshShell; 2],
        work: Option<&BooleanWork<'_>>,
    ) -> ClassicResult<Self> {
        let count = shells[0].len().saturating_add(shells[1].len());
        let admit_bytes = |capacity: usize| -> ClassicResult<()> {
            if let Some(work) = work {
                work.limit(
                    "intermediate_prepared_bytes",
                    capacity.saturating_mul(size_of::<Option<Bounds>>()),
                    work.limits.max_intermediate_prepared_bytes,
                )?;
            }
            Ok(())
        };
        if let Some(work) = work {
            work.limit("input_faces", count, work.limits.max_faces)?;
        }
        admit_bytes(count)?;
        let mut entries = Vec::with_capacity(count);
        admit_bytes(entries.capacity())?;
        let mut scanned = 0_usize;
        for shell in shells {
            for face in shell.iter() {
                if let Some(work) = work {
                    work.check()?;
                }
                let bounds = face.with_surface(|mesh| {
                    // Missing tessellation stays unknown here: the existing pair
                    // pipeline reports its original shell/face error in order.
                    let Some(mesh) = mesh else { return Ok(None) };
                    scanned = scanned.saturating_add(mesh.positions().len());
                    if let Some(work) = work {
                        work.limit("mesh_vertices", scanned, work.limits.max_mesh_vertices)?;
                    }
                    let mut bounds = Bounds {
                        min: [f64::INFINITY; 3],
                        max: [f64::NEG_INFINITY; 3],
                    };
                    let mut finite = !mesh.positions().is_empty();
                    for point in mesh.positions() {
                        if let Some(work) = work {
                            work.check()?;
                        }
                        for axis in 0..3 {
                            finite &= point[axis].is_finite();
                            bounds.min[axis] = bounds.min[axis].min(point[axis]);
                            bounds.max[axis] = bounds.max[axis].max(point[axis]);
                        }
                    }
                    Ok::<_, ShapeOpsError>(finite.then_some(bounds))
                })?;
                entries.push(bounds);
            }
        }
        Ok(Self {
            entries,
            second: shells[0].len(),
        })
    }

    pub(super) fn disjoint(
        &self,
        first: usize,
        second: usize,
        work: Option<&BooleanWork<'_>>,
    ) -> ClassicResult<bool> {
        if let Some(work) = work {
            work.check()?;
        }
        let (Some(first), Some(second)) = (self.entries[first], self.entries[self.second + second])
        else {
            return Ok(false);
        };
        Ok((0..3)
            .any(|axis| first.max[axis] < second.min[axis] || second.max[axis] < first.min[axis]))
    }
}

#[cfg(test)]
#[path = "seed_bounds_regressions.rs"]
mod regressions;
