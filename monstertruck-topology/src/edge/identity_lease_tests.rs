use super::*;
use crate::compress::CompressedShell;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

// Deliberately neither Clone nor Debug: leasing must impose neither bound on C.
struct OwnedCurve {
    coefficients: Arc<Vec<u8>>,
    drops: Arc<AtomicUsize>,
}

impl Drop for OwnedCurve {
    fn drop(&mut self) { self.drops.fetch_add(1, Ordering::SeqCst); }
}

#[test]
fn lease_preserves_identity_but_releases_geometry_after_last_opposed_use() {
    let vertices = Vertex::from_points([(); 2]);
    let drops = Arc::new(AtomicUsize::new(0));
    let coefficients = Arc::new(vec![7_u8; 4096]);
    let coefficients_expiration = Arc::downgrade(&coefficients);
    let edge = Edge::new(
        &vertices[0],
        &vertices[1],
        OwnedCurve {
            coefficients,
            drops: Arc::clone(&drops),
        },
    );
    let id = edge.id();
    let lease = edge.identity_lease();
    let lease_clone = lease.clone();
    assert_eq!(edge.count(), 1);
    assert_eq!(lease.id(), id);
    assert_eq!(lease_clone.id(), id);
    assert_eq!(vertices[0].count(), 2);
    assert_eq!(vertices[1].count(), 2);
    edge.with_curve(|curve| assert_eq!(curve.coefficients.len(), 4096));
    assert!(format!("{lease:?}").contains(&format!("{id:?}")));

    let opposed = edge.inverse();
    let cloned = edge.clone();
    let opposed_lease = opposed.identity_lease();
    assert_eq!(opposed_lease.id(), id);
    assert_eq!(cloned.identity_lease().id(), id);
    assert_eq!(edge.count(), 3);
    drop(edge);
    drop(cloned);
    assert_eq!(opposed.count(), 1);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert!(coefficients_expiration.upgrade().is_some());
    drop(opposed);

    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(coefficients_expiration.upgrade().is_none());
    assert_eq!(vertices[0].count(), 1);
    assert_eq!(vertices[1].count(), 1);
    assert_eq!(lease.id(), id);
    assert_eq!(lease_clone.id(), id);
    assert_eq!(opposed_lease.id(), id);
    assert_eq!(lease.curve.strong_count(), 0);
    assert!(lease.curve.upgrade().is_none());

    // Cloning an expired lease cannot resurrect geometry or its owned buffers.
    let expired_clone = lease.clone();
    drop(lease);
    drop(lease_clone);
    drop(opposed_lease);
    assert_eq!(expired_clone.id(), id);
    assert!(expired_clone.curve.upgrade().is_none());
    assert!(coefficients_expiration.upgrade().is_none());
    drop(expired_clone);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn historical_identity_cannot_alias_rebuilt_or_generated_edges() {
    let vertices = Vertex::from_points([0_u8, 1]);
    let original = Edge::new(&vertices[0], &vertices[1], (0_u32, 2_u32));
    let original_id = original.id();
    let retained = HashMap::from([(original_id, original.identity_lease())]);
    let rebuilt = original.mapped(|point| *point, |curve| *curve);
    let rebuilt_lease = rebuilt.identity_lease();
    assert_ne!(rebuilt.id(), original_id);
    assert_eq!(rebuilt.curve(), original.curve());
    assert!(!retained.contains_key(&rebuilt.id()));
    drop(original);
    drop(rebuilt);

    // The old allocation is reserved, so these inequalities are deterministic;
    // this is not an allocator-dependent attempt to provoke address recycling.
    let generated = Edge::new(&vertices[0], &vertices[1], (0_u32, 185_u32));
    let same_geometry = Edge::new(&vertices[0], &vertices[1], (0_u32, 2_u32));
    for edge in [&generated, &same_geometry] {
        assert_ne!(edge.id(), original_id);
        assert_ne!(edge.id(), rebuilt_lease.id());
        assert!(!retained.contains_key(&edge.id()));
    }
    assert_eq!(retained[&original_id].id(), original_id);
    assert!(retained[&original_id].curve.upgrade().is_none());
    assert_eq!(generated.curve(), (0, 185));
    assert_eq!(same_geometry.curve(), (0, 2));
}

#[test]
fn leases_preserve_send_sync_for_send_but_not_sync_curve_values() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Edge<(), Cell<u8>>>();
    assert_send_sync::<EdgeIdentityLease<Cell<u8>>>();

    let vertices = Vertex::from_points([(); 2]);
    let edge = Edge::new(&vertices[0], &vertices[1], Cell::new(5_u8));
    let lease = edge.identity_lease();
    let id = lease.id();
    drop(edge);
    let returned = std::thread::spawn(move || lease).join().unwrap();
    assert_eq!(returned.id(), id);
    assert!(returned.curve.upgrade().is_none());
}

#[test]
fn retained_charge_includes_inline_curve_counts_and_overalignment() {
    #[repr(align(64))]
    struct Aligned;

    fn expected<C>() -> usize {
        let (layout, _) = std::alloc::Layout::new::<[AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<Mutex<C>>())
            .unwrap();
        layout.pad_to_align().size()
    }
    let vertices = Vertex::from_points([(); 2]);
    let empty = Edge::new(&vertices[0], &vertices[1], ());
    let aligned = Edge::new(&vertices[0], &vertices[1], Aligned);
    let empty_lease = empty.identity_lease();
    let aligned_lease = aligned.identity_lease();
    assert_eq!(empty_lease.retained_allocation_bytes(), expected::<()>());
    assert_eq!(
        aligned_lease.retained_allocation_bytes(),
        expected::<Aligned>()
    );
    assert!(aligned_lease.retained_allocation_bytes() > empty_lease.retained_allocation_bytes());
    let charge = aligned_lease.retained_allocation_bytes();
    let cloned = aligned_lease.clone();
    drop(aligned);
    assert_eq!(aligned_lease.retained_allocation_bytes(), charge);
    assert_eq!(cloned.retained_allocation_bytes(), charge);
}

#[test]
fn compressed_serialization_contract_is_independent_of_native_identity_leases() {
    fn assert_serde<T: serde::Serialize + serde::de::DeserializeOwned>() {}
    assert_serde::<CompressedShell<u8, (u8, u8), u8>>();

    let vertices = Vertex::from_points([0_u8, 1, 2]);
    let edges = [
        Edge::new(&vertices[0], &vertices[1], (0_u8, 1_u8)),
        Edge::new(&vertices[1], &vertices[2], (1, 2)),
        Edge::new(&vertices[2], &vertices[0], (2, 0)),
    ];
    let leases = edges.each_ref().map(Edge::identity_lease);
    let shell: Shell<_, _, _> = vec![Face::new(
        vec![wire![&edges[0], &edges[1], &edges[2]]],
        9_u8,
    )]
    .into();
    let compressed = shell.compress();
    drop(edges);
    drop(shell);
    assert!(leases.iter().all(|lease| lease.curve.upgrade().is_none()));

    let extracted = Shell::extract(compressed.clone()).unwrap();
    assert_eq!(extracted.compress(), compressed);
    for edge in extracted[0].absolute_boundaries()[0].iter() {
        assert!(leases.iter().all(|lease| edge.id() != lease.id()));
    }
}
