use super::*;
use monstertruck_traits::BoundedCurve;

#[test]
fn one_bucket_does_not_turn_a_regular_segment_into_a_single_point_branch() {
    let a = Point3::new(-0.75 * TOLERANCE, 0.0, 0.0);
    let b = Point3::new(0.75 * TOLERANCE, 0.0, 0.0);
    assert_eq!(PointIndex::from(a), PointIndex::from(b));
    assert!(!a.near(&b));
    let result = construct_polylines(&[(a, b)]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 2);
    assert!(!result[0].front().near(&result[0].back()));
}

#[test]
fn certified_neighboring_bucket_contacts_stitch_one_complete_branch() {
    let a = Point3::new(-10.0 * TOLERANCE, 0.0, 0.0);
    let p = Point3::new(0.9 * TOLERANCE, 0.0, 0.0);
    let q = Point3::new(1.1 * TOLERANCE, 0.0, 0.0);
    let b = Point3::new(10.0 * TOLERANCE, 0.0, 0.0);
    assert_ne!(PointIndex::from(p), PointIndex::from(q));
    assert!(p.near(&q));
    let result = construct_polylines(&[(a, p), (q, b)]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 3);
    assert!(result[0].front().near(&a) || result[0].back().near(&a));
    assert!(result[0].front().near(&b) || result[0].back().near(&b));
}

#[test]
fn adjacent_cells_do_not_merge_unrelated_contacts_outside_kernel_residual() {
    let a = Point3::new(-10.0 * TOLERANCE, 0.0, 0.0);
    let p = Point3::new(0.1 * TOLERANCE, 0.0, 0.0);
    let q = Point3::new(1.9 * TOLERANCE, 0.0, 0.0);
    let b = Point3::new(10.0 * TOLERANCE, 0.0, 0.0);
    assert!(!p.near(&q));
    let result = construct_polylines(&[(a, p), (q, b)]);
    assert_eq!(result.len(), 2);
    assert!(result.iter().all(|branch| branch.len() == 2));
}
