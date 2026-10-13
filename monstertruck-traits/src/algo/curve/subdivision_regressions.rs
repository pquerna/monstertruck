use super::*;
use crate::polynomial::PolynomialCurve;

// Published recursive assembly retained only as a regression oracle. Geometry,
// probe seeds, stopping rules and evaluation order must not change when output
// storage stops allocating/copying whole subtrees.
fn reference<C>(
    curve: &C,
    range: (f64, f64),
    ends: (C::Point, C::Point),
    tol: f64,
    trials: usize,
) -> (Vec<f64>, Vec<C::Point>)
where
    C: ParametricCurve,
    C::Point: EuclideanSpace<Scalar = f64> + MetricSpace<Metric = f64> + HashGen<f64>,
{
    let seed = ends.0.midpoint(ends.1);
    let p = 0.5 + (0.2 * HashGen::hash1(seed) - 0.1);
    let t = range.0 * (1.0 - p) + range.1 * p;
    let mid = ends.0 + (ends.1 - ends.0) * p;
    let dist2 = curve.evaluate(t).distance2(mid);
    let mid_param = (range.0 + range.1) / 2.0;
    let range_is_halvable = range.0 < mid_param && mid_param < range.1;
    if dist2 < tol * tol || trials == 0 || !dist2.is_finite() || !range_is_halvable {
        (vec![range.0, range.1], vec![ends.0, ends.1])
    } else {
        let mid_value = curve.evaluate(mid_param);
        let (mut parameters, mut points) = reference(
            curve,
            (range.0, mid_param),
            (ends.0, mid_value),
            tol,
            trials - 1,
        );
        parameters.pop();
        points.pop();
        let (right_parameters, right_points) = reference(
            curve,
            (mid_param, range.1),
            (mid_value, ends.1),
            tol,
            trials - 1,
        );
        parameters.extend(right_parameters);
        points.extend(right_points);
        (parameters, points)
    }
}

fn bits(points: &[Point3]) -> Vec<[u64; 3]> {
    points
        .iter()
        .map(|point| [point.x.to_bits(), point.y.to_bits(), point.z.to_bits()])
        .collect()
}

#[test]
fn native_polynomial_subdivision_keeps_published_probe_and_sample_sequence() {
    let curves = [
        PolynomialCurve::<Point3>(vec![
            Vector3::new(-3.0, 2.0, 1.0),
            Vector3::new(7.0, -4.0, 0.0),
            Vector3::new(0.0, 5.0, 2.0),
        ]),
        PolynomialCurve::<Point3>(vec![
            Vector3::zero(),
            Vector3::new(100.0, 0.0, 0.0),
            Vector3::new(-180.0, 20.0, 0.0),
            Vector3::new(120.0, -15.0, 4.0),
        ]),
    ];
    for curve in &curves {
        for range in [
            (-1.0, 2.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (0.5, 0.5),
            (1.0, f64::from_bits(1.0_f64.to_bits() + 1)),
        ] {
            let ends = (curve.evaluate(range.0), curve.evaluate(range.1));
            let expected = reference(curve, range, ends, 1.0e-5, 100);
            let actual = parameter_division(curve, range, 1.0e-5);
            assert_eq!(
                actual
                    .0
                    .iter()
                    .map(|parameter| parameter.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .0
                    .iter()
                    .map(|parameter| parameter.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(bits(&actual.1), bits(&expected.1));
            assert_eq!(actual.0.len(), actual.1.len());
            assert_eq!(actual.0.first().copied(), Some(range.0));
            assert_eq!(actual.0.last().copied(), Some(range.1));
        }
    }
}

#[test]
fn native_parabola_chords_keep_full_ordered_coverage_at_original_tolerance() {
    let curve =
        PolynomialCurve::<Point3>(vec![Vector3::zero(), Vector3::unit_x(), Vector3::unit_y()]);
    let tolerance = 1.0e-5;
    let (parameters, points) = parameter_division(&curve, (0.0, 1.0), tolerance);
    for (parameters, points) in parameters.windows(2).zip(points.windows(2)) {
        assert!(parameters[0] < parameters[1]);
        assert_eq!(points[0], curve.evaluate(parameters[0]));
        assert_eq!(points[1], curve.evaluate(parameters[1]));
        // The unchanged hashed probe lies in [0.4, 0.6]. For this exact
        // quadratic, its maximum midpoint deviation is at most 25/24 of
        // that probe's deviation; this verifies actual chord geometry.
        for index in 0..=16 {
            let fraction = index as f64 / 16.0;
            let parameter = parameters[0] * (1.0 - fraction) + parameters[1] * fraction;
            let chord = points[0] + (points[1] - points[0]) * fraction;
            assert!(
                curve.evaluate(parameter).distance(chord) <= tolerance * (25.0 / 24.0) + 1.0e-14
            );
        }
    }
}
