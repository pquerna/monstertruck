use super::{
    ValidatedGeometryBudget, ValidatedGeometryError, ValidatedInterval, ZERO, interval_range,
    surface_box_smooth, validated_sin_cos, validated_surface_jet,
};
use crate::{Matrix4, Surface};
use std::sync::atomic::AtomicBool;

type Interval = ValidatedInterval;
type Error = ValidatedGeometryError;

fn cross(a: [Interval; 3], b: [Interval; 3]) -> [Interval; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn finite(normal: [Interval; 3]) -> Result<[Interval; 3], Error> {
    if normal.into_iter().all(Interval::is_finite) {
        Ok(normal)
    } else {
        Err(Error::ConditioningFailure)
    }
}

fn affine_cofactor(
    matrix: &Matrix4,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<[[Interval; 3]; 3], Error> {
    budget.charge(64, cancel)?;
    if !(0..4).all(|i| (0..4).all(|j| matrix[i][j].is_finite())) {
        return Err(Error::InvalidInput);
    }
    if !(0..3).all(|i| matrix[i][3] == 0.0) || matrix[3][3] != 1.0 {
        return Err(Error::Unsupported);
    }
    let columns: [[Interval; 3]; 3] =
        std::array::from_fn(|j| std::array::from_fn(|i| Interval::point(matrix[j][i])));
    let cofactor = [
        cross(columns[1], columns[2]),
        cross(columns[2], columns[0]),
        cross(columns[0], columns[1]),
    ];
    cofactor
        .into_iter()
        .try_for_each(|column| finite(column).map(|_| ()))?;
    let determinant = (0..3)
        .map(|i| columns[0][i] * cofactor[0][i])
        .fold(ZERO, |a, b| a + b);
    if !determinant.is_finite() {
        Err(Error::ConditioningFailure)
    } else if determinant.contains_zero() {
        Err(Error::Singular)
    } else {
        Ok(cofactor)
    }
}

/// Encloses a continuous, nowhere-zero oriented geometric normal on a UV box.
///
/// At every parameter, the enclosed field is a positive multiple of the native
/// source-geometric normal, not necessarily a unit vector. Its component intervals
/// may all contain zero: that does not invalidate the separate regularity proof.
/// Ordinary charts require a certified C1 source box and a whole-box nonzero
/// `du × dv`. Unqualified creases and singular charts refuse certification.
///
/// Analytic spheres instead use their coefficient-defined unit radial field,
/// including collapsed polar charts. A positive finite radius, finite center,
/// and certified nonsingular affine processor prove geometric regularity. The
/// processor's cofactor maps this field with the native reflection convention;
/// inversion swaps UV and negates it. No zero chart derivative is normalized.
/// Projective, singular, and nonfinite processor transforms are refused.
///
/// Source orientation follows the coefficient derivatives on regular charts and
/// the canonical analytic primitive extension at spherical poles. Historical
/// NURBS nearby-normal averaging and revolution endpoint heuristics are excluded,
/// as in the other validated coefficient queries. Torus charts must prove their
/// canonical radial orientation factor positive. Uncertified conditions refuse.
/// All coefficient scans and fixed-size arithmetic share the caller's work and
/// cancellation allowance; no subdivision, sampling, or heap storage is added.
pub fn validated_surface_geometric_normal(
    surface: &Surface,
    domain: [[f64; 2]; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<[Interval; 3], Error> {
    budget.charge(1, cancel)?;
    let uv = [interval_range(domain[0])?, interval_range(domain[1])?];
    match surface {
        Surface::SphericalSurface(processor) => {
            let sphere = processor.entity();
            if !sphere.radius().is_finite() || !(0..3).all(|i| sphere.center()[i].is_finite()) {
                return Err(Error::InvalidInput);
            }
            if sphere.radius() <= 0.0 {
                return Err(Error::Singular);
            }
            let cofactor = affine_cofactor(processor.transform(), budget, cancel)?;
            let [u, v] = if processor.orientation() {
                uv
            } else {
                [uv[1], uv[0]]
            };
            let (su, cu) = validated_sin_cos(u, budget, cancel)?;
            let (sv, cv) = validated_sin_cos(v, budget, cancel)?;
            let radial = [su * cv, su * sv, cu];
            budget.charge(32, cancel)?;
            let normal = std::array::from_fn(|i| {
                let component = (0..3)
                    .map(|j| cofactor[j][i] * radial[j])
                    .fold(ZERO, |a, b| a + b);
                if processor.orientation() {
                    component
                } else {
                    -component
                }
            });
            // The radial field has norm one identically, even when its interval
            // hull contains the origin. An invertible cofactor cannot kill it.
            finite(normal)
        }
        _ => {
            if let Surface::RevolutionSurface(processor) = surface {
                affine_cofactor(processor.transform(), budget, cancel)?;
            }
            if let Surface::ToroidalSurface(processor) = surface {
                affine_cofactor(processor.transform(), budget, cancel)?;
                let torus = processor.entity();
                let minor = Interval::new(torus.small_radius(), torus.small_radius())?;
                let major = Interval::new(torus.large_radius(), torus.large_radius())?;
                let v = uv[usize::from(processor.orientation())];
                let (_, cosine) = validated_sin_cos(v, budget, cancel)?;
                let factor = minor * (major + minor * cosine);
                if !factor.is_finite() {
                    return Err(Error::ConditioningFailure);
                }
                if factor.lower <= 0.0 {
                    return Err(Error::Unsupported);
                }
            }
            if !surface_box_smooth(surface, uv[0], uv[1], budget, cancel)? {
                return Err(Error::NonSmooth);
            }
            let jet = validated_surface_jet(surface, domain, budget, cancel)?;
            budget.charge(32, cancel)?;
            let normal = finite(cross(jet.du, jet.dv))?;
            if normal.into_iter().all(Interval::contains_zero) {
                return Err(Error::Singular);
            }
            Ok(normal)
        }
    }
}

#[cfg(test)]
mod geometric_normal_regressions {
    use super::*;
    use crate::{
        BsplineSurface, Curve, Invertible, KnotVector, Line, NurbsSurface, ParametricSurface3D,
        Plane, Point3, Processor, RevolutionSurface, Sphere, Torus, Vector3, Vector4,
    };
    use std::f64::consts::{FRAC_PI_2, PI, TAU};

    fn enclose(surface: &Surface, uv: [[f64; 2]; 2]) -> Result<[Interval; 3], Error> {
        validated_surface_geometric_normal(
            surface,
            uv,
            &mut ValidatedGeometryBudget::new(1_000_000),
            &AtomicBool::new(false),
        )
    }

    fn sphere(matrix: Matrix4, radius: f64, inverted: bool) -> Surface {
        let mut processor =
            Processor::with_transform(Sphere::new(Point3::new(5.0, -7.0, 11.0), radius), matrix);
        if inverted {
            processor.invert();
        }
        Surface::SphericalSurface(processor)
    }

    fn affine(sign: f64) -> Matrix4 {
        Matrix4::from_cols(
            Vector4::new(2.0, 0.0, 0.0, 0.0),
            Vector4::new(1.0, 3.0, 0.0, 0.0),
            Vector4::new(-0.5, 0.75, 4.0 * sign, 0.0),
            Vector4::new(17.0, -19.0, 23.0, 1.0),
        )
    }

    fn assert_contains(normal: [Interval; 3], expected: [f64; 3]) {
        for (bound, value) in normal.into_iter().zip(expected) {
            assert!(bound.is_finite());
            let roundoff = 2.0e-13 * value.abs().max(1.0);
            assert!(
                bound.lower - roundoff <= value && value <= bound.upper + roundoff,
                "{bound:?} does not enclose independent normal component {value}",
            );
        }
    }

    // Hand-solved inverse-transpose for the triangular affine matrix, multiplied
    // by its determinant. This does not reuse the implementation's cross products.
    fn expected_affine(u: f64, v: f64, sign: f64, inverted: bool) -> [f64; 3] {
        let n = [u.sin() * v.cos(), u.sin() * v.sin(), u.cos()];
        let result = [
            sign * 12.0 * n[0],
            sign * (8.0 * n[1] - 4.0 * n[0]),
            2.25 * n[0] - 1.5 * n[1] + 6.0 * n[2],
        ];
        result.map(|x| if inverted { -x } else { x })
    }

    #[test]
    fn spherical_poles_and_whole_patches_have_independent_affine_normals() {
        for sign in [1.0, -1.0] {
            for inverted in [false, true] {
                let source = sphere(affine(sign), 2.5, inverted);
                for chart in [
                    [[0.0, 0.0], [0.0, TAU]],
                    [[PI, PI], [0.0, TAU]],
                    [[0.0, 0.2], [0.3, 0.8]],
                    [[1.1, 1.4], [0.3, 0.7]],
                    [[PI - 0.2, PI], [0.3, 0.8]],
                ] {
                    let uv = if inverted {
                        [chart[1], chart[0]]
                    } else {
                        chart
                    };
                    let normal = enclose(&source, uv).expect("regular affine sphere patch");
                    for i in 0..=8 {
                        for j in 0..=8 {
                            let u = chart[0][0] + (chart[0][1] - chart[0][0]) * i as f64 / 8.0;
                            let v = chart[1][0] + (chart[1][1] - chart[1][0]) * j as f64 / 8.0;
                            assert_contains(normal, expected_affine(u, v, sign, inverted));
                        }
                    }
                }
                let chart = [0.6, 0.4];
                let uv = if inverted {
                    [chart[1], chart[0]]
                } else {
                    chart
                };
                let normal = enclose(&source, [[uv[0]; 2], [uv[1]; 2]]).unwrap();
                let expected = expected_affine(chart[0], chart[1], sign, inverted);
                assert_contains(normal, expected);
                assert!(normal.into_iter().all(|bound| bound.width() < 1.0e-10));
                let native = source.normal(uv[0], uv[1]);
                let dot = (0..3).map(|i| expected[i] * native[i]).sum::<f64>();
                assert!(dot > 0.0);
            }
        }
    }

    #[test]
    fn spherical_broad_hull_may_contain_origin_without_losing_regularity() {
        let source = sphere(affine(-1.0), 3.0, false);
        let normal = enclose(&source, [[0.0, PI], [0.0, TAU]]).unwrap();
        assert!(normal.into_iter().all(Interval::contains_zero));
        for (u, v) in [(0.0, 0.0), (PI, 0.0), (FRAC_PI_2, 0.0), (FRAC_PI_2, PI)] {
            assert_contains(normal, expected_affine(u, v, -1.0, false));
        }
    }

    #[test]
    fn spherical_ellipsoid_normal_uses_inverse_scale_not_position_direction() {
        let source = sphere(Matrix4::from_nonuniform_scale(2.0, 3.0, 4.0), 7.0, false);
        let normal = enclose(&source, [[0.6; 2], [0.4; 2]]).unwrap();
        assert_contains(
            normal,
            [
                12.0 * 0.6_f64.sin() * 0.4_f64.cos(),
                8.0 * 0.6_f64.sin() * 0.4_f64.sin(),
                6.0 * 0.6_f64.cos(),
            ],
        );
        for scale in [0.25, 4.0] {
            let source = sphere(Matrix4::from_scale(scale), 2.0, false);
            assert_contains(
                enclose(&source, [[0.0; 2], [0.0, TAU]]).unwrap(),
                [0.0, 0.0, scale * scale],
            );
        }
    }

    #[test]
    fn spherical_invalid_geometry_and_processor_transforms_refuse() {
        for radius in [0.0, -1.0] {
            assert_eq!(
                enclose(&sphere(affine(1.0), radius, false), [[0.0; 2]; 2]),
                Err(Error::Singular)
            );
        }
        for radius in [f64::NAN, f64::INFINITY] {
            assert_eq!(
                enclose(&sphere(affine(1.0), radius, false), [[0.0; 2]; 2]),
                Err(Error::InvalidInput)
            );
        }
        for bad in [f64::NAN, f64::INFINITY] {
            let mut matrix = affine(1.0);
            matrix[3][0] = bad;
            assert_eq!(
                enclose(&sphere(matrix, 1.0, false), [[0.0; 2]; 2]),
                Err(Error::InvalidInput)
            );
            let source = Surface::SphericalSurface(Processor::new(Sphere::new(
                Point3::new(bad, 0.0, 0.0),
                1.0,
            )));
            assert_eq!(enclose(&source, [[0.0; 2]; 2]), Err(Error::InvalidInput));
        }
        for matrix in [
            Matrix4::from_nonuniform_scale(1.0, 1.0, 0.0),
            Matrix4::from_cols(
                Vector4::new(1.0, 2.0, 3.0, 0.0),
                Vector4::new(2.0, 4.0, 6.0, 0.0),
                Vector4::new(0.0, 0.0, 1.0, 0.0),
                Vector4::new(0.0, 0.0, 0.0, 1.0),
            ),
        ] {
            assert_eq!(
                enclose(&sphere(matrix, 1.0, false), [[0.0; 2]; 2]),
                Err(Error::Singular)
            );
        }
        for row in [0, 3] {
            let mut matrix = affine(1.0);
            matrix[row][3] = 0.5;
            assert_eq!(
                enclose(&sphere(matrix, 1.0, false), [[0.0; 2]; 2]),
                Err(Error::Unsupported)
            );
        }
        let source = sphere(affine(1.0), 1.0, false);
        assert_eq!(
            enclose(&source, [[1.0, 0.0], [0.0, 1.0]]),
            Err(Error::InvalidInput)
        );
        assert_eq!(
            enclose(&source, [[0.0, f64::INFINITY], [0.0, 1.0]]),
            Err(Error::InvalidInput)
        );
    }

    #[test]
    fn ordinary_whole_chart_normal_preserves_orientation_and_refuses_collapse() {
        let mut source = Surface::Plane(Plane::new(
            Point3::new(5.0, -7.0, 11.0),
            Point3::new(7.0, -7.0, 11.0),
            Point3::new(6.0, -4.0, 11.0),
        ));
        for inverted in [false, true] {
            if inverted {
                source.invert();
            }
            let normal = enclose(&source, [[-10.0, 10.0], [-20.0, 20.0]]).unwrap();
            assert_contains(normal, [0.0, 0.0, if inverted { -6.0 } else { 6.0 }]);
            assert!(!normal[2].contains_zero());
        }
        let collapsed = Surface::Plane(Plane::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        ));
        assert_eq!(enclose(&collapsed, [[0.0, 1.0]; 2]), Err(Error::Singular));
    }

    #[test]
    fn ordinary_crease_union_is_not_continuous_normal_authority() {
        let surface = Surface::BsplineSurface(BsplineSurface::new(
            (
                KnotVector::from(vec![0.0, 0.0, 1.0, 2.0, 2.0]),
                KnotVector::from(vec![0.0, 0.0, 1.0, 1.0]),
            ),
            vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 1.0)],
                vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 0.0, 1.0)],
                vec![Point3::new(2.0, 0.25, 0.0), Point3::new(2.0, 0.25, 1.0)],
            ],
        ));
        let left = enclose(&surface, [[0.49, 0.51], [0.49, 0.51]])
            .expect("regular whole patch on the first side of the crease");
        let right = enclose(&surface, [[1.49, 1.51], [0.49, 0.51]])
            .expect("regular whole patch on the second side of the crease");
        assert_contains(left, [0.0, -1.0, 0.0]);
        assert_contains(right, [0.25, -1.0, 0.0]);
        assert!(left[1].upper < 0.0 && right[1].upper < 0.0);
        assert_eq!(
            enclose(&surface, [[0.5, 1.5], [0.0, 1.0]]),
            Err(Error::NonSmooth)
        );
    }

    #[test]
    fn tiny_regular_nurbs_and_torus_source_orientation_do_not_use_heuristics() {
        let knots = KnotVector::from(vec![0.0, 0.0, 1.0, 1.0]);
        for scale in [1.0, 1.0e-10] {
            let source = Surface::NurbsSurface(NurbsSurface::new(BsplineSurface::new(
                (knots.clone(), knots.clone()),
                vec![
                    vec![
                        Vector4::new(0.0, 0.0, 0.0, 1.0),
                        Vector4::new(0.0, scale, 0.0, 1.0),
                    ],
                    vec![
                        Vector4::new(scale, 0.0, 0.0, 1.0),
                        Vector4::new(scale, scale, 0.0, 1.0),
                    ],
                ],
            )));
            let normal = enclose(&source, [[0.0, 1.0]; 2]).unwrap();
            assert_contains(normal, [0.0, 0.0, scale * scale]);
            assert!(normal[2].lower > 0.0);
        }
        let ring = Surface::ToroidalSurface(Processor::with_transform(
            Torus::new(Point3::new(0.0, 0.0, 0.0), 3.0, 1.0),
            affine(-1.0),
        ));
        let normal = enclose(&ring, [[0.0; 2], [0.0; 2]]).unwrap();
        assert_contains(normal, [-48.0, 16.0, 9.0]);
        let spindle = Surface::ToroidalSurface(Processor::new(Torus::new(
            Point3::new(0.0, 0.0, 0.0),
            0.5,
            1.0,
        )));
        assert_eq!(
            enclose(&spindle, [[0.0; 2], [PI; 2]]),
            Err(Error::Unsupported)
        );
    }

    #[test]
    fn regular_revolution_source_normals_keep_affine_pose_and_inverted_chart() {
        for sign in [1.0, -1.0] {
            for inverted in [false, true] {
                let mut processor = Processor::with_transform(
                    RevolutionSurface::by_revolution(
                        Curve::Line(Line(Point3::new(2.0, 0.0, 0.0), Point3::new(2.0, 0.0, 1.0))),
                        Point3::new(0.0, 0.0, 0.0),
                        Vector3::unit_z(),
                    ),
                    affine(sign),
                );
                if inverted {
                    processor.invert();
                }
                let source = Surface::RevolutionSurface(processor);
                let chart = [[0.0, 1.0], [0.0, 0.3]];
                let domain = if inverted {
                    [chart[1], chart[0]]
                } else {
                    chart
                };
                let normal = enclose(&source, domain).unwrap();
                for v in [0.0_f64, 0.15, 0.3] {
                    // Increasing axial profile crossed with angular tangent is
                    // inward, unlike the conventional outward cylinder normal.
                    let expected = [
                        -24.0 * sign * v.cos(),
                        sign * (8.0 * v.cos() - 16.0 * v.sin()),
                        -4.5 * v.cos() + 3.0 * v.sin(),
                    ]
                    .map(|x| if inverted { -x } else { x });
                    assert_contains(normal, expected);
                }
            }
        }
        let cylinder = |matrix| {
            Surface::RevolutionSurface(Processor::with_transform(
                RevolutionSurface::by_revolution(
                    Curve::Line(Line(Point3::new(2.0, 0.0, 0.0), Point3::new(2.0, 0.0, 1.0))),
                    Point3::new(0.0, 0.0, 0.0),
                    Vector3::unit_z(),
                ),
                matrix,
            ))
        };
        let mut projective = affine(1.0);
        projective[0][3] = 0.5;
        assert_eq!(
            enclose(&cylinder(projective), [[0.0, 0.1]; 2]),
            Err(Error::Unsupported)
        );
        assert_eq!(
            enclose(
                &cylinder(Matrix4::from_nonuniform_scale(1.0, 1.0, 0.0)),
                [[0.0, 0.1]; 2]
            ),
            Err(Error::Singular),
        );
    }

    #[test]
    fn normal_cancellation_and_shared_work_exhaustion_never_return_authority() {
        let source = sphere(affine(1.0), 1.0, false);
        let domain = [[0.0, 0.1], [0.0, TAU]];
        for allowance in [0, 1, 65, 106] {
            let mut budget = ValidatedGeometryBudget::new(allowance);
            assert_eq!(
                validated_surface_geometric_normal(
                    &source,
                    domain,
                    &mut budget,
                    &AtomicBool::new(false)
                ),
                Err(Error::ResourceLimit),
            );
            assert_eq!(budget.remaining(), 0);
        }
        assert_eq!(
            validated_surface_geometric_normal(
                &source,
                domain,
                &mut ValidatedGeometryBudget::new(1000),
                &AtomicBool::new(true)
            ),
            Err(Error::Cancelled),
        );
        let mut budget = ValidatedGeometryBudget::new(1_000_000);
        validated_surface_geometric_normal(&source, domain, &mut budget, &AtomicBool::new(false))
            .unwrap();
        let mut shared = ValidatedGeometryBudget::new(budget.consumed());
        validated_surface_geometric_normal(&source, domain, &mut shared, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(
            validated_surface_geometric_normal(
                &source,
                domain,
                &mut shared,
                &AtomicBool::new(false)
            ),
            Err(Error::ResourceLimit),
        );
    }
}
