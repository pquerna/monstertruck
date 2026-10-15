use super::*;

fn dyadic(value: f64) -> (bool, u64, i32) {
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let mantissa = bits & ((1_u64 << 52) - 1);
    if exponent == 0 {
        (bits >> 63 != 0, mantissa, -1074)
    } else {
        (bits >> 63 != 0, mantissa | (1_u64 << 52), exponent - 1075)
    }
}

fn exact_product(a: f64, b: f64) -> (bool, u128, i32) {
    let (sa, ma, ea) = dyadic(a);
    let (sb, mb, eb) = dyadic(b);
    let product = u128::from(ma) * u128::from(mb);
    if product == 0 {
        (false, 0, 0)
    } else {
        let shift = product.trailing_zeros();
        (sa != sb, product >> shift, ea + eb + shift as i32)
    }
}

fn homogeneous_endpoint(curve: &Curve2D, back: bool) -> Option<[f64; 3]> {
    match curve {
        Curve2D::Line(line) => {
            let point = if back { line.1 } else { line.0 };
            Some([point.x, point.y, 1.0])
        }
        Curve2D::Polyline(curve) => {
            let point = if back {
                curve.as_slice().last()?
            } else {
                curve.as_slice().first()?
            };
            Some([point.x, point.y, 1.0])
        }
        Curve2D::BsplineCurve(curve) => {
            let points = curve.control_points();
            let knots = curve.knot_vector();
            if points.is_empty() || knots.len() <= points.len() {
                return None;
            }
            let degree = knots.len().checked_sub(points.len() + 1)?;
            if degree >= points.len() {
                return None;
            }
            let endpoint = if back {
                knots[points.len()]
            } else {
                knots[degree]
            };
            let clamped = if back {
                knots
                    .iter()
                    .rev()
                    .take(degree + 1)
                    .all(|value| *value == endpoint)
            } else {
                knots
                    .iter()
                    .take(degree + 1)
                    .all(|value| *value == endpoint)
            };
            if !clamped {
                return None;
            }
            let point = if back {
                points.last()?
            } else {
                points.first()?
            };
            Some([point.x, point.y, 1.0])
        }
        Curve2D::NurbsCurve(curve) => {
            let points = curve.control_points();
            let knots = curve.knot_vector();
            if points.is_empty() || knots.len() <= points.len() {
                return None;
            }
            let degree = knots.len().checked_sub(points.len() + 1)?;
            if degree >= points.len() {
                return None;
            }
            let endpoint = if back {
                knots[points.len()]
            } else {
                knots[degree]
            };
            let clamped = if back {
                knots
                    .iter()
                    .rev()
                    .take(degree + 1)
                    .all(|value| *value == endpoint)
            } else {
                knots
                    .iter()
                    .take(degree + 1)
                    .all(|value| *value == endpoint)
            };
            if !clamped {
                return None;
            }
            let point = if back {
                points.last()?
            } else {
                points.first()?
            };
            Some([point.x, point.y, point.z])
        }
        Curve2D::Conic(_) | Curve2D::TrimmedCurve(_) => None,
    }
}

fn same_conic_endpoint(a: &Curve2D, aback: bool, b: &Curve2D, bback: bool) -> bool {
    macro_rules! same {
        ($a:expr, $b:expr) => {{
            let a = $a;
            let b = $b;
            let arange = a.entity().range();
            let brange = b.entity().range();
            let ta = if aback == a.orientation() {
                arange.1
            } else {
                arange.0
            };
            let tb = if bback == b.orientation() {
                brange.1
            } else {
                brange.0
            };
            a.transform() == b.transform() && ta == tb
        }};
    }
    match (a, b) {
        (Curve2D::Conic(Conic2D::Ellipse(a)), Curve2D::Conic(Conic2D::Ellipse(b))) => same!(a, b),
        (Curve2D::Conic(Conic2D::Hyperbola(a)), Curve2D::Conic(Conic2D::Hyperbola(b))) => {
            same!(a, b)
        }
        (Curve2D::Conic(Conic2D::Parabola(a)), Curve2D::Conic(Conic2D::Parabola(b))) => same!(a, b),
        _ => false,
    }
}

pub(super) fn endpoints_equal(a: &Curve2D, aback: bool, b: &Curve2D, bback: bool) -> bool {
    match (
        homogeneous_endpoint(a, aback),
        homogeneous_endpoint(b, bback),
    ) {
        (Some(a), Some(b))
            if a.into_iter().chain(b).all(f64::is_finite) && a[2] != 0.0 && b[2] != 0.0 =>
        {
            exact_product(a[0], b[2]) == exact_product(b[0], a[2])
                && exact_product(a[1], b[2]) == exact_product(b[1], a[2])
        }
        _ => same_conic_endpoint(a, aback, b, bback),
    }
}
