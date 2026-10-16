//! Whole original curve support in an immutable convex planar operand face.
//! Only literal affine boundary coefficients and actual shared vertex incidence
//! describe the material trim. Generated current-boundary slits are not authority.

use super::chart_boundary::{curve_control, curve_data, plane_value};
use super::*;

#[derive(Clone, Copy)]
pub(super) struct Enclosure {
    pub(super) lo: f64,
    pub(super) hi: f64,
}

impl Enclosure {
    pub(super) fn point(value: f64) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }
    pub(super) fn add(self, other: Self) -> Self {
        Self {
            lo: (self.lo + other.lo).next_down(),
            hi: (self.hi + other.hi).next_up(),
        }
    }
    fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
    pub(super) fn sub(self, other: Self) -> Self { self.add(other.neg()) }
    pub(super) fn mul(self, other: Self) -> Self {
        let products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ];
        if products.into_iter().any(f64::is_nan) {
            return Self {
                lo: f64::NAN,
                hi: f64::NAN,
            };
        }
        Self {
            lo: products
                .into_iter()
                .fold(f64::INFINITY, f64::min)
                .next_down(),
            hi: products
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max)
                .next_up(),
        }
    }
    pub(super) fn div(self, value: f64) -> Self {
        let reciprocal = Self {
            lo: (1.0 / value).next_down(),
            hi: (1.0 / value).next_up(),
        };
        self.mul(reciprocal)
    }
    pub(super) fn quotient(self, denominator: Self) -> Option<Self> {
        if !denominator.finite() || (denominator.lo <= 0.0 && denominator.hi >= 0.0) {
            return None;
        }
        Some(self.mul(Self {
            lo: (1.0 / denominator.hi).next_down(),
            hi: (1.0 / denominator.lo).next_up(),
        }))
    }
    pub(super) fn finite(self) -> bool {
        self.lo.is_finite() && self.hi.is_finite() && self.lo <= self.hi
    }
    pub(super) fn magnitude(self) -> f64 { self.lo.abs().max(self.hi.abs()) }
}

pub(super) fn sum(values: impl IntoIterator<Item = Enclosure>) -> Enclosure {
    values
        .into_iter()
        .fold(Enclosure::point(0.0), Enclosure::add)
}

fn source_control(source: OriginalCurveCoefficients<'_>, index: usize) -> Option<Vector4> {
    match source {
        OriginalCurveCoefficients::Line(line) => match index {
            0 => Some(line.0.to_homogeneous()),
            1 => Some(line.1.to_homogeneous()),
            _ => None,
        },
        _ => curve_control(source, index),
    }
}

pub(super) fn contains_curve<C>(
    source: OriginalCurveCoefficients<'_>,
    boundaries: &[Wire<Point3, C>],
    plane: &OriginalPlane,
    work: Option<&BooleanWork<'_>>,
) -> std::result::Result<bool, AttachmentFailure>
where
    C: ParametricCurve3D + BoundedCurve + TryIntoHomogeneousBsplineCurve,
{
    let [boundary] = boundaries else {
        return Ok(false);
    };
    let count = match source {
        OriginalCurveCoefficients::Line(_) => 2,
        _ => {
            let Some((_, count)) = curve_data(source) else {
                return Ok(false);
            };
            count
        }
    };
    if boundary.len() < 3 || !boundary.is_closed() || !boundary.is_simple() {
        return Ok(false);
    }
    let mut weight_sign = None;
    for index in 0..count {
        if let Some(work) = work {
            work.check()?;
        }
        let Some(control) = source_control(source, index) else {
            return Ok(false);
        };
        if ![control.x, control.y, control.z, control.w]
            .into_iter()
            .all(f64::is_finite)
            || control.w == 0.0
            || weight_sign.is_some_and(|positive| positive != control.w.is_sign_positive())
        {
            return Ok(false);
        }
        weight_sign = Some(control.w.is_sign_positive());
    }
    for edge in boundary {
        if let Some(work) = work {
            work.check()?;
        }
        let affine = edge.with_curve(|curve| {
            let (controls, domain) = segment_coefficients(curve.original_curve_coefficients()?)?;
            (curve.parameter_range() == domain
                && coefficient_endpoint(controls[0], edge.absolute_front().point())?
                && coefficient_endpoint(controls[1], edge.absolute_back().point())?
                && plane_value(plane, controls[0]) == Some(0.0)
                && plane_value(plane, controls[1]) == Some(0.0))
            .then_some(())
        });
        if affine.is_none() {
            return Ok(false);
        }
        let origin = edge.front().point().to_vec();
        let Some(direction) = difference(edge.back().point().to_vec(), origin) else {
            return Ok(false);
        };
        let Some(normal) = cross(plane.normal, direction) else {
            return Ok(false);
        };
        if normal == Vector3::zero() {
            return Ok(false);
        }
        let mut positive = None;
        for vertex in boundary.iter().map(Edge::front) {
            if let Some(work) = work {
                work.check()?;
            }
            if vertex == edge.front() || vertex == edge.back() {
                continue;
            }
            let point = vertex.point();
            let side = sum((0..3).map(|axis| {
                Enclosure::point(point[axis])
                    .sub(Enclosure::point(origin[axis]))
                    .mul(Enclosure::point(normal[axis]))
            }));
            if !side.finite() {
                return Ok(false);
            }
            let sign = if side.lo > 0.0 {
                true
            } else if side.hi < 0.0 {
                false
            } else if projected_difference(normal, point.to_vec(), origin) == Some(0.0) {
                continue;
            } else {
                return Ok(false);
            };
            if positive.is_some_and(|previous| previous != sign) {
                return Ok(false);
            }
            positive = Some(sign);
        }
        let Some(positive) = positive else {
            return Ok(false);
        };
        for index in 0..count {
            if let Some(work) = work {
                work.check()?;
            }
            let Some(control) = source_control(source, index) else {
                return Ok(false);
            };
            let side = sum((0..3).map(|axis| {
                Enclosure::point(control[axis])
                    .sub(Enclosure::point(origin[axis]).mul(Enclosure::point(control.w)))
                    .mul(Enclosure::point(normal[axis]))
            }));
            if !side.finite() {
                return Ok(false);
            }
            let direction = positive == control.w.is_sign_positive();
            if (direction && side.lo < 0.0) || (!direction && side.hi > 0.0) {
                let exact = difference(
                    control.truncate(),
                    Vector3::new(
                        match exact_mul(origin.x, control.w) {
                            Some(value) => value,
                            None => return Ok(false),
                        },
                        match exact_mul(origin.y, control.w) {
                            Some(value) => value,
                            None => return Ok(false),
                        },
                        match exact_mul(origin.z, control.w) {
                            Some(value) => value,
                            None => return Ok(false),
                        },
                    ),
                )
                .and_then(|delta| exact_dot(normal, delta));
                if exact != Some(0.0) {
                    return Ok(false);
                }
            }
        }
    }
    // Positive rational weights put the entire curve in its control hull.
    // Every hull point satisfies every proved original convex trim halfspace.
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use monstertruck_modeling as modeling;

    #[test]
    fn original_convex_trim_support_covers_the_whole_original_control_hull() {
        let vertices = Vertex::from_points([
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(1.0, -1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(-1.0, 1.0, 0.0),
        ]);
        let boundary: modeling::Wire = (0..4)
            .map(|index| modeling::builder::line(&vertices[index], &vertices[(index + 1) % 4]))
            .collect();
        let curve = |height| {
            modeling::Curve::BsplineCurve(BsplineCurve::new(
                KnotVector::bezier_knot(2),
                vec![
                    Point3::new(-0.5, 0.0, 0.0),
                    Point3::new(0.0, height, 0.0),
                    Point3::new(0.5, 0.0, 0.0),
                ],
            ))
        };
        let contained = curve(0.5);
        let escaping = curve(4.0);
        let plane = OriginalPlane::new(&Plane::xy()).unwrap();
        assert!(
            contains_curve(
                contained.original_curve_coefficients().unwrap(),
                std::slice::from_ref(&boundary),
                &plane,
                None,
            )
            .unwrap()
        );
        assert!(
            !contains_curve(
                escaping.original_curve_coefficients().unwrap(),
                std::slice::from_ref(&boundary),
                &plane,
                None,
            )
            .unwrap()
        );
        assert!(
            !contains_curve(
                contained.original_curve_coefficients().unwrap(),
                &[boundary.clone(), boundary.inverse()],
                &plane,
                None,
            )
            .unwrap()
        );
        let negative = modeling::Curve::NurbsCurve(NurbsCurve::new(BsplineCurve::new(
            KnotVector::bezier_knot(2),
            vec![
                Vector4::new(1.0, 0.0, 0.0, -2.0),
                Vector4::new(0.0, -1.0, 0.0, -2.0),
                Vector4::new(-1.0, 0.0, 0.0, -2.0),
            ],
        )));
        assert!(
            contains_curve(
                negative.original_curve_coefficients().unwrap(),
                std::slice::from_ref(&boundary),
                &plane,
                None,
            )
            .unwrap()
        );
    }
}
