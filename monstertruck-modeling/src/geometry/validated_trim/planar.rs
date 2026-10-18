use super::*;
use crate::{
    BsplineCurve, Invertible, KnotVector, Matrix3, Matrix4, NurbsCurve, Plane, Processor,
    TrimmedCurve,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Dyadic {
    coefficient: i128,
    exponent: i32,
}

impl Dyadic {
    pub(super) fn scalar(value: f64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }
        let bits = value.to_bits();
        let raw_exponent = ((bits >> 52) & 0x7ff) as i32;
        let mantissa = bits & ((1_u64 << 52) - 1);
        let (mantissa, exponent) = if raw_exponent == 0 {
            (mantissa, -1074)
        } else {
            (mantissa | (1_u64 << 52), raw_exponent - 1075)
        };
        let coefficient = if bits >> 63 == 0 {
            i128::from(mantissa)
        } else {
            -i128::from(mantissa)
        };
        Some(
            Self {
                coefficient,
                exponent,
            }
            .normalized(),
        )
    }

    fn normalized(self) -> Self {
        if self.coefficient == 0 {
            Self {
                coefficient: 0,
                exponent: 0,
            }
        } else {
            let shift = self.coefficient.unsigned_abs().trailing_zeros();
            Self {
                coefficient: self.coefficient >> shift,
                exponent: self.exponent + shift as i32,
            }
        }
    }

    pub(super) fn multiply(self, rhs: Self) -> Option<Self> {
        Some(
            Self {
                coefficient: self.coefficient.checked_mul(rhs.coefficient)?,
                exponent: self.exponent.checked_add(rhs.exponent)?,
            }
            .normalized(),
        )
    }

    pub(super) fn add(self, rhs: Self) -> Option<Self> {
        if self.coefficient == 0 {
            return Some(rhs);
        }
        if rhs.coefficient == 0 {
            return Some(self);
        }
        let exponent = self.exponent.min(rhs.exponent);
        let left_shift = u32::try_from(self.exponent - exponent).ok()?;
        let right_shift = u32::try_from(rhs.exponent - exponent).ok()?;
        if left_shift >= 127 || right_shift >= 127 {
            return None;
        }
        let left = self
            .coefficient
            .checked_mul(1_i128.checked_shl(left_shift)?)?;
        let right = rhs
            .coefficient
            .checked_mul(1_i128.checked_shl(right_shift)?)?;
        Some(
            Self {
                coefficient: left.checked_add(right)?,
                exponent,
            }
            .normalized(),
        )
    }

    pub(super) fn approximate_quotient(self, denominator: Self) -> f64 {
        (self.coefficient as f64 / denominator.coefficient as f64)
            * 2.0_f64.powi(self.exponent - denominator.exponent)
    }

    /// Returns a quotient only when its exact value is finite binary64.
    pub(super) fn exact_quotient(self, denominator: Self) -> Option<f64> {
        if denominator.coefficient == 0
            || self.coefficient.checked_rem(denominator.coefficient)? != 0
        {
            return None;
        }
        Self {
            coefficient: self.coefficient.checked_div(denominator.coefficient)?,
            exponent: self.exponent.checked_sub(denominator.exponent)?,
        }
        .normalized()
        .exact_float()
    }

    pub(super) fn compare(self, other: Self) -> std::cmp::Ordering {
        let sign = self.coefficient.signum().cmp(&other.coefficient.signum());
        if !sign.is_eq() {
            return sign;
        }
        if self.coefficient == 0 {
            return std::cmp::Ordering::Equal;
        }
        let order = self.leading_exponent().cmp(&other.leading_exponent());
        let order = if order.is_eq() {
            let exponent = self.exponent.min(other.exponent);
            let left = self.coefficient.unsigned_abs() << (self.exponent - exponent) as u32;
            let right = other.coefficient.unsigned_abs() << (other.exponent - exponent) as u32;
            left.cmp(&right)
        } else {
            order
        };
        if self.coefficient < 0 {
            order.reverse()
        } else {
            order
        }
    }

    pub(super) fn negated(self) -> Option<Self> {
        Some(Self {
            coefficient: self.coefficient.checked_neg()?,
            exponent: self.exponent,
        })
    }

    pub(super) fn unsigned_coefficient(self) -> u128 { self.coefficient.unsigned_abs() }

    pub(super) fn divided_coefficient(self, divisor: u128) -> Option<Self> {
        let divisor = i128::try_from(divisor).ok()?;
        if divisor == 0 || self.coefficient % divisor != 0 {
            return None;
        }
        Some(Self {
            coefficient: self.coefficient / divisor,
            exponent: self.exponent,
        })
    }

    pub(super) fn leading_exponent(self) -> i32 {
        if self.coefficient == 0 {
            i32::MIN
        } else {
            self.exponent + 127 - self.coefficient.unsigned_abs().leading_zeros() as i32
        }
    }

    pub(super) fn shifted(self, exponent: i32) -> Option<Self> {
        Some(Self {
            coefficient: self.coefficient,
            exponent: self.exponent.checked_add(exponent)?,
        })
    }

    pub(super) fn exact_float(self) -> Option<f64> {
        if self.coefficient == 0 {
            return Some(0.0);
        }
        let magnitude = self.coefficient.unsigned_abs();
        let width = 128 - magnitude.leading_zeros();
        if width > 53 || self.exponent < -1074 {
            return None;
        }
        let top = self.exponent.checked_add(width as i32 - 1)?;
        if top > 1023 {
            return None;
        }
        let sign = if self.coefficient < 0 { 1_u64 << 63 } else { 0 };
        let bits = if top >= -1022 {
            let significand = (magnitude as u64) << (53 - width);
            (((top + 1023) as u64) << 52) | (significand & ((1_u64 << 52) - 1))
        } else {
            (magnitude as u64) << u32::try_from(self.exponent + 1074).ok()?
        };
        Some(f64::from_bits(sign | bits))
    }

    /// Retains an inexact scalar between adjacent floats, never as one rounded knot.
    pub(super) fn float_bounds(self) -> Option<[f64; 2]> {
        if let Some(value) = self.exact_float() {
            return Some([value; 2]);
        }
        let width = 128 - self.coefficient.unsigned_abs().leading_zeros();
        let shift = width.checked_sub(53)?;
        if shift == 0 {
            return None;
        }
        let magnitude = (self.coefficient.unsigned_abs() >> shift) as i128;
        let toward_zero = Self {
            coefficient: if self.coefficient < 0 {
                -magnitude
            } else {
                magnitude
            },
            exponent: self.exponent.checked_add(shift as i32)?,
        }
        .exact_float()?;
        let bounds = if self.coefficient < 0 {
            [toward_zero.next_down(), toward_zero]
        } else {
            [toward_zero, toward_zero.next_up()]
        };
        bounds.into_iter().all(f64::is_finite).then_some(bounds)
    }
}

pub(super) fn plane_coefficients(plane: &Plane) -> Option<[[Dyadic; 3]; 3]> {
    let origin = plane.origin();
    let zero = Dyadic::scalar(0.0)?;
    let mut result = [[zero; 3]; 3];
    let points = plane.defining_points();
    for axis in 0..3 {
        result[0][axis] = Dyadic::scalar(origin[axis])?;
        if let Some(axes) = plane.stored_axes() {
            result[1][axis] = Dyadic::scalar(axes[0][axis])?;
            result[2][axis] = Dyadic::scalar(axes[1][axis])?;
        } else {
            result[1][axis] =
                Dyadic::scalar(points[1][axis])?.add(Dyadic::scalar(-origin[axis])?)?;
            result[2][axis] =
                Dyadic::scalar(points[2][axis])?.add(Dyadic::scalar(-origin[axis])?)?;
        }
    }
    Some(result)
}

/// Extracts a planar UV curve only after an exact whole-coefficient carrier proof.
///
/// Complete original homogeneous controls are projected with an exactly checked
/// plane identity, retaining original parameters and knots. Homogeneous UV storage
/// avoids rounding nondyadic coordinate quotients. Plane-embedded analytic pcurves
/// are recharted with exactly representable affine `Point2` coefficients,
/// retaining their original conic pose factors. Homogeneous common scaling is
/// reserved for rational controls; projective or nondyadic `Point2` operators
/// refuse this exact lane rather than claim that an undivided or rounded matrix
/// is equivalent. Selected generic trims retain every basis, interval and
/// independent sense; spatial poses participate in the complete carrier proof.
/// Checked dyadic overflow, unrepresentable homogeneous factors, excessive nesting,
/// implicit intersections and unrelated carriers return typed refusals.
///
/// The returned UV coefficients are new necessary chart storage; no 3D curve,
/// carrier or BREP is copied. The caller can store this result in its existing
/// [`TrimmedFace`] and use [`prepare_face_parameter_domain`].
pub fn certified_planar_parameter_curve(
    curve: &Curve,
    plane: &Plane,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Curve2D, TrimDomainError> {
    let identity = Matrix4::from_scale(1.0);
    let mut poses = [&identity; 32];
    project(curve, plane, &mut poses, 0, budget, cancel)
}

fn project<'a>(
    curve: &'a Curve,
    plane: &Plane,
    poses: &mut [&'a Matrix4; 32],
    depth: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Curve2D, TrimDomainError> {
    budget.charge(1, cancel)?;
    if depth >= poses.len() {
        return Err(TrimDomainError::ResourceLimit);
    }
    match curve {
        Curve::TrimmedCurve(curve) => {
            poses[depth] = curve.transform();
            let basis = project(
                curve.entity().curve().as_ref(),
                plane,
                poses,
                depth + 1,
                budget,
                cancel,
            )?;
            let mut selected = Processor::with_transform(
                TrimmedCurve::new(Box::new(basis), curve.entity().range()),
                Matrix3::from_scale(1.0),
            );
            if !curve.orientation() {
                selected.invert();
            }
            Ok(Curve2D::TrimmedCurve(selected))
        }
        Curve::Line(line) => {
            let controls = [
                [line.0.x, line.0.y, line.0.z, 1.0],
                [line.1.x, line.1.y, line.1.z, 1.0],
            ];
            let points =
                rechart::controls(controls.into_iter(), plane, &poses[..depth], budget, cancel)?;
            Ok(Curve2D::NurbsCurve(NurbsCurve::new(BsplineCurve::new(
                KnotVector::from(vec![0.0, 0.0, 1.0, 1.0]),
                points,
            ))))
        }
        Curve::BsplineCurve(curve) => {
            budget.charge(curve.knot_vector().len(), cancel)?;
            let controls = curve
                .control_points()
                .iter()
                .map(|point| [point.x, point.y, point.z, 1.0]);
            let points = rechart::controls(controls, plane, &poses[..depth], budget, cancel)?;
            Ok(Curve2D::NurbsCurve(NurbsCurve::new(BsplineCurve::new(
                curve.knot_vector().clone(),
                points,
            ))))
        }
        Curve::NurbsCurve(curve) => {
            budget.charge(curve.knot_vector().len(), cancel)?;
            let controls = curve
                .control_points()
                .iter()
                .map(|point| [point.x, point.y, point.z, point.w]);
            let points = rechart::controls(controls, plane, &poses[..depth], budget, cancel)?;
            Ok(Curve2D::NurbsCurve(NurbsCurve::new(BsplineCurve::new(
                curve.knot_vector().clone(),
                points,
            ))))
        }
        Curve::ParameterCurve(curve) => {
            let Surface::Plane(source) = curve.surface().as_ref() else {
                return Err(ValidatedGeometryError::Unsupported.into());
            };
            if depth == 0 && source == plane {
                let count = curve_storage_units(curve.curve(), budget, cancel)?;
                budget.charge(count, cancel)?;
                Ok(curve.curve().clone())
            } else {
                rechart::parameter_curve(
                    curve.curve(),
                    source,
                    plane,
                    &poses[..depth],
                    budget,
                    cancel,
                )
            }
        }
        Curve::IntersectionCurve(_) => Err(ValidatedGeometryError::Unsupported.into()),
    }
}
