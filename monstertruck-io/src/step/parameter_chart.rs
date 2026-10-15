//! Bounded orientation charts shared by STEP readers and writers.
//!
//! These maps report coordinate conversion and its arithmetic enclosure. They
//! do not, by themselves, certify geometric equivalence or authorize UV trims.

/// A finite closed parameter enclosure.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ParameterEnclosure {
    /// Lower included endpoint.
    pub lower: f64,
    /// Upper included endpoint.
    pub upper: f64,
}

impl ParameterEnclosure {
    /// Constructs a finite ordered enclosure.
    pub fn new(lower: f64, upper: f64) -> Option<Self> {
        (lower.is_finite() && upper.is_finite() && lower <= upper).then_some(Self { lower, upper })
    }

    /// Constructs an enclosure of one finite datum.
    pub fn point(value: f64) -> Option<Self> { Self::new(value, value) }

    /// Whether this enclosure contains a finite value.
    pub fn contains(self, value: f64) -> bool {
        value.is_finite() && self.lower <= value && value <= self.upper
    }

    /// Reflects a complete enclosure through zero.
    pub fn reflected(self) -> Self {
        Self {
            lower: -self.upper,
            upper: -self.lower,
        }
    }

    /// Encloses a sum, retaining exact additions without unnecessary widening.
    pub fn sum(self, other: Self) -> Option<Self> {
        Self::new(self.lower, self.upper)?;
        Self::new(other.lower, other.upper)?;
        let lower = enclosed_sum(self.lower, other.lower)?.lower;
        let upper = enclosed_sum(self.upper, other.upper)?.upper;
        Self::new(lower, upper)
    }
}

// TwoSum's residual records the rounding direction without widening exact sums.
// Nonfinite compensation intermediates refuse even when the rounded sum is finite.
fn enclosed_sum(left: f64, right: f64) -> Option<ParameterEnclosure> {
    let sum = left + right;
    if !sum.is_finite() {
        return None;
    }
    let displaced = sum - left;
    let recovered = sum - displaced;
    let left_error = left - recovered;
    let right_error = right - displaced;
    let residual = left_error + right_error;
    if [displaced, recovered, left_error, right_error, residual]
        .iter()
        .any(|value| !value.is_finite())
    {
        return None;
    }
    ParameterEnclosure::new(
        if residual < 0.0 { sum.next_down() } else { sum },
        if residual > 0.0 { sum.next_up() } else { sum },
    )
}

/// Encloses an explicit integer turn shift without treating binary period as exact.
pub(crate) fn periodic_parameter_shift(turns: f64, period: f64) -> Option<ParameterChart> {
    if !period.is_finite()
        || period <= 0.0
        || !turns.is_finite()
        || turns.fract() != 0.0
        || turns.abs() > 9_007_199_254_740_992.0
    {
        return None;
    }
    if turns == 0.0 {
        return Some(ParameterChart::identity());
    }
    let nominal = turns * period;
    let first = turns * period.next_down();
    let second = turns * period.next_up();
    let enclosure =
        ParameterEnclosure::new(first.min(second).next_down(), first.max(second).next_up())?;
    ParameterChart::translation(nominal, enclosure)
}

/// An affine orientation chart with an enclosed translation coefficient.
///
/// The scale is exactly `1` or `-1`: these are the native/STEP orientation
/// changes supported here, not an arbitrary expression or unit-conversion DSL.
/// Active source and target domains are retained by the owning curve/surface
/// metadata, independently of this composable coefficient map.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ParameterChart {
    /// Exact orientation scale, either `1` or `-1`.
    pub scale: f64,
    /// Nominal translation used for emitted numeric values.
    pub offset: f64,
    /// Translation enclosure, including supplied period and arithmetic error.
    pub offset_enclosure: ParameterEnclosure,
}

impl ParameterChart {
    /// Constructs a checked finite orientation chart.
    pub fn new(scale: f64, offset: f64, offset_enclosure: ParameterEnclosure) -> Option<Self> {
        let enclosure = ParameterEnclosure::new(offset_enclosure.lower, offset_enclosure.upper)?;
        ((scale == 1.0 || scale == -1.0) && enclosure.contains(offset)).then_some(Self {
            scale,
            offset,
            offset_enclosure: enclosure,
        })
    }

    /// The exact identity chart.
    pub const fn identity() -> Self {
        Self {
            scale: 1.0,
            offset: 0.0,
            offset_enclosure: ParameterEnclosure {
                lower: 0.0,
                upper: 0.0,
            },
        }
    }

    /// Constructs a checked translation with its coefficient enclosure.
    pub fn translation(offset: f64, enclosure: ParameterEnclosure) -> Option<Self> {
        Self::new(1.0, offset, enclosure)
    }
    /// Reflects through the sum of two finite domain endpoints.
    pub fn reflection(domain: (f64, f64)) -> Option<Self> {
        ParameterEnclosure::new(domain.0, domain.1)?;
        let enclosure =
            ParameterEnclosure::point(domain.0)?.sum(ParameterEnclosure::point(domain.1)?)?;
        Self::new(-1.0, domain.0 + domain.1, enclosure)
    }

    /// Applies the nominal chart to a finite parameter.
    pub fn nominal(self, value: f64) -> Option<f64> {
        Self::new(self.scale, self.offset, self.offset_enclosure)?;
        if !value.is_finite() {
            return None;
        }
        let result = self.scale * value + self.offset;
        result.is_finite().then_some(result)
    }

    /// Encloses a finite parameter after conversion, including arithmetic error.
    pub fn enclose(self, value: f64) -> Option<ParameterEnclosure> {
        self.enclose_interval(ParameterEnclosure::point(value)?)
    }

    /// Encloses a complete input interval after conversion.
    pub fn enclose_interval(self, value: ParameterEnclosure) -> Option<ParameterEnclosure> {
        Self::new(self.scale, self.offset, self.offset_enclosure)?;
        let value = ParameterEnclosure::new(value.lower, value.upper)?;
        let scaled = if self.scale == 1.0 {
            value
        } else {
            value.reflected()
        };
        scaled.sum(self.offset_enclosure)
    }

    /// Composes `self(inner(value))`, preserving coefficient uncertainty.
    pub fn compose(self, inner: Self) -> Option<Self> {
        Self::new(self.scale, self.offset, self.offset_enclosure)?;
        Self::new(inner.scale, inner.offset, inner.offset_enclosure)?;
        let translated = if self.scale == 1.0 {
            inner.offset_enclosure
        } else {
            inner.offset_enclosure.reflected()
        };
        Self::new(
            self.scale * inner.scale,
            self.scale * inner.offset + self.offset,
            translated.sum(self.offset_enclosure)?,
        )
    }

    /// Inverts this orientation chart with the same enclosed coefficient.
    pub fn inverse(self) -> Option<Self> {
        Self::new(self.scale, self.offset, self.offset_enclosure)?;
        let offset = -self.scale * self.offset;
        let enclosure = if self.scale == 1.0 {
            self.offset_enclosure.reflected()
        } else {
            self.offset_enclosure
        };
        Self::new(self.scale, offset, enclosure)
    }
}

/// A bounded affine UV chart consisting of oriented axes and an optional swap.
///
/// Source and target active domains remain owned by the surface/face. This map
/// carries coefficient uncertainty and does not establish trim correspondence.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SurfaceParameterChart {
    /// Output-axis orientation charts.
    pub axes: [ParameterChart; 2],
    /// Whether the input axes are exchanged before applying the axis charts.
    pub swapped: bool,
}

impl SurfaceParameterChart {
    /// The exact identity UV chart.
    pub const fn identity() -> Self {
        Self {
            axes: [ParameterChart::identity(); 2],
            swapped: false,
        }
    }

    /// Applies the nominal UV chart.
    pub fn nominal(self, uv: [f64; 2]) -> Option<[f64; 2]> {
        let [u, v] = if self.swapped { [uv[1], uv[0]] } else { uv };
        Some([self.axes[0].nominal(u)?, self.axes[1].nominal(v)?])
    }

    /// Encloses the image of a complete input UV box.
    pub fn enclose(self, uv: [ParameterEnclosure; 2]) -> Option<[ParameterEnclosure; 2]> {
        let [u, v] = if self.swapped { [uv[1], uv[0]] } else { uv };
        Some([
            self.axes[0].enclose_interval(u)?,
            self.axes[1].enclose_interval(v)?,
        ])
    }

    /// Composes two bounded chart changes as `self(inner(uv))`.
    pub fn compose(self, inner: Self) -> Option<Self> {
        let input_axes = if self.swapped { [1, 0] } else { [0, 1] };
        Some(Self {
            axes: [
                self.axes[0].compose(inner.axes[input_axes[0]])?,
                self.axes[1].compose(inner.axes[input_axes[1]])?,
            ],
            swapped: self.swapped != inner.swapped,
        })
    }

    /// Inverts both oriented axes and their permutation.
    pub fn inverse(self) -> Option<Self> {
        let axes = if self.swapped {
            [self.axes[1].inverse()?, self.axes[0].inverse()?]
        } else {
            [self.axes[0].inverse()?, self.axes[1].inverse()?]
        };
        Some(Self {
            axes,
            swapped: self.swapped,
        })
    }

    /// The exact linear part, for derivative chain rules.
    pub fn jacobian(self) -> [[f64; 2]; 2] {
        if self.swapped {
            [[0.0, self.axes[0].scale], [self.axes[1].scale, 0.0]]
        } else {
            [[self.axes[0].scale, 0.0], [0.0, self.axes[1].scale]]
        }
    }
}

#[cfg(test)]
mod parameter_chart_regressions {
    use super::*;

    #[test]
    fn composition_and_inverse_retain_reflection_and_coefficient_error() {
        let reverse =
            ParameterChart::new(-1.0, 10.0, ParameterEnclosure::point(10.0).unwrap()).unwrap();
        let shift =
            ParameterChart::translation(3.0, ParameterEnclosure::new(2.0, 4.0).unwrap()).unwrap();
        let combined = reverse.compose(shift).unwrap();
        assert_eq!(combined.nominal(1.0), Some(6.0));
        assert_eq!(combined.enclose(1.0), ParameterEnclosure::new(5.0, 7.0));
        assert_eq!(reverse.inverse().unwrap().nominal(6.0), Some(4.0));
        assert!(ParameterChart::new(0.0, 0.0, ParameterEnclosure::point(0.0).unwrap()).is_none());
        assert!(combined.enclose(f64::INFINITY).is_none());
    }

    #[test]
    fn mapped_endpoint_rounding_is_not_reported_as_an_exact_sum() {
        let shift =
            ParameterChart::translation(1.0, ParameterEnclosure::point(1.0).unwrap()).unwrap();
        let enclosure = shift.enclose(f64::EPSILON / 4.0).unwrap();
        assert_eq!(enclosure.lower, 1.0);
        assert_eq!(enclosure.upper, 1.0f64.next_up());
        assert_eq!(shift.nominal(f64::EPSILON / 4.0), Some(1.0));
    }
}

#[cfg(test)]
mod surface_chart_composition_regressions {
    use super::*;

    #[test]
    fn signed_axis_permutation_composes_with_actual_input_axes() {
        let shift = |value| {
            ParameterChart::translation(value, ParameterEnclosure::point(value).unwrap()).unwrap()
        };
        let inner = SurfaceParameterChart {
            axes: [shift(1.0), shift(4.0)],
            swapped: false,
        };
        let outer = SurfaceParameterChart {
            axes: [
                ParameterChart::identity(),
                ParameterChart::reflection((2.0, 8.0)).unwrap(),
            ],
            swapped: true,
        };
        let composed = outer.compose(inner).unwrap();
        let uv = [2.0, 3.0];
        assert_eq!(composed.nominal(uv), Some([7.0, 7.0]));
        assert_eq!(
            composed.nominal(uv),
            outer.nominal(inner.nominal(uv).unwrap())
        );
        assert_eq!(composed.inverse().unwrap().nominal([7.0, 7.0]), Some(uv));
        let enclosure = composed
            .enclose(uv.map(|value| ParameterEnclosure::point(value).unwrap()))
            .unwrap();
        assert!(enclosure[0].contains(7.0) && enclosure[1].contains(7.0));
    }
}

#[cfg(test)]
mod compensated_sum_regressions {
    use super::*;

    #[test]
    fn finite_sum_with_overflowed_compensation_refuses_instead_of_claiming_exactness() {
        let left = -1.5 * 2.0f64.powi(971);
        let right = f64::MAX;
        // The exact result is strictly between MAX.next_down().next_down()
        // and MAX.next_down(); it is not the rounded singleton. Refusal is the
        // explicit finite-conditioning limit of this compensated primitive.
        assert_eq!(
            ParameterEnclosure::point(left)
                .unwrap()
                .sum(ParameterEnclosure::point(right).unwrap()),
            None
        );
    }
}
