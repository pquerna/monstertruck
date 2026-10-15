//! Validated analytic chart phases, without libm accuracy assumptions.

use super::validated_interval::{
    ValidatedGeometryBudget, ValidatedGeometryError, ValidatedInterval, validated_sin_cos,
};
use std::sync::atomic::AtomicBool;

type Interval = ValidatedInterval;

fn point(value: f64) -> Result<Interval, ValidatedGeometryError> { Interval::new(value, value) }

fn corner_phase(
    y: f64,
    x: f64,
    uncertainty: f64,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Interval, ValidatedGeometryError> {
    let half_pi = Interval::new(
        std::f64::consts::FRAC_PI_2.next_down(),
        std::f64::consts::FRAC_PI_2.next_up(),
    )?;
    let pi = Interval::new(
        std::f64::consts::PI.next_down(),
        std::f64::consts::PI.next_up(),
    )?;
    if x == 0.0 {
        return if y > 0.0 {
            Ok(half_pi)
        } else if y < 0.0 {
            Ok(-half_pi)
        } else {
            Err(ValidatedGeometryError::Singular)
        };
    }
    if y == 0.0 {
        return if x > 0.0 { point(0.0) } else { Ok(pi) };
    }
    let scale = x.abs().max(y.abs());
    let signs = (x > 0.0, y > 0.0);
    let x = Interval::new((x / scale).next_down(), (x / scale).next_up())?;
    let y = Interval::new((y / scale).next_down(), (y / scale).next_up())?;
    let mut bracket = match signs {
        (true, true) => Interval::new(0.0, half_pi.upper)?,
        (false, true) => Interval::new(half_pi.lower, pi.upper)?,
        (false, false) => Interval::new(-pi.upper, -half_pi.lower)?,
        (true, false) => Interval::new(-half_pi.upper, 0.0)?,
    };
    let equation = |angle: f64, budget: &mut ValidatedGeometryBudget| {
        let (sine, cosine) = validated_sin_cos(point(angle)?, budget, cancel)?;
        Ok::<_, ValidatedGeometryError>(y * cosine - x * sine)
    };
    loop {
        budget.charge(1, cancel)?;
        if bracket.width() <= uncertainty {
            return Ok(bracket);
        }
        let middle = bracket.midpoint();
        if middle <= bracket.lower || middle >= bracket.upper {
            return Err(ValidatedGeometryError::ConditioningFailure);
        }
        let value = equation(middle, budget)?;
        if value.lower > 0.0 {
            bracket.lower = middle;
        } else if value.upper < 0.0 {
            bracket.upper = middle;
        } else {
            let radius = (uncertainty / 4.0).next_up();
            let lower = (middle - radius).next_down().max(bracket.lower);
            let upper = (middle + radius).next_up().min(bracket.upper);
            let left = equation(lower, budget)?;
            let right = equation(upper, budget)?;
            if left.lower > 0.0 && right.upper < 0.0 {
                bracket = Interval::new(lower, upper)?;
            } else {
                return Err(ValidatedGeometryError::ConditioningFailure);
            }
        }
    }
}

/// Encloses the continuously lifted phase of a rectangular vector enclosure.
///
/// The rectangle must exclude the origin. Every corner phase is isolated by
/// strict signs of a rigorously enclosed sine/cosine equation on its quadrant.
/// Angular extrema of an origin-excluding rectangle occur at its corners.
/// The chosen integer turns use an outward enclosure of mathematical `2*pi`,
/// not an assertion that binary64 `TAU` is the exact period. `reference` selects
/// the covering-chart branch; a sector crossing that branch cut refuses.
///
/// `uncertainty` bounds each isolated corner phase, not the angular spread of
/// the input rectangle. All bisection and Taylor work charges are cumulative.
pub fn validated_atan2(
    y: Interval,
    x: Interval,
    reference: f64,
    uncertainty: f64,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Interval, ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    if !x.is_finite()
        || !y.is_finite()
        || !reference.is_finite()
        || reference.abs() > 1_099_511_627_776.0
        || !uncertainty.is_finite()
        || uncertainty <= 0.0
    {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    if x.contains_zero() && y.contains_zero() {
        return Err(ValidatedGeometryError::Singular);
    }
    let period = Interval::new(
        std::f64::consts::TAU.next_down(),
        std::f64::consts::TAU.next_up(),
    )?;
    let mut result: Option<Interval> = None;
    for (cx, cy) in [
        (x.lower, y.lower),
        (x.lower, y.upper),
        (x.upper, y.lower),
        (x.upper, y.upper),
    ] {
        let phase = corner_phase(cy, cx, uncertainty, budget, cancel)?;
        if phase.width() > uncertainty {
            return Err(ValidatedGeometryError::ConditioningFailure);
        }
        let turns = ((reference - phase.midpoint()) / std::f64::consts::TAU).round();
        let lifted = phase + period * point(turns)?;
        if !lifted.is_finite() {
            return Err(ValidatedGeometryError::ConditioningFailure);
        }
        result = Some(result.map_or(lifted, |old| old.hull(lifted)));
    }
    let result = result.ok_or(ValidatedGeometryError::InvalidInput)?;
    if result.width() >= std::f64::consts::PI.next_down() {
        Err(ValidatedGeometryError::AmbiguousBranch)
    } else {
        Ok(result)
    }
}

/// Encloses the principal analytic arcsine of an interval within `[-1, 1]`.
///
/// Square-root endpoints use outward-rounded IEEE square root. The phase is
/// then certified by [`validated_atan2`], including the two pole limits.
pub fn validated_asin(
    value: Interval,
    uncertainty: f64,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Interval, ValidatedGeometryError> {
    budget.charge(1, cancel)?;
    if !value.is_finite()
        || value.lower < -1.0
        || value.upper > 1.0
        || !uncertainty.is_finite()
        || uncertainty <= 0.0
    {
        return Err(ValidatedGeometryError::InvalidInput);
    }
    let endpoint = |coordinate: f64, budget: &mut ValidatedGeometryBudget| {
        if coordinate.abs() == 1.0 {
            let half_pi = Interval::new(
                std::f64::consts::FRAC_PI_2.next_down(),
                std::f64::consts::FRAC_PI_2.next_up(),
            )?;
            Ok::<_, ValidatedGeometryError>(if coordinate > 0.0 { half_pi } else { -half_pi })
        } else {
            let squared_cosine = point(1.0)? - point(coordinate)?.square();
            let cosine = Interval::new(
                squared_cosine.lower.max(0.0).sqrt().next_down().max(0.0),
                squared_cosine.upper.max(0.0).sqrt().next_up(),
            )?;
            validated_atan2(point(coordinate)?, cosine, 0.0, uncertainty, budget, cancel)
        }
    };
    let lower = endpoint(value.lower, budget)?;
    let upper = endpoint(value.upper, budget)?;
    Interval::new(lower.lower, upper.upper)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_enclosures_cover_quadrants_and_lifted_branch() {
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(100_000);
        for (x, y, expected) in [
            (1.0, 1.0, std::f64::consts::FRAC_PI_4),
            (-1.0, 1.0, 3.0 * std::f64::consts::FRAC_PI_4),
            (-1.0, -1.0, -3.0 * std::f64::consts::FRAC_PI_4),
            (1.0, -1.0, -std::f64::consts::FRAC_PI_4),
        ] {
            let phase = validated_atan2(
                point(y).expect("finite"),
                point(x).expect("finite"),
                expected,
                1.0e-12,
                &mut budget,
                &cancel,
            )
            .expect("validated quadrant");
            assert!(phase.lower <= expected && expected <= phase.upper);
        }
        let phase = validated_atan2(
            Interval::new(-0.01, 0.01).expect("finite"),
            point(-1.0).expect("finite"),
            std::f64::consts::PI,
            1.0e-12,
            &mut budget,
            &cancel,
        )
        .expect("lift across negative x");
        assert!(phase.lower < std::f64::consts::PI && phase.upper > std::f64::consts::PI);
    }

    #[test]
    fn phase_origin_cancellation_and_work_refusals_are_typed() {
        let cancel = AtomicBool::new(false);
        let mut budget = ValidatedGeometryBudget::new(0);
        assert_eq!(
            validated_atan2(
                point(1.0).expect("finite"),
                point(1.0).expect("finite"),
                0.0,
                1.0e-12,
                &mut budget,
                &cancel
            ),
            Err(ValidatedGeometryError::ResourceLimit)
        );
        let mut budget = ValidatedGeometryBudget::new(100);
        assert_eq!(
            validated_atan2(
                Interval::new(-1.0, 1.0).expect("finite"),
                Interval::new(-1.0, 1.0).expect("finite"),
                0.0,
                1.0e-12,
                &mut budget,
                &cancel
            ),
            Err(ValidatedGeometryError::Singular)
        );
        cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            validated_atan2(
                point(1.0).expect("finite"),
                point(1.0).expect("finite"),
                0.0,
                1.0e-12,
                &mut budget,
                &cancel
            ),
            Err(ValidatedGeometryError::Cancelled)
        );
    }
}
