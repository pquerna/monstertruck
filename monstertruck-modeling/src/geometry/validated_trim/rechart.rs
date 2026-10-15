use super::planar::{Dyadic, plane_coefficients};
use super::*;
use crate::{Matrix3, Matrix4, Plane, Processor, TrimmedCurve, Vector3};

fn refusal() -> TrimDomainError { ValidatedGeometryError::Unresolved.into() }
fn value(number: f64) -> Result<Dyadic, TrimDomainError> {
    Dyadic::scalar(number).ok_or_else(refusal)
}
fn add(a: Dyadic, b: Dyadic) -> Result<Dyadic, TrimDomainError> { a.add(b).ok_or_else(refusal) }
fn multiply(a: Dyadic, b: Dyadic) -> Result<Dyadic, TrimDomainError> {
    a.multiply(b).ok_or_else(refusal)
}
fn negate(number: Dyadic) -> Result<Dyadic, TrimDomainError> {
    number.negated().ok_or_else(refusal)
}

struct Projection {
    coefficients: [[Dyadic; 3]; 3],
    axes: [usize; 2],
    determinant: Dyadic,
}

impl Projection {
    fn new(plane: &Plane) -> Result<Self, TrimDomainError> {
        let coefficients = plane_coefficients(plane).ok_or_else(refusal)?;
        let u = plane.axis_u();
        let v = plane.axis_v();
        let axes = [[0, 1], [0, 2], [1, 2]]
            .into_iter()
            .max_by(|a, b| {
                (u[a[0]] * v[a[1]] - u[a[1]] * v[a[0]])
                    .abs()
                    .total_cmp(&(u[b[0]] * v[b[1]] - u[b[1]] * v[b[0]]).abs())
            })
            .ok_or(TrimDomainError::InvalidChart)?;
        let [a, b] = axes;
        let determinant = add(
            multiply(coefficients[1][a], coefficients[2][b])?,
            negate(multiply(coefficients[1][b], coefficients[2][a])?)?,
        )?;
        if determinant == value(0.0)? {
            return Err(TrimDomainError::InvalidChart);
        }
        Ok(Self {
            coefficients,
            axes,
            determinant,
        })
    }

    fn project(&self, world: [Dyadic; 4]) -> Result<[Dyadic; 3], TrimDomainError> {
        let [a, b] = self.axes;
        let delta_a = add(
            world[a],
            negate(multiply(self.coefficients[0][a], world[3])?)?,
        )?;
        let delta_b = add(
            world[b],
            negate(multiply(self.coefficients[0][b], world[3])?)?,
        )?;
        let result = [
            add(
                multiply(delta_a, self.coefficients[2][b])?,
                negate(multiply(delta_b, self.coefficients[2][a])?)?,
            )?,
            add(
                multiply(self.coefficients[1][a], delta_b)?,
                negate(multiply(self.coefficients[1][b], delta_a)?)?,
            )?,
            multiply(self.determinant, world[3])?,
        ];
        for (axis, coordinate) in world.iter().enumerate().take(3) {
            let reconstruction = add(
                add(
                    multiply(self.coefficients[0][axis], result[2])?,
                    multiply(self.coefficients[1][axis], result[0])?,
                )?,
                multiply(self.coefficients[2][axis], result[1])?,
            )?;
            if reconstruction != multiply(*coordinate, self.determinant)? {
                return Err(refusal());
            }
        }
        Ok(result)
    }
}

pub(super) fn pose(
    mut point: [Dyadic; 4],
    poses: &[&Matrix4],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<[Dyadic; 4], TrimDomainError> {
    for matrix in poses.iter().rev() {
        budget.charge(1, cancel)?;
        let mut transformed = [value(0.0)?; 4];
        for row in 0..4 {
            for column in 0..4 {
                transformed[row] = add(
                    transformed[row],
                    multiply(value(matrix[column][row])?, point[column])?,
                )?;
            }
        }
        point = transformed;
    }
    Ok(point)
}

fn common_scale(values: &[[Dyadic; 3]]) -> Result<(u128, i32), TrimDomainError> {
    let mut divisor = 0_u128;
    let mut exponent = i32::MIN;
    for number in values.iter().flatten() {
        let mut a = divisor;
        let mut b = number.unsigned_coefficient();
        while b != 0 {
            let remainder = a % b;
            a = b;
            b = remainder;
        }
        divisor = a;
        exponent = exponent.max(number.leading_exponent());
    }
    if divisor == 0 || exponent == i32::MIN {
        return Err(refusal());
    }
    Ok((divisor, exponent.checked_neg().ok_or_else(refusal)?))
}

fn scaled(number: Dyadic, divisor: u128, exponent: i32) -> Result<f64, TrimDomainError> {
    number
        .divided_coefficient(divisor)
        .and_then(|number| number.shifted(exponent))
        .and_then(Dyadic::exact_float)
        .ok_or_else(refusal)
}

fn matrix(values: [[Dyadic; 3]; 3]) -> Result<Matrix3, TrimDomainError> {
    let (divisor, exponent) = common_scale(&values)?;
    let mut columns = [Vector3::new(0.0, 0.0, 0.0); 3];
    for column in 0..3 {
        for row in 0..3 {
            columns[column][row] = scaled(values[column][row], divisor, exponent)?;
        }
    }
    Ok(Matrix3::from_cols(columns[0], columns[1], columns[2]))
}

fn transformed(curve: Curve2D, transform: Matrix3) -> Result<Curve2D, TrimDomainError> {
    if (0..3).all(|column| {
        (0..3).all(|row| {
            if row == column {
                transform[column][row] == transform[2][2]
            } else {
                transform[column][row] == 0.0
            }
        })
    }) && transform[2][2] != 0.0
    {
        return Ok(curve);
    }
    let range = curve.range_tuple();
    if ![range.0, range.1].into_iter().all(f64::is_finite) || range.0 >= range.1 {
        return Err(TrimDomainError::InvalidChart);
    }
    Ok(Curve2D::TrimmedCurve(Processor::with_transform(
        TrimmedCurve::new(Box::new(curve), range),
        transform,
    )))
}

pub(super) fn parameter_curve(
    curve: &Curve2D,
    source: &Plane,
    target: &Plane,
    poses: &[&Matrix4],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Curve2D, TrimDomainError> {
    let projection = Projection::new(target)?;
    let source = plane_coefficients(source).ok_or_else(refusal)?;
    let mut frame = [[value(0.0)?; 4]; 3];
    let mut projected = [[value(0.0)?; 3]; 3];
    for (column, source_column) in [1, 2, 0].into_iter().enumerate() {
        budget.charge(1, cancel)?;
        let weight = value(if source_column == 0 { 1.0 } else { 0.0 })?;
        frame[column] = pose(
            [
                source[source_column][0],
                source[source_column][1],
                source[source_column][2],
                weight,
            ],
            poses,
            budget,
            cancel,
        )?;
        projected[column] = projection.project(frame[column])?;
    }
    budget.charge(1, cancel)?;
    let transform = matrix(projected);
    let count = curve_storage_units(curve, budget, cancel)?;
    budget.charge(count, cancel)?;
    let curve = curve.clone();
    if let Ok(transform) = transform {
        return transformed(curve, transform);
    }
    // Retain exact factors rather than round a composed coefficient matrix.
    let [a, b] = projection.axes;
    let first = matrix(frame.map(|point| [point[a], point[b], point[3]]))?;
    let zero = value(0.0)?;
    let translated = Matrix3::from_cols(
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(-target.origin()[a], -target.origin()[b], 1.0),
    );
    let inverse = matrix([
        [
            projection.coefficients[2][b],
            negate(projection.coefficients[1][b])?,
            zero,
        ],
        [
            negate(projection.coefficients[2][a])?,
            projection.coefficients[1][a],
            zero,
        ],
        [zero, zero, projection.determinant],
    ])?;
    budget.charge(3, cancel)?;
    transformed(
        transformed(transformed(curve, first)?, translated)?,
        inverse,
    )
}

pub(super) fn controls(
    controls: impl ExactSizeIterator<Item = [f64; 4]>,
    target: &Plane,
    poses: &[&Matrix4],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Vec<Vector3>, TrimDomainError> {
    let projection = Projection::new(target)?;
    budget.charge(controls.len(), cancel)?;
    let mut projected = Vec::with_capacity(controls.len());
    for point in controls {
        budget.charge(1, cancel)?;
        let point = [
            value(point[0])?,
            value(point[1])?,
            value(point[2])?,
            value(point[3])?,
        ];
        projected.push(projection.project(pose(point, poses, budget, cancel)?)?);
    }
    budget.charge(projected.len().saturating_mul(3), cancel)?;
    let (divisor, exponent) = common_scale(&projected)?;
    projected
        .into_iter()
        .map(|point| {
            budget.charge(1, cancel)?;
            Ok(Vector3::new(
                scaled(point[0], divisor, exponent)?,
                scaled(point[1], divisor, exponent)?,
                scaled(point[2], divisor, exponent)?,
            ))
        })
        .collect()
}
