use super::planar::Dyadic;
use super::*;
use crate::{
    BsplineCurve, Invertible, KnotVector, Line, Matrix3, Matrix4, NurbsCurve, Point2, Processor,
    TrimmedCurve, Vector4,
};

/// An explicitly selected native tensor-boundary chart branch.
///
/// Identical periodic seam rows can satisfy more than one selection. Selection
/// therefore belongs to the original oriented source use, not spatial proximity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeSurfaceBoundary {
    /// Constant first-axis lower endpoint, varying second axis.
    UFront,
    /// Constant first-axis upper endpoint, varying second axis.
    UBack,
    /// Constant second-axis lower endpoint, varying first axis.
    VFront,
    /// Constant second-axis upper endpoint, varying first axis.
    VBack,
}

pub(super) enum ExplicitCurve<'a> {
    Line(&'a Line<Point3>),
    Polynomial(&'a BsplineCurve<Point3>),
    Rational(&'a NurbsCurve<Vector4>),
}

impl ExplicitCurve<'_> {
    pub(super) fn count(&self) -> usize {
        match self {
            Self::Line(_) => 2,
            Self::Polynomial(curve) => curve.control_points().len(),
            Self::Rational(curve) => curve.control_points().len(),
        }
    }

    pub(super) fn knots_len(&self) -> usize {
        match self {
            Self::Line(_) => 4,
            Self::Polynomial(curve) => curve.knot_vector().len(),
            Self::Rational(curve) => curve.knot_vector().len(),
        }
    }

    pub(super) fn knot(&self, index: usize) -> f64 {
        match self {
            Self::Line(_) => [0.0, 0.0, 1.0, 1.0][index],
            Self::Polynomial(curve) => curve.knot_vector()[index],
            Self::Rational(curve) => curve.knot_vector()[index],
        }
    }

    pub(super) fn control(&self, index: usize) -> [f64; 4] {
        match self {
            Self::Line(curve) => {
                let point = if index == 0 { curve.0 } else { curve.1 };
                [point.x, point.y, point.z, 1.0]
            }
            Self::Polynomial(curve) => {
                let point = curve.control_points()[index];
                [point.x, point.y, point.z, 1.0]
            }
            Self::Rational(curve) => {
                let point = curve.control_points()[index];
                [point.x, point.y, point.z, point.w]
            }
        }
    }
}

fn dyadic(value: f64) -> Result<Dyadic, TrimDomainError> {
    Dyadic::scalar(value).ok_or_else(|| ValidatedGeometryError::Unresolved.into())
}

fn difference(a: f64, b: f64) -> Result<Dyadic, TrimDomainError> {
    dyadic(a)?
        .add(dyadic(-b)?)
        .ok_or_else(|| ValidatedGeometryError::Unresolved.into())
}

fn product(a: Dyadic, b: Dyadic) -> Result<Dyadic, TrimDomainError> {
    a.multiply(b)
        .ok_or_else(|| ValidatedGeometryError::Unresolved.into())
}

pub(super) fn knot_domain(
    knots: &KnotVector,
    count: usize,
) -> Result<(usize, [f64; 2]), TrimDomainError> {
    let degree = knots
        .len()
        .checked_sub(count.saturating_add(1))
        .ok_or(TrimDomainError::InvalidChart)?;
    if count == 0 || degree == 0 || degree >= count {
        return Err(TrimDomainError::InvalidChart);
    }
    let range = [knots[degree], knots[count]];
    if !range.into_iter().all(f64::is_finite)
        || range[0] >= range[1]
        || !knots.iter().all(|knot| knot.is_finite())
        || !knots.windows(2).all(|pair| pair[0] <= pair[1])
        || !knots.iter().take(degree + 1).all(|knot| *knot == range[0])
        || !knots.iter().skip(count).all(|knot| *knot == range[1])
    {
        return Err(ValidatedGeometryError::Unsupported.into());
    }
    Ok((degree, range))
}

fn source_control(
    source: &ExplicitCurve<'_>,
    index: usize,
    poses: &[&Matrix4],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<[Dyadic; 4], TrimDomainError> {
    let point = source.control(index);
    rechart::pose(
        [
            dyadic(point[0])?,
            dyadic(point[1])?,
            dyadic(point[2])?,
            dyadic(point[3])?,
        ],
        poses,
        budget,
        cancel,
    )
}

fn range_matches(
    mut curve: &Curve2D,
    range: [f64; 2],
    axis: usize,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<bool, TrimDomainError> {
    if !range.into_iter().all(f64::is_finite) || range[0] >= range[1] {
        return Err(TrimDomainError::InvalidChart);
    }
    let domain = curve.range_tuple();
    let mut parameters = [dyadic(domain.0)?, dyadic(domain.1)?];
    let mut depth = 0;
    while let Curve2D::TrimmedCurve(selected) = curve {
        budget.charge(1, cancel)?;
        depth += 1;
        if depth > 32 {
            return Err(TrimDomainError::ResourceLimit);
        }
        if *selected.transform() != Matrix3::from_scale(1.0) {
            return Ok(false);
        }
        if !selected.orientation() {
            let (a, b) = selected.entity().range();
            let sum = dyadic(a)?
                .add(dyadic(b)?)
                .ok_or(TrimDomainError::InvalidChart)?;
            for parameter in &mut parameters {
                *parameter = sum
                    .add(parameter.negated().ok_or(TrimDomainError::InvalidChart)?)
                    .ok_or(TrimDomainError::Geometry(
                        ValidatedGeometryError::Unresolved,
                    ))?;
            }
        }
        curve = selected.entity().curve().as_ref();
    }
    let Curve2D::BsplineCurve(line) = curve else {
        return Ok(false);
    };
    if line.control_points().len() != 2 || line.knot_vector().len() != 4 {
        return Ok(false);
    }
    let knots = line.knot_vector();
    let span = difference(knots[3], knots[0])?;
    let controls = line.control_points();
    let delta = difference(controls[1][axis], controls[0][axis])?;
    let mut numerators = [dyadic(0.0)?; 2];
    for (index, parameter) in parameters.into_iter().enumerate() {
        budget.charge(1, cancel)?;
        let offset = parameter
            .add(dyadic(-knots[0])?)
            .ok_or(TrimDomainError::InvalidChart)?;
        numerators[index] = product(dyadic(controls[0][axis])?, span)?
            .add(product(delta, offset)?)
            .ok_or(TrimDomainError::Geometry(
                ValidatedGeometryError::Unresolved,
            ))?;
    }
    let expected = [
        product(dyadic(range[0])?, span)?,
        product(dyadic(range[1])?, span)?,
    ];
    Ok(numerators == expected || numerators == [expected[1], expected[0]])
}

fn extract(
    source: ExplicitCurve<'_>,
    knots: &(KnotVector, KnotVector),
    counts: [usize; 2],
    control: impl Fn(usize, usize) -> [f64; 4],
    selection: (NativeSurfaceBoundary, Option<[f64; 2]>),
    poses: &[&Matrix4],
    execution: (&mut ValidatedGeometryBudget, &AtomicBool),
) -> Result<Curve2D, TrimDomainError> {
    let (boundary, restriction) = selection;
    let (budget, cancel) = execution;
    budget.charge(
        knots
            .0
            .len()
            .saturating_add(knots.1.len())
            .saturating_add(source.knots_len()),
        cancel,
    )?;
    let (u_degree, u) = knot_domain(&knots.0, counts[0])?;
    let (v_degree, v) = knot_domain(&knots.1, counts[1])?;
    let (varying, complete_count, degree, complete_range) = match boundary {
        NativeSurfaceBoundary::UFront | NativeSurfaceBoundary::UBack => {
            (&knots.1, counts[1], v_degree, v)
        }
        NativeSurfaceBoundary::VFront | NativeSurfaceBoundary::VBack => {
            (&knots.0, counts[0], u_degree, u)
        }
    };
    let range = restriction.unwrap_or(complete_range);
    if !range.into_iter().all(f64::is_finite)
        || range[0] >= range[1]
        || range[0] < complete_range[0]
        || range[1] > complete_range[1]
    {
        return Err(TrimDomainError::InvalidChart);
    }
    let (first_control, last_control, first_interior) = if restriction.is_some() {
        let mut endpoint =
            |value: f64| -> Result<(usize, usize), TrimDomainError> {
                budget.charge(varying.len(), cancel)?;
                let first = varying.iter().position(|knot| *knot == value).ok_or(
                    TrimDomainError::Geometry(ValidatedGeometryError::Unsupported),
                )?;
                let multiplicity = varying
                    .iter()
                    .skip(first)
                    .take_while(|knot| **knot == value)
                    .count();
                if value == complete_range[0] {
                    Ok((0, degree + 1))
                } else if value == complete_range[1] {
                    Ok((complete_count - 1, first))
                } else if multiplicity == degree {
                    Ok((
                        first.checked_sub(1).ok_or(TrimDomainError::InvalidChart)?,
                        first + multiplicity,
                    ))
                } else {
                    Err(ValidatedGeometryError::Unsupported.into())
                }
            };
        let (first, interior) = endpoint(range[0])?;
        let (last, _) = endpoint(range[1])?;
        (first, last, interior)
    } else {
        (0, complete_count - 1, degree + 1)
    };
    let count = last_control
        .checked_sub(first_control)
        .and_then(|count| count.checked_add(1))
        .ok_or(TrimDomainError::InvalidChart)?;
    let knot_count = count
        .checked_add(degree + 1)
        .ok_or(TrimDomainError::ResourceLimit)?;
    if source.count() != count || source.knots_len() != knot_count {
        return Err(ValidatedGeometryError::Unresolved.into());
    }
    let knot = |index: usize| {
        if index <= degree {
            range[0]
        } else if index >= count {
            range[1]
        } else {
            varying[first_interior + index - degree - 1]
        }
    };
    let native_range = [source.knot(degree), source.knot(count)];
    if !native_range.into_iter().all(f64::is_finite) || native_range[0] >= native_range[1] {
        return Err(TrimDomainError::InvalidChart);
    }
    let row = |index| match boundary {
        NativeSurfaceBoundary::UFront => control(0, first_control + index),
        NativeSurfaceBoundary::UBack => control(counts[0] - 1, first_control + index),
        NativeSurfaceBoundary::VFront => control(first_control + index, 0),
        NativeSurfaceBoundary::VBack => control(first_control + index, counts[1] - 1),
    };
    let native_span = difference(native_range[1], native_range[0])?;
    let tensor_span = difference(range[1], range[0])?;
    for reverse in [false, true] {
        let mut matches = true;
        for index in 0..knot_count {
            budget.charge(1, cancel)?;
            let native = source.knot(index);
            if !native.is_finite() || (index > 0 && native < source.knot(index - 1)) {
                return Err(TrimDomainError::InvalidChart);
            }
            let tensor_offset = if reverse {
                difference(range[1], knot(knot_count - 1 - index))?
            } else {
                difference(knot(index), range[0])?
            };
            if product(difference(native, native_range[0])?, tensor_span)?
                != product(tensor_offset, native_span)?
            {
                matches = false;
                break;
            }
        }
        if !matches {
            continue;
        }
        let mut pivot = None;
        for index in 0..count {
            budget.charge(1, cancel)?;
            let a = source_control(&source, index, poses, budget, cancel)?;
            let b = row(if reverse { count - 1 - index } else { index });
            if a[3] != dyadic(0.0)? && b[3] != 0.0 {
                pivot = Some((a[3], dyadic(b[3])?));
                break;
            }
        }
        let Some((source_pivot, row_pivot)) = pivot else {
            return Err(ValidatedGeometryError::DenominatorContainsZero.into());
        };
        for index in 0..count {
            budget.charge(1, cancel)?;
            let a = source_control(&source, index, poses, budget, cancel)?;
            let b = row(if reverse { count - 1 - index } else { index });
            for axis in 0..4 {
                if product(a[axis], row_pivot)? != product(dyadic(b[axis])?, source_pivot)? {
                    matches = false;
                    break;
                }
            }
            if !matches {
                break;
            }
        }
        if !matches {
            continue;
        }
        let (mut front, mut back) = match boundary {
            NativeSurfaceBoundary::UFront => {
                (Point2::new(u[0], range[0]), Point2::new(u[0], range[1]))
            }
            NativeSurfaceBoundary::UBack => {
                (Point2::new(u[1], range[0]), Point2::new(u[1], range[1]))
            }
            NativeSurfaceBoundary::VFront => {
                (Point2::new(range[0], v[0]), Point2::new(range[1], v[0]))
            }
            NativeSurfaceBoundary::VBack => {
                (Point2::new(range[0], v[1]), Point2::new(range[1], v[1]))
            }
        };
        if reverse {
            std::mem::swap(&mut front, &mut back);
        }
        budget.charge(1, cancel)?;
        return Ok(Curve2D::BsplineCurve(BsplineCurve::new(
            KnotVector::from(vec![
                native_range[0],
                native_range[0],
                native_range[1],
                native_range[1],
            ]),
            vec![front, back],
        )));
    }
    Err(ValidatedGeometryError::Unresolved.into())
}

/// Proves an explicit source edge is exactly a selected tensor boundary isocurve.
///
/// Every homogeneous control coefficient must equal the complete selected row
/// up to one nonzero global projective scale. All native knots must match under
/// an exactly checked affine parameter map, including reversed control order.
/// Endpoint sampling, nearest-point reconstruction and analytic resemblance are
/// never used. Supports complete clamped multi-span tensor boundaries and
/// explicit lines, polynomial B-splines and NURBS edges. The returned degree-one
/// UV curve has the original spatial edge's exact parameter interval.
///
/// The caller must bind `boundary` to the original oriented source use. Coincident
/// seam rows can prove several branches; this function does not claim the selected
/// branch is uniquely determined by the spatial curve. Native denominator jets
/// must separately certify regularity before using this correspondence.
pub fn certified_surface_isocurve_parameter_curve(
    curve: &Curve,
    surface: &Surface,
    boundary: NativeSurfaceBoundary,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Curve2D, TrimDomainError> {
    parameter_curve(curve, surface, boundary, None, budget, cancel)
}

/// Proves exact tensor-boundary correspondence on a selected source interval.
///
/// The selected varying-axis interval is original source-use input. Its endpoints
/// must be clamped carrier endpoints or actual native knots of multiplicity equal
/// to degree, where restriction is a genuine contiguous control-row subset.
/// No rounded de Boor cutting or nearest-point reconstruction is used. Every
/// restricted homogeneous coefficient and every restricted knot is then verified
/// against the spatial curve exactly, allowing global projective scale and affine
/// parameter reversal. The UV curve retains the spatial curve's native domain.
/// Other restriction endpoints return typed `Unsupported`.
pub fn certified_surface_boundary_parameter_curve(
    curve: &Curve,
    surface: &Surface,
    boundary: NativeSurfaceBoundary,
    variable_range: [f64; 2],
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Curve2D, TrimDomainError> {
    parameter_curve(
        curve,
        surface,
        boundary,
        Some(variable_range),
        budget,
        cancel,
    )
}

fn parameter_curve(
    curve: &Curve,
    surface: &Surface,
    boundary: NativeSurfaceBoundary,
    restriction: Option<[f64; 2]>,
    budget: &mut ValidatedGeometryBudget,
    cancel: &AtomicBool,
) -> Result<Curve2D, TrimDomainError> {
    let identity = Matrix4::from_scale(1.0);
    let mut poses = [&identity; 32];
    parameter_curve_at(
        curve,
        surface,
        boundary,
        restriction,
        &mut poses,
        0,
        (budget, cancel),
    )
}

fn parameter_curve_at<'a>(
    curve: &'a Curve,
    surface: &Surface,
    boundary: NativeSurfaceBoundary,
    restriction: Option<[f64; 2]>,
    poses: &mut [&'a Matrix4; 32],
    depth: usize,
    execution: (&mut ValidatedGeometryBudget, &AtomicBool),
) -> Result<Curve2D, TrimDomainError> {
    let (budget, cancel) = execution;
    budget.charge(1, cancel)?;
    if depth >= poses.len() {
        return Err(TrimDomainError::ResourceLimit);
    }
    if let Curve::TrimmedCurve(selected) = curve {
        poses[depth] = selected.transform();
        let underlying = selected.entity().curve().as_ref();
        let basis = match parameter_curve_at(
            underlying,
            surface,
            boundary,
            None,
            poses,
            depth + 1,
            (budget, cancel),
        ) {
            Ok(basis) => basis,
            Err(TrimDomainError::Geometry(ValidatedGeometryError::Unresolved))
                if restriction.is_some() =>
            {
                parameter_curve_at(
                    underlying,
                    surface,
                    boundary,
                    restriction,
                    poses,
                    depth + 1,
                    (budget, cancel),
                )?
            }
            Err(error) => return Err(error),
        };
        let mut output = Processor::with_transform(
            TrimmedCurve::new(Box::new(basis), selected.entity().range()),
            Matrix3::from_scale(1.0),
        );
        if !selected.orientation() {
            output.invert();
        }
        let output = Curve2D::TrimmedCurve(output);
        if let Some(range) = restriction {
            let axis = match boundary {
                NativeSurfaceBoundary::UFront | NativeSurfaceBoundary::UBack => 1,
                _ => 0,
            };
            if !range_matches(&output, range, axis, budget, cancel)? {
                return Err(ValidatedGeometryError::Unresolved.into());
            }
        }
        return Ok(output);
    }
    let curve = match curve {
        Curve::Line(curve) => ExplicitCurve::Line(curve),
        Curve::BsplineCurve(curve) => ExplicitCurve::Polynomial(curve),
        Curve::NurbsCurve(curve) => ExplicitCurve::Rational(curve),
        _ => return Err(ValidatedGeometryError::Unsupported.into()),
    };
    match surface {
        Surface::BsplineSurface(surface) => {
            let controls = surface.control_points();
            let columns = controls.first().ok_or(TrimDomainError::InvalidChart)?.len();
            budget.charge(controls.len(), cancel)?;
            if columns == 0 || !controls.iter().all(|row| row.len() == columns) {
                return Err(TrimDomainError::InvalidChart);
            }
            extract(
                curve,
                surface.knot_vectors(),
                [controls.len(), columns],
                |i, j| {
                    let point = controls[i][j];
                    [point.x, point.y, point.z, 1.0]
                },
                (boundary, restriction),
                &poses[..depth],
                (budget, cancel),
            )
        }
        Surface::NurbsSurface(surface) => {
            let controls = surface.control_points();
            let columns = controls.first().ok_or(TrimDomainError::InvalidChart)?.len();
            budget.charge(controls.len(), cancel)?;
            if columns == 0 || !controls.iter().all(|row| row.len() == columns) {
                return Err(TrimDomainError::InvalidChart);
            }
            extract(
                curve,
                surface.knot_vectors(),
                [controls.len(), columns],
                |i, j| {
                    let point = controls[i][j];
                    [point.x, point.y, point.z, point.w]
                },
                (boundary, restriction),
                &poses[..depth],
                (budget, cancel),
            )
        }
        _ => Err(ValidatedGeometryError::Unsupported.into()),
    }
}
