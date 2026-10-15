use super::{Result, *};
use crate::step::{ParameterChart, ParameterEnclosure, SurfaceParameterChart};
use monstertruck_geometry::prelude::*;
use monstertruck_mesh::PolylineCurve;
use monstertruck_modeling::{
    Conic2D as ModelingConic2D, Curve as ModelingCurve, Curve2D as ModelingCurve2D,
    Surface as ModelingSurface,
};

impl StepFormat for Point2 {
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        f.write_fmt(format_args!(
            "#{idx} = CARTESIAN_POINT('', {coordinates});\n",
            coordinates = SliceDisplay(AsRef::<[f64; 2]>::as_ref(self)),
        ))
    }
}
impl_const_step_length!(Point2, 1);

impl StepFormat for Point3 {
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        f.write_fmt(format_args!(
            "#{idx} = CARTESIAN_POINT('', {coordinates});\n",
            coordinates = SliceDisplay(AsRef::<[f64; 3]>::as_ref(self)),
        ))
    }
}
impl_const_step_length!(Point3, 1);

/// class for display `DIRECTION`.
#[derive(Clone, Debug, Copy)]
pub struct VectorAsDirection<V>(pub V);

impl StepFormat for VectorAsDirection<Vector2> {
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        f.write_fmt(format_args!(
            "#{idx} = DIRECTION('', {direction_ratios});\n",
            direction_ratios = SliceDisplay(AsRef::<[f64; 2]>::as_ref(&self.0)),
        ))
    }
}

impl StepFormat for VectorAsDirection<Vector3> {
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        f.write_fmt(format_args!(
            "#{idx} = DIRECTION('', {direction_ratios});\n",
            direction_ratios = SliceDisplay(AsRef::<[f64; 3]>::as_ref(&self.0)),
        ))
    }
}
impl_const_step_length!(VectorAsDirection<V>, 1, <V>);

impl<V> StepFormat for V
where
    V: InnerSpace<Scalar = f64>,
    VectorAsDirection<V>: StepFormat,
{
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let magnitude = self.magnitude();
        let direction_idx = idx + 1;
        f.write_fmt(format_args!(
            "#{idx} = VECTOR('', #{direction_idx}, {magnitude});\n{direction}",
            direction = StepDisplay::new(VectorAsDirection(*self / magnitude), direction_idx),
            magnitude = FloatDisplay(magnitude),
        ))
    }
}
impl_const_step_length!(Vector2, 2);
impl_const_step_length!(Vector3, 2);

/// Wrapper that displays a transform matrix as a STEP `AXIS2_PLACEMENT_*` entity.
///
/// STEP encodes an oriented coordinate frame -- a placement -- as a
/// location point plus one (`AXIS2_PLACEMENT_2D`) or two
/// (`AXIS2_PLACEMENT_3D`) direction vectors. This wrapper takes a
/// transform matrix from `cgmath` (column-major) and emits the four
/// referenced entities: the placement itself, the location point, and
/// the direction(s).
#[derive(Clone, Copy, Debug)]
pub struct MatrixAsAxis<M>(pub M);

impl StepFormat for MatrixAsAxis<Matrix3> {
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let location_idx = idx + 1;
        let ref_direction_idx = idx + 2;
        let location = self.0[2].to_point();
        let ref_direction = VectorAsDirection(self.0[0].truncate());
        f.write_fmt(format_args!(
            "#{idx} = AXIS2_PLACEMENT_2D('', #{location_idx}, #{ref_direction_idx});\n",
        ))?;
        StepFormat::fmt(&location, location_idx, f)?;
        StepFormat::fmt(&ref_direction, ref_direction_idx, f)
    }
}
impl_const_step_length!(MatrixAsAxis<Matrix3>, 3);

impl StepFormat for MatrixAsAxis<Matrix4> {
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let location_idx = idx + 1;
        let axis_idx = idx + 2;
        let ref_direction_idx = idx + 3;
        let location = self.0[3].to_point();
        let axis = VectorAsDirection(self.0[2].truncate());
        let ref_direction = VectorAsDirection(self.0[0].truncate());
        f.write_fmt(format_args!(
            "#{idx} = AXIS2_PLACEMENT_3D('', #{location_idx}, #{axis_idx}, #{ref_direction_idx});\n",
        ))?;
        StepFormat::fmt(&location, location_idx, f)?;
        StepFormat::fmt(&axis, axis_idx, f)?;
        StepFormat::fmt(&ref_direction, ref_direction_idx, f)
    }
}
impl_const_step_length!(MatrixAsAxis<Matrix4>, 4);

impl<P> StepFormat for Line<P>
where
    P: EuclideanSpace + ConstStepLength + StepFormat,
    P::Diff: StepFormat,
{
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let pnt_idx = idx + 1;
        let dir_idx = idx + 1 + P::LENGTH;
        f.write_fmt(format_args!(
            "#{idx} = LINE('', #{pnt_idx}, #{dir_idx});\n{pnt}{dir}",
            pnt = StepDisplay::new(self.0, pnt_idx),
            dir = StepDisplay::new(self.1 - self.0, dir_idx),
        ))
    }
}

impl<P> StepLength for Line<P>
where
    P: EuclideanSpace + ConstStepLength,
    P::Diff: ConstStepLength,
{
    #[inline(always)]
    fn step_length(&self) -> usize { <Self as ConstStepLength>::LENGTH }
}

impl<P> ConstStepLength for Line<P>
where
    P: EuclideanSpace + ConstStepLength,
    P::Diff: ConstStepLength,
{
    const LENGTH: usize = 1 + P::LENGTH + P::Diff::LENGTH;
}

impl<P> StepCurve for Line<P> {
    fn step_reversed_parameter_chart(&self) -> Option<ParameterChart> {
        ParameterChart::reflection((0.0, 0.0))
    }
}

impl<P> StepFormat for PolylineCurve<P>
where P: Copy + ConstStepLength + StepFormat
{
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        f.write_fmt(format_args!(
            "#{idx} = POLYLINE('', {range});\n",
            range = IndexSliceDisplay(idx + 1..=idx + self.0.len())
        ))?;
        let closure = |(i, p): (usize, &P)| p.fmt(idx + 1 + i * P::LENGTH, f);
        self.0.iter().enumerate().try_for_each(closure)
    }
}

impl<P: ConstStepLength> StepLength for PolylineCurve<P> {
    #[inline(always)]
    fn step_length(&self) -> usize { 1 + self.0.len() * P::LENGTH }
}

impl<P> StepCurve for PolylineCurve<P> {
    fn step_reversed_parameter_chart(&self) -> Option<ParameterChart> {
        ParameterChart::reflection((0.0, self.len().checked_sub(1)? as f64))
    }
}

impl<P> StepFormat for BsplineCurve<P>
where P: Copy + ConstStepLength + StepFormat
{
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let (knots, multi) = self.knot_vector().to_single_multi();
        let control_points_instances = self
            .control_points()
            .iter()
            .enumerate()
            .map(|(i, p)| StepDisplay::new(*p, idx + 1 + i * P::LENGTH))
            .collect::<Vec<_>>();
        f.write_fmt(format_args!(
            "#{idx} = B_SPLINE_CURVE_WITH_KNOTS('', {degree}, {control_points_list}, .UNSPECIFIED., .U., .U., {knot_multiplicities}, {knots}, .UNSPECIFIED.);\n{control_points_instances}",
            degree = self.degree(),
            control_points_list = IndexSliceDisplay((idx + 1..=idx + self.control_points().len() * P::LENGTH).step_by(P::LENGTH)),
			knot_multiplicities = SliceDisplay(&multi),
            knots = SliceDisplay(&knots),
            control_points_instances = SliceDisplay(&control_points_instances),
		))
    }
}

impl<P> StepLength for BsplineCurve<P> {
    #[inline(always)]
    fn step_length(&self) -> usize { self.control_points().len() + 1 }
}

impl<P> StepCurve for BsplineCurve<P> {
    fn step_reversed_parameter_chart(&self) -> Option<ParameterChart> {
        ParameterChart::reflection((
            *self.knot_vector().get(self.degree())?,
            *self.knot_vector().get(self.control_points().len())?,
        ))
    }
}

impl<V> StepFormat for NurbsCurve<V>
where
    V: Homogeneous<Scalar = f64>,
    V::Point: ConstStepLength + StepFormat,
{
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let (knots, multi) = self.knot_vector().to_single_multi();
        let control_points_instances = self
            .control_points()
            .iter()
            .enumerate()
            .map(|(i, v)| StepDisplay::new(v.to_point(), idx + 1 + i * V::Point::LENGTH))
            .collect::<Vec<_>>();
        let weights = self
            .control_points()
            .iter()
            .map(|v| v.weight())
            .collect::<Vec<_>>();
        f.write_fmt(format_args!(
            "#{idx} = (
    BOUNDED_CURVE()
    B_SPLINE_CURVE({degree}, {control_points_list}, .UNSPECIFIED., .U., .U.)
    B_SPLINE_CURVE_WITH_KNOTS({knot_multiplicities}, {knots}, .UNSPECIFIED.)
    CURVE()
    GEOMETRIC_REPRESENTATION_ITEM()
    RATIONAL_B_SPLINE_CURVE({weights})
    REPRESENTATION_ITEM('')
);\n{control_points_instances}",
            degree = self.degree(),
            control_points_list = IndexSliceDisplay(
                (idx + 1..=idx + self.control_points().len() * V::Point::LENGTH)
                    .step_by(V::Point::LENGTH)
            ),
            knot_multiplicities = SliceDisplay(&multi),
            knots = SliceDisplay(&knots),
            weights = SliceDisplay(&weights),
            control_points_instances = SliceDisplay(&control_points_instances),
        ))
    }
}

impl<V> StepLength for NurbsCurve<V> {
    #[inline(always)]
    fn step_length(&self) -> usize { self.control_points().len() + 1 }
}

impl<V> StepCurve for NurbsCurve<V> {
    fn step_reversed_parameter_chart(&self) -> Option<ParameterChart> {
        ParameterChart::reflection((
            *self.knot_vector().get(self.degree())?,
            *self.knot_vector().get(self.control_points().len())?,
        ))
    }
}

impl StepFormat for Processor<TrimmedCurve<UnitCircle<Point2>>, Matrix3> {
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let transform = *self.transform();
        conic_frame_orientation_2d(&transform).ok_or(std::fmt::Error)?;
        let position_idx = idx + 1;
        let location_idx = idx + 2;
        let ref_direction_idx = idx + 3;
        let r0 = transform[0].magnitude();
        let r1 = transform[1].magnitude();
        let ref_direction = VectorAsDirection(transform[0].truncate() / r0);
        let location = transform[2].to_point();
        if r0 == r1 {
            let r = FloatDisplay(r0);
            f.write_fmt(format_args!("#{idx} = CIRCLE('', #{position_idx}, {r});\n"))?;
        } else {
            let (r0, r1) = (FloatDisplay(r0), FloatDisplay(r1));
            f.write_fmt(format_args!(
                "#{idx} = ELLIPSE('', #{position_idx}, {r0}, {r1});\n"
            ))?;
        }
        f.write_fmt(format_args!(
            "#{position_idx} = AXIS2_PLACEMENT_2D('', #{location_idx}, #{ref_direction_idx});\n",
        ))?;
        StepFormat::fmt(&location, location_idx, f)?;
        StepFormat::fmt(&ref_direction, ref_direction_idx, f)
    }
}
impl_const_step_length!(Processor<TrimmedCurve<UnitCircle<Point2>>, Matrix3>, 4);

impl StepFormat for Processor<TrimmedCurve<UnitCircle<Point3>>, Matrix4> {
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let transform = self.transform();
        let position_idx = idx + 1;
        let location_idx = idx + 2;
        let axis_idx = idx + 3;
        let ref_direction_idx = idx + 4;
        let location = transform[3].to_point();
        let axis = VectorAsDirection(transform[2].truncate().normalize());
        let r0 = transform[0].magnitude();
        let r1 = transform[1].magnitude();
        let ref_direction = VectorAsDirection(transform[0].truncate() / r0);
        if r0 == r1 {
            let r = FloatDisplay(r0);
            f.write_fmt(format_args!("#{idx} = CIRCLE('', #{position_idx}, {r});\n"))?;
        } else {
            let (r0, r1) = (FloatDisplay(r0), FloatDisplay(r1));
            f.write_fmt(format_args!(
                "#{idx} = ELLIPSE('', #{position_idx}, {r0}, {r1});\n"
            ))?;
        }
        f.write_fmt(format_args!(
            "#{position_idx} = AXIS2_PLACEMENT_3D('', #{location_idx}, #{axis_idx}, #{ref_direction_idx});\n",
        ))?;
        StepFormat::fmt(&location, location_idx, f)?;
        StepFormat::fmt(&axis, axis_idx, f)?;
        StepFormat::fmt(&ref_direction, ref_direction_idx, f)
    }
}
impl_const_step_length!(Processor<TrimmedCurve<UnitCircle<Point3>>, Matrix4>, 5);

impl StepFormat for Processor<TrimmedCurve<UnitHyperbola<Point2>>, Matrix3> {
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let transform = *self.transform();
        conic_frame_orientation_2d(&transform).ok_or(std::fmt::Error)?;
        let position_idx = idx + 1;
        let location_idx = idx + 2;
        let ref_direction_idx = idx + 3;
        let r0 = transform[0].magnitude();
        let r1 = transform[1].magnitude();
        let ref_direction_raw = VectorAsDirection(transform[0].truncate() / r0);
        let ref_direction = StepDisplay::new(ref_direction_raw, ref_direction_idx);
        let location = StepDisplay::new(transform[2].to_point(), location_idx);
        let (r0, r1) = (FloatDisplay(r0), FloatDisplay(r1));
        f.write_fmt(format_args!(
            "#{idx} = HYPERBOLA('', #{position_idx}, {r0}, {r1});
#{position_idx} = AXIS2_PLACEMENT_2D('', #{location_idx}, #{ref_direction_idx});
{location}{ref_direction}"
        ))
    }
}
impl_const_step_length!(Processor<TrimmedCurve<UnitHyperbola<Point2>>, Matrix3>, 4);

impl StepFormat for Processor<TrimmedCurve<UnitHyperbola<Point3>>, Matrix4> {
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let transform = self.transform();
        let position_idx = idx + 1;
        let location_idx = idx + 2;
        let axis_idx = idx + 3;
        let ref_direction_idx = idx + 4;
        let location = StepDisplay::new(transform[3].to_point(), location_idx);
        let axis_raw = VectorAsDirection(transform[2].truncate().normalize());
        let axis = StepDisplay::new(axis_raw, axis_idx);
        let r0 = transform[0].magnitude();
        let r1 = transform[1].magnitude();
        let ref_direction_raw = VectorAsDirection(transform[0].truncate() / r0);
        let ref_direction = StepDisplay::new(ref_direction_raw, ref_direction_idx);
        let (r0, r1) = (FloatDisplay(r0), FloatDisplay(r1));
        f.write_fmt(format_args!(
            "#{idx} = HYPERBOLA('', #{position_idx}, {r0}, {r1});
#{position_idx} = AXIS2_PLACEMENT_3D('', #{location_idx}, #{axis_idx}, #{ref_direction_idx});
{location}{axis}{ref_direction}"
        ))
    }
}
impl_const_step_length!(Processor<TrimmedCurve<UnitHyperbola<Point3>>, Matrix4>, 5);

impl StepFormat for Processor<TrimmedCurve<UnitParabola<Point2>>, Matrix3> {
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let transform = *self.transform();
        conic_frame_orientation_2d(&transform).ok_or(std::fmt::Error)?;
        let position_idx = idx + 1;
        let location_idx = idx + 2;
        let ref_direction_idx = idx + 3;
        let r0 = transform[0].magnitude();
        let r1 = transform[1].magnitude();
        let focal_dist = FloatDisplay(r1 * r1 / r0);
        let ref_direction_raw = VectorAsDirection(transform[0].truncate() / r0);
        let ref_direction = StepDisplay::new(ref_direction_raw, ref_direction_idx);
        let location = StepDisplay::new(transform[2].to_point(), location_idx);
        f.write_fmt(format_args!(
            "#{idx} = PARABOLA('', #{position_idx}, {focal_dist});
#{position_idx} = AXIS2_PLACEMENT_2D('', #{location_idx}, #{ref_direction_idx});
{location}{ref_direction}"
        ))
    }
}
impl_const_step_length!(Processor<TrimmedCurve<UnitParabola<Point2>>, Matrix3>, 4);

impl StepFormat for Processor<TrimmedCurve<UnitParabola<Point3>>, Matrix4> {
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let transform = self.transform();
        let position_idx = idx + 1;
        let location_idx = idx + 2;
        let axis_idx = idx + 3;
        let ref_direction_idx = idx + 4;
        let location = StepDisplay::new(transform[3].to_point(), location_idx);
        let axis_raw = VectorAsDirection(transform[2].truncate().normalize());
        let axis = StepDisplay::new(axis_raw, axis_idx);
        let r0 = transform[0].magnitude();
        let r1 = transform[1].magnitude();
        let focal_dist = FloatDisplay(r1 * r1 / r0);
        let ref_direction_raw = VectorAsDirection(transform[0].truncate() / r0);
        let ref_direction = StepDisplay::new(ref_direction_raw, ref_direction_idx);
        f.write_fmt(format_args!(
            "#{idx} = PARABOLA('', #{position_idx}, {focal_dist});
#{position_idx} = AXIS2_PLACEMENT_3D('', #{location_idx}, #{axis_idx}, #{ref_direction_idx});
{location}{axis}{ref_direction}"
        ))
    }
}
impl_const_step_length!(Processor<TrimmedCurve<UnitParabola<Point3>>, Matrix4>, 5);

fn circle_reversed_parameter_chart() -> Option<ParameterChart> {
    let period = std::f64::consts::TAU;
    ParameterChart::new(
        -1.0,
        period,
        ParameterEnclosure::new(period.next_down(), period.next_up())?,
    )
}

fn exact_dyadic_product(a: f64, b: f64) -> (bool, u128, i32) {
    let parts = |value: f64| {
        let bits = value.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & ((1u64 << 52) - 1);
        if exponent == 0 {
            (fraction, -1074)
        } else {
            (fraction | (1u64 << 52), exponent - 1075)
        }
    };
    let (am, ae) = parts(a);
    let (bm, be) = parts(b);
    let product = u128::from(am) * u128::from(bm);
    if product == 0 {
        return (false, 0, 0);
    }
    let zeros = product.trailing_zeros();
    (
        a.is_sign_negative() != b.is_sign_negative(),
        product >> zeros,
        ae + be + zeros as i32,
    )
}

// AXIS2_PLACEMENT_2D supplies positive-handed Y. A reflected native frame
// therefore changes the conic parameter, even when Processor is forward.
fn conic_frame_orientation_2d(matrix: &Matrix3) -> Option<f64> {
    if !(0..3).all(|column| (0..3).all(|row| matrix[column][row].is_finite()))
        || matrix[0].z != 0.0
        || matrix[1].z != 0.0
        || matrix[2].z != 1.0
    {
        return None;
    }
    let x = matrix[0].truncate();
    let y = matrix[1].truncate();
    let radii = [x.magnitude(), y.magnitude()];
    if radii
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || exact_dyadic_product(x.x, y.x) != exact_dyadic_product(-x.y, y.y)
    {
        return None;
    }
    let positive = if x.x != 0.0 {
        x.x.is_sign_negative() == y.y.is_sign_negative()
    } else {
        x.y.is_sign_negative() != y.x.is_sign_negative()
    };
    Some(if positive { 1.0 } else { -1.0 })
}

fn parabola_frame_orientation_2d(matrix: &Matrix3) -> Option<f64> {
    let orientation = conic_frame_orientation_2d(matrix)?;
    let x = matrix[0].truncate();
    let y = matrix[1].truncate();
    // Anisotropic scaling changes STEP's dimensionless parabola parameter.
    // This orientation-only chart deliberately refuses that unproved scale.
    if y.x != -orientation * x.y || y.y != orientation * x.x {
        return None;
    }
    Some(orientation)
}

fn uniform_axis_parabola_frame_3d(matrix: &Matrix4) -> bool {
    if !(0..4).all(|column| (0..4).all(|row| matrix[column][row].is_finite()))
        || (0..3).any(|column| matrix[column].w != 0.0)
        || matrix[3].w != 1.0
    {
        return false;
    }
    let x = matrix[0].truncate();
    let y = matrix[1].truncate();
    let ix = (0..3).find(|&index| x[index] != 0.0);
    let iy = (0..3).find(|&index| y[index] != 0.0);
    let (Some(ix), Some(iy)) = (ix, iy) else {
        return false;
    };
    ix != iy
        && (0..3).all(|index| (index == ix || x[index] == 0.0) && (index == iy || y[index] == 0.0))
        && x[ix].abs() == y[iy].abs()
}

macro_rules! primitive_processor_step_curve {
    ($curve:ident, $reversal:expr, $parabola:expr, $period:expr) => {
        impl StepCurve for Processor<TrimmedCurve<$curve<Point2>>, Matrix3> {
            fn same_sense(&self) -> bool {
                self.step_parameter_chart()
                    .is_some_and(|chart| chart.scale > 0.0)
            }
            fn step_parameter_chart(&self) -> Option<ParameterChart> {
                let orientation = if $parabola {
                    parabola_frame_orientation_2d(self.transform())?
                } else {
                    conic_frame_orientation_2d(self.transform())?
                };
                let frame = ParameterChart::new(orientation, 0.0, ParameterEnclosure::point(0.0)?)?;
                let local = if self.orientation() {
                    ParameterChart::identity()
                } else {
                    ParameterChart::reflection(self.entity().range())?
                };
                frame.compose(local)
            }
            fn step_reversed_parameter_chart(&self) -> Option<ParameterChart> { $reversal }
            fn step_parameter_period(&self) -> Option<f64> { $period }
        }
        impl StepCurve for Processor<TrimmedCurve<$curve<Point3>>, Matrix4> {
            fn same_sense(&self) -> bool { self.orientation() }
            fn step_parameter_chart(&self) -> Option<ParameterChart> {
                if $parabola && !uniform_axis_parabola_frame_3d(self.transform()) {
                    return None;
                }
                if self.orientation() {
                    Some(ParameterChart::identity())
                } else {
                    ParameterChart::reflection(self.entity().range())
                }
            }
            fn step_reversed_parameter_chart(&self) -> Option<ParameterChart> { $reversal }
            fn step_parameter_period(&self) -> Option<f64> { $period }
        }
    };
}
primitive_processor_step_curve!(
    UnitCircle,
    circle_reversed_parameter_chart(),
    false,
    Some(std::f64::consts::TAU)
);
primitive_processor_step_curve!(
    UnitHyperbola,
    ParameterChart::reflection((0.0, 0.0)),
    false,
    None
);
primitive_processor_step_curve!(
    UnitParabola,
    ParameterChart::reflection((0.0, 0.0)),
    true,
    None
);

fn directed_trim_header(
    start: f64,
    end: f64,
    forward: bool,
    idx: usize,
    basis: usize,
    f: &mut Formatter<'_>,
) -> Result {
    f.write_fmt(format_args!(
        "#{idx} = TRIMMED_CURVE('', #{basis}, (PARAMETER_VALUE({})), \
         (PARAMETER_VALUE({})), {}, .PARAMETER.);\n",
        FloatDisplay(start),
        FloatDisplay(end),
        if forward { ".T." } else { ".F." },
    ))
}

fn finish_trim_chart<C: StepCurve>(
    basis: &C,
    mut chart: ParameterChart,
    range: (f64, f64),
    period: Option<f64>,
) -> Option<ParameterChart> {
    if let Some(period) = period {
        let first = chart.nominal(range.0)?;
        let second = chart.nominal(range.1)?;
        let lower = first.min(second);
        let length = first.max(second) - lower;
        if !period.is_finite()
            || period <= 0.0
            || !length.is_finite()
            || length <= 0.0
            || length > period.next_up()
        {
            return None;
        }
        // STEP AdjustPeriodic happens BEFORE root Reverse. In particular a
        // reflected circle with negative selectors must not acquire a spurious
        // extra period in the emitted TRIMMED_CURVE parameter chart.
        let normalize = crate::step::periodic_parameter_shift(-(lower / period).floor(), period)?;
        chart = normalize.compose(chart)?;
    }
    if chart.scale > 0.0 {
        Some(chart)
    } else {
        basis.step_reversed_parameter_chart()?.compose(chart)
    }
}

fn intrinsic_basis_chart<C: StepCurve + BoundedCurve>(basis: &C) -> Option<ParameterChart> {
    let chart = basis.step_parameter_chart()?;
    if basis.same_sense() {
        Some(chart)
    } else {
        finish_trim_chart(
            basis,
            chart,
            basis.range_tuple(),
            basis.step_parameter_period(),
        )
    }
}

impl<C, M> StepFormat for Processor<TrimmedCurve<Box<C>>, M>
where
    C: StepFormat + StepLength + StepCurve + BoundedCurve + Transformed<M>,
    M: Copy + One + PartialEq,
{
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let original = self.entity().curve().as_ref();
        let transformed = if *self.transform() == M::one() {
            None
        } else {
            Some(original.transformed(*self.transform()))
        };
        let basis = transformed.as_ref().unwrap_or(original);
        let basis_idx = idx + 1;
        let local = if self.orientation() {
            ParameterChart::identity()
        } else {
            ParameterChart::reflection(self.entity().range()).ok_or(std::fmt::Error)?
        };
        let chart = intrinsic_basis_chart(basis)
            .and_then(|chart| chart.compose(local))
            .ok_or(std::fmt::Error)?;
        let range = self.entity().range();
        let period = if basis.same_sense() {
            basis.step_parameter_period()
        } else {
            None
        };
        finish_trim_chart(basis, chart, range, period).ok_or(std::fmt::Error)?;
        directed_trim_header(
            chart.nominal(range.0).ok_or(std::fmt::Error)?,
            chart.nominal(range.1).ok_or(std::fmt::Error)?,
            chart.scale > 0.0,
            idx,
            basis_idx,
            f,
        )?;
        if basis.same_sense() {
            StepFormat::fmt(basis, basis_idx, f)
        } else {
            let chart = basis.step_parameter_chart().ok_or(std::fmt::Error)?;
            let range = basis.range_tuple();
            directed_trim_header(
                chart.nominal(range.0).ok_or(std::fmt::Error)?,
                chart.nominal(range.1).ok_or(std::fmt::Error)?,
                chart.scale > 0.0,
                basis_idx,
                basis_idx + 1,
                f,
            )?;
            StepFormat::fmt(basis, basis_idx + 1, f)
        }
    }
}

impl<C, M> StepLength for Processor<TrimmedCurve<Box<C>>, M>
where
    C: StepLength + StepCurve + Transformed<M>,
    M: Copy + One + PartialEq,
{
    fn step_length(&self) -> usize {
        let original = self.entity().curve().as_ref();
        let transformed = if *self.transform() == M::one() {
            None
        } else {
            Some(original.transformed(*self.transform()))
        };
        let basis = transformed.as_ref().unwrap_or(original);
        1 + usize::from(!basis.same_sense()) + basis.step_length()
    }
}

impl<C, M> StepCurve for Processor<TrimmedCurve<Box<C>>, M>
where
    C: StepCurve + BoundedCurve + Transformed<M>,
    M: Copy + One + PartialEq,
{
    // Both the active restriction and its direction are emitted in the entity.
    fn same_sense(&self) -> bool { true }
    fn step_parameter_chart(&self) -> Option<ParameterChart> {
        let original = self.entity().curve().as_ref();
        let transformed = if *self.transform() == M::one() {
            None
        } else {
            Some(original.transformed(*self.transform()))
        };
        let basis = transformed.as_ref().unwrap_or(original);
        let local = if self.orientation() {
            ParameterChart::identity()
        } else {
            ParameterChart::reflection(self.entity().range())?
        };
        let chart = intrinsic_basis_chart(basis)?.compose(local)?;
        let period = if basis.same_sense() {
            basis.step_parameter_period()
        } else {
            None
        };
        finish_trim_chart(basis, chart, self.entity().range(), period)
    }
    fn step_reversed_parameter_chart(&self) -> Option<ParameterChart> {
        let original = self.entity().curve().as_ref();
        if *self.transform() == M::one() {
            original.step_reversed_parameter_chart()
        } else {
            original
                .transformed(*self.transform())
                .step_reversed_parameter_chart()
        }
    }
}

impl<C, S0, S1> StepFormat for IntersectionCurve<C, S0, S1>
where
    C: StepLength + StepFormat,
    S0: StepLength + StepFormat,
    S1: StepLength + StepFormat,
{
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let curve_idx = idx + 1;
        let surface0_idx = curve_idx + self.leader().step_length();
        let surface1_idx = surface0_idx + self.surface0().step_length();
        f.write_fmt(format_args!(
            "#{idx} = INTERSECTION_CURVE('', #{curve_idx}, (#{surface0_idx}, #{surface1_idx}), .CURVE_3D.);\n"
        ))?;
        self.leader().fmt(curve_idx, f)?;
        self.surface0().fmt(surface0_idx, f)?;
        self.surface1().fmt(surface1_idx, f)
    }
}

impl<C: StepLength, S0: StepLength, S1: StepLength> StepLength for IntersectionCurve<C, S0, S1> {
    #[inline(always)]
    fn step_length(&self) -> usize {
        1 + self.leader().step_length()
            + self.surface0().step_length()
            + self.surface1().step_length()
    }
}

impl<C, S0, S1, T0, T1> StepFormat for SurfaceCurve<C, S0, S1, T0, T1>
where
    C: StepLength + StepFormat,
    S0: StepLength + StepFormat,
    S1: StepLength + StepFormat,
    T0: StepLength + StepFormat,
    T1: StepLength + StepFormat,
{
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let curve_idx = idx + 1;
        let boundary0_idx = curve_idx + self.leader().step_length();
        let boundary1_idx = boundary0_idx + self.boundary0().map_or(0, StepLength::step_length);
        let surface0_idx = boundary1_idx + self.boundary1().map_or(0, StepLength::step_length);
        let surface1_idx = surface0_idx + self.surface0().step_length();
        let assoc0 = self
            .boundary0()
            .map(|_| format!("#{boundary0_idx}"))
            .unwrap_or_else(|| format!("#{surface0_idx}"));
        let assoc1 = self
            .boundary1()
            .map(|_| format!("#{boundary1_idx}"))
            .unwrap_or_else(|| format!("#{surface1_idx}"));
        f.write_fmt(format_args!(
            "#{idx} = INTERSECTION_CURVE('', #{curve_idx}, ({assoc0}, {assoc1}), .CURVE_3D.);\n"
        ))?;
        self.leader().fmt(curve_idx, f)?;
        if let Some(boundary) = self.boundary0() {
            boundary.fmt(boundary0_idx, f)?;
        }
        if let Some(boundary) = self.boundary1() {
            boundary.fmt(boundary1_idx, f)?;
        }
        self.surface0().fmt(surface0_idx, f)?;
        self.surface1().fmt(surface1_idx, f)
    }
}

impl<C, S0, S1, T0, T1> StepLength for SurfaceCurve<C, S0, S1, T0, T1>
where
    C: StepLength,
    S0: StepLength,
    S1: StepLength,
    T0: StepLength,
    T1: StepLength,
{
    fn step_length(&self) -> usize {
        1 + self.leader().step_length()
            + self.boundary0().map_or(0, StepLength::step_length)
            + self.boundary1().map_or(0, StepLength::step_length)
            + self.surface0().step_length()
            + self.surface1().step_length()
    }
}

impl<C, S0, S1> ConstStepLength for IntersectionCurve<C, S0, S1>
where
    C: ConstStepLength,
    S0: ConstStepLength,
    S1: ConstStepLength,
{
    const LENGTH: usize = 1 + C::LENGTH + S0::LENGTH + S1::LENGTH;
}

impl<C: StepCurve, S0, S1> StepCurve for IntersectionCurve<C, S0, S1> {
    #[inline(always)]
    fn same_sense(&self) -> bool { self.leader().same_sense() }
    fn step_parameter_chart(&self) -> Option<ParameterChart> {
        self.leader().step_parameter_chart()
    }
    fn step_reversed_parameter_chart(&self) -> Option<ParameterChart> {
        self.leader().step_reversed_parameter_chart()
    }
    fn step_parameter_period(&self) -> Option<f64> { self.leader().step_parameter_period() }
}
impl<C: StepCurve, S0, S1, T0, T1> StepCurve for SurfaceCurve<C, S0, S1, T0, T1> {
    // SurfaceCurve::invert reverses the leader and both optional boundaries;
    // there is no independent orientation flag.
    fn same_sense(&self) -> bool { self.leader().same_sense() }
    fn step_parameter_chart(&self) -> Option<ParameterChart> {
        self.leader().step_parameter_chart()
    }
    fn step_reversed_parameter_chart(&self) -> Option<ParameterChart> {
        self.leader().step_reversed_parameter_chart()
    }
    fn step_parameter_period(&self) -> Option<f64> { self.leader().step_parameter_period() }
}

impl<C, S> StepFormat for ParameterCurve<C, S>
where
    C: StepFormat + Transformed<Matrix3>,
    S: StepFormat + StepLength + StepSurface,
{
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let chart = self
            .surface()
            .step_parameter_chart()
            .ok_or(std::fmt::Error)?;
        fmt_parameter_curve_with_chart(self.curve(), self.surface(), chart, idx, f)
    }
}

/// Writes stored UV using the caller's explicit, bounded chart convention.
///
/// Generic native parameter curves pass internal UV; source parameter-curve
/// wrappers pass raw STEP UV. Neither convention is inferred from `S`'s type.
pub(crate) fn fmt_parameter_curve_with_chart<C, S>(
    curve: &C,
    surface: &S,
    chart: SurfaceParameterChart,
    idx: usize,
    f: &mut Formatter<'_>,
) -> Result
where
    C: StepFormat + Transformed<Matrix3>,
    S: StepFormat + StepLength,
{
    let surface_idx = idx + 1;
    let repr_idx = surface_idx + surface.step_length();
    let context_idx = repr_idx + 1;
    let curve_idx = repr_idx + 2;
    with_pcurve_step_curve(curve, chart, |curve| {
        let curve = StepDisplay::new(curve, curve_idx);
        let surface = StepDisplay::new(surface, surface_idx);
        f.write_fmt(format_args!(
            "#{idx} = PCURVE('', #{surface_idx}, #{repr_idx});
{surface}#{repr_idx} = DEFINITIONAL_REPRESENTATION('', (#{curve_idx}), #{context_idx});
#{context_idx} = (
    GEOMETRIC_REPRESENTATION_CONTEXT(2)
    PARAMETRIC_REPRESENTATION_CONTEXT()
    REPRESENTATION_CONTEXT('2D SPACE', '')
);
{curve}"
        ))
    })
    .ok_or(std::fmt::Error)?
}

/// Borrows the same emitted UV view for scalar charts, entity counts and writing.
pub(crate) fn with_pcurve_step_curve<C: Transformed<Matrix3>, R>(
    curve: &C,
    chart: SurfaceParameterChart,
    visitor: impl FnOnce(&C) -> R,
) -> Option<R> {
    chart.enclose([ParameterEnclosure::point(0.0)?; 2])?;
    let jacobian = chart.jacobian();
    let matrix = Matrix3::from_cols(
        Vector3::new(jacobian[0][0], jacobian[1][0], 0.0),
        Vector3::new(jacobian[0][1], jacobian[1][1], 0.0),
        Vector3::new(chart.axes[0].offset, chart.axes[1].offset, 1.0),
    );
    let mapped = if matrix == Matrix3::identity() {
        None
    } else {
        Some(curve.transformed(matrix))
    };
    Some(visitor(mapped.as_ref().unwrap_or(curve)))
}

impl<C, S> StepLength for ParameterCurve<C, S>
where
    C: StepLength + Transformed<Matrix3>,
    S: StepLength + StepSurface,
{
    fn step_length(&self) -> usize {
        let length = self
            .surface()
            .step_parameter_chart()
            .and_then(|chart| with_pcurve_step_curve(self.curve(), chart, StepLength::step_length))
            .unwrap_or_else(|| self.curve().step_length());
        3 + length + self.surface().step_length()
    }
}

impl<C, S> ConstStepLength for ParameterCurve<C, S>
where
    C: ConstStepLength + Transformed<Matrix3>,
    S: ConstStepLength + StepSurface,
{
    const LENGTH: usize = 3 + C::LENGTH + S::LENGTH;
}

impl<C, S> StepCurve for ParameterCurve<C, S>
where
    C: StepCurve + Transformed<Matrix3>,
    S: StepSurface,
{
    fn same_sense(&self) -> bool {
        self.surface()
            .step_parameter_chart()
            .and_then(|chart| with_pcurve_step_curve(self.curve(), chart, StepCurve::same_sense))
            .unwrap_or(false)
    }
    fn step_parameter_chart(&self) -> Option<ParameterChart> {
        self.surface()
            .step_parameter_chart()
            .and_then(|chart| {
                with_pcurve_step_curve(self.curve(), chart, StepCurve::step_parameter_chart)
            })
            .flatten()
    }
    fn step_reversed_parameter_chart(&self) -> Option<ParameterChart> {
        self.surface()
            .step_parameter_chart()
            .and_then(|chart| {
                with_pcurve_step_curve(
                    self.curve(),
                    chart,
                    StepCurve::step_reversed_parameter_chart,
                )
            })
            .flatten()
    }
    fn step_parameter_period(&self) -> Option<f64> {
        self.surface()
            .step_parameter_chart()
            .and_then(|chart| {
                with_pcurve_step_curve(self.curve(), chart, StepCurve::step_parameter_period)
            })
            .flatten()
    }
}

impl StepFormat for ModelingConic2D {
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        self.step_parameter_chart().ok_or(std::fmt::Error)?;
        let (range, chart) = match self {
            Self::Ellipse(curve) => (curve.range_tuple(), curve.step_parameter_chart()),
            Self::Hyperbola(curve) => (curve.range_tuple(), curve.step_parameter_chart()),
            Self::Parabola(curve) => (curve.range_tuple(), curve.step_parameter_chart()),
        };
        let chart = chart.ok_or(std::fmt::Error)?;
        let basis = idx + 1;
        directed_trim_header(
            chart.nominal(range.0).ok_or(std::fmt::Error)?,
            chart.nominal(range.1).ok_or(std::fmt::Error)?,
            chart.scale > 0.0,
            idx,
            basis,
            f,
        )?;
        match self {
            ModelingConic2D::Ellipse(x) => StepFormat::fmt(x, basis, f),
            ModelingConic2D::Hyperbola(x) => StepFormat::fmt(x, basis, f),
            ModelingConic2D::Parabola(x) => StepFormat::fmt(x, basis, f),
        }
    }
}

impl StepLength for ModelingConic2D {
    fn step_length(&self) -> usize {
        match self {
            ModelingConic2D::Ellipse(x) => 1 + x.step_length(),
            ModelingConic2D::Hyperbola(x) => 1 + x.step_length(),
            ModelingConic2D::Parabola(x) => 1 + x.step_length(),
        }
    }
}

impl StepCurve for ModelingConic2D {
    // TRIMMED_CURVE encodes the profile direction in the emitted entity itself.
    fn same_sense(&self) -> bool { true }
    fn step_parameter_chart(&self) -> Option<ParameterChart> {
        match self {
            Self::Ellipse(curve) => finish_trim_chart(
                curve,
                curve.step_parameter_chart()?,
                curve.range_tuple(),
                curve.step_parameter_period(),
            ),
            Self::Hyperbola(curve) => finish_trim_chart(
                curve,
                curve.step_parameter_chart()?,
                curve.range_tuple(),
                curve.step_parameter_period(),
            ),
            Self::Parabola(curve) => finish_trim_chart(
                curve,
                curve.step_parameter_chart()?,
                curve.range_tuple(),
                curve.step_parameter_period(),
            ),
        }
    }
    fn step_reversed_parameter_chart(&self) -> Option<ParameterChart> {
        match self {
            Self::Ellipse(curve) => curve.step_reversed_parameter_chart(),
            Self::Hyperbola(curve) => curve.step_reversed_parameter_chart(),
            Self::Parabola(curve) => curve.step_reversed_parameter_chart(),
        }
    }
}

impl StepFormat for ModelingCurve2D {
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        match self {
            ModelingCurve2D::Line(x) => StepFormat::fmt(x, idx, f),
            ModelingCurve2D::Polyline(x) => StepFormat::fmt(x, idx, f),
            ModelingCurve2D::Conic(x) => StepFormat::fmt(x, idx, f),
            ModelingCurve2D::BsplineCurve(x) => StepFormat::fmt(x, idx, f),
            ModelingCurve2D::NurbsCurve(x) => StepFormat::fmt(x, idx, f),
            ModelingCurve2D::TrimmedCurve(x) => StepFormat::fmt(x, idx, f),
        }
    }
}

impl StepLength for ModelingCurve2D {
    fn step_length(&self) -> usize {
        match self {
            ModelingCurve2D::Line(_) => Line::<Point2>::LENGTH,
            ModelingCurve2D::Polyline(x) => x.step_length(),
            ModelingCurve2D::Conic(x) => x.step_length(),
            ModelingCurve2D::BsplineCurve(x) => x.step_length(),
            ModelingCurve2D::NurbsCurve(x) => x.step_length(),
            ModelingCurve2D::TrimmedCurve(x) => x.step_length(),
        }
    }
}

impl StepCurve for ModelingCurve2D {
    fn same_sense(&self) -> bool {
        match self {
            Self::Conic(curve) => curve.same_sense(),
            Self::TrimmedCurve(curve) => curve.same_sense(),
            _ => true,
        }
    }
    fn step_parameter_chart(&self) -> Option<ParameterChart> {
        match self {
            Self::Line(curve) => curve.step_parameter_chart(),
            Self::Polyline(curve) => curve.step_parameter_chart(),
            Self::Conic(curve) => curve.step_parameter_chart(),
            Self::BsplineCurve(curve) => curve.step_parameter_chart(),
            Self::NurbsCurve(curve) => curve.step_parameter_chart(),
            Self::TrimmedCurve(curve) => curve.step_parameter_chart(),
        }
    }
    fn step_reversed_parameter_chart(&self) -> Option<ParameterChart> {
        match self {
            Self::Line(curve) => curve.step_reversed_parameter_chart(),
            Self::Polyline(curve) => curve.step_reversed_parameter_chart(),
            Self::Conic(curve) => curve.step_reversed_parameter_chart(),
            Self::BsplineCurve(curve) => curve.step_reversed_parameter_chart(),
            Self::NurbsCurve(curve) => curve.step_reversed_parameter_chart(),
            Self::TrimmedCurve(curve) => curve.step_reversed_parameter_chart(),
        }
    }
}

impl StepFormat for ModelingCurve {
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        match self {
            ModelingCurve::Line(x) => StepFormat::fmt(x, idx, f),
            ModelingCurve::BsplineCurve(x) => StepFormat::fmt(x, idx, f),
            ModelingCurve::NurbsCurve(x) => StepFormat::fmt(x, idx, f),
            ModelingCurve::TrimmedCurve(x) => StepFormat::fmt(x, idx, f),
            ModelingCurve::ParameterCurve(x) => StepFormat::fmt(x, idx, f),
            ModelingCurve::IntersectionCurve(x) => StepFormat::fmt(x, idx, f),
        }
    }
}

impl StepLength for ModelingCurve {
    fn step_length(&self) -> usize {
        match self {
            ModelingCurve::Line(_) => Line::<Point3>::LENGTH,
            ModelingCurve::BsplineCurve(x) => x.step_length(),
            ModelingCurve::NurbsCurve(x) => x.step_length(),
            ModelingCurve::TrimmedCurve(x) => x.step_length(),
            ModelingCurve::ParameterCurve(x) => x.step_length(),
            ModelingCurve::IntersectionCurve(x) => x.step_length(),
        }
    }
}

/// Preserves an affine pcurve chart without normalizing its plane axes.
///
/// A STEP PLANE has an orthonormal UV frame; using it directly for a raw affine
/// frame would silently change analytic profile radii, shear and placement.
struct AffinePlaneChart<'a>(&'a Plane);

impl StepFormat for AffinePlaneChart<'_> {
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let line = idx + 1;
        let origin = idx + 2;
        let u = idx + 3;
        let v = idx + 5;
        f.write_fmt(format_args!(
            "#{idx} = SURFACE_OF_LINEAR_EXTRUSION('', #{line}, #{v});\n\
             #{line} = LINE('', #{origin}, #{u});\n{}{}{}",
            StepDisplay::new(self.0.origin(), origin),
            StepDisplay::new(self.0.axis_u(), u),
            StepDisplay::new(self.0.axis_v(), v),
        ))
    }
}

impl_const_step_length!(AffinePlaneChart<'_>, 7);
impl StepSurface for AffinePlaneChart<'_> {}

impl StepCurve for ModelingCurve {
    fn same_sense(&self) -> bool {
        match self {
            Self::ParameterCurve(curve) => curve.same_sense(),
            Self::IntersectionCurve(curve) => curve.same_sense(),
            Self::TrimmedCurve(curve) => curve.same_sense(),
            _ => true,
        }
    }
    fn step_parameter_chart(&self) -> Option<ParameterChart> {
        match self {
            Self::Line(curve) => curve.step_parameter_chart(),
            Self::BsplineCurve(curve) => curve.step_parameter_chart(),
            Self::NurbsCurve(curve) => curve.step_parameter_chart(),
            Self::ParameterCurve(curve) => curve.step_parameter_chart(),
            Self::IntersectionCurve(curve) => curve.step_parameter_chart(),
            Self::TrimmedCurve(curve) => curve.step_parameter_chart(),
        }
    }
    fn step_reversed_parameter_chart(&self) -> Option<ParameterChart> {
        match self {
            Self::Line(curve) => curve.step_reversed_parameter_chart(),
            Self::BsplineCurve(curve) => curve.step_reversed_parameter_chart(),
            Self::NurbsCurve(curve) => curve.step_reversed_parameter_chart(),
            Self::ParameterCurve(curve) => curve.step_reversed_parameter_chart(),
            Self::IntersectionCurve(curve) => curve.step_reversed_parameter_chart(),
            Self::TrimmedCurve(curve) => curve.step_reversed_parameter_chart(),
        }
    }
    fn step_parameter_period(&self) -> Option<f64> {
        match self {
            Self::Line(curve) => curve.step_parameter_period(),
            Self::BsplineCurve(curve) => curve.step_parameter_period(),
            Self::NurbsCurve(curve) => curve.step_parameter_period(),
            Self::ParameterCurve(curve) => curve.step_parameter_period(),
            Self::IntersectionCurve(curve) => curve.step_parameter_period(),
            Self::TrimmedCurve(curve) => curve.step_parameter_period(),
        }
    }
}

impl StepFormat for Plane {
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let axis2_placement_idx = idx + 1;
        let location_idx = idx + 2;
        let z_axis_idx = idx + 3;
        let x_axis_idx = idx + 4;
        f.write_fmt(format_args!(
            "#{idx} = PLANE('', #{axis2_placement_idx});
#{axis2_placement_idx} = AXIS2_PLACEMENT_3D('', #{location_idx}, #{z_axis_idx}, #{x_axis_idx});
{location}{z_axis}{x_axis}",
            location = StepDisplay::new(self.origin(), location_idx),
            z_axis = StepDisplay::new(VectorAsDirection(self.normal()), z_axis_idx),
            x_axis = StepDisplay::new(VectorAsDirection(self.axis_u().normalize()), x_axis_idx)
        ))
    }
}
impl_const_step_length!(Plane, 5);

impl StepSurface for Plane {}

// A deliberately exact admission: signed-permutation similarities have no
// hidden shear, projective denominator or rounded-rotation interpretation.
// Other frames need a separately quantified world-space conversion bound.
fn axis_similarity(transform: &Matrix4) -> Option<(f64, Matrix3)> {
    for column in 0..4 {
        for row in 0..4 {
            if !transform[column][row].is_finite() {
                return None;
            }
        }
    }
    if (0..3).any(|column| transform[column][3] != 0.0) || transform[3][3] != 1.0 {
        return None;
    }
    let mut axes = [Vector3::zero(); 3];
    let mut scale = 0.0;
    let mut rows = 0u8;
    for column in 0..3 {
        let vector = transform[column].truncate();
        let row = (0..3).find(|&row| vector[row] != 0.0)?;
        if (0..3).any(|other| other != row && vector[other] != 0.0) || rows & (1 << row) != 0 {
            return None;
        }
        rows |= 1 << row;
        let magnitude = vector[row].abs();
        if column == 0 {
            scale = magnitude;
        } else if magnitude != scale {
            return None;
        }
        axes[column][row] = vector[row].signum();
    }
    if axes[0].cross(axes[1]) != axes[2] {
        return None;
    }
    Some((scale, Matrix3::from_cols(axes[0], axes[1], axes[2])))
}

fn exact_scaled_value(scale: f64, value: f64) -> Option<f64> {
    if !scale.is_finite() || !value.is_finite() {
        return None;
    }
    if scale == 0.0 || value == 0.0 {
        return Some(0.0);
    }
    let power_of_two = |number: f64| {
        let bits = number.abs().to_bits();
        bits & ((1u64 << 52) - 1) == 0 && bits >> 52 != 0
    };
    let product = scale * value;
    if !product.is_finite() {
        return None;
    }
    if scale.abs() == 1.0
        || value.abs() == 1.0
        || ((power_of_two(scale) || power_of_two(value)) && product.is_normal())
    {
        Some(product)
    } else {
        None
    }
}

fn exact_transformed_center(transform: &Matrix4, center: Point3) -> Option<Point3> {
    let mut result = [0.0; 3];
    for row in 0..3 {
        let column = (0..3).find(|&column| transform[column][row] != 0.0)?;
        let product = exact_scaled_value(transform[column][row], center[column])?;
        let sum = ParameterEnclosure::point(product)?
            .sum(ParameterEnclosure::point(transform[3][row])?)?;
        if sum.lower != sum.upper {
            return None;
        }
        result[row] = sum.lower;
    }
    Some(Point3::new(result[0], result[1], result[2]))
}

impl StepFormat for Processor<Sphere, Matrix4> {
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let sphere = *self.entity();
        let transform = self.transform();
        let position_idx = idx + 1;
        let location_idx = idx + 2;
        let axis_idx = idx + 3;
        let ref_direction_idx = idx + 4;
        let (scale, rotation) = axis_similarity(transform).ok_or(std::fmt::Error)?;
        let location =
            exact_transformed_center(transform, sphere.center()).ok_or(std::fmt::Error)?;
        let axis = VectorAsDirection(rotation[2]);
        let ref_direction = VectorAsDirection(rotation[0]);
        let radius = exact_scaled_value(scale, sphere.radius())
            .filter(|radius| *radius > 0.0)
            .ok_or(std::fmt::Error)?;
        let r = FloatDisplay(radius);
        f.write_fmt(format_args!(
            "#{idx} = SPHERICAL_SURFACE('', #{position_idx}, {r});
#{position_idx} = AXIS2_PLACEMENT_3D('', #{location_idx}, #{axis_idx}, #{ref_direction_idx});\n"
        ))?;
        StepFormat::fmt(&location, location_idx, f)?;
        StepFormat::fmt(&axis, axis_idx, f)?;
        StepFormat::fmt(&ref_direction, ref_direction_idx, f)
    }
}
impl_const_step_length!(Processor<Sphere, Matrix4>, 5);

impl StepFormat for Sphere {
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        StepFormat::fmt(&Processor::new(*self), idx, f)
    }
}
impl_const_step_length!(Sphere, 5);
fn native_sphere_parameter_chart(forward: bool) -> Option<SurfaceParameterChart> {
    let half_pi = std::f64::consts::FRAC_PI_2;
    Some(SurfaceParameterChart {
        axes: [
            ParameterChart::identity(),
            ParameterChart::new(
                -1.0,
                half_pi,
                ParameterEnclosure::new(half_pi.next_down(), half_pi.next_up())?,
            )?,
        ],
        swapped: forward,
    })
}

impl StepSurface for Sphere {
    fn step_parameter_chart(&self) -> Option<SurfaceParameterChart> {
        native_sphere_parameter_chart(true)
    }
}

impl StepSurface for Processor<Sphere, Matrix4> {
    fn same_sense(&self) -> bool { self.orientation() }
    fn step_parameter_chart(&self) -> Option<SurfaceParameterChart> {
        let (scale, _) = axis_similarity(self.transform())?;
        exact_transformed_center(self.transform(), self.entity().center())?;
        exact_scaled_value(scale, self.entity().radius()).filter(|radius| *radius > 0.0)?;
        native_sphere_parameter_chart(self.orientation())
    }
}

impl StepFormat for Processor<Torus, Matrix4> {
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let torus = *self.entity();
        let transform = self.transform();
        let position_idx = idx + 1;
        let location_idx = idx + 2;
        let axis_idx = idx + 3;
        let ref_direction_idx = idx + 4;
        let (scale, rotation) = axis_similarity(transform).ok_or(std::fmt::Error)?;
        let location =
            exact_transformed_center(transform, torus.center()).ok_or(std::fmt::Error)?;
        let axis = VectorAsDirection(rotation[2]);
        let ref_direction = VectorAsDirection(rotation[0]);
        let greater = FloatDisplay(
            exact_scaled_value(scale, torus.large_radius())
                .filter(|radius| *radius > 0.0)
                .ok_or(std::fmt::Error)?,
        );
        let lesser = FloatDisplay(
            exact_scaled_value(scale, torus.small_radius())
                .filter(|radius| *radius > 0.0)
                .ok_or(std::fmt::Error)?,
        );
        f.write_fmt(format_args!(
            "#{idx} = TOROIDAL_SURFACE('', #{position_idx}, {greater}, {lesser});
#{position_idx} = AXIS2_PLACEMENT_3D('', #{location_idx}, #{axis_idx}, #{ref_direction_idx});\n",
        ))?;
        StepFormat::fmt(&location, location_idx, f)?;
        StepFormat::fmt(&axis, axis_idx, f)?;
        StepFormat::fmt(&ref_direction, ref_direction_idx, f)
    }
}
impl_const_step_length!(Processor<Torus, Matrix4>, 5);

impl StepSurface for Processor<Torus, Matrix4> {
    #[inline(always)]
    fn same_sense(&self) -> bool { self.orientation() }
    fn step_parameter_chart(&self) -> Option<SurfaceParameterChart> {
        let (scale, _) = axis_similarity(self.transform())?;
        exact_transformed_center(self.transform(), self.entity().center())?;
        exact_scaled_value(scale, self.entity().large_radius()).filter(|radius| *radius > 0.0)?;
        exact_scaled_value(scale, self.entity().small_radius()).filter(|radius| *radius > 0.0)?;
        Some(SurfaceParameterChart {
            axes: [ParameterChart::identity(); 2],
            swapped: !self.orientation(),
        })
    }
}

impl StepFormat for Torus {
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        StepFormat::fmt(&Processor::new(*self), idx, f)
    }
}
impl_const_step_length!(Torus, 5);
impl StepSurface for Torus {}

impl<P> StepFormat for BsplineSurface<P>
where P: Copy + StepFormat
{
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let control_points = self.control_points();
        let control_points_instances = self
            .control_points()
            .iter()
            .flatten()
            .enumerate()
            .map(|(i, p)| StepDisplay::new(*p, idx + i + 1))
            .collect::<Vec<_>>();
        let mut counter = 0;
        let control_points_list = control_points
            .iter()
            .map(|slice| {
                counter += slice.len();
                IndexSliceDisplay(idx + counter - slice.len() + 1..=idx + counter)
            })
            .collect::<Vec<_>>();
        let (uknots, umulti) = self.knot_vector_u().to_single_multi();
        let (vknots, vmulti) = self.knot_vector_v().to_single_multi();
        f.write_fmt(format_args!(
            "#{idx} = B_SPLINE_SURFACE_WITH_KNOTS('', {u_degree}, {v_degree}, {control_points_list}, .UNSPECIFIED., .U., .U., .U., \
{u_multiplicities}, {v_multiplicities}, {u_knots}, {v_knots}, .UNSPECIFIED.);\n{control_points_instances}",
            u_degree = self.udegree(),
            v_degree = self.vdegree(),
            control_points_list = SliceDisplay(&control_points_list),
            u_multiplicities = SliceDisplay(&umulti),
            v_multiplicities = SliceDisplay(&vmulti),
            u_knots = SliceDisplay(&uknots),
            v_knots = SliceDisplay(&vknots),
            control_points_instances = SliceDisplay(&control_points_instances),
        ))
    }
}

impl<P> StepLength for BsplineSurface<P> {
    #[inline(always)]
    fn step_length(&self) -> usize { 1 + self.control_points().iter().map(Vec::len).sum::<usize>() }
}
impl<P> StepSurface for BsplineSurface<P> {}

impl<V> StepFormat for NurbsSurface<V>
where
    V: Homogeneous<Scalar = f64>,
    V::Point: Copy + StepFormat,
{
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let control_points_instances = self
            .control_points()
            .iter()
            .flatten()
            .enumerate()
            .map(|(i, v)| StepDisplay::new(v.to_point(), idx + i + 1))
            .collect::<Vec<_>>();
        let mut counter = 0;
        let control_points_list = self
            .control_points()
            .iter()
            .map(|slice| {
                counter += slice.len();
                IndexSliceDisplay(idx + counter - slice.len() + 1..=idx + counter)
            })
            .collect::<Vec<_>>();
        let weights = self
            .control_points()
            .iter()
            .map(|slice| slice.iter().map(|v| v.weight()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let weights = weights
            .iter()
            .map(|slice| SliceDisplay(slice))
            .collect::<Vec<_>>();
        let (uknots, umulti) = self.knot_vector_u().to_single_multi();
        let (vknots, vmulti) = self.knot_vector_v().to_single_multi();
        f.write_fmt(format_args!(
            "#{idx} = (
    BOUNDED_SURFACE()
    B_SPLINE_SURFACE({u_degree}, {v_degree}, {control_points_list}, .UNSPECIFIED., .U., .U., .U.)
    B_SPLINE_SURFACE_WITH_KNOTS({u_multiplicities}, {v_multiplicities}, {u_knots}, {v_knots}, .UNSPECIFIED.)
    GEOMETRIC_REPRESENTATION_ITEM()
    RATIONAL_B_SPLINE_SURFACE({weights})
    REPRESENTATION_ITEM('')
    SURFACE()
);\n{control_points_instances}",
            u_degree = self.udegree(),
            v_degree = self.vdegree(),
            control_points_list = SliceDisplay(&control_points_list),
            u_multiplicities = SliceDisplay(&umulti),
            v_multiplicities = SliceDisplay(&vmulti),
            u_knots = SliceDisplay(&uknots),
            v_knots = SliceDisplay(&vknots),
            control_points_instances = SliceDisplay(&control_points_instances),
            weights = SliceDisplay(&weights),
        ))
    }
}

impl<V> StepLength for NurbsSurface<V> {
    #[inline(always)]
    fn step_length(&self) -> usize { 1 + self.control_points().iter().map(Vec::len).sum::<usize>() }
}
impl<V> StepSurface for NurbsSurface<V> {}

impl<C> StepFormat for ExtrusionSurface<C, Vector3>
where C: StepLength + StepFormat
{
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let curve = self.entity_curve();
        let curve_idx = idx + 1;
        let vector_idx = idx + 1 + curve.step_length();
        let vector = self.extruding_vector();
        f.write_fmt(format_args!(
            "#{idx} = SURFACE_OF_LINEAR_EXTRUSION('', #{curve_idx}, #{vector_idx});\n{}{}",
            StepDisplay::new(curve, curve_idx),
            StepDisplay::new(vector, vector_idx),
        ))
    }
}
impl<C: StepLength> StepLength for ExtrusionSurface<C, Vector3> {
    fn step_length(&self) -> usize { 1 + self.entity_curve().step_length() + Vector3::LENGTH }
}
impl<C: ConstStepLength> ConstStepLength for ExtrusionSurface<C, Vector3> {
    const LENGTH: usize = 1 + C::LENGTH + Vector3::LENGTH;
}
impl<C: StepCurve> StepSurface for ExtrusionSurface<C, Vector3> {
    fn same_sense(&self) -> bool { self.entity_curve().same_sense() }
    fn step_parameter_chart(&self) -> Option<SurfaceParameterChart> {
        Some(SurfaceParameterChart {
            axes: [
                self.entity_curve().step_parameter_chart()?,
                ParameterChart::identity(),
            ],
            swapped: false,
        })
    }
}

impl<C: StepCurve, T: One> StepSurface for Processor<ExtrusionSurface<C, Vector3>, T> {
    #[inline(always)]
    fn same_sense(&self) -> bool { self.orientation() == self.entity().same_sense() }
    fn step_parameter_chart(&self) -> Option<SurfaceParameterChart> {
        let mut chart = self.entity().step_parameter_chart()?;
        chart.swapped = !self.orientation();
        Some(chart)
    }
}

impl<C> StepFormat for RevolutionSurface<C>
where C: StepLength + StepFormat
{
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let curve = self.entity_curve();
        let curve_idx = idx + 1;
        let axis_idx = curve_idx + curve.step_length();
        let location_idx = axis_idx + 1;
        let dir_idx = location_idx + 1;
        f.write_fmt(format_args!(
            "#{idx} = SURFACE_OF_REVOLUTION('', #{curve_idx}, #{axis_idx});
{curve}#{axis_idx} = AXIS1_PLACEMENT('', #{location_idx}, #{dir_idx});\n{location}{dir}",
            curve = StepDisplay::new(curve, curve_idx),
            location = StepDisplay::new(self.origin(), location_idx),
            dir = StepDisplay::new(VectorAsDirection(self.axis()), dir_idx),
        ))
    }
}

impl<C: StepLength> StepLength for RevolutionSurface<C> {
    #[inline(always)]
    fn step_length(&self) -> usize { 4 + self.entity_curve().step_length() }
}

impl<C: ConstStepLength> ConstStepLength for RevolutionSurface<C> {
    const LENGTH: usize = 4 + C::LENGTH;
}

impl<C: StepCurve> StepSurface for RevolutionSurface<C> {
    #[inline(always)]
    fn same_sense(&self) -> bool { !self.entity_curve().same_sense() }
    fn step_parameter_chart(&self) -> Option<SurfaceParameterChart> {
        Some(SurfaceParameterChart {
            axes: [
                ParameterChart::identity(),
                self.entity_curve().step_parameter_chart()?,
            ],
            swapped: true,
        })
    }
}

impl<C> StepFormat for Processor<RevolutionSurface<C>, Matrix4>
where C: StepLength + Transformed<Matrix4> + StepFormat
{
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        let surface = self.entity();
        let transform = self.transform();
        let (scale, rotation) = axis_similarity(transform).ok_or(std::fmt::Error)?;
        // Nontrivial scale/translation would need a bounded conversion of the
        // complete profile coefficients, not merely a near-similarity guess.
        if scale != 1.0 || transform[3].truncate() != Vector3::zero() {
            return Err(std::fmt::Error);
        }
        let axis = surface.axis();
        if (0..3).filter(|&row| axis[row] != 0.0).count() != 1 || axis.magnitude2() != 1.0 {
            return Err(std::fmt::Error);
        }
        let curve = surface.entity_curve().transformed(*transform);
        let axis = rotation * surface.axis();
        let origin = transform.transform_point(surface.origin());
        let surface = RevolutionSurface::by_revolution(curve, origin, axis);
        StepFormat::fmt(&surface, idx, f)
    }
}
impl<C: StepLength> StepLength for Processor<RevolutionSurface<C>, Matrix4> {
    fn step_length(&self) -> usize { self.entity().step_length() }
}

impl<C: StepCurve, T: One> StepSurface for Processor<RevolutionSurface<C>, T> {
    #[inline(always)]
    fn same_sense(&self) -> bool { self.orientation() == self.entity().same_sense() }
    fn step_parameter_chart(&self) -> Option<SurfaceParameterChart> {
        let mut chart = self.entity().step_parameter_chart()?;
        chart.swapped = self.orientation();
        Some(chart)
    }
}

impl StepFormat for ModelingSurface {
    fn fmt(&self, idx: usize, f: &mut Formatter<'_>) -> Result {
        match self {
            ModelingSurface::Plane(x) => StepFormat::fmt(&AffinePlaneChart(x), idx, f),
            ModelingSurface::BsplineSurface(x) => StepFormat::fmt(x, idx, f),
            ModelingSurface::NurbsSurface(x) => StepFormat::fmt(x, idx, f),
            ModelingSurface::ExtrusionSurface(x) => StepFormat::fmt(x, idx, f),
            ModelingSurface::RevolutionSurface(x) => StepFormat::fmt(x, idx, f),
            ModelingSurface::TsplineSurface(t_mesh) => {
                let bsp = t_mesh.to_bspline_surface(4);
                StepFormat::fmt(&bsp, idx, f)
            }
            ModelingSurface::SphericalSurface(x) => StepFormat::fmt(x, idx, f),
            ModelingSurface::ToroidalSurface(x) => StepFormat::fmt(x, idx, f),
        }
    }
}

impl StepLength for ModelingSurface {
    fn step_length(&self) -> usize {
        match self {
            ModelingSurface::Plane(_) => AffinePlaneChart::LENGTH,
            ModelingSurface::BsplineSurface(x) => x.step_length(),
            ModelingSurface::NurbsSurface(x) => x.step_length(),
            ModelingSurface::ExtrusionSurface(x) => x.step_length(),
            ModelingSurface::RevolutionSurface(x) => x.step_length(),
            ModelingSurface::TsplineSurface(t_mesh) => {
                let bsp = t_mesh.to_bspline_surface(4);
                bsp.step_length()
            }
            ModelingSurface::SphericalSurface(x) => x.step_length(),
            ModelingSurface::ToroidalSurface(x) => x.step_length(),
        }
    }
}

impl StepSurface for ModelingSurface {
    fn same_sense(&self) -> bool {
        match self {
            Self::ExtrusionSurface(surface) => surface.same_sense(),
            Self::RevolutionSurface(surface) => surface.same_sense(),
            Self::SphericalSurface(surface) => surface.orientation(),
            Self::ToroidalSurface(surface) => surface.same_sense(),
            _ => true,
        }
    }
    fn step_parameter_chart(&self) -> Option<SurfaceParameterChart> {
        match self {
            Self::ExtrusionSurface(surface) => surface.step_parameter_chart(),
            Self::RevolutionSurface(surface) => {
                let (scale, _) = axis_similarity(surface.transform())?;
                let axis = surface.entity().axis();
                if scale != 1.0
                    || surface.transform()[3].truncate() != Vector3::zero()
                    || (0..3).filter(|&row| axis[row] != 0.0).count() != 1
                    || axis.magnitude2() != 1.0
                {
                    return None;
                }
                surface.step_parameter_chart()
            }
            Self::ToroidalSurface(surface) => surface.step_parameter_chart(),
            Self::SphericalSurface(surface) => surface.step_parameter_chart(),
            Self::Plane(_) | Self::BsplineSurface(_) | Self::NurbsSurface(_) => {
                Some(SurfaceParameterChart::identity())
            }
            Self::TsplineSurface(_) => None,
        }
    }
}

#[cfg(test)]
mod exact_native_extrusion_save_regressions {
    use super::*;
    use step_p21::tables::IntoOwned;

    fn original(range: (f64, f64), inverted: bool) -> ModelingSurface {
        let mut conic =
            Processor::<_, Matrix3>::new(TrimmedCurve::new(UnitCircle::<Point2>::new(), range));
        if inverted {
            conic.invert();
        }
        let profile = ModelingCurve::ParameterCurve(ParameterCurve::new(
            ModelingCurve2D::Conic(ModelingConic2D::Ellipse(conic)),
            Box::new(ModelingSurface::Plane(Plane::from_origin_vectors(
                Point3::new(7.0, 11.0, 13.0),
                Vector3::new(2.0, 0.0, 0.0),
                Vector3::new(0.0, 3.0, 0.0),
            ))),
        ));
        ModelingSurface::ExtrusionSurface(ExtrusionSurface::by_extrusion(
            profile,
            Vector3::new(0.0, 0.0, 5.0),
        ))
    }

    fn reimport_source(surface: &ModelingSurface) -> crate::step::load::step_geometry::Surface {
        let entities = format!("{}", StepDisplay::new(surface, 1));
        let exchange = format!(
            "ISO-10303-21;\nHEADER;\n\
             FILE_DESCRIPTION(('native exact extrusion'),'2;1');\n\
             FILE_NAME('native.step','2026-01-01T00:00:00',(''),(''),'','','');\n\
             FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n\
             {entities}ENDSEC;\nEND-ISO-10303-21;\n",
        );
        let table = crate::step::load::Table::from_step(&exchange).expect("emitted STEP exchange");
        let holder = table
            .surface_of_linear_extrusion
            .get(&1)
            .expect("root extrusion")
            .clone();
        let record = holder
            .into_owned(&table)
            .expect("resolve actual emitted profile and chart");
        let extrusion: ExtrusionSurface<Box<crate::step::load::step_geometry::Curve3D>, Vector3> =
            (&record).try_into().expect("STEP extrusion geometry");
        crate::step::load::step_geometry::Surface::SweepSurface(
            crate::step::load::step_geometry::SweepSurface::ExtrusionSurface(extrusion),
        )
    }

    fn reimport(surface: &ModelingSurface) -> (ModelingSurface, SurfaceParameterChart) {
        let surface = reimport_source(surface);
        let chart = surface
            .raw_surface_parameter_chart()
            .expect("explicit raw STEP UV chart");
        (
            ModelingSurface::try_from(&surface).expect("source-to-native conversion"),
            chart,
        )
    }

    #[test]
    fn generic_parameter_curve_on_source_carrier_uses_internal_not_raw_uv() {
        use step_p21::tables::EntityTable;
        let surface = reimport_source(&original((0.2, 5.7), true));
        let curve = ParameterCurve::new(
            crate::step::load::step_geometry::Curve2D::Line(Line(
                Point2::new(0.37, -3.0),
                Point2::new(0.91, 9.0),
            )),
            surface,
        );
        let exchange = format!(
            "{}",
            CompleteStepDisplay::new(StepDisplay::new(&curve, 1), StepHeaderDescriptor::default(),)
        );
        let table = crate::step::load::Table::from_step(&exchange).expect("actual generic PCURVE");
        let ast = EntityTable::<crate::step::load::CurveAnyHolder>::get_owned(&table, 1)
            .expect("resolve emitted generic PCURVE");
        let source = crate::step::load::step_geometry::Curve3D::try_from(&ast)
            .expect("checked raw source parameter curve");
        let native = ModelingCurve::try_from(&source).expect("source curve conversion");
        let crate::step::load::step_geometry::Curve3D::ParameterCurve(declared) = &source else {
            panic!("actual emitted PCURVE must expose its checked source composition");
        };
        for t in [0.0_f64, 0.37, 1.0] {
            let theta = 5.9 - (0.37 + 0.54 * t);
            let height = -3.0 + 12.0 * t;
            let expected = Point3::new(
                7.0 + 2.0 * theta.cos(),
                11.0 + 3.0 * theta.sin(),
                13.0 + 5.0 * height,
            );
            let derivative = Vector3::new(1.08 * theta.sin(), -1.62 * theta.cos(), 60.0);
            assert!(curve.evaluate(t).distance(expected) < 1.0e-11);
            assert!(source.evaluate(t).distance(expected) < 1.0e-11);
            assert!(native.evaluate(t).distance(expected) < 1.0e-11);
            assert!(declared.evaluate(t).distance(expected) < 1.0e-11);
            assert!((curve.derivative(t) - derivative).magnitude() < 1.0e-11);
            assert!((source.derivative(t) - derivative).magnitude() < 1.0e-11);
            assert!((native.derivative(t) - derivative).magnitude() < 1.0e-11);
            assert!((declared.derivative(t) - derivative).magnitude() < 1.0e-11);
        }
    }

    fn directional_derivative(
        surface: &ModelingSurface,
        uv: [f64; 2],
        direction: [f64; 2],
        order: usize,
    ) -> Vector3 {
        assert!((1..=3).contains(&order));
        let mut derivative = Vector3::zero();
        for m in 0..=order {
            let binomial = match (order, m) {
                (2, 1) => 2.0,
                (3, 1 | 2) => 3.0,
                _ => 1.0,
            };
            let coefficient =
                binomial * direction[0].powi(m as i32) * direction[1].powi((order - m) as i32);
            if coefficient != 0.0 {
                derivative += surface.derivative_mn(m, order - m, uv[0], uv[1]) * coefficient;
            }
        }
        derivative
    }

    fn assert_roundtrip(range: (f64, f64), inverted: bool) {
        let original = original(range, inverted);
        let exported_chart = original
            .step_parameter_chart()
            .expect("emitted surface UV chart");
        let (native, imported_chart) = reimport(&original);
        let a = exported_chart.jacobian();
        let b = imported_chart.jacobian();
        let jacobian = [
            [
                b[0][0] * a[0][0] + b[0][1] * a[1][0],
                b[0][0] * a[0][1] + b[0][1] * a[1][1],
            ],
            [
                b[1][0] * a[0][0] + b[1][1] * a[1][0],
                b[1][0] * a[0][1] + b[1][1] * a[1][1],
            ],
        ];
        for u in [range.0, 0.37, 0.91, range.1] {
            for v in [-3.0, 9.0, 12.0] {
                let angle = if inverted { range.0 + range.1 - u } else { u };
                let sign = if inverted { -1.0 } else { 1.0 };
                let expected = Point3::new(
                    7.0 + 2.0 * angle.cos(),
                    11.0 + 3.0 * angle.sin(),
                    13.0 + 5.0 * v,
                );
                let du = Vector3::new(-2.0 * sign * angle.sin(), 3.0 * sign * angle.cos(), 0.0);
                let dv = Vector3::new(0.0, 0.0, 5.0);
                let du2 = Vector3::new(-2.0 * angle.cos(), -3.0 * angle.sin(), 0.0);
                let du3 = Vector3::new(2.0 * sign * angle.sin(), -3.0 * sign * angle.cos(), 0.0);
                let raw_uv = exported_chart
                    .nominal([u, v])
                    .expect("exported source coordinates");
                let native_uv = imported_chart
                    .nominal(raw_uv)
                    .expect("decoded native coordinates");
                for (label, surface, uv, jacobian) in [
                    ("original", &original, [u, v], [[1.0, 0.0], [0.0, 1.0]]),
                    ("reimported", &native, native_uv, jacobian),
                ] {
                    let u_direction = [jacobian[0][0], jacobian[1][0]];
                    let v_direction = [jacobian[0][1], jacobian[1][1]];
                    assert!(
                        surface.evaluate(uv[0], uv[1]).distance(expected) < 1.0e-11,
                        "{label} mapped chart at ({u}, {v})"
                    );
                    assert!(
                        (directional_derivative(surface, uv, u_direction, 1) - du).magnitude()
                            < 1.0e-11,
                        "{label} first profile derivative at ({u}, {v})"
                    );
                    assert!(
                        (directional_derivative(surface, uv, u_direction, 2) - du2).magnitude()
                            < 1.0e-11,
                        "{label} second profile derivative at ({u}, {v})"
                    );
                    assert!(
                        (directional_derivative(surface, uv, u_direction, 3) - du3).magnitude()
                            < 1.0e-11,
                        "{label} third profile derivative at ({u}, {v})"
                    );
                    assert!(
                        (directional_derivative(surface, uv, v_direction, 1) - dv).magnitude()
                            < 1.0e-11
                    );
                    let determinant =
                        jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0];
                    assert!(
                        (surface.normal(uv[0], uv[1]) * determinant.signum()
                            - du.cross(dv).normalize())
                        .magnitude()
                            < 1.0e-11
                    );
                }
            }
        }
    }

    #[test]
    fn analytic_extrusion_serializes_native_profile_and_affine_chart_without_fitting() {
        assert_roundtrip((0.0, std::f64::consts::TAU), false);
    }

    #[test]
    fn inverted_analytic_profile_retains_curve_and_surface_sense() {
        assert_roundtrip((0.2, 5.7), true);
    }

    fn reimport_curve(curve: &ModelingCurve) -> ModelingCurve {
        use step_p21::tables::EntityTable;
        let entities = format!("{}", StepDisplay::new(curve, 1));
        let exchange = format!(
            "ISO-10303-21;\nHEADER;\n\
             FILE_DESCRIPTION(('native curve'),'2;1');\n\
             FILE_NAME('native.step','2026-01-01T00:00:00',(''),(''),'','','');\n\
             FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n\
             {entities}ENDSEC;\nEND-ISO-10303-21;\n",
        );
        let table = crate::step::load::Table::from_step(&exchange).expect("emitted curve STEP");
        let ast = EntityTable::<crate::step::load::CurveAnyHolder>::get_owned(&table, 1)
            .expect("resolve emitted curve");
        let source = crate::step::load::step_geometry::Curve3D::try_from(&ast)
            .expect("decode emitted source curve");
        ModelingCurve::try_from(&source).expect("convert emitted curve to native")
    }

    #[test]
    fn nested_trim_selectors_use_the_emitted_inverse_basis_chart() {
        let ModelingSurface::ExtrusionSurface(surface) = original((0.2, 5.7), true) else {
            unreachable!()
        };
        let curve = ModelingCurve::TrimmedCurve(Processor::new(TrimmedCurve::new(
            Box::new(surface.entity_curve().clone()),
            (1.0, 2.0),
        )));
        let imported = reimport_curve(&curve);
        for t in [1.0, 1.37, 2.0] {
            let theta: f64 = 5.9 - t;
            let expected = Point3::new(7.0 + 2.0 * theta.cos(), 11.0 + 3.0 * theta.sin(), 13.0);
            let derivative = Vector3::new(2.0 * theta.sin(), -3.0 * theta.cos(), 0.0);
            assert!(imported.evaluate(t).distance(expected) < 1.0e-11);
            assert!((imported.derivative(t) - derivative).magnitude() < 1.0e-11);
        }
    }

    #[test]
    fn extrusion_pcurve_coordinates_use_the_emitted_profile_chart() {
        let surface = original((0.2, 5.7), true);
        let curve = ModelingCurve::ParameterCurve(ParameterCurve::new(
            ModelingCurve2D::Line(Line(Point2::new(0.37, -3.0), Point2::new(0.91, 9.0))),
            Box::new(surface),
        ));
        let imported = reimport_curve(&curve);
        for t in [0.0, 0.37, 1.0] {
            let u = 0.37 + 0.54 * t;
            let v = -3.0 + 12.0 * t;
            let theta: f64 = 5.9 - u;
            let expected = Point3::new(
                7.0 + 2.0 * theta.cos(),
                11.0 + 3.0 * theta.sin(),
                13.0 + 5.0 * v,
            );
            let derivative = Vector3::new(1.08 * theta.sin(), -1.62 * theta.cos(), 60.0);
            assert!(imported.evaluate(t).distance(expected) < 1.0e-11);
            assert!((imported.derivative(t) - derivative).magnitude() < 1.0e-11);
        }
    }

    #[test]
    fn revolution_pcurve_coordinates_use_profile_gauge_and_axis_permutation() {
        let mut ellipse = Processor::<_, Matrix3>::new(TrimmedCurve::new(
            UnitCircle::<Point2>::new(),
            (0.2, 5.7),
        ));
        ellipse.invert();
        let profile = ModelingCurve::ParameterCurve(ParameterCurve::new(
            ModelingCurve2D::Conic(ModelingConic2D::Ellipse(ellipse)),
            Box::new(ModelingSurface::Plane(Plane::from_origin_vectors(
                Point3::new(7.0, 0.0, 13.0),
                Vector3::new(2.0, 0.0, 0.0),
                Vector3::new(0.0, 0.0, 3.0),
            ))),
        ));
        let surface = ModelingSurface::RevolutionSurface(Processor::new(
            RevolutionSurface::by_revolution(profile, Point3::origin(), Vector3::unit_z()),
        ));
        let curve = ModelingCurve::ParameterCurve(ParameterCurve::new(
            ModelingCurve2D::Line(Line(Point2::new(0.37, -3.0), Point2::new(0.91, 9.0))),
            Box::new(surface),
        ));
        let imported = reimport_curve(&curve);
        for t in [0.0, 0.37, 1.0] {
            let theta: f64 = 5.9 - (0.37 + 0.54 * t);
            let angle: f64 = -3.0 + 12.0 * t;
            let radius = 7.0 + 2.0 * theta.cos();
            let radial_derivative = 1.08 * theta.sin();
            let expected = Point3::new(
                radius * angle.cos(),
                radius * angle.sin(),
                13.0 + 3.0 * theta.sin(),
            );
            let derivative = Vector3::new(
                radial_derivative * angle.cos() - 12.0 * radius * angle.sin(),
                radial_derivative * angle.sin() + 12.0 * radius * angle.cos(),
                -1.62 * theta.cos(),
            );
            assert!(imported.evaluate(t).distance(expected) < 1.0e-11);
            assert!((imported.derivative(t) - derivative).magnitude() < 1.0e-11);
        }
    }

    #[test]
    fn analytic_surface_writer_refuses_unproved_affine_carriers() {
        let mut shear = Matrix4::identity();
        shear[1][0] = 0.125;
        let mut projective = Matrix4::identity();
        projective[0][3] = 0.125;
        for matrix in [
            Matrix4::from_nonuniform_scale(2.0, 3.0, 1.0),
            shear,
            projective,
            Matrix4::from_angle_z(Rad(0.37)),
        ] {
            let sphere = ModelingSurface::SphericalSurface(Processor::with_transform(
                Sphere::new(Point3::origin(), 2.0),
                matrix,
            ));
            let revolution = ModelingSurface::RevolutionSurface(Processor::with_transform(
                RevolutionSurface::by_revolution(
                    ModelingCurve::Line(Line(
                        Point3::new(2.0, 0.0, 0.0),
                        Point3::new(2.0, 0.0, 1.0),
                    )),
                    Point3::origin(),
                    Vector3::unit_z(),
                ),
                matrix,
            ));
            for surface in [sphere, revolution] {
                let mut output = String::new();
                assert!(
                    std::fmt::write(
                        &mut output,
                        format_args!("{}", StepDisplay::new(&surface, 1))
                    )
                    .is_err()
                );
            }
        }
    }
}

#[cfg(test)]
mod native_material_orientation_regressions {
    use super::*;
    use monstertruck_topology::compress::{
        CompressedEdge, CompressedEdgeIndex, CompressedFace, CompressedShell,
    };

    fn reimport_patch(
        surface: ModelingSurface,
        uv: [Point2; 4],
        orientation: bool,
    ) -> CompressedShell<
        Point3,
        crate::step::load::step_geometry::Curve3D,
        crate::step::load::step_geometry::Surface,
    > {
        let vertices = uv.iter().map(|p| surface.evaluate(p.x, p.y)).collect();
        let edges = (0..4)
            .map(|index| {
                let next = (index + 1) % 4;
                CompressedEdge {
                    vertices: (index, next),
                    curve: ModelingCurve::ParameterCurve(ParameterCurve::new(
                        ModelingCurve2D::Line(Line(uv[index], uv[next])),
                        Box::new(surface.clone()),
                    )),
                }
            })
            .collect();
        let shell = CompressedShell {
            vertices,
            edges,
            faces: vec![CompressedFace {
                boundaries: vec![
                    (0..4)
                        .map(|index| CompressedEdgeIndex {
                            index,
                            orientation: true,
                        })
                        .collect(),
                ],
                orientation,
                surface,
            }],
            vertex_stable_ids: None,
            edge_stable_ids: None,
            face_stable_ids: None,
        };
        let exchange = format!(
            "{}",
            CompleteStepDisplay::new(StepModel::from(&shell), StepHeaderDescriptor::default(),)
        );
        let table = crate::step::load::Table::from_step(&exchange).expect("actual emitted shell");
        table
            .to_compressed_shell(table.shell.values().next().expect("source shell"))
            .expect("reload actual face and material sense")
    }

    #[test]
    fn asymmetric_revolution_face_preserves_material_normal_under_uv_exchange() {
        let surface =
            ModelingSurface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
                ModelingCurve::Line(Line(Point3::new(2.0, 0.0, 1.0), Point3::new(3.0, 0.0, 4.0))),
                Point3::origin(),
                Vector3::unit_z(),
            )));
        let uv = [
            Point2::new(0.2, 0.3),
            Point2::new(0.8, 0.3),
            Point2::new(0.8, 1.1),
            Point2::new(0.2, 1.1),
        ];
        let shell = reimport_patch(surface, uv, false);
        let face = &shell.faces[0];
        for (profile, angle) in [(0.31f64, 0.47f64), (0.67, 0.93)] {
            let radius = 2.0 + profile;
            let expected = Point3::new(
                radius * angle.cos(),
                radius * angle.sin(),
                1.0 + 3.0 * profile,
            );
            let angular = Vector3::new(-radius * angle.sin(), radius * angle.cos(), 0.0);
            let axial = Vector3::new(angle.cos(), angle.sin(), 3.0);
            assert!(face.surface.evaluate(angle, profile).distance(expected) < 1.0e-11);
            assert!((face.surface.derivative_u(angle, profile) - angular).magnitude() < 1.0e-11);
            assert!((face.surface.derivative_v(angle, profile) - axial).magnitude() < 1.0e-11);
            let native_normal = face.surface.normal(angle, profile);
            let material_normal = if face.orientation {
                native_normal
            } else {
                native_normal * -1.0
            };
            let expected_outward =
                Vector3::new(3.0 * angle.cos(), 3.0 * angle.sin(), -1.0).normalize();
            assert!((material_normal - expected_outward).magnitude() < 1.0e-11);
            assert!(material_normal.dot(Vector3::new(angle.cos(), angle.sin(), 0.0)) > 0.0);
        }
    }

    #[test]
    fn affine_plane_face_keeps_unbounded_line_profile_outside_placeholder_range() {
        let surface = ModelingSurface::Plane(Plane::from_origin_vectors(
            Point3::new(7.0, 11.0, 13.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(0.0, 3.0, 0.0),
        ));
        let uv = [
            Point2::new(4.0, -2.0),
            Point2::new(8.0, -2.0),
            Point2::new(8.0, 3.0),
            Point2::new(4.0, 3.0),
        ];
        let shell = reimport_patch(surface, uv, true);
        let face = &shell.faces[0];
        for u in [4.0, 5.25, 7.6, 8.0] {
            for v in [-2.0, -0.3, 2.4, 3.0] {
                let expected = Point3::new(7.0 + 2.0 * u, 11.0 + 3.0 * v, 13.0);
                assert!(face.surface.evaluate(u, v).distance(expected) < 1.0e-11);
                assert!(
                    (face.surface.derivative_u(u, v) - Vector3::new(2.0, 0.0, 0.0)).magnitude()
                        < 1.0e-11
                );
                assert!(
                    (face.surface.derivative_v(u, v) - Vector3::new(0.0, 3.0, 0.0)).magnitude()
                        < 1.0e-11
                );
                let normal = face.surface.normal(u, v);
                let material = if face.orientation {
                    normal
                } else {
                    normal * -1.0
                };
                assert!((material - Vector3::unit_z()).magnitude() < 1.0e-11);
            }
        }
    }
}

#[cfg(test)]
mod native_edge_sense_regressions {
    use super::*;
    use monstertruck_topology::compress::{
        CompressedEdge, CompressedEdgeIndex, CompressedFace, CompressedShell,
    };

    #[test]
    fn closed_circle_edge_retains_negative_native_to_step_parameter_direction() {
        let mut circle = Processor::<_, Matrix4>::new(TrimmedCurve::new(
            UnitCircle::<Point3>::new(),
            (0.0, std::f64::consts::TAU),
        ));
        circle.invert();
        let shell = CompressedShell {
            vertices: vec![Point3::new(1.0, 0.0, 0.0)],
            edges: vec![CompressedEdge {
                vertices: (0, 0),
                curve: circle,
            }],
            faces: vec![CompressedFace {
                boundaries: vec![vec![CompressedEdgeIndex {
                    index: 0,
                    orientation: true,
                }]],
                orientation: false,
                surface: ModelingSurface::Plane(Plane::from_origin_vectors(
                    Point3::origin(),
                    Vector3::unit_x(),
                    Vector3::unit_y(),
                )),
            }],
            vertex_stable_ids: None,
            edge_stable_ids: None,
            face_stable_ids: None,
        };
        let exchange = format!(
            "{}",
            CompleteStepDisplay::new(StepModel::from(&shell), StepHeaderDescriptor::default(),)
        );
        let table =
            crate::step::load::Table::from_step(&exchange).expect("emitted closed-circle shell");
        let shell = table
            .to_compressed_shell(table.shell.values().next().expect("source shell"))
            .expect("reload actual oriented edge");
        let curve = &shell.edges[0].curve;
        for angle in [0.37f64, 1.19, 4.73] {
            let expected = Point3::new(angle.cos(), -angle.sin(), 0.0);
            let derivative = Vector3::new(-angle.sin(), -angle.cos(), 0.0);
            assert!(curve.evaluate(angle).distance(expected) < 1.0e-11);
            assert!((curve.derivative(angle) - derivative).magnitude() < 1.0e-11);
        }
    }
}

#[cfg(test)]
mod native_reflected_pcurve_regressions {
    use super::*;
    use step_p21::tables::EntityTable;

    #[test]
    fn ellipse_on_revolution_preserves_world_curve_through_emitted_scalar_trim_chart() {
        let ellipse = Processor::with_transform(
            TrimmedCurve::new(UnitCircle::<Point2>::new(), (0.2, 2.1)),
            Matrix3::from_cols(
                Vector3::new(0.2, 0.0, 0.0),
                Vector3::new(0.0, 0.3, 0.0),
                Vector3::new(0.5, 0.7, 1.0),
            ),
        );
        let surface =
            ModelingSurface::RevolutionSurface(Processor::new(RevolutionSurface::by_revolution(
                ModelingCurve::Line(Line(Point3::new(2.0, 0.0, 1.0), Point3::new(3.0, 0.0, 4.0))),
                Point3::origin(),
                Vector3::unit_z(),
            )));
        let original = ModelingCurve::ParameterCurve(ParameterCurve::new(
            ModelingCurve2D::Conic(ModelingConic2D::Ellipse(ellipse)),
            Box::new(surface),
        ));
        let emitted = original
            .step_parameter_chart()
            .expect("actual emitted scalar chart");
        let exchange = format!(
            "{}",
            CompleteStepDisplay::new(
                StepDisplay::new(&original, 1),
                StepHeaderDescriptor::default(),
            )
        );
        let table =
            crate::step::load::Table::from_step(&exchange).expect("actual reflected UV PCURVE");
        let ast = EntityTable::<crate::step::load::CurveAnyHolder>::get_owned(&table, 1)
            .expect("resolve actual emitted scalar trim");
        let source = crate::step::load::step_geometry::Curve3D::try_from(&ast)
            .expect("checked source parameter curve");
        let decoded = source
            .raw_parameter_chart()
            .expect("source scalar gauge")
            .expect("bounded affine scalar map")
            .compose(emitted)
            .expect("scalar composition");
        let native = ModelingCurve::try_from(&source).expect("native parameter curve");
        for t in [0.37f64, 0.91, 1.73] {
            let radius = 2.5 + 0.2 * t.cos();
            let angle = 0.7 + 0.3 * t.sin();
            let dr = -0.2 * t.sin();
            let da = 0.3 * t.cos();
            let expected = Point3::new(
                radius * angle.cos(),
                radius * angle.sin(),
                2.5 + 0.6 * t.cos(),
            );
            let derivative = Vector3::new(
                dr * angle.cos() - radius * angle.sin() * da,
                dr * angle.sin() + radius * angle.cos() * da,
                -0.6 * t.sin(),
            );
            let parameter = decoded.nominal(t).expect("source active scalar parameter");
            assert!(original.evaluate(t).distance(expected) < 1.0e-11);
            assert!(source.evaluate(parameter).distance(expected) < 1.0e-11);
            assert!(native.evaluate(parameter).distance(expected) < 1.0e-11);
            assert!((original.derivative(t) - derivative).magnitude() < 1.0e-11);
            assert!(
                (source.derivative(parameter) * decoded.scale - derivative).magnitude() < 1.0e-11
            );
            assert!(
                (native.derivative(parameter) * decoded.scale - derivative).magnitude() < 1.0e-11
            );
        }
    }

    #[test]
    fn two_dimensional_conic_writer_refuses_unproved_shear_and_projective_frames() {
        let frames = [
            Matrix3::from_cols(
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.25, 1.0, 0.0),
                Vector3::unit_z(),
            ),
            Matrix3::from_cols(
                Vector3::new(1.0, 0.0, 0.1),
                Vector3::unit_y(),
                Vector3::unit_z(),
            ),
        ];
        for frame in frames {
            let curve = Processor::with_transform(
                TrimmedCurve::new(UnitCircle::<Point2>::new(), (0.2, 2.1)),
                frame,
            );
            let mut output = String::new();
            assert!(
                std::fmt::write(&mut output, format_args!("{}", StepDisplay::new(&curve, 1)))
                    .is_err()
            );
            assert!(curve.step_parameter_chart().is_none());
        }
    }
}
