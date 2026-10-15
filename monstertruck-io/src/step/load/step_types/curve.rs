//! Curve entities: lines, conics, polylines, B-spline and NURBS curves, and
//! the standalone curve sets built from them.
//!
//! The `pcurve`/`surface_curve` cluster is the exception: it lives in the
//! parent module because [`topology`](super::topology) reads its fields.

use super::*;
use monstertruck_modeling::geometry::MAX_NATIVE_GEOMETRY_NESTING;
use serde::{
    Deserializer,
    de::{Error as DeserializeError, MapAccess, SeqAccess, Visitor, value::SeqAccessDeserializer},
};
use std::fmt;
use step_p21::{
    error::Error as StepEntityError,
    tables::{EntityTable, Holder as HolderTrait, IntoOwned, WithVisitor},
};

/// `curve`
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CurveAny {
    Line(Box<Line>),
    BoundedCurve(Box<BoundedCurveAny>),
    Conic(Box<Conic>),
    Pcurve(Box<Pcurve>),
    SurfaceCurve(Box<SurfaceCurve>),
    TrimmedCurve(Box<StepTrimmedCurve>),
}

impl TryFrom<&CurveAny> for Curve2D {
    type Error = StepConvertingError;
    #[inline(always)]
    fn try_from(curve: &CurveAny) -> Result<Self, Self::Error> {
        validate_owned_curve_nesting(curve, 0)?;
        curve2d_from_validated(curve)
    }
}

impl TryFrom<&CurveAny> for Curve3D {
    type Error = StepConvertingError;
    #[inline(always)]
    fn try_from(curve: &CurveAny) -> Result<Self, Self::Error> {
        validate_owned_curve_nesting(curve, 0)?;
        curve3d_from_validated(curve)
    }
}

fn curve2d_from_validated(curve: &CurveAny) -> Result<Curve2D, StepConvertingError> {
    match curve {
        CurveAny::Line(line) => Ok(Curve2D::Line(line.as_ref().try_into()?)),
        CurveAny::BoundedCurve(curve) => curve.as_ref().try_into(),
        CurveAny::Conic(curve) => Ok(Curve2D::Conic(curve.as_ref().try_into()?)),
        CurveAny::TrimmedCurve(curve) => trimmed_curve2d_from_validated(curve),
        CurveAny::Pcurve(_) => Err("Pcurves cannot be parsed to 2D curves.".into()),
        CurveAny::SurfaceCurve(_) => Err("Surface curves cannot be parsed to 2D curves.".into()),
    }
}

fn curve3d_from_validated(curve: &CurveAny) -> Result<Curve3D, StepConvertingError> {
    match curve {
        CurveAny::Line(line) => Ok(Curve3D::Line(line.as_ref().try_into()?)),
        CurveAny::BoundedCurve(curve) => curve.as_ref().try_into(),
        CurveAny::Conic(curve) => Ok(Curve3D::Conic(curve.as_ref().try_into()?)),
        CurveAny::Pcurve(curve) => Ok(Curve3D::ParameterCurve(curve.as_ref().try_into()?)),
        CurveAny::SurfaceCurve(curve) => curve.as_ref().try_into(),
        CurveAny::TrimmedCurve(curve) => trimmed_curve3d_from_validated(curve),
    }
}

pub(crate) fn validate_source_nesting_depth(depth: usize) -> Result<(), ExactConversionError> {
    if depth > MAX_NATIVE_GEOMETRY_NESTING {
        Err(ExactConversionError::SourceGeometryNestingExceeded)
    } else {
        Ok(())
    }
}

fn validate_owned_pcurve_nesting(curve: &Pcurve, depth: usize) -> Result<(), ExactConversionError> {
    validate_source_nesting_depth(depth)?;
    curve.basis_surface.validate_curve_nesting(depth + 1)?;
    curve
        .reference_to_curve
        .representation_item
        .iter()
        .try_for_each(|curve| validate_owned_curve_nesting(curve, depth + 1))
}

pub(crate) fn validate_owned_curve_nesting(
    curve: &CurveAny,
    depth: usize,
) -> Result<(), ExactConversionError> {
    validate_source_nesting_depth(depth)?;
    match curve {
        CurveAny::TrimmedCurve(curve) => {
            validate_owned_curve_nesting(&curve.basis_curve, depth + 1)
        }
        CurveAny::SurfaceCurve(curve) => {
            validate_owned_curve_nesting(&curve.curve_3d, depth + 1)?;
            curve
                .associated_geometry
                .iter()
                .try_for_each(|associated| match associated {
                    PcurveOrSurface::Pcurve(curve) => {
                        validate_owned_pcurve_nesting(curve, depth + 1)
                    }
                    PcurveOrSurface::Surface(surface) => surface.validate_curve_nesting(depth + 1),
                })
        }
        CurveAny::Pcurve(curve) => validate_owned_pcurve_nesting(curve, depth),
        CurveAny::Line(_) | CurveAny::BoundedCurve(_) | CurveAny::Conic(_) => Ok(()),
    }
}

/// An explicitly typed STEP source parameter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(field = parameter_value)]
pub struct ParameterValue(pub f64);

impl<'de> Deserialize<'de> for ParameterValueHolder {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        f64::deserialize(deserializer).map(Self)
    }
}

/// The declared representation preferred by a STEP trim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrimmingPreference {
    /// Cartesian endpoints are authoritative.
    Cartesian,
    /// Explicit source parameters are authoritative.
    Parameter,
    /// The producer specifies no preference.
    Unspecified,
}

/// One declared endpoint selector of a STEP trimmed curve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(generate_deserialize)]
pub enum TrimmingSelect {
    #[holder(use_place_holder)]
    CartesianPoint(Box<CartesianPoint>),
    #[holder(use_place_holder)]
    ParameterValue(Box<ParameterValue>),
}

/// A STEP `TRIMMED_CURVE`, retaining its declared basis and endpoint selectors.
///
/// Numeric endpoint ordering follows STEP/OCCT trim selection, including
/// genuine periodic seam crossings and full periods. The source wrapper keeps
/// its raw canonical parameter gauge separately from its decoded native range.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepTrimmedCurve {
    /// Source entity label.
    pub label: String,
    /// Original basis curve, without geometric reconstruction.
    pub basis_curve: CurveAny,
    /// Declared first endpoint selectors.
    pub trim_1: Vec<TrimmingSelect>,
    /// Declared second endpoint selectors.
    pub trim_2: Vec<TrimmingSelect>,
    /// Whether the trim follows its basis parameter direction.
    pub sense_agreement: bool,
    /// The producer's endpoint-selector preference.
    pub master_representation: TrimmingPreference,
}

/// Reference-preserving holder for a STEP `TRIMMED_CURVE`.
#[derive(Clone, Debug, PartialEq)]
pub struct StepTrimmedCurveHolder {
    pub label: String,
    pub basis_curve: PlaceHolder<CurveAnyHolder>,
    pub trim_1: Vec<PlaceHolder<TrimmingSelectHolder>>,
    pub trim_2: Vec<PlaceHolder<TrimmingSelectHolder>>,
    pub sense_agreement: bool,
    pub master_representation: TrimmingPreference,
}

impl HolderTrait for StepTrimmedCurveHolder {
    fn name() -> &'static str { "TRIMMED_CURVE" }
    fn attr_len() -> usize { 6 }
}

/// Deserializer for an inline STEP trim entity.
#[doc(hidden)]
#[derive(Debug)]
pub struct StepTrimmedCurveHolderVisitor;

impl<'de> Visitor<'de> for StepTrimmedCurveHolderVisitor {
    type Value = StepTrimmedCurveHolder;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("TRIMMED_CURVE")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, sequence: A) -> Result<Self::Value, A::Error> {
        let (label, basis_curve, trim_1, trim_2, sense_agreement, master_representation) =
            Deserialize::deserialize(SeqAccessDeserializer::new(sequence))?;
        Ok(StepTrimmedCurveHolder {
            label,
            basis_curve,
            trim_1,
            trim_2,
            sense_agreement,
            master_representation,
        })
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let name: String = map
            .next_key()?
            .ok_or_else(|| A::Error::custom("missing trim entity name"))?;
        if name != "TRIMMED_CURVE" {
            return Err(A::Error::custom("expected TRIMMED_CURVE"));
        }
        map.next_value()
    }
}

impl WithVisitor for StepTrimmedCurveHolder {
    type Visitor = StepTrimmedCurveHolderVisitor;
    fn visitor_new() -> Self::Visitor { StepTrimmedCurveHolderVisitor }
}

impl<'de> Deserialize<'de> for StepTrimmedCurveHolder {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(StepTrimmedCurveHolderVisitor)
    }
}

impl IntoOwned for StepTrimmedCurveHolder {
    type Table = Table;
    type Owned = StepTrimmedCurve;

    fn into_owned(self, table: &Table) -> Result<Self::Owned, StepEntityError> {
        table
            .validate_trimmed_curve_source(&self)
            .map_err(|error| StepEntityError::DeserializeFailed(error.to_string()))?;
        Ok(StepTrimmedCurve {
            label: self.label,
            basis_curve: self.basis_curve.into_owned(table)?,
            trim_1: self
                .trim_1
                .into_iter()
                .map(|selector| selector.into_owned(table))
                .collect::<Result<_, _>>()?,
            trim_2: self
                .trim_2
                .into_iter()
                .map(|selector| selector.into_owned(table))
                .collect::<Result<_, _>>()?,
            sense_agreement: self.sense_agreement,
            master_representation: self.master_representation,
        })
    }
}

impl EntityTable<StepTrimmedCurveHolder> for Table {
    fn get_owned(&self, entity_id: u64) -> Result<StepTrimmedCurve, StepEntityError> {
        let holder = self
            .trimmed_curve
            .get(&entity_id)
            .ok_or(StepEntityError::UnknownEntity(entity_id))?;
        self.validate_trimmed_curve_source(holder)
            .map_err(|error| StepEntityError::DeserializeFailed(error.to_string()))?;
        holder.clone().into_owned(self)
    }

    fn owned_iter<'table>(
        &'table self,
    ) -> Box<dyn Iterator<Item = Result<StepTrimmedCurve, StepEntityError>> + 'table> {
        Box::new(
            self.trimmed_curve
                .keys()
                .map(|id| EntityTable::<StepTrimmedCurveHolder>::get_owned(self, *id)),
        )
    }
}

impl StepTrimmedCurve {
    fn explicit_parameter_range(&self) -> Result<(f64, f64), StepConvertingError> {
        if self.master_representation == TrimmingPreference::Cartesian
            && self
                .trim_1
                .iter()
                .chain(&self.trim_2)
                .any(|selector| matches!(selector, TrimmingSelect::CartesianPoint(_)))
        {
            return Err(ExactConversionError::CartesianTrimSelectors.into());
        }
        let endpoint = |selectors: &[TrimmingSelect]| {
            let mut parameters = selectors.iter().filter_map(|selector| match selector {
                TrimmingSelect::ParameterValue(parameter) => Some(parameter.0),
                TrimmingSelect::CartesianPoint(_) => None,
            });
            let parameter = parameters
                .next()
                .ok_or(ExactConversionError::CartesianTrimSelectors)?;
            if !parameter.is_finite() || parameters.next().is_some() {
                return Err(ExactConversionError::InvalidTrimParameterSelectors);
            }
            Ok(parameter)
        };
        let first = endpoint(&self.trim_1)?;
        let second = endpoint(&self.trim_2)?;
        Ok((first, second))
    }
}

impl TryFrom<&StepTrimmedCurve> for Curve2D {
    type Error = StepConvertingError;

    fn try_from(curve: &StepTrimmedCurve) -> Result<Self, Self::Error> {
        validate_owned_curve_nesting(&curve.basis_curve, 1)?;
        trimmed_curve2d_from_validated(curve)
    }
}

impl TryFrom<&StepTrimmedCurve> for Curve3D {
    type Error = StepConvertingError;

    fn try_from(curve: &StepTrimmedCurve) -> Result<Self, Self::Error> {
        validate_owned_curve_nesting(&curve.basis_curve, 1)?;
        trimmed_curve3d_from_validated(curve)
    }
}

fn trimmed_curve2d_from_validated(
    curve: &StepTrimmedCurve,
) -> Result<Curve2D, StepConvertingError> {
    let range = curve.explicit_parameter_range()?;
    let basis = curve2d_from_validated(&curve.basis_curve)?;
    let selection = super::trim_parameterization::select(
        range,
        curve.sense_agreement,
        super::trim_parameterization::basis_2d(&curve.basis_curve, &basis)?,
    )?;
    let mut trim = Processor::new(geom::TrimmedCurve::new(
        Box::new(basis),
        selection.internal_range,
    ));
    if selection.reversed {
        trim.invert();
    }
    Ok(Curve2D::TrimmedCurve(SourceTrimmedCurve::declared(
        trim,
        selection.raw_to_internal,
        selection.raw_range,
        selection.selected_basis_range,
        selection.raw_parameter_bounds,
        selection.selected_basis_bounds,
    )))
}

fn trimmed_curve3d_from_validated(
    curve: &StepTrimmedCurve,
) -> Result<Curve3D, StepConvertingError> {
    let range = curve.explicit_parameter_range()?;
    let basis = curve3d_from_validated(&curve.basis_curve)?;
    let selection = super::trim_parameterization::select(
        range,
        curve.sense_agreement,
        super::trim_parameterization::basis_3d(&curve.basis_curve, &basis)?,
    )?;
    let mut trim = Processor::new(geom::TrimmedCurve::new(
        Box::new(basis),
        selection.internal_range,
    ));
    if selection.reversed {
        trim.invert();
    }
    Ok(Curve3D::TrimmedCurve(SourceTrimmedCurve::declared(
        trim,
        selection.raw_to_internal,
        selection.raw_range,
        selection.selected_basis_range,
        selection.raw_parameter_bounds,
        selection.selected_basis_bounds,
    )))
}

/// `line`
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(field = line)]
#[holder(generate_deserialize)]
pub struct Line {
    pub label: String,
    #[holder(use_place_holder)]
    pub pnt: CartesianPoint,
    #[holder(use_place_holder)]
    pub dir: Vector,
}
impl<'a, P> TryFrom<&'a Line> for geom::Line<P>
where
    P: EuclideanSpace + From<&'a CartesianPoint>,
    P::Diff: TryFrom<&'a Vector, Error = VectorSemanticError>,
{
    type Error = VectorSemanticError;
    #[inline(always)]
    fn try_from(line: &'a Line) -> std::result::Result<Self, Self::Error> {
        let p = P::from(&line.pnt);
        let q = p + P::Diff::try_from(&line.dir)?;
        Ok(Self(p, q))
    }
}

/// `bounded_curve`
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(generate_deserialize)]
pub enum BoundedCurveAny {
    #[holder(use_place_holder)]
    Polyline(Box<Polyline>),
    #[holder(use_place_holder)]
    BsplineCurve(Box<BsplineCurveAny>),
}

impl TryFrom<&BoundedCurveAny> for Curve2D {
    type Error = StepConvertingError;
    #[inline(always)]
    fn try_from(value: &BoundedCurveAny) -> Result<Self, Self::Error> {
        use BoundedCurveAny::*;
        Ok(match value {
            Polyline(x) => Self::Polyline(x.as_ref().into()),
            BsplineCurve(x) => x.as_ref().try_into()?,
        })
    }
}

impl TryFrom<&BoundedCurveAny> for Curve3D {
    type Error = StepConvertingError;
    #[inline(always)]
    fn try_from(value: &BoundedCurveAny) -> Result<Self, Self::Error> {
        use BoundedCurveAny::*;
        Ok(match value {
            Polyline(x) => Self::Polyline(x.as_ref().into()),
            BsplineCurve(x) => x.as_ref().try_into()?,
        })
    }
}

/// `polyline`
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(field = polyline)]
#[holder(generate_deserialize)]
pub struct Polyline {
    pub label: String,
    #[holder(use_place_holder)]
    pub points: Vec<CartesianPoint>,
}
impl<'a, P: From<&'a CartesianPoint>> From<&'a Polyline> for PolylineCurve<P> {
    #[inline(always)]
    fn from(poly: &'a Polyline) -> Self { Self(poly.points.iter().map(|pt| P::from(pt)).collect()) }
}

/// `b_spline_curve_form`
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BsplineCurveForm {
    PolylineForm,
    CircularArc,
    EllipticArc,
    ParabolicArc,
    HyperbolicArc,
    Unspecified,
}

/// `knot_type`
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KnotType {
    UniformKnots,
    Unspecified,
    QuasiUniformKnots,
    PiecewiseBezierKnots,
}

/// `b_spline_curve_with_knots`
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(field = b_spline_curve_with_knots)]
#[holder(generate_deserialize)]
pub struct BsplineCurveWithKnots {
    pub label: String,
    pub degree: i64,
    #[holder(use_place_holder)]
    pub control_points_list: Vec<CartesianPoint>,
    pub curve_form: BsplineCurveForm,
    pub closed_curve: Logical,
    pub self_intersect: Logical,
    pub knot_multiplicities: Vec<i64>,
    pub knots: Vec<f64>,
    pub knot_spec: KnotType,
}
impl<P: for<'a> From<&'a CartesianPoint>> TryFrom<&BsplineCurveWithKnots> for BsplineCurve<P> {
    type Error = StepConvertingError;
    #[inline(always)]
    fn try_from(curve: &BsplineCurveWithKnots) -> Result<Self, StepConvertingError> {
        let knots = curve.knots.clone();
        let multi = curve
            .knot_multiplicities
            .iter()
            .map(|n| *n as usize)
            .collect();
        let knots = KnotVector::from_single_multi(knots, multi)?;
        let ctrpts = curve.control_points_list.iter().map(Into::into).collect();
        Ok(Self::try_new(knots, ctrpts)?)
    }
}

/// `bezier_curve`
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(field = bezier_curve)]
#[holder(generate_deserialize)]
pub struct BezierCurve {
    pub label: String,
    pub degree: i64,
    #[holder(use_place_holder)]
    pub control_points_list: Vec<CartesianPoint>,
    pub curve_form: BsplineCurveForm,
    pub closed_curve: Logical,
    pub self_intersect: Logical,
}
impl<P: for<'a> From<&'a CartesianPoint>> TryFrom<&BezierCurve> for BsplineCurve<P> {
    type Error = StepConvertingError;
    #[inline(always)]
    fn try_from(curve: &BezierCurve) -> Result<Self, StepConvertingError> {
        let degree = curve.degree as usize;
        let knots = KnotVector::bezier_knot(degree);
        let ctrpts = curve.control_points_list.iter().map(Into::into).collect();
        Ok(Self::try_new(knots, ctrpts)?)
    }
}

/// `quasi_uniform_curve`
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(field = quasi_uniform_curve)]
#[holder(generate_deserialize)]
pub struct QuasiUniformCurve {
    pub label: String,
    pub degree: i64,
    #[holder(use_place_holder)]
    pub control_points_list: Vec<CartesianPoint>,
    pub curve_form: BsplineCurveForm,
    pub closed_curve: Logical,
    pub self_intersect: Logical,
}
impl<P: for<'a> From<&'a CartesianPoint>> TryFrom<&QuasiUniformCurve> for BsplineCurve<P> {
    type Error = StepConvertingError;
    #[inline(always)]
    fn try_from(curve: &QuasiUniformCurve) -> Result<Self, StepConvertingError> {
        let knots = quasi_uniform_knots(curve.control_points_list.len(), curve.degree as usize);
        let ctrpts = curve.control_points_list.iter().map(Into::into).collect();
        Ok(Self::try_new(knots, ctrpts)?)
    }
}

/// `uniform_curve`
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(field = uniform_curve)]
#[holder(generate_deserialize)]
pub struct UniformCurve {
    pub label: String,
    pub degree: i64,
    #[holder(use_place_holder)]
    pub control_points_list: Vec<CartesianPoint>,
    pub curve_form: BsplineCurveForm,
    pub closed_curve: Logical,
    pub self_intersect: Logical,
}
impl<P: for<'a> From<&'a CartesianPoint>> TryFrom<&UniformCurve> for BsplineCurve<P> {
    type Error = StepConvertingError;
    #[inline(always)]
    fn try_from(curve: &UniformCurve) -> Result<Self, StepConvertingError> {
        let knots = uniform_knots(curve.control_points_list.len(), curve.degree as usize)?;
        let ctrpts = curve.control_points_list.iter().map(Into::into).collect();
        Ok(Self::try_new(knots, ctrpts)?)
    }
}

/// Entity that does not exist in AP042.
/// Curve before rationalization of [`RationalBsplineCurve`] defined by a complex entity
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(generate_deserialize)]
pub enum NonRationalBsplineCurve {
    #[holder(use_place_holder)]
    BsplineCurveWithKnots(BsplineCurveWithKnots),
    #[holder(use_place_holder)]
    BezierCurve(BezierCurve),
    #[holder(use_place_holder)]
    QuasiUniformCurve(QuasiUniformCurve),
    #[holder(use_place_holder)]
    UniformCurve(UniformCurve),
}

impl<P: for<'a> From<&'a CartesianPoint>> TryFrom<&NonRationalBsplineCurve> for BsplineCurve<P> {
    type Error = StepConvertingError;
    #[inline(always)]
    fn try_from(curve: &NonRationalBsplineCurve) -> Result<Self, StepConvertingError> {
        use NonRationalBsplineCurve::*;
        match curve {
            BsplineCurveWithKnots(x) => x.try_into(),
            BezierCurve(x) => x.try_into(),
            QuasiUniformCurve(x) => x.try_into(),
            UniformCurve(x) => x.try_into(),
        }
    }
}

/// `rational_b_spline_curve` as complex entity
///
/// This struct is an ad hoc implementation that differs from the definition by EXPRESS:
/// in AP042, rationalized curves are defined as complex entities,
/// but here the curves before rationalization are held as internal variables.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(field = rational_b_spline_curve)]
#[holder(generate_deserialize)]
pub struct RationalBsplineCurve {
    #[holder(use_place_holder)]
    pub non_rational_b_spline_curve: NonRationalBsplineCurve,
    pub weights_data: Vec<f64>,
}
impl<V> TryFrom<&RationalBsplineCurve> for NurbsCurve<V>
where
    V: Homogeneous<Scalar = f64>,
    V::Point: for<'a> From<&'a CartesianPoint>,
{
    type Error = StepConvertingError;
    #[inline(always)]
    fn try_from(curve: &RationalBsplineCurve) -> Result<Self, StepConvertingError> {
        Ok(Self::try_from_bspline_and_weights(
            BsplineCurve::try_from(&curve.non_rational_b_spline_curve)?,
            curve.weights_data.clone(),
        )?)
    }
}

/// b_spline_curve
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(generate_deserialize)]
pub enum BsplineCurveAny {
    #[holder(use_place_holder)]
    NonRationalBsplineCurve(Box<NonRationalBsplineCurve>),
    #[holder(use_place_holder)]
    RationalBsplineCurve(Box<RationalBsplineCurve>),
}

impl TryFrom<&BsplineCurveAny> for Curve2D {
    type Error = StepConvertingError;
    #[inline(always)]
    fn try_from(value: &BsplineCurveAny) -> Result<Self, Self::Error> {
        use BsplineCurveAny::*;
        Ok(match value {
            NonRationalBsplineCurve(bsp) => Self::BsplineCurve(bsp.as_ref().try_into()?),
            RationalBsplineCurve(bsp) => Self::NurbsCurve(bsp.as_ref().try_into()?),
        })
    }
}

impl TryFrom<&BsplineCurveAny> for Curve3D {
    type Error = StepConvertingError;
    #[inline(always)]
    fn try_from(value: &BsplineCurveAny) -> Result<Self, Self::Error> {
        use BsplineCurveAny::*;
        Ok(match value {
            NonRationalBsplineCurve(bsp) => Self::BsplineCurve(bsp.as_ref().try_into()?),
            RationalBsplineCurve(bsp) => Self::NurbsCurve(bsp.as_ref().try_into()?),
        })
    }
}

/// `conic`
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(generate_deserialize)]
pub enum Conic {
    #[holder(use_place_holder)]
    Circle(Circle),
    #[holder(use_place_holder)]
    Ellipse(Ellipse),
    #[holder(use_place_holder)]
    Hyperbola(Hyperbola),
    #[holder(use_place_holder)]
    Parabola(Parabola),
}

impl TryFrom<&Conic> for Conic2D {
    type Error = StepConvertingError;
    #[inline(always)]
    fn try_from(value: &Conic) -> Result<Self, Self::Error> {
        Ok(match value {
            Conic::Circle(value) => Conic2D::Ellipse(value.try_into()?),
            Conic::Ellipse(value) => Conic2D::Ellipse(value.try_into()?),
            Conic::Hyperbola(value) => Conic2D::Hyperbola(value.try_into()?),
            Conic::Parabola(value) => Conic2D::Parabola(value.try_into()?),
        })
    }
}

impl TryFrom<&Conic> for Conic3D {
    type Error = StepConvertingError;
    #[inline(always)]
    fn try_from(value: &Conic) -> Result<Self, Self::Error> {
        Ok(match value {
            Conic::Circle(value) => Conic3D::Ellipse(value.try_into()?),
            Conic::Ellipse(value) => Conic3D::Ellipse(value.try_into()?),
            Conic::Hyperbola(value) => Conic3D::Hyperbola(value.try_into()?),
            Conic::Parabola(value) => Conic3D::Parabola(value.try_into()?),
        })
    }
}

/// `circle`
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(field = circle)]
#[holder(generate_deserialize)]
pub struct Circle {
    pub label: String,
    #[holder(use_place_holder)]
    pub position: Axis2Placement,
    pub radius: f64,
}

impl TryFrom<&Circle> for step_geometry::Ellipse<Point2, Matrix3> {
    type Error = StepConvertingError;
    #[inline(always)]
    fn try_from(circle: &Circle) -> Result<Self, Self::Error> {
        let transform = Matrix3::try_from(&circle.position)? * Matrix3::from_scale(circle.radius);
        Ok(
            Processor::new(geom::TrimmedCurve::new(UnitCircle::new(), (0.0, 2.0 * PI)))
                .transformed(transform),
        )
    }
}

impl TryFrom<&Circle> for step_geometry::Ellipse<Point3, Matrix4> {
    type Error = StepConvertingError;
    #[inline(always)]
    fn try_from(circle: &Circle) -> Result<Self, Self::Error> {
        let transform = Matrix4::try_from(&circle.position)? * Matrix4::from_scale(circle.radius);
        Ok(
            Processor::new(geom::TrimmedCurve::new(UnitCircle::new(), (0.0, 2.0 * PI)))
                .transformed(transform),
        )
    }
}

/// `ellipse`
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(field = ellipse)]
#[holder(generate_deserialize)]
pub struct Ellipse {
    pub label: String,
    #[holder(use_place_holder)]
    pub position: Axis2Placement,
    pub semi_axis_1: f64,
    pub semi_axis_2: f64,
}

impl TryFrom<&Ellipse> for step_geometry::Ellipse<Point2, Matrix3> {
    type Error = StepConvertingError;
    #[inline(always)]
    fn try_from(ellipse: &Ellipse) -> Result<Self, Self::Error> {
        let (r0, r1) = (ellipse.semi_axis_1, ellipse.semi_axis_2);
        let transform =
            Matrix3::try_from(&ellipse.position)? * Matrix3::from_nonuniform_scale(r0, r1);
        Ok(
            Processor::new(geom::TrimmedCurve::new(UnitCircle::new(), (0.0, 2.0 * PI)))
                .transformed(transform),
        )
    }
}

impl TryFrom<&Ellipse> for step_geometry::Ellipse<Point3, Matrix4> {
    type Error = StepConvertingError;
    #[inline(always)]
    fn try_from(ellipse: &Ellipse) -> Result<Self, Self::Error> {
        let (r0, r1) = (ellipse.semi_axis_1, ellipse.semi_axis_2);
        let transform = Matrix4::try_from(&ellipse.position)?
            * Matrix4::from_nonuniform_scale(r0, r1, f64::min(r0, r1));
        Ok(
            Processor::new(geom::TrimmedCurve::new(UnitCircle::new(), (0.0, 2.0 * PI)))
                .transformed(transform),
        )
    }
}

/// `hyperbola`
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(field = hyperbola)]
#[holder(generate_deserialize)]
pub struct Hyperbola {
    pub label: String,
    #[holder(use_place_holder)]
    pub position: Axis2Placement,
    pub semi_axis: f64,
    pub semi_imag_axis: f64,
}

impl TryFrom<&Hyperbola> for step_geometry::Hyperbola<Point2, Matrix3> {
    type Error = StepConvertingError;
    #[inline(always)]
    fn try_from(hyperbola: &Hyperbola) -> Result<Self, Self::Error> {
        let (r0, r1) = (hyperbola.semi_axis, hyperbola.semi_imag_axis);
        let transform =
            Matrix3::try_from(&hyperbola.position)? * Matrix3::from_nonuniform_scale(r0, r1);
        Ok(
            Processor::new(geom::TrimmedCurve::new(UnitHyperbola::new(), (-1.0, 1.0)))
                .transformed(transform),
        )
    }
}

impl TryFrom<&Hyperbola> for step_geometry::Hyperbola<Point3, Matrix4> {
    type Error = StepConvertingError;
    #[inline(always)]
    fn try_from(hyperbola: &Hyperbola) -> Result<Self, Self::Error> {
        let (r0, r1) = (hyperbola.semi_axis, hyperbola.semi_imag_axis);
        let transform = Matrix4::try_from(&hyperbola.position)?
            * Matrix4::from_nonuniform_scale(r0, r1, f64::min(r0, r1));
        Ok(
            Processor::new(geom::TrimmedCurve::new(UnitHyperbola::new(), (-1.0, 1.0)))
                .transformed(transform),
        )
    }
}

/// `parabola`
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(field = parabola)]
#[holder(generate_deserialize)]
pub struct Parabola {
    pub label: String,
    #[holder(use_place_holder)]
    pub position: Axis2Placement,
    pub focal_dist: f64,
}

impl TryFrom<&Parabola> for step_geometry::Parabola<Point2, Matrix3> {
    type Error = StepConvertingError;
    #[inline(always)]
    fn try_from(parabola: &Parabola) -> Result<Self, Self::Error> {
        let transform =
            Matrix3::try_from(&parabola.position)? * Matrix3::from_scale(parabola.focal_dist);
        Ok(
            Processor::new(geom::TrimmedCurve::new(UnitParabola::new(), (-1.0, 1.0)))
                .transformed(transform),
        )
    }
}

impl TryFrom<&Parabola> for step_geometry::Parabola<Point3, Matrix4> {
    type Error = StepConvertingError;
    #[inline(always)]
    fn try_from(parabola: &Parabola) -> Result<Self, Self::Error> {
        let transform =
            Matrix4::try_from(&parabola.position)? * Matrix4::from_scale(parabola.focal_dist);
        Ok(
            Processor::new(geom::TrimmedCurve::new(UnitParabola::new(), (-1.0, 1.0)))
                .transformed(transform),
        )
    }
}

/// Element of a [`GeometricCurveSet`] -- a point or curve reference.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(generate_deserialize)]
pub enum GeometricSetSelect {
    #[holder(use_place_holder)]
    Curve(Box<CurveAny>),
    #[holder(use_place_holder)]
    Point(Box<CartesianPoint>),
}

/// `geometric_curve_set` -- a set of standalone 3D curves (and optionally points).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(field = geometric_curve_set)]
#[holder(generate_deserialize)]
pub struct GeometricCurveSet {
    /// Label from the STEP entity.
    pub label: String,
    /// Elements of the set (curves and/or points).
    #[holder(use_place_holder)]
    pub elements: Vec<GeometricSetSelect>,
}

#[cfg(test)]
mod trimmed_curve_source_regressions {
    use super::*;
    use monstertruck_modeling::{
        Curve as NativeCurve, Curve2D as NativeCurve2D, Surface as NativeSurface,
    };

    fn source_table(data: &str) -> Table {
        Table::from_step(&format!(
            "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('trim regression'),'2;1');\n\
             FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n{data}\nENDSEC;\nEND-ISO-10303-21;"
        )).expect("numeric source trim entities parse")
    }

    fn ellipse_source() -> Table {
        source_table(
            "#1=CARTESIAN_POINT('',(7.,11.,13.));\n\
             #2=DIRECTION('',(0.,0.,1.));\n\
             #3=DIRECTION('',(1.,0.,0.));\n\
             #4=AXIS2_PLACEMENT_3D('',#1,#2,#3);\n\
             #5=ELLIPSE('',#4,2.,3.);\n\
             #6=TRIMMED_CURVE('',#5,(PARAMETER_VALUE(5.7)),(PARAMETER_VALUE(0.2)),.F.,.PARAMETER.);\n\
             #7=VECTOR('',#2,5.);\n\
             #8=SURFACE_OF_LINEAR_EXTRUSION('',#6,#7);",
        )
    }

    #[test]
    fn numeric_inverse_trim_retains_original_radians_and_native_identity_chart() {
        let table = ellipse_source();
        let source = EntityTable::<CurveAnyHolder>::get_owned(&table, 6)
            .expect("declared numeric trim resolves");
        let source = Curve3D::try_from(&source).expect("source trim is analytic");
        let (native, map) = source
            .to_modeling_with_parameter_map()
            .expect("ordinary native trim preserves its source parameter");
        assert!(matches!(native, NativeCurve::TrimmedCurve(_)));
        assert_eq!(map.kind, CurveParameterMapKind::Identity);
        assert_eq!(source.range_tuple(), (0.2, 5.7));
        assert_eq!(native.range_tuple(), (0.2, 5.7));
        for parameter in [0.2, 0.37, 0.91, 5.7] {
            let angle: f64 = 0.2 + 5.7 - parameter;
            let expected = Point3::new(7.0 + 2.0 * angle.cos(), 11.0 + 3.0 * angle.sin(), 13.0);
            let derivative = Vector3::new(2.0 * angle.sin(), -3.0 * angle.cos(), 0.0);
            assert!(source.subs(parameter).near(&expected));
            assert!(native.subs(parameter).near(&expected));
            assert!(source.der(parameter).near(&derivative));
            assert!(native.der(parameter).near(&derivative));
            assert_eq!(map.source_to_native(parameter), Some(parameter));
        }
    }

    #[test]
    fn inverse_trim_cuts_keep_the_original_anchor_and_repeated_forward_ranges() {
        let table = ellipse_source();
        let source = EntityTable::<CurveAnyHolder>::get_owned(&table, 6)
            .expect("numeric source trim resolves");
        let original = Curve3D::try_from(&source).expect("source trim converts");
        let mut first = original.clone();
        let mut second = first.cut(0.91);
        let third = second.cut(1.7);
        for (curve, expected_range) in [
            (&first, (0.2, 0.91)),
            (&second, (0.91, 1.7)),
            (&third, (1.7, 5.7)),
        ] {
            assert_eq!(curve.range_tuple(), expected_range);
            for parameter in [
                expected_range.0,
                expected_range.0 + (expected_range.1 - expected_range.0) * 0.37,
                expected_range.1,
            ] {
                assert!(curve.subs(parameter).near(&original.subs(parameter)));
                assert!(curve.der(parameter).near(&original.der(parameter)));
            }
        }
    }

    #[test]
    fn declared_inverse_profile_extrudes_in_its_original_nondefault_domain() {
        let table = ellipse_source();
        let source = table.surface_of_linear_extrusion[&8]
            .clone()
            .into_owned(&table)
            .expect("trimmed extrusion holder resolves");
        let source = StepExtrusionSurface::try_from(&source).expect("generic source extrusion");
        let source = Surface::SweepSurface(SweepSurface::ExtrusionSurface(source));
        let native = NativeSurface::try_from(&source).expect("generic native extrusion");
        for u in [0.2, 0.37, 0.91, 5.7] {
            let angle = 0.2 + 5.7 - u;
            for v in [-3.0, 9.0, 12.0] {
                assert!(native.subs(u, v).near(&Point3::new(
                    7.0 + 2.0 * angle.cos(),
                    11.0 + 3.0 * angle.sin(),
                    13.0 + 5.0 * v,
                )));
                assert!(native.uder(u, v).near(&Vector3::new(
                    2.0 * angle.sin(),
                    -3.0 * angle.cos(),
                    0.0,
                )));
                assert!(native.vder(u, v).near(&Vector3::new(0.0, 0.0, 5.0)));
            }
        }
    }

    #[test]
    fn actual_native_writer_roundtrip_preserves_inverse_profile_points_derivatives_and_normals() {
        let table = ellipse_source();
        let owned = table.surface_of_linear_extrusion[&8]
            .clone()
            .into_owned(&table)
            .expect("declared trimmed profile resolves");
        let source = Surface::SweepSurface(SweepSurface::ExtrusionSurface(
            StepExtrusionSurface::try_from(&owned).expect("source extrusion"),
        ));
        let native = NativeSurface::try_from(&source).expect("native generic extrusion");
        let emitted = format!("{}", crate::step::save::StepDisplay::new(&native, 100));
        assert!(emitted.contains("TRIMMED_CURVE"));
        assert!(emitted.contains("PARAMETER_VALUE"));
        let reparsed = source_table(&emitted);
        let owned = reparsed.surface_of_linear_extrusion[&100]
            .clone()
            .into_owned(&reparsed)
            .expect("actual emitted extrusion holder resolves");
        let source = Surface::SweepSurface(SweepSurface::ExtrusionSurface(
            StepExtrusionSurface::try_from(&owned).expect("emitted source extrusion"),
        ));
        let raw_to_decoded = source
            .raw_surface_parameter_chart()
            .expect("source raw chart");
        let (roundtrip, source_to_native) = source
            .to_modeling_with_parameter_map()
            .expect("emitted native extrusion and bounded raw chart");
        let native_to_raw = crate::step::save::StepSurface::step_parameter_chart(&native)
            .expect("actual writer's emitted surface chart");
        for u in [0.2, 0.37, 0.91, 5.7] {
            let angle = 0.2 + 5.7 - u;
            for v in [-3.0, 9.0, 12.0] {
                let original_bounds = [
                    crate::step::ParameterEnclosure::point(u).expect("finite u"),
                    crate::step::ParameterEnclosure::point(v).expect("finite v"),
                ];
                let raw = native_to_raw
                    .nominal([u, v])
                    .expect("finite emitted coordinates");
                let decoded = raw_to_decoded
                    .nominal(raw)
                    .expect("finite decoded coordinates");
                let restored_bounds = raw_to_decoded
                    .enclose(
                        native_to_raw
                            .enclose(original_bounds)
                            .expect("writer coordinate enclosure"),
                    )
                    .expect("reader coordinate enclosure");
                assert!(restored_bounds[0].contains(u));
                assert!(restored_bounds[1].contains(v));
                let (u, v) = source_to_native
                    .source_to_native((raw[0], raw[1]))
                    .expect("raw declared coordinates map to the actual native chart");
                assert!(
                    source
                        .subs(decoded[0], decoded[1])
                        .near(&roundtrip.subs(u, v))
                );
                assert!(roundtrip.subs(u, v).near(&Point3::new(
                    7.0 + 2.0 * angle.cos(),
                    11.0 + 3.0 * angle.sin(),
                    13.0 + 5.0 * v,
                )));
                assert!(roundtrip.uder(u, v).near(&Vector3::new(
                    2.0 * angle.sin(),
                    -3.0 * angle.cos(),
                    0.0,
                )));
                assert!(roundtrip.vder(u, v).near(&Vector3::new(0.0, 0.0, 5.0)));
                assert!(roundtrip.normal(u, v).near(
                    &Vector3::new(-15.0 * angle.cos(), -10.0 * angle.sin(), 0.0).normalize(),
                ));
            }
        }
    }

    #[test]
    fn two_dimensional_line_spline_and_rational_basis_keep_their_numeric_domains() {
        let table = source_table(
            "#1=CARTESIAN_POINT('',(2.,4.));\n\
             #2=DIRECTION('',(1.,0.));\n\
             #3=VECTOR('',#2,2.);\n\
             #4=LINE('',#1,#3);\n\
             #5=TRIMMED_CURVE('',#4,(PARAMETER_VALUE(7.)),(PARAMETER_VALUE(-3.)),.F.,.PARAMETER.);\n\
             #6=CARTESIAN_POINT('',(0.,0.));\n\
             #7=CARTESIAN_POINT('',(4.,8.));\n\
             #8=B_SPLINE_CURVE_WITH_KNOTS('',1,(#6,#7),.UNSPECIFIED.,.F.,.F.,(2,2),(2.,6.),.UNSPECIFIED.);\n\
             #9=TRIMMED_CURVE('',#8,(PARAMETER_VALUE(2.5)),(PARAMETER_VALUE(5.5)),.F.,.PARAMETER.);\n\
             #10=(BOUNDED_CURVE() B_SPLINE_CURVE(1,(#6,#7),.UNSPECIFIED.,.F.,.F.) \
             B_SPLINE_CURVE_WITH_KNOTS((2,2),(2.,6.),.UNSPECIFIED.) CURVE() GEOMETRIC_REPRESENTATION_ITEM() \
             RATIONAL_B_SPLINE_CURVE((1.,2.)) REPRESENTATION_ITEM(''));\n\
             #11=TRIMMED_CURVE('',#10,(PARAMETER_VALUE(2.5)),(PARAMETER_VALUE(5.5)),.T.,.PARAMETER.);",
        );
        for (id, range) in [(5, (-3.0, 7.0)), (9, (2.5, 5.5)), (11, (2.5, 5.5))] {
            let owned = EntityTable::<CurveAnyHolder>::get_owned(&table, id)
                .expect("numeric basis resolves");
            let source = Curve2D::try_from(&owned).expect("generic 2D source trim");
            let native = NativeCurve2D::try_from(&source).expect("generic 2D native trim");
            assert_eq!(source.range_tuple(), range);
            assert_eq!(native.range_tuple(), range);
            for t in [range.0, range.0 + (range.1 - range.0) * 0.37, range.1] {
                assert!(source.subs(t).near(&native.subs(t)));
                assert!(source.der(t).near(&native.der(t)));
                if id == 5 {
                    assert!(
                        native
                            .subs(t)
                            .near(&Point2::new(2.0 + 2.0 * (4.0 - t), 4.0))
                    );
                }
            }
        }
    }

    #[test]
    fn trim_cycle_and_depth_refuse_before_recursive_holder_materialization() {
        let cyclic = source_table(
            "#1=TRIMMED_CURVE('',#2,(PARAMETER_VALUE(0.2)),(PARAMETER_VALUE(5.7)),.T.,.PARAMETER.);\n\
             #2=SURFACE_CURVE('',#1,(),.CURVE_3D.);",
        );
        assert!(matches!(
            cyclic.validate_curve_source(&PlaceHolder::Ref(Name::Entity(1))),
            Err(ExactConversionError::CyclicSourceGeometry { entity: 1 })
        ));
        assert!(
            EntityTable::<CurveAnyHolder>::get_owned(&cyclic, 1)
                .expect_err("cyclic trim must refuse before clone")
                .to_string()
                .contains("cyclic")
        );
        let chain: String = (1..=MAX_NATIVE_GEOMETRY_NESTING + 2).map(|id| format!(
            "#{id}=TRIMMED_CURVE('',#{},(PARAMETER_VALUE(0.2)),(PARAMETER_VALUE(5.7)),.T.,.PARAMETER.);\n",
            id + 1,
        )).collect();
        let deep = source_table(&chain);
        assert_eq!(
            deep.validate_curve_source(&PlaceHolder::Ref(Name::Entity(1))),
            Err(ExactConversionError::SourceGeometryNestingExceeded)
        );
        assert!(
            EntityTable::<CurveAnyHolder>::get_owned(&deep, 1)
                .expect_err("deep trim must refuse before clone")
                .to_string()
                .contains("nesting")
        );
    }

    #[test]
    fn owned_inline_trim_depth_is_bounded_before_two_or_three_dimensional_conversion() {
        let table = source_table(
            "#1=CARTESIAN_POINT('',(0.,0.));\n#2=DIRECTION('',(1.,0.));\n\
             #3=VECTOR('',#2,1.);\n#4=LINE('',#1,#3);",
        );
        let basis = EntityTable::<CurveAnyHolder>::get_owned(&table, 4).expect("line resolves");
        let deep = (0..MAX_NATIVE_GEOMETRY_NESTING + 1).fold(basis, |basis_curve, _| {
            CurveAny::TrimmedCurve(Box::new(StepTrimmedCurve {
                label: String::new(),
                basis_curve,
                trim_1: vec![TrimmingSelect::ParameterValue(Box::new(ParameterValue(
                    0.2,
                )))],
                trim_2: vec![TrimmingSelect::ParameterValue(Box::new(ParameterValue(
                    0.9,
                )))],
                sense_agreement: true,
                master_representation: TrimmingPreference::Parameter,
            }))
        });
        for error in [
            Curve2D::try_from(&deep).expect_err("2D depth bound"),
            Curve3D::try_from(&deep).expect_err("3D depth bound"),
        ] {
            assert!(matches!(
                error.downcast_ref::<ExactConversionError>(),
                Some(ExactConversionError::SourceGeometryNestingExceeded)
            ));
        }
    }

    #[test]
    fn cartesian_only_and_ambiguous_numeric_trim_selectors_refuse_typed() {
        let table = source_table(
            "#1=CARTESIAN_POINT('',(0.,0.));\n#2=DIRECTION('',(1.,0.));\n\
             #3=VECTOR('',#2,1.);\n#4=LINE('',#1,#3);\n\
             #5=TRIMMED_CURVE('',#4,(#1),(#1),.T.,.CARTESIAN.);\n\
             #6=TRIMMED_CURVE('',#4,(PARAMETER_VALUE(0.2),PARAMETER_VALUE(0.3)),\
             (PARAMETER_VALUE(0.9)),.T.,.PARAMETER.);",
        );
        for (id, expected) in [
            (5, ExactConversionError::CartesianTrimSelectors),
            (6, ExactConversionError::InvalidTrimParameterSelectors),
        ] {
            let owned = EntityTable::<CurveAnyHolder>::get_owned(&table, id)
                .expect("trim selectors resolve");
            let error = Curve2D::try_from(&owned).expect_err("unproven trim selectors must refuse");
            assert_eq!(
                error.downcast_ref::<ExactConversionError>(),
                Some(&expected)
            );
        }
    }

    #[test]
    fn parameter_preference_preserves_numeric_selectors_and_real_periodic_branch_lifts() {
        let table = source_table(
            "#1=CARTESIAN_POINT('',(0.,0.));\n#2=DIRECTION('',(1.,0.));\n\
             #3=AXIS2_PLACEMENT_2D('',#1,#2);\n#4=CIRCLE('',#3,2.);\n\
             #9=VECTOR('',#2,1.);\n#10=LINE('',#1,#9);\n\
             #11=CARTESIAN_POINT('',(0.2,0.));\n#12=CARTESIAN_POINT('',(5.7,0.));\n\
             #5=TRIMMED_CURVE('',#10,(#11,PARAMETER_VALUE(0.2)),\
             (PARAMETER_VALUE(5.7),#12),.T.,.PARAMETER.);\n\
             #6=TRIMMED_CURVE('',#4,(PARAMETER_VALUE(5.7)),\
             (PARAMETER_VALUE(0.2)),.T.,.PARAMETER.);\n\
             #7=TRIMMED_CURVE('',#4,(PARAMETER_VALUE(0.2)),\
             (PARAMETER_VALUE(0.2)),.T.,.PARAMETER.);",
        );
        let mixed =
            EntityTable::<CurveAnyHolder>::get_owned(&table, 5).expect("mixed selectors resolve");
        let curve = Curve2D::try_from(&mixed).expect("numeric selectors are authoritative");
        assert_eq!(curve.range_tuple(), (0.2, 5.7));
        for (id, range) in [
            (6, (5.7, 0.2 + std::f64::consts::TAU)),
            (7, (0.2, 0.2 + std::f64::consts::TAU)),
        ] {
            let owned = EntityTable::<CurveAnyHolder>::get_owned(&table, id)
                .expect("periodic selectors resolve");
            let source =
                Curve2D::try_from(&owned).expect("ordinary periodic selection has a bounded chart");
            let native = NativeCurve2D::try_from(&source).expect("ordinary native periodic trim");
            assert_eq!(source.range_tuple(), range);
            for parameter in [range.0, range.0 + (range.1 - range.0) * 0.37, range.1] {
                let expected = Point2::new(2.0 * parameter.cos(), 2.0 * parameter.sin());
                assert!(source.subs(parameter).near(&expected));
                assert!(native.subs(parameter).near(&expected));
            }
        }
    }

    #[test]
    fn literal_short_reverse_ellipse_crosses_the_seam_without_becoming_the_long_arc() {
        let table = source_table(
            "#1=CARTESIAN_POINT('',(7.,11.,13.));\n\
             #2=DIRECTION('',(0.,0.,1.));\n#3=DIRECTION('',(1.,0.,0.));\n\
             #4=AXIS2_PLACEMENT_3D('',#1,#2,#3);\n#5=ELLIPSE('',#4,2.,3.);\n\
             #6=TRIMMED_CURVE('',#5,(PARAMETER_VALUE(0.2)),(PARAMETER_VALUE(5.7)),.F.,.PARAMETER.);\n\
             #11=CARTESIAN_POINT('',(7.,11.));\n#12=DIRECTION('',(1.,0.));\n\
             #13=AXIS2_PLACEMENT_2D('',#11,#12);\n#14=ELLIPSE('',#13,2.,3.);\n\
             #15=TRIMMED_CURVE('',#14,(PARAMETER_VALUE(0.2)),(PARAMETER_VALUE(5.7)),.F.,.PARAMETER.);",
        );
        let owned =
            EntityTable::<CurveAnyHolder>::get_owned(&table, 6).expect("literal STEP reverse trim");
        let source = Curve3D::try_from(&owned).expect("short reverse branch");
        let (native, map) = source
            .to_modeling_with_parameter_map()
            .expect("bounded short reverse chart");
        let owned2 =
            EntityTable::<CurveAnyHolder>::get_owned(&table, 15).expect("literal 2D reverse trim");
        let source2 = Curve2D::try_from(&owned2).expect("2D short reverse branch");
        let native2 = NativeCurve2D::try_from(&source2).expect("2D short reverse native trim");
        assert_eq!(source.period(), None);
        assert_eq!(source2.period(), None);
        let tau = std::f64::consts::TAU;
        let range = (5.7 - tau, 0.2);
        assert_eq!(source.range_tuple(), range);
        assert_eq!(native.range_tuple(), range);
        let SourceParameterGauge::Declared {
            selected_basis_range,
            raw_to_internal,
            ..
        } = map.raw_step_gauge
        else {
            panic!("raw declaration must survive decoding");
        };
        assert_eq!(
            (selected_basis_range.lower, selected_basis_range.upper),
            range
        );
        assert!(raw_to_internal.offset_enclosure.lower < raw_to_internal.offset_enclosure.upper);
        for fraction in [0.0, 0.37, 0.91, 1.0] {
            let parameter = range.0 + (range.1 - range.0) * fraction;
            let angle = 0.2 - (0.2 + tau - 5.7) * fraction;
            let point = Point3::new(7.0 + 2.0 * angle.cos(), 11.0 + 3.0 * angle.sin(), 13.0);
            let derivative = Vector3::new(2.0 * angle.sin(), -3.0 * angle.cos(), 0.0);
            let point2 = Point2::new(point.x, point.y);
            let derivative2 = Vector2::new(derivative.x, derivative.y);
            assert!(source2.subs(parameter).near(&point2));
            assert!(native2.subs(parameter).near(&point2));
            assert!(source2.der(parameter).near(&derivative2));
            assert!(native2.der(parameter).near(&derivative2));
            assert!(source.subs(parameter).near(&point));
            assert!(native.subs(parameter).near(&point));
            assert!(source.der(parameter).near(&derivative));
            assert!(native.der(parameter).near(&derivative));
        }
    }

    #[test]
    fn negative_reverse_circle_selectors_normalize_before_reversal_and_preserve_nested_endpoints() {
        let table = source_table(
            "#1=CARTESIAN_POINT('',(0.,0.));\n#2=DIRECTION('',(1.,0.));\n\
             #3=AXIS2_PLACEMENT_2D('',#1,#2);\n#4=CIRCLE('',#3,2.);\n\
             #5=TRIMMED_CURVE('',#4,(PARAMETER_VALUE(-0.2)),(PARAMETER_VALUE(-2.1)),.F.,.PARAMETER.);\n\
             #6=TRIMMED_CURVE('',#5,(PARAMETER_VALUE(0.2)),(PARAMETER_VALUE(2.1)),.T.,.PARAMETER.);\n\
             #7=TRIMMED_CURVE('',#5,(PARAMETER_VALUE(0.199)),(PARAMETER_VALUE(2.1)),.T.,.PARAMETER.);",
        );
        for id in [5, 6] {
            let owned = EntityTable::<CurveAnyHolder>::get_owned(&table, id)
                .expect("literal reflected-circle trim");
            let source = Curve2D::try_from(&owned)
                .expect("canonical reverse trim including named endpoints");
            let native = NativeCurve2D::try_from(&source).expect("unchanged analytic restriction");
            assert_eq!(source.range_tuple(), (-2.1, -0.2));
            assert_eq!(native.range_tuple(), (-2.1, -0.2));
            assert_eq!(source.period(), None);
            let SourceParameterGauge::Declared {
                raw_to_internal,
                raw_range,
                raw_parameter_bounds,
                selected_basis_range,
                ..
            } = source
                .raw_parameter_gauge()
                .expect("declared canonical chart")
            else {
                panic!("explicit trim has a declared chart");
            };
            assert_eq!((raw_range.lower, raw_range.upper), (0.2, 2.1));
            assert!(raw_parameter_bounds[0].contains(0.2));
            assert!(raw_parameter_bounds[1].contains(2.1));
            if id == 5 {
                assert_eq!(
                    (selected_basis_range.lower, selected_basis_range.upper),
                    (-2.1, -0.2)
                );
                assert!(
                    raw_to_internal.offset_enclosure.lower < raw_to_internal.offset_enclosure.upper
                );
            }
            for parameter in [0.2, 0.91, 2.1] {
                let decoded = raw_to_internal
                    .nominal(parameter)
                    .expect("canonical-to-decoded scalar");
                let point = Point2::new(2.0 * parameter.cos(), -2.0 * parameter.sin());
                let derivative = Vector2::new(-2.0 * parameter.sin(), -2.0 * parameter.cos());
                assert!(source.subs(decoded).near(&point));
                assert!(native.subs(decoded).near(&point));
                assert!((source.der(decoded) * raw_to_internal.scale).near(&derivative));
            }
        }
        let owned =
            EntityTable::<CurveAnyHolder>::get_owned(&table, 7).expect("literal escaped trim");
        let error = Curve2D::try_from(&owned)
            .expect_err("finite explicit trim cannot wrap to admit an escape");
        assert_eq!(
            error.downcast_ref::<ExactConversionError>(),
            Some(&ExactConversionError::InvalidTrimParameterSelectors)
        );
    }

    #[test]
    fn nonperiodic_selector_order_not_sense_alone_determines_physical_line_traversal() {
        for (first, second, sense) in [
            (-3.0, 7.0, true),
            (-3.0, 7.0, false),
            (7.0, -3.0, true),
            (7.0, -3.0, false),
        ] {
            let sense = if sense { ".T." } else { ".F." };
            let table = source_table(&format!(
                "#1=CARTESIAN_POINT('',(2.,4.));\n#2=DIRECTION('',(2.,0.));\n\
                 #3=VECTOR('',#2,3.);\n#4=LINE('',#1,#3);\n\
                 #5=TRIMMED_CURVE('',#4,(PARAMETER_VALUE({first:.1})),\
                 (PARAMETER_VALUE({second:.1})),{sense},.PARAMETER.);\n\
                 #6=CARTESIAN_POINT('',(2.,4.,6.));\n#7=DIRECTION('',(0.,0.,2.));\n\
                 #8=VECTOR('',#7,3.);\n#9=LINE('',#6,#8);\n\
                 #10=TRIMMED_CURVE('',#9,(PARAMETER_VALUE({first:.1})),\
                 (PARAMETER_VALUE({second:.1})),{sense},.PARAMETER.);"
            ));
            let source2 =
                EntityTable::<CurveAnyHolder>::get_owned(&table, 5).expect("2D line trim");
            let source2 = Curve2D::try_from(&source2).expect("2D source line trim");
            let native2 = NativeCurve2D::try_from(&source2).expect("2D native line trim");
            let source3 =
                EntityTable::<CurveAnyHolder>::get_owned(&table, 10).expect("3D line trim");
            let source3 = Curve3D::try_from(&source3).expect("3D source line trim");
            let native3 = NativeCurve::try_from(&source3).expect("3D native line trim");
            for fraction in [0.0, 0.37, 1.0] {
                let parameter = -3.0 + 10.0 * fraction;
                let original = first + (second - first) * fraction;
                let point2 = Point2::new(2.0 + 3.0 * original, 4.0);
                let point3 = Point3::new(2.0, 4.0, 6.0 + 3.0 * original);
                assert!(source2.subs(parameter).near(&point2));
                assert!(native2.subs(parameter).near(&point2));
                assert!(source3.subs(parameter).near(&point3));
                assert!(native3.subs(parameter).near(&point3));
            }
        }
    }

    #[test]
    fn nested_reverse_basis_maps_raw_selectors_and_rejects_escape_from_its_active_interval() {
        let mut table = ellipse_source();
        let shift = std::f64::consts::TAU - (0.2 + 5.7);
        let selector = |value| {
            PlaceHolder::Owned(TrimmingSelectHolder::ParameterValue(Box::new(
                ParameterValueHolder(value),
            )))
        };
        table.trimmed_curve.insert(
            9,
            StepTrimmedCurveHolder {
                label: String::new(),
                basis_curve: PlaceHolder::Ref(Name::Entity(6)),
                trim_1: vec![selector(1.0 + shift)],
                trim_2: vec![selector(2.0 + shift)],
                sense_agreement: true,
                master_representation: TrimmingPreference::Parameter,
            },
        );
        let owned = EntityTable::<CurveAnyHolder>::get_owned(&table, 9).expect("nested raw source");
        let source = Curve3D::try_from(&owned)
            .expect("mapped outer selectors lie in the inner source domain");
        let native = NativeCurve::try_from(&source).expect("nested native restriction");
        let (start, end) = source.range_tuple();
        assert!((start - 1.0).abs() < 1.0e-14 && (end - 2.0).abs() < 1.0e-14);
        for parameter in [start, start + (end - start) * 0.37, end] {
            let angle = 0.2 + 5.7 - parameter;
            let expected = Point3::new(7.0 + 2.0 * angle.cos(), 11.0 + 3.0 * angle.sin(), 13.0);
            assert!(source.subs(parameter).near(&expected));
            assert!(native.subs(parameter).near(&expected));
        }
        let outer = table
            .trimmed_curve
            .get_mut(&9)
            .expect("outer source declaration");
        outer.trim_1 = vec![selector(0.2)];
        outer.trim_2 = vec![selector(5.7)];
        let owned = EntityTable::<CurveAnyHolder>::get_owned(&table, 9)
            .expect("out-of-domain source resolves");
        let error =
            Curve3D::try_from(&owned).expect_err("an explicit trim is not periodic authority");
        assert_eq!(
            error.downcast_ref::<ExactConversionError>(),
            Some(&ExactConversionError::InvalidTrimParameterSelectors)
        );
    }

    fn vector_consumer_source() -> Table {
        source_table(
            "#1=CARTESIAN_POINT('',(7.,11.,13.));\n#2=DIRECTION('',(0.,0.,2.));\n\
             #3=VECTOR('',#2,3.);\n#4=LINE('',#1,#3);\n\
             #5=DIRECTION('',(2.,0.,0.));\n#6=VECTOR('',#5,2.);\n#7=LINE('',#1,#6);\n\
             #8=SURFACE_OF_LINEAR_EXTRUSION('',#7,#3);\n\
             #11=CARTESIAN_POINT('',(2.,4.));\n#12=DIRECTION('',(0.,2.));\n\
             #13=VECTOR('',#12,3.);\n#14=LINE('',#11,#13);\n\
             #19=(GEOMETRIC_REPRESENTATION_CONTEXT(2) PARAMETRIC_REPRESENTATION_CONTEXT() \
             REPRESENTATION_CONTEXT('',''));\n\
             #20=DEFINITIONAL_REPRESENTATION('',(#14),#19);\n#21=PCURVE('',#33,#20);\n\
             #30=DIRECTION('',(0.,0.,1.));\n#31=DIRECTION('',(1.,0.,0.));\n\
             #32=AXIS2_PLACEMENT_3D('',#1,#30,#31);\n#33=PLANE('',#32);",
        )
    }

    #[test]
    fn actual_table_line_pcurve_and_extrusion_normalize_direction_ratios_before_magnitude() {
        let table = vector_consumer_source();
        let line2 = EntityTable::<CurveAnyHolder>::get_owned(&table, 14).expect("2D LINE source");
        let line2 = Curve2D::try_from(&line2).expect("normalized 2D vector");
        let line3 = EntityTable::<CurveAnyHolder>::get_owned(&table, 4).expect("3D LINE source");
        let line3 = Curve3D::try_from(&line3).expect("normalized 3D vector");
        let pcurve =
            EntityTable::<CurveAnyHolder>::get_owned(&table, 21).expect("actual PCURVE source");
        let pcurve = Curve3D::try_from(&pcurve).expect("normalized PCURVE UV vector");
        let native_pcurve = NativeCurve::try_from(&pcurve).expect("native declared PCURVE");
        let extrusion = table.surface_of_linear_extrusion[&8]
            .clone()
            .into_owned(&table)
            .expect("actual extrusion source");
        let extrusion = Surface::SweepSurface(SweepSurface::ExtrusionSurface(
            StepExtrusionSurface::try_from(&extrusion).expect("normalized extrusion axis"),
        ));
        let native_extrusion =
            NativeSurface::try_from(&extrusion).expect("native generic extrusion");
        for parameter in [-2.0, 0.37, 3.0] {
            assert!(
                line2
                    .subs(parameter)
                    .near(&Point2::new(2.0, 4.0 + 3.0 * parameter))
            );
            assert!(
                line3
                    .subs(parameter)
                    .near(&Point3::new(7.0, 11.0, 13.0 + 3.0 * parameter))
            );
            let expected = Point3::new(9.0, 15.0 + 3.0 * parameter, 13.0);
            assert!(pcurve.subs(parameter).near(&expected));
            assert!(native_pcurve.subs(parameter).near(&expected));
            assert!(pcurve.der(parameter).near(&Vector3::new(0.0, 3.0, 0.0)));
        }
        assert!(
            native_extrusion
                .subs(0.37, 2.0)
                .near(&Point3::new(7.74, 11.0, 19.0))
        );
        assert!(
            native_extrusion
                .uder(0.37, 2.0)
                .near(&Vector3::new(2.0, 0.0, 0.0))
        );
        assert!(
            native_extrusion
                .vder(0.37, 2.0)
                .near(&Vector3::new(0.0, 0.0, 3.0))
        );
    }

    #[test]
    fn invalid_vector_semantics_propagate_through_actual_line_pcurve_and_extrusion_consumers() {
        let mut table = vector_consumer_source();
        table
            .direction
            .get_mut(&2)
            .expect("3D direction source")
            .direction_ratios = vec![0.0; 3];
        let line = EntityTable::<CurveAnyHolder>::get_owned(&table, 4)
            .expect("LINE resolves without repair");
        let error = Curve3D::try_from(&line).expect_err("zero LINE direction must refuse");
        assert!(error.downcast_ref::<VectorSemanticError>().is_some());
        let extrusion = table.surface_of_linear_extrusion[&8]
            .clone()
            .into_owned(&table)
            .expect("extrusion resolves without repair");
        let error = StepExtrusionSurface::try_from(&extrusion)
            .expect_err("zero extrusion axis must refuse");
        assert!(error.downcast_ref::<VectorSemanticError>().is_some());
        table
            .direction
            .get_mut(&12)
            .expect("UV direction source")
            .direction_ratios = vec![0.0; 2];
        let pcurve =
            EntityTable::<CurveAnyHolder>::get_owned(&table, 21).expect("PCURVE source resolves");
        let error = Curve3D::try_from(&pcurve).expect_err("zero UV LINE direction must refuse");
        assert!(error.downcast_ref::<VectorSemanticError>().is_some());
    }

    #[test]
    fn bare_line_sweep_domains_do_not_inherit_native_zero_to_one_placeholders() {
        let table = vector_consumer_source();
        let owned = table.surface_of_linear_extrusion[&8]
            .clone()
            .into_owned(&table)
            .expect("bare LINE profile source");
        let source = Surface::SweepSurface(SweepSurface::ExtrusionSurface(
            StepExtrusionSurface::try_from(&owned).expect("bare LINE extrusion"),
        ));
        let (native, map) = source
            .to_modeling_with_parameter_map()
            .expect("unbounded source profile map");
        let SurfaceParameterMap::Extrusion(profile) = map else {
            panic!("extrusion map");
        };
        assert_eq!(profile.source_range, (0.0, 1.0));
        assert_eq!(profile.source_domain, CurveParameterDomain::Unbounded);
        for u in [4.0, 6.0, 8.0] {
            for v in [-2.0, 3.0] {
                assert_eq!(map.source_to_native((u, v)), Some((u, v)));
                assert!(
                    native
                        .subs(u, v)
                        .near(&Point3::new(7.0 + 2.0 * u, 11.0, 13.0 + 3.0 * v))
                );
            }
        }
    }

    #[test]
    fn raw_and_decoded_pcurve_modes_preserve_world_geometry_native_conversion_and_export() {
        let table = source_table(
            "#1=CARTESIAN_POINT('',(7.,11.,13.));\n#2=DIRECTION('',(1.,0.,0.));\n\
             #3=VECTOR('',#2,2.);\n#4=LINE('',#1,#3);\n\
             #5=TRIMMED_CURVE('',#4,(PARAMETER_VALUE(8.)),(PARAMETER_VALUE(4.)),.T.,.PARAMETER.);\n\
             #6=DIRECTION('',(0.,0.,2.));\n#7=VECTOR('',#6,3.);\n\
             #8=SURFACE_OF_LINEAR_EXTRUSION('',#5,#7);\n\
             #9=CARTESIAN_POINT('',(-7.,0.));\n#10=DIRECTION('',(1.,0.));\n\
             #11=VECTOR('',#10,2.);\n#12=LINE('',#9,#11);\n\
             #13=TRIMMED_CURVE('',#12,(PARAMETER_VALUE(0.)),(PARAMETER_VALUE(1.)),.T.,.PARAMETER.);\n\
             #14=(GEOMETRIC_REPRESENTATION_CONTEXT(2) PARAMETRIC_REPRESENTATION_CONTEXT() \
             REPRESENTATION_CONTEXT('',''));\n\
             #15=DEFINITIONAL_REPRESENTATION('',(#13),#14);\n#16=PCURVE('',#8,#15);",
        );
        let owned =
            EntityTable::<CurveAnyHolder>::get_owned(&table, 16).expect("raw source PCURVE");
        let source = Curve3D::try_from(&owned).expect("checked raw admission");
        let Curve3D::ParameterCurve(raw) = source else {
            panic!("literal PCURVE");
        };
        assert_ne!(
            raw.input_parameter_chart(),
            crate::step::SurfaceParameterChart::identity()
        );
        assert_eq!(raw.input_parameter_chart().axes[0].offset, 12.0);
        let decoded_uv = Curve2D::TrimmedCurve(SourceTrimmedCurve::internal(Processor::new(
            TrimmedCurve::new(
                Box::new(Curve2D::Line(Line(
                    Point2::new(5.0, 0.0),
                    Point2::new(7.0, 0.0),
                ))),
                (0.0, 1.0),
            ),
        )));
        let decoded =
            StepParameterCurve::new(Box::new(decoded_uv), Box::new(raw.surface().clone()));
        assert_eq!(
            decoded.input_parameter_chart(),
            crate::step::SurfaceParameterChart::identity()
        );
        let (uv, carrier, chart) = raw.clone().decompose();
        let rebuilt = StepParameterCurve::try_from_parts(uv, carrier, chart)
            .expect("chart-carrying reconstruction");
        assert_eq!(rebuilt.input_parameter_chart(), raw.input_parameter_chart());
        let (_, surface_map) = raw
            .surface()
            .to_modeling_with_parameter_map()
            .expect("bounded profile map");
        assert_eq!(surface_map.source_to_native((-7.0, 0.0)), Some((5.0, 0.0)));
        assert_eq!(surface_map.source_to_native((-9.0, 0.0)), None);
        for pcurve in [&raw, &decoded, &rebuilt] {
            let native = NativeCurve::try_from(&Curve3D::ParameterCurve(pcurve.clone()))
                .expect("native conversion applies the admitted input chart once");
            let emitted = format!("{}", crate::step::save::StepDisplay::new(pcurve, 100));
            let emitted_table = source_table(&emitted);
            let emitted_source = EntityTable::<CurveAnyHolder>::get_owned(&emitted_table, 100)
                .expect("actual emitted PCURVE source");
            let emitted_source = Curve3D::try_from(&emitted_source).expect("re-admitted raw chart");
            for parameter in [0.0, 0.37, 1.0] {
                let expected = Point3::new(21.0 - 4.0 * parameter, 11.0, 13.0);
                let derivative = Vector3::new(-4.0, 0.0, 0.0);
                assert!(pcurve.subs(parameter).near(&expected));
                assert!(pcurve.der(parameter).near(&derivative));
                assert!(native.subs(parameter).near(&expected));
                assert!(native.der(parameter).near(&derivative));
                assert!(emitted_source.subs(parameter).near(&expected));
                assert!(emitted_source.der(parameter).near(&derivative));
            }
        }
    }

    fn rotated_circle_pcurve_source() -> Table {
        source_table(
            "#1=CARTESIAN_POINT('',(0.,0.));\n#2=DIRECTION('',(3.,4.));\n\
             #3=AXIS2_PLACEMENT_2D('',#1,#2);\n#4=CIRCLE('',#3,2.);\n\
             #5=CARTESIAN_POINT('',(7.,11.,13.));\n#6=DIRECTION('',(0.,0.,1.));\n\
             #7=DIRECTION('',(1.,0.,0.));\n#8=AXIS2_PLACEMENT_3D('',#5,#6,#7);\n\
             #9=PLANE('',#8);\n\
             #10=(GEOMETRIC_REPRESENTATION_CONTEXT(2) PARAMETRIC_REPRESENTATION_CONTEXT() \
             REPRESENTATION_CONTEXT('',''));\n\
             #11=DEFINITIONAL_REPRESENTATION('',(#4),#10);\n#12=PCURVE('',#9,#11);",
        )
    }

    #[test]
    fn actual_table_circle_and_pcurve_normalize_two_dimensional_placement_ratios() {
        let table = rotated_circle_pcurve_source();
        let circle =
            EntityTable::<CurveAnyHolder>::get_owned(&table, 4).expect("literal 2D CIRCLE");
        let circle = Curve2D::try_from(&circle).expect("checked 2D circle frame");
        let native_circle = NativeCurve2D::try_from(&circle).expect("analytic native 2D circle");
        let pcurve =
            EntityTable::<CurveAnyHolder>::get_owned(&table, 12).expect("literal circle PCURVE");
        let pcurve = Curve3D::try_from(&pcurve).expect("checked raw circle PCURVE");
        let native_pcurve = NativeCurve::try_from(&pcurve).expect("analytic native circle PCURVE");
        assert!(circle.subs(0.0).near(&Point2::new(1.2, 1.6)));
        assert!(circle.der(0.0).near(&Vector2::new(-1.6, 1.2)));
        for parameter in [0.0_f64, 0.37, 1.4] {
            let (sin, cos) = parameter.sin_cos();
            let uv = Point2::new(1.2 * cos - 1.6 * sin, 1.6 * cos + 1.2 * sin);
            let tangent = Vector2::new(-1.2 * sin - 1.6 * cos, -1.6 * sin + 1.2 * cos);
            let point = Point3::new(7.0 + uv.x, 11.0 + uv.y, 13.0);
            let derivative = Vector3::new(tangent.x, tangent.y, 0.0);
            assert!(circle.subs(parameter).near(&uv));
            assert!(circle.der(parameter).near(&tangent));
            assert!(native_circle.subs(parameter).near(&uv));
            assert!(native_circle.der(parameter).near(&tangent));
            assert!(pcurve.subs(parameter).near(&point));
            assert!(pcurve.der(parameter).near(&derivative));
            assert!(native_pcurve.subs(parameter).near(&point));
            assert!(native_pcurve.der(parameter).near(&derivative));
        }
    }

    #[test]
    fn invalid_two_dimensional_placement_ratios_refuse_through_circle_and_pcurve() {
        for ratios in [
            vec![0.0, 0.0],
            vec![3.0, 4.0, 0.0],
            vec![f64::NAN, 1.0],
            vec![1.0, f64::INFINITY],
        ] {
            let mut table = rotated_circle_pcurve_source();
            table
                .direction
                .get_mut(&2)
                .expect("2D placement direction")
                .direction_ratios = ratios;
            let circle = EntityTable::<CurveAnyHolder>::get_owned(&table, 4)
                .expect("invalid semantic circle source still resolves");
            let error = Curve2D::try_from(&circle).expect_err("invalid 2D frame must refuse");
            assert!(error.downcast_ref::<VectorSemanticError>().is_some());
            let pcurve = EntityTable::<CurveAnyHolder>::get_owned(&table, 12)
                .expect("invalid semantic PCURVE source still resolves");
            let error = Curve3D::try_from(&pcurve).expect_err("invalid UV frame must refuse");
            assert!(error.downcast_ref::<VectorSemanticError>().is_some());
        }
    }
}
