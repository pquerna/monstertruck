//! Points, directions, vectors and axis placements.
//!
//! The coordinate-carrying entities every other family refers to: a
//! `cartesian_point` is a position, a `direction` a unit vector, and the
//! `axis*_placement` entities the local frames that curves, surfaces and
//! transformations are expressed in.

use super::*;

/// `cartesian_point`
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(field = cartesian_point)]
#[holder(generate_deserialize)]
pub struct CartesianPoint {
    pub label: String,
    pub coordinates: Vec<f64>,
}
impl From<&CartesianPoint> for Point2 {
    #[inline(always)]
    fn from(pt: &CartesianPoint) -> Self {
        let pt = &pt.coordinates;
        match pt.len() {
            0 => Point2::origin(),
            1 => Point2::new(pt[0], 0.0),
            _ => Point2::new(pt[0], pt[1]),
        }
    }
}
impl From<&CartesianPoint> for Point3 {
    #[inline(always)]
    fn from(pt: &CartesianPoint) -> Self {
        let pt = &pt.coordinates;
        match pt.len() {
            0 => Point3::origin(),
            1 => Point3::new(pt[0], 0.0, 0.0),
            2 => Point3::new(pt[0], pt[1], 0.0),
            _ => Point3::new(pt[0], pt[1], pt[2]),
        }
    }
}

/// `direction`
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(field = direction)]
#[holder(generate_deserialize)]
pub struct Direction {
    pub label: String,
    pub direction_ratios: Vec<f64>,
}
impl From<&Direction> for Vector2 {
    #[inline(always)]
    fn from(dir: &Direction) -> Self {
        let dir = &dir.direction_ratios;
        match dir.len() {
            0 => Vector2::zero(),
            1 => Vector2::new(dir[0], 0.0),
            _ => Vector2::new(dir[0], dir[1]),
        }
    }
}
impl From<&Direction> for Vector3 {
    #[inline(always)]
    fn from(dir: &Direction) -> Self {
        let dir = &dir.direction_ratios;
        match dir.len() {
            0 => Vector3::zero(),
            1 => Vector3::new(dir[0], 0.0, 0.0),
            2 => Vector3::new(dir[0], dir[1], 0.0),
            _ => Vector3::new(dir[0], dir[1], dir[2]),
        }
    }
}

/// `vector`
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(field = vector)]
#[holder(generate_deserialize)]
pub struct Vector {
    pub label: String,
    #[holder(use_place_holder)]
    pub orientation: Direction,
    pub magnitude: f64,
}
/// Invalid STEP direction, geometric vector or two-dimensional placement data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum VectorSemanticError {
    /// A geometric entity and its consuming geometry have different dimensions.
    #[error("STEP geometric entity has dimension {actual}; expected {expected}.")]
    InvalidDimension { expected: usize, actual: usize },
    /// A direction ratio, magnitude or placement coordinate is nonfinite.
    #[error("STEP direction, vector or placement contains a nonfinite coefficient.")]
    NonFinite,
    /// All direction ratios vanish.
    #[error("STEP geometric DIRECTION has no orientation.")]
    ZeroDirection,
    /// A geometric vector has no positive magnitude.
    #[error("STEP geometric VECTOR requires a positive magnitude.")]
    NonPositiveMagnitude,
}

fn unit_direction<const N: usize>(direction: &Direction) -> Result<[f64; N], VectorSemanticError> {
    let ratios = &direction.direction_ratios;
    if ratios.len() != N {
        return Err(VectorSemanticError::InvalidDimension {
            expected: N,
            actual: ratios.len(),
        });
    }
    if ratios.iter().any(|ratio| !ratio.is_finite()) {
        return Err(VectorSemanticError::NonFinite);
    }
    let scale = ratios
        .iter()
        .fold(0.0f64, |max, value| max.max(value.abs()));
    if scale == 0.0 {
        return Err(VectorSemanticError::ZeroDirection);
    }
    // DIRECTION ratios specify orientation, not vector length. Scaling first
    // keeps normalization finite for legal subnormal and very large ratios.
    let mut result = [0.0; N];
    for (target, ratio) in result.iter_mut().zip(ratios) {
        *target = ratio / scale;
    }
    let norm = result.iter().map(|value| value * value).sum::<f64>().sqrt();
    for value in &mut result {
        *value /= norm;
    }
    Ok(result)
}

fn geometric_vector<const N: usize>(vector: &Vector) -> Result<[f64; N], VectorSemanticError> {
    if !vector.magnitude.is_finite() {
        return Err(VectorSemanticError::NonFinite);
    }
    if vector.magnitude <= 0.0 {
        return Err(VectorSemanticError::NonPositiveMagnitude);
    }
    let mut result = unit_direction::<N>(&vector.orientation)?;
    for value in &mut result {
        *value *= vector.magnitude;
    }
    Ok(result)
}

impl TryFrom<&Vector> for Vector2 {
    type Error = VectorSemanticError;
    fn try_from(vector: &Vector) -> Result<Self, Self::Error> {
        let [x, y] = geometric_vector::<2>(vector)?;
        Ok(Self::new(x, y))
    }
}
impl TryFrom<&Vector> for Vector3 {
    type Error = VectorSemanticError;
    fn try_from(vector: &Vector) -> Result<Self, Self::Error> {
        let [x, y, z] = geometric_vector::<3>(vector)?;
        Ok(Self::new(x, y, z))
    }
}

/// `placement`
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(field = placement)]
#[holder(generate_deserialize)]
pub struct Placement {
    pub label: String,
    #[holder(use_place_holder)]
    pub location: CartesianPoint,
}
impl From<&Placement> for Point2 {
    #[inline(always)]
    fn from(p: &Placement) -> Self { Self::from(&p.location) }
}
impl From<&Placement> for Point3 {
    #[inline(always)]
    fn from(p: &Placement) -> Self { Self::from(&p.location) }
}

/// `axis1_placement`
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(field = axis1_placement)]
#[holder(generate_deserialize)]
pub struct Axis1Placement {
    pub label: String,
    #[holder(use_place_holder)]
    pub location: CartesianPoint,
    #[holder(use_place_holder)]
    pub direction: Option<Direction>,
}

impl Axis1Placement {
    pub fn direction(&self) -> Vector3 {
        self.direction
            .as_ref()
            .map(Vector3::from)
            .unwrap_or_else(Vector3::unit_z)
    }
}

/// `axis2_placement`
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(generate_deserialize)]
pub enum Axis2Placement {
    #[holder(use_place_holder)]
    Axis2Placement2d(Axis2Placement2d),
    #[holder(use_place_holder)]
    Axis2Placement3d(Axis2Placement3d),
}

impl TryFrom<&Axis2Placement> for Matrix3 {
    type Error = StepConvertingError;
    #[inline(always)]
    fn try_from(axis: &Axis2Placement) -> Result<Self, StepConvertingError> {
        use Axis2Placement::*;
        match axis {
            Axis2Placement2d(axis) => Ok(Matrix3::try_from(axis)?),
            Axis2Placement3d(_) => Err(VectorSemanticError::InvalidDimension {
                expected: 2,
                actual: 3,
            }
            .into()),
        }
    }
}
impl TryFrom<&Axis2Placement> for Matrix4 {
    type Error = StepConvertingError;
    #[inline(always)]
    fn try_from(axis: &Axis2Placement) -> Result<Self, StepConvertingError> {
        use Axis2Placement::*;
        match axis {
            Axis2Placement2d(_) => Err("This is not a 3D axis placement.".into()),
            Axis2Placement3d(axis) => Ok(Matrix4::from(axis)),
        }
    }
}

/// `axis2_placement_2d`
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(field = axis2_placement_2d)]
#[holder(generate_deserialize)]
pub struct Axis2Placement2d {
    pub label: String,
    #[holder(use_place_holder)]
    pub location: CartesianPoint,
    #[holder(use_place_holder)]
    pub ref_direction: Option<Direction>,
}

impl TryFrom<&Axis2Placement2d> for Matrix3 {
    type Error = VectorSemanticError;
    fn try_from(axis: &Axis2Placement2d) -> Result<Self, Self::Error> {
        if axis.location.coordinates.len() != 2 {
            return Err(VectorSemanticError::InvalidDimension {
                expected: 2,
                actual: axis.location.coordinates.len(),
            });
        }
        if axis
            .location
            .coordinates
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(VectorSemanticError::NonFinite);
        }
        let z = Point2::from(&axis.location);
        let x = match &axis.ref_direction {
            Some(direction) => {
                let [x, y] = unit_direction::<2>(direction)?;
                Vector2::new(x, y)
            }
            None => Vector2::unit_x(),
        };
        let y = Vector2::new(-x.y, x.x);
        Ok(Matrix3::from_cols(
            x.extend(0.0),
            y.extend(0.0),
            z.to_vec().extend(1.0),
        ))
    }
}

/// `axis2_placement_3d`
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Holder)]
#[holder(table = Table)]
#[holder(field = axis2_placement_3d)]
#[holder(generate_deserialize)]
pub struct Axis2Placement3d {
    pub label: String,
    #[holder(use_place_holder)]
    pub location: CartesianPoint,
    #[holder(use_place_holder)]
    pub axis: Option<Direction>,
    #[holder(use_place_holder)]
    pub ref_direction: Option<Direction>,
}

impl From<&Axis2Placement3d> for Matrix4 {
    #[inline(always)]
    fn from(axis: &Axis2Placement3d) -> Matrix4 {
        let w = Point3::from(&axis.location);
        let z = match &axis.axis {
            Some(axis) => Vector3::from(axis),
            None => Vector3::unit_z(),
        };
        // Pick a fallback reference direction that is not parallel to `z`.
        let fallback = match z.near(&Vector3::unit_x()) {
            true => Vector3::unit_y(),
            false => Vector3::unit_x(),
        };
        let x = match &axis.ref_direction {
            Some(axis) => Vector3::from(axis),
            None => fallback,
        };
        // Gram-Schmidt: remove the `z` component of `x`. ISO 10303 permits
        // `ref_direction` to be parallel to `axis`; in that case the projected
        // vector is zero and `normalize()` would produce `NaN`, which later
        // panics in `nonpositive_tolerance!` during meshing. Fall back to an
        // arbitrary direction orthogonal to `z` when the projection degenerates.
        let projected = x - x.dot(z) * z;
        let x = match projected.magnitude2().so_small() {
            true => (fallback - fallback.dot(z) * z).normalize(),
            false => projected.normalize(),
        };
        let y = z.cross(x);
        Matrix4::from_cols(
            x.extend(0.0),
            y.extend(0.0),
            z.extend(0.0),
            w.to_vec().extend(1.0),
        )
    }
}

#[cfg(test)]
mod step_vector_semantic_regressions {
    use super::*;

    fn vector(ratios: &[f64], magnitude: f64) -> Vector {
        Vector {
            label: String::new(),
            orientation: Direction {
                label: String::new(),
                direction_ratios: ratios.to_vec(),
            },
            magnitude,
        }
    }

    #[test]
    fn declared_magnitude_is_independent_of_direction_ratio_length() {
        for ratio in [2.0, f64::MAX, f64::from_bits(1)] {
            assert_eq!(
                Vector3::try_from(&vector(&[0.0, 0.0, ratio], 3.0)).unwrap(),
                Vector3::new(0.0, 0.0, 3.0)
            );
            assert_eq!(
                Vector2::try_from(&vector(&[0.0, ratio], 3.0)).unwrap(),
                Vector2::new(0.0, 3.0)
            );
        }
    }

    #[test]
    fn invalid_geometric_vectors_have_typed_failures() {
        assert_eq!(
            Vector3::try_from(&vector(&[0.0, 0.0, 0.0], 3.0)),
            Err(VectorSemanticError::ZeroDirection)
        );
        assert_eq!(
            Vector3::try_from(&vector(&[0.0, 0.0, f64::INFINITY], 3.0)),
            Err(VectorSemanticError::NonFinite)
        );
        assert_eq!(
            Vector2::try_from(&vector(&[0.0, 1.0], f64::NAN)),
            Err(VectorSemanticError::NonFinite)
        );
        for magnitude in [0.0, -3.0] {
            assert_eq!(
                Vector3::try_from(&vector(&[0.0, 0.0, 2.0], magnitude)),
                Err(VectorSemanticError::NonPositiveMagnitude)
            );
        }
        assert_eq!(
            Vector2::try_from(&vector(&[0.0, 0.0, 2.0], 3.0)),
            Err(VectorSemanticError::InvalidDimension {
                expected: 2,
                actual: 3
            })
        );
    }
}
