//! Curve-select resolution that preserves recursive source refusal diagnostics.

use super::*;
use serde::{
    Deserializer,
    de::{Error as DeserializeError, MapAccess, Visitor},
};
use std::fmt;
use step_p21::{
    error::Error as StepEntityError,
    tables::{EntityTable, Holder as HolderTrait, IntoOwned, WithVisitor},
};

/// Reference-preserving holder for a source curve select.
#[derive(Clone, Debug, PartialEq)]
pub enum CurveAnyHolder {
    Line(Box<LineHolder>),
    BoundedCurve(Box<BoundedCurveAnyHolder>),
    Conic(Box<ConicHolder>),
    Pcurve(Box<PcurveHolder>),
    SurfaceCurve(Box<SurfaceCurveHolder>),
    TrimmedCurve(Box<StepTrimmedCurveHolder>),
}

impl HolderTrait for CurveAnyHolder {
    fn name() -> &'static str { "CURVE_ANY" }
    fn attr_len() -> usize { 0 }
}

impl IntoOwned for CurveAnyHolder {
    type Table = Table;
    type Owned = CurveAny;

    fn into_owned(self, table: &Table) -> Result<Self::Owned, StepEntityError> {
        table
            .validate_curve_holder_source(&self)
            .map_err(|error| StepEntityError::DeserializeFailed(error.to_string()))?;
        Ok(match self {
            Self::Line(curve) => CurveAny::Line(Box::new((*curve).into_owned(table)?)),
            Self::BoundedCurve(curve) => {
                CurveAny::BoundedCurve(Box::new((*curve).into_owned(table)?))
            }
            Self::Conic(curve) => CurveAny::Conic(Box::new((*curve).into_owned(table)?)),
            Self::Pcurve(curve) => CurveAny::Pcurve(Box::new((*curve).into_owned(table)?)),
            Self::SurfaceCurve(curve) => {
                CurveAny::SurfaceCurve(Box::new((*curve).into_owned(table)?))
            }
            Self::TrimmedCurve(curve) => {
                CurveAny::TrimmedCurve(Box::new((*curve).into_owned(table)?))
            }
        })
    }
}

/// Deserializer for the STEP curve select.
#[doc(hidden)]
#[derive(Debug)]
pub struct CurveAnyHolderVisitor;

impl<'de> Visitor<'de> for CurveAnyHolderVisitor {
    type Value = CurveAnyHolder;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a STEP curve select")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let name: String = map
            .next_key()?
            .ok_or_else(|| A::Error::custom("missing curve select name"))?;
        match name.as_str() {
            "LINE" => map
                .next_value()
                .map(|curve| CurveAnyHolder::Line(Box::new(curve))),
            "BOUNDED_CURVE" => map
                .next_value()
                .map(|curve| CurveAnyHolder::BoundedCurve(Box::new(curve))),
            "CONIC" => map
                .next_value()
                .map(|curve| CurveAnyHolder::Conic(Box::new(curve))),
            "PCURVE" => map
                .next_value()
                .map(|curve| CurveAnyHolder::Pcurve(Box::new(curve))),
            "SURFACE_CURVE" => map
                .next_value()
                .map(|curve| CurveAnyHolder::SurfaceCurve(Box::new(curve))),
            "TRIMMED_CURVE" => map
                .next_value()
                .map(|curve| CurveAnyHolder::TrimmedCurve(Box::new(curve))),
            _ => Err(A::Error::custom(format!("unsupported curve select {name}"))),
        }
    }
}

impl WithVisitor for CurveAnyHolder {
    type Visitor = CurveAnyHolderVisitor;
    fn visitor_new() -> Self::Visitor { CurveAnyHolderVisitor }
}

impl<'de> Deserialize<'de> for CurveAnyHolder {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(CurveAnyHolderVisitor)
    }
}

impl EntityTable<CurveAnyHolder> for Table {
    fn get_owned(&self, entity_id: u64) -> Result<CurveAny, StepEntityError> {
        self.validate_curve_source(&PlaceHolder::Ref(Name::Entity(entity_id)))
            .map_err(|error| StepEntityError::DeserializeFailed(error.to_string()))?;
        if self.trimmed_curve.contains_key(&entity_id) {
            EntityTable::<StepTrimmedCurveHolder>::get_owned(self, entity_id)
                .map(|curve| CurveAny::TrimmedCurve(Box::new(curve)))
        } else if self.surface_curve.contains_key(&entity_id) {
            EntityTable::<SurfaceCurveHolder>::get_owned(self, entity_id)
                .map(|curve| CurveAny::SurfaceCurve(Box::new(curve)))
        } else if self.pcurve.contains_key(&entity_id) {
            EntityTable::<PcurveHolder>::get_owned(self, entity_id)
                .map(|curve| CurveAny::Pcurve(Box::new(curve)))
        } else if self.line.contains_key(&entity_id) {
            EntityTable::<LineHolder>::get_owned(self, entity_id)
                .map(|curve| CurveAny::Line(Box::new(curve)))
        } else if let Ok(curve) = EntityTable::<BoundedCurveAnyHolder>::get_owned(self, entity_id) {
            Ok(CurveAny::BoundedCurve(Box::new(curve)))
        } else {
            EntityTable::<ConicHolder>::get_owned(self, entity_id)
                .map(|curve| CurveAny::Conic(Box::new(curve)))
        }
    }

    fn owned_iter<'table>(
        &'table self,
    ) -> Box<dyn Iterator<Item = Result<CurveAny, StepEntityError>> + 'table> {
        let lines = EntityTable::<LineHolder>::owned_iter(self)
            .map(|curve| curve.map(|curve| CurveAny::Line(Box::new(curve))));
        let bounded = EntityTable::<BoundedCurveAnyHolder>::owned_iter(self)
            .map(|curve| curve.map(|curve| CurveAny::BoundedCurve(Box::new(curve))));
        let conics = EntityTable::<ConicHolder>::owned_iter(self)
            .map(|curve| curve.map(|curve| CurveAny::Conic(Box::new(curve))));
        let recursive = self
            .pcurve
            .keys()
            .chain(self.surface_curve.keys())
            .chain(self.trimmed_curve.keys())
            .map(|id| EntityTable::<CurveAnyHolder>::get_owned(self, *id));
        Box::new(lines.chain(bounded).chain(conics).chain(recursive))
    }
}
