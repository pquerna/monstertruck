//! Conservative owned-source expansion accounting before reference resolution.
//!
//! Repeated references are charged for every expansion, including resolved
//! point/name payloads and vector/string capacities. Holder and owned cell
//! sizes conservatively bound inline storage; fixed-field accounting can
//! overlap vector cells. Native sweep profile boxes are charged separately.
//! Raw topology clone sites share the same budget.

use super::super::step_types as ast;
use super::*;
use std::mem::size_of;
use step_p21::primitive::Logical;

/// Producer-profile resource policy, not a geometric validity bound.
/// This is a conservative expanded-owned-payload cost, not peak heap usage or
/// a universal limit on representable STEP geometry.
pub const MAX_SOURCE_GEOMETRY_EXPANSION_BYTES: usize = 64 * 1024 * 1024;
/// Maximum cumulative source graph work for one conversion under this profile.
pub const MAX_SOURCE_GEOMETRY_EXPANSION_VISITS: usize = 1_000_000;

#[derive(Default)]
pub(super) struct SourceExpansionBudget {
    pub(super) expanded_bytes: usize,
    pub(super) visits: usize,
}

impl SourceExpansionBudget {
    pub(super) fn charge(&mut self, bytes: usize) -> Result<(), ExactConversionError> {
        self.expanded_bytes = self.expanded_bytes.saturating_add(bytes);
        self.visits = self.visits.saturating_add(1);
        if self.expanded_bytes > MAX_SOURCE_GEOMETRY_EXPANSION_BYTES
            || self.visits > MAX_SOURCE_GEOMETRY_EXPANSION_VISITS
        {
            Err(ExactConversionError::SourceGeometryExpansionExceeded {
                expanded_bytes: self.expanded_bytes,
                visits: self.visits,
            })
        } else {
            Ok(())
        }
    }
}

pub(super) struct ExpansionWalk<'a, 'budget> {
    table: &'a Table,
    budget: &'budget mut SourceExpansionBudget,
    resolve_references: bool,
}

impl<'a, 'budget> ExpansionWalk<'a, 'budget> {
    pub(super) fn new(table: &'a Table, budget: &'budget mut SourceExpansionBudget) -> Self {
        Self {
            table,
            budget,
            resolve_references: true,
        }
    }

    pub(super) fn raw_clone(table: &'a Table, budget: &'budget mut SourceExpansionBudget) -> Self {
        Self {
            table,
            budget,
            resolve_references: false,
        }
    }

    pub(super) fn raw_clones<T: ExpandedSource>(
        &mut self,
        value: &T,
        count: usize,
    ) -> Result<(), ExactConversionError> {
        for _ in 0..count {
            self.expand(value)?;
        }
        Ok(())
    }

    pub(super) fn expand<T: ExpandedSource>(
        &mut self,
        value: &T,
    ) -> Result<(), ExactConversionError> {
        value.expand(self)
    }

    fn charge(&mut self, bytes: usize) -> Result<(), ExactConversionError> {
        self.budget.charge(bytes)
    }

    fn generated_degree_knots(&mut self, degree: i64) -> Result<(), ExactConversionError> {
        if !self.resolve_references {
            return Ok(());
        }
        let bytes = usize::try_from(degree)
            .ok()
            .and_then(|degree| degree.checked_add(1))
            .and_then(|degree| degree.checked_mul(2 * size_of::<f64>()))
            .unwrap_or(usize::MAX);
        self.charge(bytes)
    }

    fn expanded_knots(&mut self, multiplicities: &[i64]) -> Result<(), ExactConversionError> {
        if !self.resolve_references {
            return Ok(());
        }
        let count = multiplicities
            .iter()
            .try_fold(0_usize, |count, multiplicity| {
                count.checked_add(usize::try_from(*multiplicity).ok()?)
            });
        self.charge(
            count
                .and_then(|count| count.checked_mul(size_of::<f64>()))
                .unwrap_or(usize::MAX),
        )
    }
}

pub(super) trait ExpandedSource {
    type Owned;
    fn expand(&self, walk: &mut ExpansionWalk<'_, '_>) -> Result<(), ExactConversionError>;
    fn contains(_table: &Table, _id: u64) -> bool { false }
    fn reference(_walk: &mut ExpansionWalk<'_, '_>, _id: u64) -> Result<(), ExactConversionError> {
        Ok(())
    }
}

macro_rules! scalar {
    ($($ty:ty),+ $(,)?) => {$(
        impl ExpandedSource for $ty {
            type Owned = Self;
            fn expand(&self, walk: &mut ExpansionWalk<'_, '_>) -> Result<(), ExactConversionError> {
                walk.charge(size_of::<Self>())
            }
        }
    )+};
}
scalar!(
    f64,
    i64,
    bool,
    Logical,
    KnotType,
    BsplineCurveForm,
    BsplineSurfaceForm,
    TrimmingPreference,
    SurfaceCurveEntityKind,
    PreferredSurfaceCurveRepresentation
);

impl ExpandedSource for String {
    type Owned = Self;
    fn expand(&self, walk: &mut ExpansionWalk<'_, '_>) -> Result<(), ExactConversionError> {
        walk.charge(size_of::<Self>().saturating_add(self.capacity()))
    }
}

impl<T: ExpandedSource> ExpandedSource for Vec<T> {
    type Owned = Vec<T::Owned>;
    fn expand(&self, walk: &mut ExpansionWalk<'_, '_>) -> Result<(), ExactConversionError> {
        let cell = size_of::<T>().max(size_of::<T::Owned>());
        walk.charge(self.capacity().saturating_mul(cell))?;
        self.iter().try_for_each(|value| value.expand(walk))
    }
}

impl<T: ExpandedSource> ExpandedSource for Option<T> {
    type Owned = Option<T::Owned>;
    fn expand(&self, walk: &mut ExpansionWalk<'_, '_>) -> Result<(), ExactConversionError> {
        walk.charge(size_of::<Self::Owned>())?;
        match self {
            Some(value) => value.expand(walk),
            None => Ok(()),
        }
    }
}

impl<T: ExpandedSource> ExpandedSource for Box<T> {
    type Owned = Box<T::Owned>;
    fn expand(&self, walk: &mut ExpansionWalk<'_, '_>) -> Result<(), ExactConversionError> {
        walk.charge(size_of::<Self::Owned>())?;
        self.as_ref().expand(walk)
    }
    fn contains(table: &Table, id: u64) -> bool { T::contains(table, id) }
    fn reference(walk: &mut ExpansionWalk<'_, '_>, id: u64) -> Result<(), ExactConversionError> {
        T::reference(walk, id)
    }
}

impl<T: ExpandedSource> ExpandedSource for PlaceHolder<T> {
    type Owned = T::Owned;
    fn expand(&self, walk: &mut ExpansionWalk<'_, '_>) -> Result<(), ExactConversionError> {
        walk.charge(size_of::<<Self as ExpandedSource>::Owned>())?;
        match self {
            Self::Owned(value) => value.expand(walk),
            Self::Ref(Name::Entity(id)) if walk.resolve_references => T::reference(walk, *id),
            Self::Ref(Name::ConstantEntity(name) | Name::ConstantValue(name)) => name.expand(walk),
            Self::Ref(_) => Ok(()),
        }
    }
}

macro_rules! source_struct {
    ($holder:ident => $owned:ident, $table:ident; $($field:ident),+ $(,)?) => {
        source_struct!($holder => $owned, $table; [$($field),+]; degrees []; knots []);
    };
    ($holder:ident => $owned:ident, $table:ident; [$($field:ident),+]; degrees [$($degree:ident),*]; knots [$($knots:ident),*] $(; owned_boxes [$($boxed:ty),*])?) => {
        impl ExpandedSource for $holder {
            type Owned = ast::$owned;
            fn expand(&self, walk: &mut ExpansionWalk<'_, '_>) -> Result<(), ExactConversionError> {
                walk.charge(size_of::<Self>().max(size_of::<Self::Owned>()))?;
                $($(if walk.resolve_references {
                    // Charge the native owned profile allocation in addition to source payloads.
                    walk.charge(size_of::<$boxed>())?;
                })*)?
                $(self.$field.expand(walk)?;)+
                $(walk.generated_degree_knots(self.$degree)?;)*
                $(walk.expanded_knots(&self.$knots)?;)*
                Ok(())
            }
            fn contains(table: &Table, id: u64) -> bool { table.$table.contains_key(&id) }
            fn reference(walk: &mut ExpansionWalk<'_, '_>, id: u64) -> Result<(), ExactConversionError> {
                let table = walk.table;
                match table.$table.get(&id) { Some(value) => value.expand(walk), None => Ok(()) }
            }
        }
    };
}

macro_rules! source_select {
    ($holder:ident => $owned:ident; $($variant:ident($payload:ty)),+ $(,)?) => {
        impl ExpandedSource for $holder {
            type Owned = ast::$owned;
            fn expand(&self, walk: &mut ExpansionWalk<'_, '_>) -> Result<(), ExactConversionError> {
                walk.charge(size_of::<Self::Owned>())?;
                match self { $(Self::$variant(value) => value.expand(walk)),+ }
            }
            fn contains(table: &Table, id: u64) -> bool { false $(|| <$payload>::contains(table, id))+ }
            fn reference(walk: &mut ExpansionWalk<'_, '_>, id: u64) -> Result<(), ExactConversionError> {
                $(if <$payload>::contains(walk.table, id) { return <$payload>::reference(walk, id); })+
                Ok(())
            }
        }
    };
}

source_struct!(CartesianPointHolder => CartesianPoint, cartesian_point; label, coordinates);
source_struct!(DirectionHolder => Direction, direction; label, direction_ratios);
source_struct!(VectorHolder => Vector, vector; label, orientation, magnitude);
source_struct!(PlacementHolder => Placement, placement; label, location);
source_struct!(Axis1PlacementHolder => Axis1Placement, axis1_placement; label, location, direction);
source_struct!(Axis2Placement2dHolder => Axis2Placement2d, axis2_placement_2d; label, location, ref_direction);
source_struct!(Axis2Placement3dHolder => Axis2Placement3d, axis2_placement_3d; label, location, axis, ref_direction);
source_select!(Axis2PlacementHolder => Axis2Placement; Axis2Placement2d(Axis2Placement2dHolder), Axis2Placement3d(Axis2Placement3dHolder));
source_struct!(LineHolder => Line, line; label, pnt, dir);
source_struct!(PolylineHolder => Polyline, polyline; label, points);
source_struct!(BsplineCurveWithKnotsHolder => BsplineCurveWithKnots, b_spline_curve_with_knots;
    [label, degree, control_points_list, curve_form, closed_curve, self_intersect, knot_multiplicities, knots, knot_spec];
    degrees [degree]; knots [knot_multiplicities]);
source_struct!(BezierCurveHolder => BezierCurve, bezier_curve;
    [label, degree, control_points_list, curve_form, closed_curve, self_intersect]; degrees [degree]; knots []);
source_struct!(QuasiUniformCurveHolder => QuasiUniformCurve, quasi_uniform_curve;
    [label, degree, control_points_list, curve_form, closed_curve, self_intersect]; degrees [degree]; knots []);
source_struct!(UniformCurveHolder => UniformCurve, uniform_curve;
    [label, degree, control_points_list, curve_form, closed_curve, self_intersect]; degrees [degree]; knots []);
source_select!(NonRationalBsplineCurveHolder => NonRationalBsplineCurve;
    BsplineCurveWithKnots(BsplineCurveWithKnotsHolder), BezierCurve(BezierCurveHolder),
    QuasiUniformCurve(QuasiUniformCurveHolder), UniformCurve(UniformCurveHolder));
source_struct!(RationalBsplineCurveHolder => RationalBsplineCurve, rational_b_spline_curve; non_rational_b_spline_curve, weights_data);
source_select!(BsplineCurveAnyHolder => BsplineCurveAny;
    NonRationalBsplineCurve(Box<NonRationalBsplineCurveHolder>), RationalBsplineCurve(Box<RationalBsplineCurveHolder>));
source_select!(BoundedCurveAnyHolder => BoundedCurveAny; Polyline(Box<PolylineHolder>), BsplineCurve(Box<BsplineCurveAnyHolder>));
source_struct!(CircleHolder => Circle, circle; label, position, radius);
source_struct!(EllipseHolder => Ellipse, ellipse; label, position, semi_axis_1, semi_axis_2);
source_struct!(HyperbolaHolder => Hyperbola, hyperbola; label, position, semi_axis, semi_imag_axis);
source_struct!(ParabolaHolder => Parabola, parabola; label, position, focal_dist);
source_select!(ConicHolder => Conic; Circle(CircleHolder), Ellipse(EllipseHolder), Hyperbola(HyperbolaHolder), Parabola(ParabolaHolder));
source_struct!(StepTrimmedCurveHolder => StepTrimmedCurve, trimmed_curve; label, basis_curve, trim_1, trim_2, sense_agreement, master_representation);
source_struct!(PcurveHolder => Pcurve, pcurve; label, basis_surface, reference_to_curve);
source_struct!(SurfaceCurveHolder => SurfaceCurve, surface_curve; label, curve_3d, associated_geometry, master_representation, kind);
source_struct!(DefinitionalRepresentationHolder => DefinitionalRepresentation, definitional_representation; label, representation_item, context_of_items);
source_struct!(DummyHolder => Dummy, dummy; record, is_simple);
source_select!(CurveAnyHolder => CurveAny; Line(Box<LineHolder>), BoundedCurve(Box<BoundedCurveAnyHolder>),
    Conic(Box<ConicHolder>), Pcurve(Box<PcurveHolder>), SurfaceCurve(Box<SurfaceCurveHolder>), TrimmedCurve(Box<StepTrimmedCurveHolder>));
source_select!(PcurveOrSurfaceHolder => PcurveOrSurface; Pcurve(Box<PcurveHolder>), Surface(Box<SurfaceAnyHolder>));
source_select!(TrimmingSelectHolder => TrimmingSelect; CartesianPoint(Box<CartesianPointHolder>), ParameterValue(Box<ParameterValueHolder>));

impl ExpandedSource for ParameterValueHolder {
    type Owned = ast::ParameterValue;
    fn expand(&self, walk: &mut ExpansionWalk<'_, '_>) -> Result<(), ExactConversionError> {
        self.0.expand(walk)
    }
    fn contains(table: &Table, id: u64) -> bool { table.parameter_value.contains_key(&id) }
    fn reference(walk: &mut ExpansionWalk<'_, '_>, id: u64) -> Result<(), ExactConversionError> {
        let table = walk.table;
        match table.parameter_value.get(&id) {
            Some(value) => value.expand(walk),
            None => Ok(()),
        }
    }
}

source_struct!(PlaneHolder => Plane, plane; label, position);
source_struct!(SphericalSurfaceHolder => SphericalSurface, spherical_surface; label, position, radius);
source_struct!(CylindricalSurfaceHolder => CylindricalSurface, cylindrical_surface; label, position, radius);
source_struct!(ToroidalSurfaceHolder => ToroidalSurface, toroidal_surface; label, position, major_radius, minor_radius);
source_struct!(DegenerateToroidalSurfaceHolder => DegenerateToroidalSurface, degenerate_toroidal_surface; label, position, major_radius, minor_radius, select_outer);
source_struct!(ConicalSurfaceHolder => ConicalSurface, conical_surface; label, position, radius, semi_angle);
source_select!(ElementarySurfaceAnyHolder => ElementarySurfaceAny; Plane(Box<PlaneHolder>), SphericalSurface(Box<SphericalSurfaceHolder>),
    CylindricalSurface(Box<CylindricalSurfaceHolder>), ToroidalSurface(Box<ToroidalSurfaceHolder>),
    DegenerateToroidalSurface(Box<DegenerateToroidalSurfaceHolder>), ConicalSurface(Box<ConicalSurfaceHolder>));
source_struct!(BsplineSurfaceWithKnotsHolder => BsplineSurfaceWithKnots, b_spline_surface_with_knots;
    [label, u_degree, v_degree, control_points_list, surface_form, u_closed, v_closed, self_intersect,
     u_multiplicities, v_multiplicities, u_knots, v_knots, knot_spec]; degrees [u_degree, v_degree]; knots [u_multiplicities, v_multiplicities]);
source_struct!(UniformSurfaceHolder => UniformSurface, uniform_surface;
    [label, u_degree, v_degree, control_points_list, surface_form, u_closed, v_closed, self_intersect]; degrees [u_degree, v_degree]; knots []);
source_struct!(QuasiUniformSurfaceHolder => QuasiUniformSurface, quasi_uniform_surface;
    [label, u_degree, v_degree, control_points_list, surface_form, u_closed, v_closed, self_intersect]; degrees [u_degree, v_degree]; knots []);
source_struct!(BezierSurfaceHolder => BezierSurface, bezier_surface;
    [label, u_degree, v_degree, control_points_list, surface_form, u_closed, v_closed, self_intersect]; degrees [u_degree, v_degree]; knots []);
source_select!(NonRationalBsplineSurfaceHolder => NonRationalBsplineSurface; BsplineSurfaceWithKnots(Box<BsplineSurfaceWithKnotsHolder>),
    UniformSurface(Box<UniformSurfaceHolder>), QuasiUniformSurface(Box<QuasiUniformSurfaceHolder>), BezierSurface(Box<BezierSurfaceHolder>));
source_struct!(RationalBsplineSurfaceHolder => RationalBsplineSurface, rational_b_spline_surface; non_rational_b_spline_surface, weights_data);
source_select!(BsplineSurfaceAnyHolder => BsplineSurfaceAny; NonRationalBsplineSurface(NonRationalBsplineSurfaceHolder), RationalBsplineSurface(RationalBsplineSurfaceHolder));
source_struct!(SurfaceOfLinearExtrusionHolder => SurfaceOfLinearExtrusion, surface_of_linear_extrusion; [label, swept_curve, extrusion_axis]; degrees []; knots []; owned_boxes [Curve3D]);
source_struct!(SurfaceOfRevolutionHolder => SurfaceOfRevolution, surface_of_revolution; [label, swept_curve, axis_position]; degrees []; knots []; owned_boxes [Curve3D]);
source_select!(SweptSurfaceAnyHolder => SweptSurfaceAny; SurfaceOfLinearExtrusion(Box<SurfaceOfLinearExtrusionHolder>), SurfaceOfRevolution(Box<SurfaceOfRevolutionHolder>));
source_select!(SurfaceAnyHolder => SurfaceAny; ElementarySurface(Box<ElementarySurfaceAnyHolder>), BsplineSurface(Box<BsplineSurfaceAnyHolder>), SweptSurface(Box<SweptSurfaceAnyHolder>));
source_struct!(VertexPointHolder => VertexPoint, vertex_point; label, vertex_geometry);
source_struct!(EdgeCurveHolder => EdgeCurve, edge_curve; label, edge_start, edge_end, edge_geometry, same_sense);

source_struct!(OrientedEdgeHolder => OrientedEdge, oriented_edge; label, edge_element, orientation);
source_select!(EdgeAnyHolder => EdgeAny; EdgeCurve(EdgeCurveHolder), OrientedEdge(OrientedEdgeHolder));
source_struct!(EdgeLoopHolder => EdgeLoop, edge_loop; label, edge_list);
source_struct!(FaceBoundHolder => FaceBound, face_bound; label, bound, orientation);
source_struct!(FaceSurfaceHolder => FaceSurface, face_surface; label, bounds, face_geometry, same_sense);
source_struct!(OrientedFaceHolder => OrientedFace, oriented_face; label, face_element, orientation);
source_select!(FaceAnyHolder => FaceAny; FaceSurface(FaceSurfaceHolder), OrientedFace(OrientedFaceHolder));
