//! Union native component intervals before mutating shared topology.
//! Coverage follows the carrier certificate's one-dimensional component, not
//! a distance weld between endpoints or a collection of nearby sampled points.

use super::native_component::NativeComponent;
use super::*;

pub(super) struct Coordinates {
    first: f64,
    second: f64,
    penultimate: f64,
    last: f64,
    minimum: f64,
    maximum: f64,
}

impl Coordinates {
    pub(super) fn extent(&self) -> f64 { self.maximum - self.minimum }
}

pub(super) struct NativeInterval<'a, C> {
    range: (f64, f64),
    origins: Option<([OriginalTrimContact<'a, C>; 2], [f64; 2])>,
}

pub(super) fn coordinates<S>(
    component: &NativeComponent,
    curve: &IntersectionCurve<PolylineCurve, S, S>,
    work: Option<&BooleanWork<'_>>,
) -> ClassicResult<Option<Coordinates>> {
    let leader = curve.leader();
    if leader.len() < 2 {
        return Ok(None);
    }
    let mut previous = None;
    let mut direction = None;
    let mut result: Option<Coordinates> = None;
    for (index, &point) in leader.iter().enumerate() {
        if let Some(work) = work {
            work.check()?;
        }
        let projected = component.project(point, previous, work);
        if let Some(work) = work {
            work.check()?;
        }
        let Some(projected) = projected else {
            return Ok(None);
        };
        let mut coordinate = projected.coordinate;
        if index + 1 == leader.len()
            && point == leader[0]
            && let (Some(period), Some(increasing), Some(result)) =
                (component.period(), direction, &result)
        {
            coordinate = result.first + if increasing { period } else { -period };
        }
        if let Some(previous) = previous {
            let difference = coordinate - previous;
            if component
                .period()
                .is_some_and(|period| difference.abs() >= period / 2.0)
            {
                return Ok(None);
            }
            if difference != 0.0 {
                let increasing = difference > 0.0;
                if direction.is_some_and(|old| old != increasing) {
                    return Ok(None);
                }
                direction = Some(increasing);
            }
        }
        if let Some(result) = &mut result {
            if index == 1 {
                result.second = coordinate;
            }
            result.penultimate = result.last;
            result.last = coordinate;
            result.minimum = result.minimum.min(coordinate);
            result.maximum = result.maximum.max(coordinate);
        } else {
            result = Some(Coordinates {
                first: coordinate,
                second: coordinate,
                penultimate: coordinate,
                last: coordinate,
                minimum: coordinate,
                maximum: coordinate,
            });
        }
        previous = Some(coordinate);
    }
    Ok(result.filter(|result| result.minimum < result.maximum && result.extent().is_finite()))
}

fn crossing<C, S>(
    origin: &OriginalTrimContact<'_, C>,
    component: &NativeComponent,
    carriers: [&S; 2],
    hint: f64,
    work: Option<&BooleanWork<'_>>,
) -> Option<f64>
where
    C: ParametricCurve3D,
    S: ParametricSurface3D + SearchParameter<SurfaceParameter, Point = Point3>,
{
    let point = component.project(origin.point, Some(hint), work)?;
    let carrier = carriers[origin.input];
    let uv = carrier.search_parameter(origin.point, None, 100)?;
    if !carrier.evaluate(uv.0, uv.1).near(&origin.point) {
        return None;
    }
    let mut boundary = origin
        .edge
        .with_curve(|curve| curve.derivative(origin.parameter));
    if !origin.edge.orientation() {
        boundary = -boundary;
    }
    let material_left = carrier.normal(uv.0, uv.1).cross(boundary);
    let sign = material_left.dot(point.tangent);
    let error = 128.0 * f64::EPSILON * material_left.magnitude() * point.tangent.magnitude();
    (sign.is_finite() && error.is_finite() && sign.abs() > error).then_some(sign)
}

struct ClippingBoundary {
    coordinate: f64,
    upper: bool,
    shift: f64,
}

fn original_endpoint_covers<C, S>(
    candidate: &Coordinates,
    origins: &[OriginalTrimContact<'_, C>; 2],
    accepted: &([OriginalTrimContact<'_, C>; 2], [f64; 2]),
    clipping: ClippingBoundary,
    component: &NativeComponent,
    carriers: [&S; 2],
    work: Option<&BooleanWork<'_>>,
) -> bool
where
    C: ParametricCurve3D,
    S: ParametricSurface3D + SearchParameter<SurfaceParameter, Point = Point3>,
{
    let (accepted, accepted_coordinates) = accepted;
    let ClippingBoundary {
        coordinate: boundary,
        upper,
        shift,
    } = clipping;
    for (endpoint, origin) in origins.iter().enumerate() {
        let (outside, adjacent) = if endpoint == 0 {
            (candidate.first, candidate.second)
        } else {
            (candidate.last, candidate.penultimate)
        };
        if (upper && outside <= boundary) || (!upper && outside >= boundary) {
            continue;
        }
        // Only the endpoint's own original trim can clip its terminal guide
        // segment. An arbitrary distant boundary cannot erase another interval.
        if boundary < outside.min(adjacent) || boundary > outside.max(adjacent) {
            continue;
        }
        for (old, accepted_origin) in accepted.iter().enumerate() {
            if accepted_coordinates[old] + shift != boundary || !origin.same_origin(accepted_origin)
            {
                continue;
            }
            if let Some(sign) = crossing(origin, component, carriers, boundary, work)
                && ((upper && sign < 0.0) || (!upper && sign > 0.0))
            {
                return true;
            }
        }
    }
    false
}

pub(super) fn covered<C, S>(
    component: &NativeComponent,
    candidate: &Coordinates,
    origins: Option<&[OriginalTrimContact<'_, C>; 2]>,
    intervals: &[NativeInterval<'_, C>],
    carriers: [&S; 2],
    work: Option<&BooleanWork<'_>>,
) -> ClassicResult<bool>
where
    C: ParametricCurve3D,
    S: ParametricSurface3D + SearchParameter<SurfaceParameter, Point = Point3>,
{
    if let Some(work) = work {
        work.check()?;
    }
    for interval in intervals {
        if let Some(work) = work {
            work.check()?;
        }
        let shift = if let Some(period) = component.period() {
            if interval.range.1 - interval.range.0 >= period {
                return Ok(true);
            }
            (((candidate.minimum + candidate.maximum) - (interval.range.0 + interval.range.1))
                / (2.0 * period))
                .round()
                * period
        } else {
            0.0
        };
        let (lower, upper) = (interval.range.0 + shift, interval.range.1 + shift);
        let mut start = candidate.minimum;
        let mut end = candidate.maximum;
        if let (Some(origins), Some(accepted)) = (origins, &interval.origins) {
            if start < lower
                && original_endpoint_covers(
                    candidate,
                    origins,
                    accepted,
                    ClippingBoundary {
                        coordinate: lower,
                        upper: false,
                        shift,
                    },
                    component,
                    carriers,
                    work,
                )
            {
                start = lower;
            }
            if end > upper
                && original_endpoint_covers(
                    candidate,
                    origins,
                    accepted,
                    ClippingBoundary {
                        coordinate: upper,
                        upper: true,
                        shift,
                    },
                    component,
                    carriers,
                    work,
                )
            {
                end = upper;
            }
        }
        if let Some(work) = work {
            work.check()?;
        }
        if start >= lower && end <= upper && start < end {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(super) fn endpoint_coordinates<C>(
    component: &NativeComponent,
    coordinates: &Coordinates,
    origins: &[OriginalTrimContact<'_, C>; 2],
    work: Option<&BooleanWork<'_>>,
) -> Option<[f64; 2]> {
    if origins[0].point.near(&origins[1].point) {
        return None;
    }
    let first = component
        .project(origins[0].point, Some(coordinates.first), work)?
        .coordinate;
    let last = component
        .project(origins[1].point, Some(coordinates.last), work)?
        .coordinate;
    if first == last
        || !(last - first).is_finite()
        || (last > first) != (coordinates.last > coordinates.first)
    {
        return None;
    }
    // Recovery may extend an incomplete mesh chain to its actual original
    // endpoint, but cannot remove an internal guide or reverse its ordering.
    if coordinates.second < first.min(last)
        || coordinates.second > first.max(last)
        || coordinates.penultimate < first.min(last)
        || coordinates.penultimate > first.max(last)
    {
        return None;
    }
    Some([first, last])
}

pub(super) fn open_interval<'a, C>(
    coordinates: [f64; 2],
    origins: [OriginalTrimContact<'a, C>; 2],
) -> NativeInterval<'a, C> {
    let [first, last] = coordinates;
    NativeInterval {
        range: (first.min(last), first.max(last)),
        origins: Some((origins, coordinates)),
    }
}

pub(super) fn closed_interval<'a, C>(coordinates: Coordinates) -> NativeInterval<'a, C> {
    NativeInterval {
        range: (coordinates.minimum, coordinates.maximum),
        origins: None,
    }
}
