module projection;

import nexus.aliases;

/// The result of a point projection.
public struct ProjectionResult {
    /// The point’s projection on the shape.
    /// This can be equal to the original point if the point was inside
    /// of the shape and the projection function doesn’t always project
    /// on the boundary.
    public Vect point;
    /// Is the point inside of the shape?
    public bool is_inside;

    public __init(Vect point, bool is_inside) {
        this.point = point;
        this.is_inside = is_inside;
    }
}
