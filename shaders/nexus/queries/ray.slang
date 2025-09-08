module ray;

import nexus.aliases;

public struct Ray {
    public Vect origin;
    public Vect dir;

    /// The point on the ray at the given parameter `t`.
    public func pt_at(t: float) -> Vect {
        return this.origin + this.dir * t;
    }
}

