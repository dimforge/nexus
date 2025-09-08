module segment;

import nexus.aliases;

public struct Segment {
    public Vect a;
    public Vect b;

    // TODO: implement the other projection functions
    public func project_local_point(pt: Vect) -> Vect {
        let ab = this.b - this.a;
        let ap = pt - this.a;
        let ab_ap = dot(ab, ap);
        let sqnab = dot(ab, ab);

        // PERF: would it be faster to do a bunch of `select` instead of `if`?
        if (ab_ap <= 0.0) {
            // Voronoï region of vertex 'a'.
            return this.a;
        } else if (ab_ap >= sqnab) {
            // Voronoï region of vertex 'b'.
            return this.b;
        } else {
            // Voronoï region of the segment interior.
            let u = ab_ap / sqnab;
            return this.a + ab * u;
        }
    }
}
