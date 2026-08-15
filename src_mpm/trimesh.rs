// TODO: move this to nexus?

use crate::mpm_shaders::PaddedVector;
use nexus_rbd::math::Vector;
use rapier::geometry::TriMesh;

/// Convert a glamx Vec3 to bvh's Point3 (different glam versions).
fn to_bvh_point(v: Vector) -> bvh::Point3 {
    bvh::Point3::new(v.x, v.y, v.z)
}

/// Convert a bvh Point3 to a glamx Vec3.
fn from_bvh_point(v: bvh::Point3) -> Vector {
    Vector::new(v.x, v.y, v.z)
}

#[derive(Default, Clone, Debug)]
pub struct ShapeBuffers {
    /// Vertex buffer for polylines and triangle meshes.
    ///
    /// Polyline and TriMesh shapes reference ranges within this buffer.
    /// The shape stores the start and end indices of its vertices in this buffer.
    pub vertices: Vec<PaddedVector>,
    /// Index buffers for polylines, triangle meshes, and convex polyhedrons.
    pub indices: Vec<u32>,
}

#[derive(Copy, Clone)]
pub struct GpuTriMesh {
    /// Index of the root AABB in the vertex buffer.
    pub bvh_vtx_root_id: u32,
    /// The root AABB left-child index.
    pub bvh_idx_root_id: u32,
    // The number of BVH nodes. Triangle indices are stored after the last bvh node.
    pub bvh_node_len: u32,
    // The total number of triangles in the mesh.
    pub num_triangles: u32,
    // The total number of vertices in the mesh.
    pub num_vertices: u32,
}

pub fn convert_trimesh_to_gpu(shape: &TriMesh, buffers: &mut ShapeBuffers) -> GpuTriMesh {
    let bvh_vtx_root_id = buffers.vertices.len();
    let bvh_idx_root_id = buffers.indices.len();
    // Append the BVH data to the vertex/index buffers.
    // TODO: we are constructing a BVH using the `bvh` crate.
    //       While the TriMesh shape technically already has a BVH, parry's BVH
    //       doesn't provide explicit access to the BVH topology. So, for now,
    //       let's just build a new BVH that exposes its internal.
    struct BvhObject {
        aabb: bvh::aabb::AABB,
        node_index: usize,
    }

    impl bvh::aabb::Bounded for BvhObject {
        fn aabb(&self) -> bvh::aabb::AABB {
            self.aabb
        }
    }

    impl bvh::bounding_hierarchy::BHShape for BvhObject {
        fn set_bh_node_index(&mut self, index: usize) {
            self.node_index = index;
        }

        fn bh_node_index(&self) -> usize {
            self.node_index
        }
    }

    let mut objects: Vec<_> = shape
        .triangles()
        .map(|tri| {
            let aabb = tri.local_aabb();
            BvhObject {
                aabb: bvh::aabb::AABB::with_bounds(
                    to_bvh_point(aabb.mins),
                    to_bvh_point(aabb.maxs),
                ),
                node_index: 0,
            }
        })
        .collect();

    let bvh = bvh::bvh::BVH::build(&mut objects);
    let flat_bvh = bvh.flatten();
    buffers.vertices.extend(flat_bvh.iter().flat_map(|n| {
        [
            PaddedVector::new(from_bvh_point(n.aabb.min)),
            PaddedVector::new(from_bvh_point(n.aabb.max)),
        ]
    }));
    let bvh_node_len = flat_bvh.len();
    buffers.indices.extend(
        flat_bvh
            .iter()
            .flat_map(|n| [n.entry_index, n.exit_index, n.shape_index]),
    );

    // Append the actual mesh vertex/index buffers.
    #[cfg(feature = "dim3")]
    {
        let pn = shape
            .pseudo_normals()
            .expect("trimeshes without pseudo-normals are not supported");
        buffers
            .vertices
            .extend(shape.vertices().iter().map(|v| PaddedVector::new(*v)));
        buffers.vertices.extend(
            pn.vertices_pseudo_normal
                .iter()
                .map(|v| PaddedVector::new(*v)),
        );
        assert_eq!(shape.vertices().len(), pn.vertices_pseudo_normal.len());
        buffers.vertices.extend(
            pn.edges_pseudo_normal
                .iter()
                .flat_map(|n| *n)
                .map(PaddedVector::new),
        );
    }
    buffers
        .indices
        .extend(shape.indices().iter().flat_map(|tri| tri.iter().copied()));
    GpuTriMesh {
        bvh_vtx_root_id: bvh_vtx_root_id as u32,
        bvh_idx_root_id: bvh_idx_root_id as u32,
        bvh_node_len: bvh_node_len as u32,
        num_triangles: shape.indices().len() as u32,
        num_vertices: shape.vertices().len() as u32,
    }
}
