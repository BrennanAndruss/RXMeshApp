#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;

int main(int argc, char** argv)
{
    rx_init(0);

    if (argc != 2) 
    {
        RXMESH_ERROR("Invalid argument. Usage: RXMeshTemplate.exe input mesh");
        exit(EXIT_SUCCESS);
    }

    RXMeshStatic rx(argv[1]);

    // Vertex Coordinates
    auto vertex_pos = *rx.get_input_vertex_coordinates();

    // Vertex Color
    auto vertex_color = *rx.add_vertex_attribute<float>("vColor", 3);
    rx.for_each_vertex(
        DEVICE, [vertex_color, vertex_pos] __device__(const VertexHandle vh)
        {
            vertex_color(vh, 0) = 0.9;
            vertex_color(vh, 1) = vertex_pos(vh, 1);
            vertex_color(vh, 2) = 0.9;
        }
    );

    // Face Normal
    auto face_normal = *rx.add_face_attribute<float>("fNormals", 3);
    rx.run_query_kernel<Op::FV, 256>(
        [=] __device__(FaceHandle face_id, VertexIterator &fv) mutable
        {
            // Get the face's three vertex coordinates
            const vec3<float> c0 = vertex_pos.to_glm<3>(fv[0]);
            const vec3<float> c1 = vertex_pos.to_glm<3>(fv[1]);
            const vec3<float> c2 = vertex_pos.to_glm<3>(fv[2]);

            // Compute the face normal
            glm::fvec3 n = cross(c1 - c0, c2 - c0);
            n = glm::normalize(n);

            // Store the normals
            face_normal.from_glm(face_id, n);
        }
    );

    // Move attributes to the host
    vertex_color.move(DEVICE, HOST);
    face_normal.move(DEVICE, HOST);

    // Visualize using Polyscope
    auto ps_mesh = rx.get_polyscope_mesh();
    ps_mesh->addVertexColorQuantity("vColor", vertex_color);
    ps_mesh->addFaceVectorQuantity("fNormal", face_normal);

    std::cout << "Showing mesh in Polyscope. Close the Polyscope window to exit."
              << std::endl;

    polyscope::show();
}
