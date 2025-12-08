#include "rxmesh/rxmesh_static.h"

#include <random>

using namespace rxmesh;

template <typename T, uint32_t blockThreads>
void compute_face_normal(RXMeshStatic&              rx,
                         rxmesh::VertexAttribute<T> coords,
                         rxmesh::FaceAttribute<T>   normals)
{
    rx.run_query_kernel<Op::FV, blockThreads>(
        [=] __device__(FaceHandle face_id, VertexIterator &fv) mutable
        {
            // Get the face's three vertex coordinates
            const vec3<T> c0 = coords.to_glm<3>(fv[0]);
            const vec3<T> c1 = coords.to_glm<3>(fv[1]);
            const vec3<T> c2 = coords.to_glm<3>(fv[2]);

            // Compute and normalize the face normal
            vec3<T> n = cross(c1 - c0, c2 - c0);
            n = normalize(n);

            // Store the normals
            normals.from_glm(face_id, n);
        }
    );
}

template <typename T, uint32_t blockThreads>
void compute_vertex_normal(RXMeshStatic&                rx,
                           rxmesh::VertexAttribute<T>   coords,
                           rxmesh::VertexAttribute<T>   normals)
{
    rx.run_query_kernel<Op::FV, blockThreads>(
        [=] __device__(FaceHandle face_id, VertexIterator &fv) mutable
        {
            // Get the face's three vertex coordinates
            const vec3<T> c0 = coords.to_glm<3>(fv[0]);
            const vec3<T> c1 = coords.to_glm<3>(fv[1]);
            const vec3<T> c2 = coords.to_glm<3>(fv[2]);

            // Compute the face normal
            vec3<T> n = cross(c1 - c0, c2 - c0);

            // Compute the face's three edge lengths
            vec3<T> l(glm::distance2(c0, c1),
                      glm::distance2(c1, c2),
                      glm::distance2(c2, c0));
            
            // Add the face's normal to its vertex
            for (uint32_t v = 0; v < 3; ++v)
            {
                for (uint32_t i = 0; i < 3; ++i)
                {
                    atomicAdd(&normals(fv[v], i), n[i] / (l[v] + l[(v + 2) % 3]));
                }
            }
        }
    );
}

template <typename T, uint32_t blockThreads>
void compute_heat_diffusion(RXMeshStatic&              rx,
                            rxmesh::VertexAttribute<T> heat_old,
                            rxmesh::VertexAttribute<T> heat_new)
{
    rx.run_query_kernel<Op::VV, blockThreads>(
        [=] __device__(VertexHandle vert_id, VertexIterator &vv) mutable
        {
            constexpr float dt = 0.1f;

            float avg = 0.0f;
            for (uint32_t i = 0; i < vv.size(); ++i)
            {
                avg += heat_old(vv[i], 0);
            }
            avg /= static_cast<float>(vv.size());
            heat_new(vert_id) = heat_old(vert_id) + dt * (avg - heat_old(vert_id));
        }
    );
}

template <typename T, uint32_t blockThreads>
void compute_heat_diffusion2(RXMeshStatic&              rx,
                             rxmesh::VertexAttribute<T> coords,
                             rxmesh::VertexAttribute<T> heat_old,
                             rxmesh::VertexAttribute<T> heat_new)
{
    rx.run_query_kernel<Op::FV, blockThreads>(
        [=] __device__(FaceHandle face_id, VertexIterator &fv) mutable
        {
            constexpr float dt = 0.1f;

            // Get the face's three vertex coordinates
            const vec3<T> c0 = coords.to_glm<3>(fv[0]);
            const vec3<T> c1 = coords.to_glm<3>(fv[1]);
            const vec3<T> c2 = coords.to_glm<3>(fv[2]);

            // Compute edge vectors
            vec3<T> e0 = c1 - c0; // opposite v2
            vec3<T> e1 = c2 - c1; // opposite v0
            vec3<T> e2 = c0 - c2; // opposite v1

            // Compute cotangents
            T cot0 = dot(-e1, e2) / length(cross(-e1, e2));
            T cot1 = dot(-e2, e0) / length(cross(-e2, e0));
            T cot2 = dot(-e0, e1) / length(cross(-e0, e1));

            // ...
        }
    );
}

int main(int argc, char** argv)
{
    rx_init(0);

    if (argc < 2) 
    {
        RXMESH_ERROR("Invalid argument. Usage: RXMeshTemplate.exe input mesh");
        exit(EXIT_SUCCESS);
    }

    int heat_iters = 1;
    if (argc >= 3)
    {
        heat_iters = atoi(argv[2]);
    }

    RXMeshStatic rx(argv[1]);

    // Vertex Coordinates
    auto vertex_pos = *rx.get_input_vertex_coordinates();

    /*
    // Example: Face and Vertex Normals
    */

    
    // rx.for_each_vertex(
    //     DEVICE, [vertex_color, vertex_pos] __device__(const VertexHandle vh)
    //     {
    //         vertex_color(vh, 0) = 0.9;
    //         vertex_color(vh, 1) = vertex_pos(vh, 1);
    //         vertex_color(vh, 2) = vertex_pos(vh, 2);
    //     }
    // );

    // Face Normal
    auto face_normal = *rx.add_face_attribute<float>("fNormals", 3);
    
    // Compute face normals
    compute_face_normal<float, 256>(rx, vertex_pos, face_normal);

    // Vertex Normal
    auto vertex_normal = *rx.add_vertex_attribute<float>("vNormals", 3);

    // Compute vertex normals
    compute_vertex_normal<float, 256>(rx, vertex_pos, vertex_normal);

    /*
    // Example: Heat Visualization
    */

    // Generate a single random source vertex
    // std::random_device rd;
    std::mt19937 rng(1234);
    std::uniform_int_distribution<uint32_t> dist(0, rx.get_num_vertices() - 1);

    uint32_t source = dist(rng);

    // Heat attributes for double buffering
    auto heat_old = *rx.add_vertex_attribute<float>("heatOld", 1);
    heat_old.reset(0.0f, rxmesh::HOST);
    heat_old(0, 0) = 500.0f;
    heat_old.move(rxmesh::HOST, rxmesh::DEVICE);

    auto heat_new = *rx.add_vertex_attribute<float>("heatNew", 1);
    heat_new.reset(0.0f, rxmesh::HOST);
    heat_new.move(rxmesh::HOST, rxmesh::DEVICE);

    // Heat diffusion calculation
    uint32_t iter = 0;
    uint32_t max_iter = heat_iters;
    while (iter < max_iter)
    {
        iter++;
        
        compute_heat_diffusion<float, 256>(rx, heat_old, heat_new);

        // Swap heat buffers
        std::swap(heat_old, heat_new);
    }

    // Set vertex color based on heat
    auto vertex_color = *rx.add_vertex_attribute<float>("heatColor", 3);

    float max_heat = 0.0f;
    heat_old.move(DEVICE, HOST);
    for (uint32_t vh = 0; vh < rx.get_num_vertices(); ++vh)
    {
        max_heat = std::max(max_heat, heat_old(vh, 0));
    }

    rx.for_each_vertex(
        DEVICE, [vertex_color, heat_old, max_heat] __device__(const VertexHandle vh)
        {
            vertex_color(vh, 0) = 1.0f;
            vertex_color(vh, 1) = heat_old(vh, 0) / max_heat;
            vertex_color(vh, 2) = 0.0f;
        }
    );

    // Move attributes to the host
    vertex_color.move(DEVICE, HOST);
    face_normal.move(DEVICE, HOST);
    vertex_normal.move(DEVICE, HOST);

    // Visualize using Polyscope
    auto ps_mesh = rx.get_polyscope_mesh();
    ps_mesh->addVertexColorQuantity("heatColor", vertex_color);
    ps_mesh->addFaceVectorQuantity("fNormal", face_normal);
    ps_mesh->addVertexVectorQuantity("vNormal", vertex_normal);

    std::cout << "Showing mesh in Polyscope. Close the Polyscope window to exit."
              << std::endl;

    polyscope::show();
}
