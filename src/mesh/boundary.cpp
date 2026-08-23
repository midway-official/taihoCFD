#include "mesh/boundary.h"

#include "mesh/mesh.h"

#include <stdexcept>

double boundaryU(const Mesh& mesh, int i, int j) {
    const int zone = mesh.zoneid(i, j);
    if (zone < 0 || static_cast<std::size_t>(zone) >= mesh.zoneu.size()) {
        throw std::runtime_error("zoneid 超出 zoneu 范围");
    }
    return mesh.zoneu[static_cast<std::size_t>(zone)];
}

double boundaryV(const Mesh& mesh, int i, int j) {
    const int zone = mesh.zoneid(i, j);
    if (zone < 0 || static_cast<std::size_t>(zone) >= mesh.zonev.size()) {
        throw std::runtime_error("zoneid 超出 zonev 范围");
    }
    return mesh.zonev[static_cast<std::size_t>(zone)];
}

void initializeBoundaryConditions(Mesh& mesh) {
    for (int j = 0; j < mesh.nx; ++j) {
        for (int i = 0; i < mesh.ny; ++i) {
            const int type = mesh.bctype(i, j);
            if (!isInterior(type) && !isMpiGhost(type)) {
                const double bc_u = boundaryU(mesh, i, j);
                const double bc_v = boundaryV(mesh, i, j);
                mesh.u(i, j) = bc_u;
                mesh.u0(i, j) = bc_u;
                mesh.u_star(i, j) = bc_u;
                mesh.v(i, j) = bc_v;
                mesh.v0(i, j) = bc_v;
                mesh.v_star(i, j) = bc_v;
            }
            if (isPressureOutlet(type)) {
                mesh.p(i, j) = 0.0;
                mesh.p_star(i, j) = 0.0;
                mesh.p_prime(i, j) = 0.0;
            }
        }
    }

    for (int j = 0; j < mesh.nx - 1; ++j) {
        for (int i = 0; i < mesh.ny; ++i) {
            const bool left_interior = isInterior(mesh.bctype(i, j));
            const bool right_interior = isInterior(mesh.bctype(i, j + 1));
            if (left_interior && !right_interior &&
                !isMpiGhost(mesh.bctype(i, j + 1))) {
                mesh.u_face(i, j) = boundaryU(mesh, i, j + 1);
            } else if (!left_interior && !isMpiGhost(mesh.bctype(i, j)) &&
                       right_interior) {
                mesh.u_face(i, j) = boundaryU(mesh, i, j);
            }
        }
    }

    for (int j = 0; j < mesh.nx; ++j) {
        for (int i = 0; i < mesh.ny - 1; ++i) {
            const bool lower_interior = isInterior(mesh.bctype(i, j));
            const bool upper_interior = isInterior(mesh.bctype(i + 1, j));
            if (lower_interior && !upper_interior &&
                !isMpiGhost(mesh.bctype(i + 1, j))) {
                mesh.v_face(i, j) = boundaryV(mesh, i + 1, j);
            } else if (!lower_interior && !isMpiGhost(mesh.bctype(i, j)) &&
                       upper_interior) {
                mesh.v_face(i, j) = boundaryV(mesh, i, j);
            }
        }
    }
}

void initializeFlowFields(Mesh& mesh) {
    mesh.u.setZero();
    mesh.u0.setZero();
    mesh.u_star.setZero();
    mesh.v.setZero();
    mesh.v0.setZero();
    mesh.v_star.setZero();
    mesh.p.setZero();
    mesh.p_star.setZero();
    mesh.p_prime.setZero();
    mesh.u_face.setZero();
    mesh.v_face.setZero();
    initializeBoundaryConditions(mesh);
}
