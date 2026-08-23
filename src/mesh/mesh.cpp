#include "mesh/mesh.h"

#include <stdexcept>

namespace {

void resizeMeshStorage(Mesh& mesh, int ny, int nx) {
    mesh.nx = nx;
    mesh.ny = ny;
    mesh.owned_j_begin = 0;
    mesh.owned_j_end = nx;

    mesh.u.resize(ny, nx);
    mesh.u0.resize(ny, nx);
    mesh.u_star.resize(ny, nx);
    mesh.v.resize(ny, nx);
    mesh.v0.resize(ny, nx);
    mesh.v_star.resize(ny, nx);

    mesh.x.resize(ny + 1, nx + 1);
    mesh.y.resize(ny + 1, nx + 1);
    mesh.x_c.resize(ny, nx);
    mesh.y_c.resize(ny, nx);
    mesh.area_e.resize(ny, nx);
    mesh.area_w.resize(ny, nx);
    mesh.area_s.resize(ny, nx);
    mesh.area_n.resize(ny, nx);
    mesh.vol.resize(ny, nx);

    mesh.p.resize(ny, nx);
    mesh.p_star.resize(ny, nx);
    mesh.p_prime.resize(ny, nx);
    mesh.u_face.resize(ny, nx - 1);
    mesh.v_face.resize(ny - 1, nx);

    mesh.cell_kind.resize(ny, nx);
    mesh.patch_id.resize(ny, nx);
    mesh.interid.resize(ny, nx);
    mesh.initializeToZero();
}

}  // namespace

Mesh::Mesh(int n_y, int n_x) {
    if (n_y < 1 || n_x < 1) {
        throw std::invalid_argument("网格尺寸必须为正数");
    }
    resizeMeshStorage(*this, n_y, n_x);
}

void Mesh::initializeToZero() {
    u.setZero();
    u0.setZero();
    u_star.setZero();
    v.setZero();
    v0.setZero();
    v_star.setZero();
    x.setZero();
    y.setZero();
    x_c.setZero();
    y_c.setZero();
    area_e.setZero();
    area_w.setZero();
    area_s.setZero();
    area_n.setZero();
    vol.setZero();
    p.setZero();
    p_star.setZero();
    p_prime.setZero();
    u_face.setZero();
    v_face.setZero();
    cell_kind.setConstant(static_cast<int>(CellKind::Interior));
    patch_id.setConstant(-1);
    interid.setConstant(-1);
    internumber = 0;
    interi.clear();
    interj.clear();
    boundary_patches.clear();
}

void Mesh::createInterId() {
    interid.setConstant(-1);
    interi.clear();
    interj.clear();
    internumber = 0;
    interi.reserve(static_cast<std::size_t>(nx * ny));
    interj.reserve(static_cast<std::size_t>(nx * ny));

    for (int j = 0; j < nx; ++j) {
        for (int i = 0; i < ny; ++i) {
            if (isInteriorCell(*this, i, j)) {
                interid(i, j) = internumber++;
                interi.push_back(i);
                interj.push_back(j);
            }
        }
    }
}
