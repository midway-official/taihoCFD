#include "mesh/mesh.h"

#include <algorithm>
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

    mesh.bctype.resize(ny, nx);
    mesh.zoneid.resize(ny, nx);
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
    bctype.setZero();
    zoneid.setZero();
    interid.setConstant(-1);
    internumber = 0;
    interi.clear();
    interj.clear();
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
            if (isInterior(bctype(i, j))) {
                interid(i, j) = internumber++;
                interi.push_back(i);
                interj.push_back(j);
            }
        }
    }
}

void Mesh::setBlock(
    int x1,
    int y1,
    int x2,
    int y2,
    int bc_value,
    int zone_value)
{
    x1 = std::clamp(x1, 0, nx - 1);
    x2 = std::clamp(x2, 0, nx - 1);
    y1 = std::clamp(y1, 0, ny - 1);
    y2 = std::clamp(y2, 0, ny - 1);
    if (x1 > x2) {
        std::swap(x1, x2);
    }
    if (y1 > y2) {
        std::swap(y1, y2);
    }
    bctype.block(y1, x1, y2 - y1 + 1, x2 - x1 + 1).setConstant(bc_value);
    zoneid.block(y1, x1, y2 - y1 + 1, x2 - x1 + 1).setConstant(zone_value);
}

void Mesh::setZoneUV(std::size_t zone_index, double u_value, double v_value) {
    if (zoneu.size() <= zone_index) {
        zoneu.resize(zone_index + 1, 0.0);
        zonev.resize(zone_index + 1, 0.0);
    }
    zoneu[zone_index] = u_value;
    zonev[zone_index] = v_value;
}
