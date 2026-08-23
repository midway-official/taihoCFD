#include "mesh/mesh.h"

#include <cmath>
#include <stdexcept>

void Mesh::initGeometry() {
    for (int j = 0; j < nx; ++j) {
        for (int i = 0; i < ny; ++i) {
            x_c(i, j) = 0.25 *
                (x(i, j) + x(i, j + 1) + x(i + 1, j) + x(i + 1, j + 1));
            y_c(i, j) = 0.25 *
                (y(i, j) + y(i, j + 1) + y(i + 1, j) + y(i + 1, j + 1));

            area_n(i, j) = std::hypot(
                x(i, j + 1) - x(i, j), y(i, j + 1) - y(i, j));
            area_s(i, j) = std::hypot(
                x(i + 1, j + 1) - x(i + 1, j),
                y(i + 1, j + 1) - y(i + 1, j));
            area_w(i, j) = std::hypot(
                x(i + 1, j) - x(i, j), y(i + 1, j) - y(i, j));
            area_e(i, j) = std::hypot(
                x(i + 1, j + 1) - x(i, j + 1),
                y(i + 1, j + 1) - y(i, j + 1));

            const double d1x = x(i + 1, j + 1) - x(i, j);
            const double d1y = y(i + 1, j + 1) - y(i, j);
            const double d2x = x(i + 1, j) - x(i, j + 1);
            const double d2y = y(i + 1, j) - y(i, j + 1);
            vol(i, j) = 0.5 * std::abs(d1x * d2y - d1y * d2x);
        }
    }
}

void Mesh::validate(bool require_physical_outer_boundary) const {
    if (nx < 5 || ny < 5) {
        throw std::runtime_error("网格过小，Rhie-Chow 模板至少需要 5x5");
    }
    if (owned_j_begin < 0 || owned_j_end > nx || owned_j_begin >= owned_j_end) {
        throw std::runtime_error("非法 owned 列范围");
    }
    if (zoneu.empty() || zoneu.size() != zonev.size()) {
        throw std::runtime_error("边界速度表无效");
    }

    for (int j = 0; j < nx; ++j) {
        for (int i = 0; i < ny; ++i) {
            const int type = bctype(i, j);
            const bool known_type =
                isInterior(type) || isMpiGhost(type) || isWall(type) ||
                isPressureOutlet(type) || isVelocityInlet(type);
            if (!known_type) {
                throw std::runtime_error("网格包含未知边界类型");
            }
            if (isInterior(type)) {
                if (i == 0 || i == ny - 1 || j == 0 || j == nx - 1) {
                    throw std::runtime_error("内部单元位于数组外边界，邻居模板将越界");
                }
            } else if (!isMpiGhost(type)) {
                const int zone = zoneid(i, j);
                if (zone < 0 || static_cast<std::size_t>(zone) >= zoneu.size()) {
                    throw std::runtime_error("物理边界 zoneid 超出 zoneuv 范围");
                }
            }

            if (!std::isfinite(x_c(i, j)) || !std::isfinite(y_c(i, j)) ||
                !std::isfinite(vol(i, j)) || vol(i, j) <= 0.0) {
                throw std::runtime_error("网格包含非有限坐标或非正体积");
            }
            if (j + 1 < nx && x_c(i, j + 1) <= x_c(i, j)) {
                throw std::runtime_error("当前离散要求 x 单元中心沿列严格递增");
            }
            if (i + 1 < ny && y_c(i + 1, j) <= y_c(i, j)) {
                throw std::runtime_error("当前离散要求 y 单元中心沿行严格递增");
            }
        }
    }

    if (require_physical_outer_boundary) {
        for (int j = 0; j < nx; ++j) {
            if (isInterior(bctype(0, j)) || isInterior(bctype(ny - 1, j))) {
                throw std::runtime_error("顶/底外边界必须显式标记边界类型");
            }
        }
        for (int i = 0; i < ny; ++i) {
            if (isInterior(bctype(i, 0)) || isInterior(bctype(i, nx - 1))) {
                throw std::runtime_error("左/右外边界必须显式标记边界类型");
            }
        }
    }
}
