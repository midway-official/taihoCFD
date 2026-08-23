#include "mesh/boundary.h"

#include "mesh/mesh.h"

#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

CellKind cellKind(const Mesh& mesh, int i, int j) {
    return static_cast<CellKind>(mesh.cell_kind(i, j));
}

const BoundaryPatch& checkedPatch(const Mesh& mesh, int i, int j) {
    if (!isPhysicalBoundaryCell(mesh, i, j)) {
        throw std::invalid_argument("请求边界条件的单元不是物理边界");
    }
    const int id = mesh.patch_id(i, j);
    if (id < 0 || static_cast<std::size_t>(id) >= mesh.boundary_patches.size()) {
        throw std::runtime_error("物理边界单元的 patchId 无效");
    }
    return mesh.boundary_patches[static_cast<std::size_t>(id)];
}

BoundaryPatch makeLegacyPatch(
    int legacy_type,
    int zone,
    double u_value,
    double v_value)
{
    BoundaryPatch patch;
    if (legacy_type > 0) {
        patch.name = "wall_" + std::to_string(legacy_type) + '_' +
            std::to_string(zone);
        patch.kind = PatchKind::Wall;
        patch.velocity.type =
            u_value == 0.0 && v_value == 0.0
            ? BoundaryConditionType::NoSlip
            : BoundaryConditionType::FixedValue;
        patch.velocity.value = {u_value, v_value};
        patch.pressure.type = BoundaryConditionType::ZeroGradient;
    } else if (legacy_type == -2) {
        patch.name = "velocityInlet_" + std::to_string(zone);
        patch.velocity.type = BoundaryConditionType::FixedValue;
        patch.velocity.value = {u_value, v_value};
        patch.pressure.type = BoundaryConditionType::ZeroGradient;
    } else if (legacy_type == -1) {
        patch.name = "pressureOutlet_" + std::to_string(zone);
        patch.velocity.type = BoundaryConditionType::ZeroGradient;
        patch.pressure.type = BoundaryConditionType::FixedValue;
        patch.pressure.value = 0.0;
    } else {
        throw std::runtime_error(
            "未知 legacy 边界类型: " + std::to_string(legacy_type));
    }
    return patch;
}

Eigen::Vector2d cellVelocity(const Mesh& mesh, int i, int j) {
    return {mesh.u(i, j), mesh.v(i, j)};
}

}  // namespace

bool isInteriorCell(const Mesh& mesh, int i, int j) {
    return cellKind(mesh, i, j) == CellKind::Interior;
}

bool isPhysicalBoundaryCell(const Mesh& mesh, int i, int j) {
    return cellKind(mesh, i, j) == CellKind::PhysicalBoundary;
}

bool isProcessorCell(const Mesh& mesh, int i, int j) {
    return cellKind(mesh, i, j) == CellKind::Processor;
}

bool isCoupledCell(const Mesh& mesh, int i, int j) {
    return isInteriorCell(mesh, i, j) || isProcessorCell(mesh, i, j);
}

bool isFixedPressureBoundaryCell(const Mesh& mesh, int i, int j) {
    return isPhysicalBoundaryCell(mesh, i, j) &&
        hasFixedPressure(boundaryPatch(mesh, i, j));
}

const BoundaryPatch& boundaryPatch(const Mesh& mesh, int i, int j) {
    return checkedPatch(mesh, i, j);
}

BoundaryPatch& boundaryPatch(Mesh& mesh, int i, int j) {
    checkedPatch(mesh, i, j);
    return mesh.boundary_patches[static_cast<std::size_t>(mesh.patch_id(i, j))];
}

bool fixesValue(BoundaryConditionType type) {
    return type == BoundaryConditionType::FixedValue ||
        type == BoundaryConditionType::NoSlip;
}

bool hasFixedPressure(const BoundaryPatch& patch) {
    return fixesValue(patch.pressure.type);
}

Eigen::Vector2d evaluateVelocityBoundary(
    const BoundaryPatch& patch,
    const Eigen::Vector2d& owner_value,
    double outward_flux)
{
    switch (patch.velocity.type) {
        case BoundaryConditionType::FixedValue:
            return patch.velocity.value;
        case BoundaryConditionType::NoSlip:
            return Eigen::Vector2d::Zero();
        case BoundaryConditionType::ZeroGradient:
            return owner_value;
        case BoundaryConditionType::InletOutlet:
            return outward_flux >= 0.0
                ? owner_value
                : patch.velocity.inlet_value;
    }
    throw std::runtime_error("未知速度边界条件");
}

double evaluatePressureBoundary(
    const BoundaryPatch& patch,
    double owner_value,
    double outward_flux)
{
    switch (patch.pressure.type) {
        case BoundaryConditionType::FixedValue:
            return patch.pressure.value;
        case BoundaryConditionType::ZeroGradient:
            return owner_value;
        case BoundaryConditionType::InletOutlet:
            return outward_flux >= 0.0
                ? owner_value
                : patch.pressure.inlet_value;
        case BoundaryConditionType::NoSlip:
            break;
    }
    throw std::runtime_error("标量压力不支持 noSlip 边界条件");
}

void importLegacyBoundaryData(
    Mesh& mesh,
    const Eigen::MatrixXi& legacy_type,
    const Eigen::MatrixXi& legacy_zone,
    const std::vector<double>& zone_u,
    const std::vector<double>& zone_v)
{
    if (legacy_type.rows() != mesh.ny || legacy_type.cols() != mesh.nx ||
        legacy_zone.rows() != mesh.ny || legacy_zone.cols() != mesh.nx ||
        zone_u.empty() || zone_u.size() != zone_v.size()) {
        throw std::invalid_argument("legacy 边界数据尺寸或速度表无效");
    }

    mesh.boundary_patches.clear();
    mesh.patch_id.setConstant(-1);
    std::map<std::pair<int, int>, int> patch_ids;
    for (int j = 0; j < mesh.nx; ++j) {
        for (int i = 0; i < mesh.ny; ++i) {
            const int type = legacy_type(i, j);
            if (type == 0) {
                mesh.cell_kind(i, j) = static_cast<int>(CellKind::Interior);
                continue;
            }
            if (type == -3) {
                mesh.cell_kind(i, j) = static_cast<int>(CellKind::Processor);
                continue;
            }

            const int zone = legacy_zone(i, j);
            if (zone < 0 || static_cast<std::size_t>(zone) >= zone_u.size()) {
                throw std::runtime_error("legacy zoneid 超出 zoneuv 范围");
            }
            const std::pair<int, int> key{type, zone};
            auto [position, inserted] = patch_ids.emplace(
                key, static_cast<int>(mesh.boundary_patches.size()));
            if (inserted) {
                mesh.boundary_patches.push_back(makeLegacyPatch(
                    type,
                    zone,
                    zone_u[static_cast<std::size_t>(zone)],
                    zone_v[static_cast<std::size_t>(zone)]));
            }
            mesh.cell_kind(i, j) =
                static_cast<int>(CellKind::PhysicalBoundary);
            mesh.patch_id(i, j) = position->second;
        }
    }
    rebuildBoundaryPatchCells(mesh);
}

void rebuildBoundaryPatchCells(Mesh& mesh) {
    for (BoundaryPatch& patch : mesh.boundary_patches) {
        patch.cells.clear();
    }
    for (int j = 0; j < mesh.nx; ++j) {
        for (int i = 0; i < mesh.ny; ++i) {
            if (!isPhysicalBoundaryCell(mesh, i, j)) {
                continue;
            }
            const int id = mesh.patch_id(i, j);
            if (id < 0 ||
                static_cast<std::size_t>(id) >= mesh.boundary_patches.size()) {
                throw std::runtime_error("无法重建边界 patch：patchId 无效");
            }
            mesh.boundary_patches[static_cast<std::size_t>(id)]
                .cells.push_back({i, j});
        }
    }
}

void validateBoundaryModel(const Mesh& mesh) {
    std::set<std::string> names;
    for (std::size_t patch_index = 0;
         patch_index < mesh.boundary_patches.size(); ++patch_index) {
        const BoundaryPatch& patch = mesh.boundary_patches[patch_index];
        if (patch.name.empty() || !names.insert(patch.name).second) {
            throw std::runtime_error("patch 名称为空或重复");
        }
        if (!patch.velocity.value.allFinite() ||
            !patch.velocity.inlet_value.allFinite() ||
            !std::isfinite(patch.pressure.value) ||
            !std::isfinite(patch.pressure.inlet_value)) {
            throw std::runtime_error("patch 边界值包含非有限数");
        }
        if (patch.pressure.type == BoundaryConditionType::NoSlip) {
            throw std::runtime_error("标量压力场不能使用 noSlip");
        }
        if (patch.kind == PatchKind::Wall &&
            !fixesValue(patch.velocity.type)) {
            throw std::runtime_error("当前 wall patch 只支持 fixedValue/noSlip 速度");
        }
        for (const CellIndex cell : patch.cells) {
            if (cell.i < 0 || cell.i >= mesh.ny ||
                cell.j < 0 || cell.j >= mesh.nx ||
                !isPhysicalBoundaryCell(mesh, cell.i, cell.j) ||
                mesh.patch_id(cell.i, cell.j) !=
                    static_cast<int>(patch_index)) {
                throw std::runtime_error("patch 单元列表与 patchId 映射不一致");
            }
        }
    }
}

void initializeBoundaryConditions(Mesh& mesh) {
    for (const BoundaryPatch& patch : mesh.boundary_patches) {
        for (const CellIndex cell : patch.cells) {
            const Eigen::Vector2d velocity = evaluateVelocityBoundary(
                patch, Eigen::Vector2d::Zero(), 0.0);
            mesh.u(cell.i, cell.j) = velocity.x();
            mesh.u0(cell.i, cell.j) = velocity.x();
            mesh.u_star(cell.i, cell.j) = velocity.x();
            mesh.v(cell.i, cell.j) = velocity.y();
            mesh.v0(cell.i, cell.j) = velocity.y();
            mesh.v_star(cell.i, cell.j) = velocity.y();
            const double pressure = evaluatePressureBoundary(patch, 0.0);
            mesh.p(cell.i, cell.j) = pressure;
            mesh.p_star(cell.i, cell.j) = pressure;
            mesh.p_prime(cell.i, cell.j) = 0.0;
        }
    }

    for (int j = 0; j < mesh.nx - 1; ++j) {
        for (int i = 0; i < mesh.ny; ++i) {
            if (isInteriorCell(mesh, i, j) &&
                isPhysicalBoundaryCell(mesh, i, j + 1)) {
                mesh.u_face(i, j) = evaluateVelocityBoundary(
                    boundaryPatch(mesh, i, j + 1),
                    cellVelocity(mesh, i, j), 0.0).x();
            } else if (isPhysicalBoundaryCell(mesh, i, j) &&
                       isInteriorCell(mesh, i, j + 1)) {
                mesh.u_face(i, j) = evaluateVelocityBoundary(
                    boundaryPatch(mesh, i, j),
                    cellVelocity(mesh, i, j + 1), 0.0).x();
            }
        }
    }

    for (int j = 0; j < mesh.nx; ++j) {
        for (int i = 0; i < mesh.ny - 1; ++i) {
            if (isInteriorCell(mesh, i, j) &&
                isPhysicalBoundaryCell(mesh, i + 1, j)) {
                mesh.v_face(i, j) = evaluateVelocityBoundary(
                    boundaryPatch(mesh, i + 1, j),
                    cellVelocity(mesh, i, j), 0.0).y();
            } else if (isPhysicalBoundaryCell(mesh, i, j) &&
                       isInteriorCell(mesh, i + 1, j)) {
                mesh.v_face(i, j) = evaluateVelocityBoundary(
                    boundaryPatch(mesh, i, j),
                    cellVelocity(mesh, i + 1, j), 0.0).y();
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
