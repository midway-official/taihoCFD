#include "io/mesh_reader.h"
#include "mesh/boundary.h"
#include "parallel/domain_decomposition.h"

#include <mpi.h>

#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int num_procs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    try {
        if (argc != 3) {
            throw std::invalid_argument(
                "usage: boundary_model_test <mesh-folder> <open|closed>");
        }
        const bool expect_open = std::string(argv[2]) == "open";
        if (!expect_open && std::string(argv[2]) != "closed") {
            throw std::invalid_argument("边界期望必须是 open 或 closed");
        }

        Mesh original = readMesh(argv[1]);
        std::set<std::string> names;
        for (const BoundaryPatch& patch : original.boundary_patches) {
            if (patch.name.empty() || !names.insert(patch.name).second) {
                throw std::runtime_error("patch 名称为空或重复");
            }
            if (patch.pressure.type == BoundaryConditionType::NoSlip) {
                throw std::runtime_error("压力场不能使用 noSlip");
            }
        }

        Mesh mesh = extractLocalMesh(original, rank, num_procs);
        initializeFlowFields(mesh);
        int local_physical = 0;
        int local_processor = 0;
        int local_fixed_pressure = 0;
        int local_fixed_velocity = 0;
        for (int j = 0; j < mesh.nx; ++j) {
            for (int i = 0; i < mesh.ny; ++i) {
                if (isProcessorCell(mesh, i, j)) {
                    ++local_processor;
                } else if (isPhysicalBoundaryCell(mesh, i, j)) {
                    ++local_physical;
                    const BoundaryPatch& patch = boundaryPatch(mesh, i, j);
                    if (hasFixedPressure(patch)) {
                        ++local_fixed_pressure;
                        if (patch.velocity.type !=
                            BoundaryConditionType::ZeroGradient) {
                            throw std::runtime_error(
                                "legacy 压力出口没有映射为 U zeroGradient");
                        }
                    }
                    if (fixesValue(patch.velocity.type)) {
                        ++local_fixed_velocity;
                    }
                } else if (!isInteriorCell(mesh, i, j)) {
                    throw std::runtime_error("单元拓扑没有明确分类");
                }
            }
        }

        int local_counts[4] = {
            local_physical,
            local_processor,
            local_fixed_pressure,
            local_fixed_velocity,
        };
        int global_counts[4] = {0, 0, 0, 0};
        MPI_Allreduce(
            local_counts, global_counts, 4, MPI_INT, MPI_SUM,
            MPI_COMM_WORLD);
        if (global_counts[0] <= 0 || global_counts[3] <= 0 ||
            (expect_open && global_counts[2] <= 0) ||
            (!expect_open && global_counts[2] != 0) ||
            (num_procs > 1 && global_counts[1] <= 0)) {
            throw std::runtime_error("patch/字段边界/MPI 拓扑统计不符合预期");
        }

        for (BoundaryPatch& patch : mesh.boundary_patches) {
            if (!hasFixedPressure(patch)) {
                continue;
            }
            patch.pressure.value = 2.5;
        }
        initializeFlowFields(mesh);
        for (const BoundaryPatch& patch : mesh.boundary_patches) {
            if (!hasFixedPressure(patch)) {
                continue;
            }
            for (const CellIndex cell : patch.cells) {
                if (mesh.p(cell.i, cell.j) != 2.5) {
                    throw std::runtime_error("非零 fixedValue 压力没有应用");
                }
            }
        }

        if (rank == 0) {
            std::cout << "patches=" << original.boundary_patches.size()
                      << " physical_cells=" << global_counts[0]
                      << " processor_cells=" << global_counts[1]
                      << " fixed_pressure_cells=" << global_counts[2]
                      << " fixed_velocity_cells=" << global_counts[3]
                      << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    MPI_Finalize();
    return 0;
}
