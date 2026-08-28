#include "parallel/domain_decomposition.h"

#include "mesh/boundary.h"

#include <stdexcept>
#include <vector>

Mesh extractLocalMesh(
    const Mesh& original,
    const ParallelContext& parallel)
{
    constexpr int ghost_layers = 2;
    parallel.validate();
    const int rank = parallel.rank;
    const int num_procs = parallel.size;
    if (original.nx < ghost_layers * num_procs) {
        throw std::runtime_error("每个 MPI 子域至少需要 2 个真实列以支持双 ghost 层");
    }

    std::vector<int> widths(static_cast<std::size_t>(num_procs));
    int remaining = original.nx;
    for (int part = 0; part < num_procs; ++part) {
        widths[static_cast<std::size_t>(part)] = remaining / (num_procs - part);
        remaining -= widths[static_cast<std::size_t>(part)];
    }
    for (int width : widths) {
        if (width < ghost_layers) {
            throw std::runtime_error("MPI 子域真实宽度小于 ghost 层数");
        }
    }

    int start = 0;
    for (int part = 0; part < rank; ++part) {
        start += widths[static_cast<std::size_t>(part)];
    }

    const int owned_width = widths[static_cast<std::size_t>(rank)];
    const int left_ghost = rank > 0 ? ghost_layers : 0;
    const int right_ghost = rank + 1 < num_procs ? ghost_layers : 0;
    const int local_nx = owned_width + left_ghost + right_ghost;
    const int original_offset = start - left_ghost;
    if (original_offset < 0 || original_offset + local_nx > original.nx) {
        throw std::runtime_error("子域 ghost 映射越过全局网格范围");
    }

    Mesh local(original.ny, local_nx);
    local.boundary_patches = original.boundary_patches;
    local.owned_j_begin = left_ghost;
    local.owned_j_end = left_ghost + owned_width;

    for (int j = 0; j < local_nx; ++j) {
        const int original_j = original_offset + j;
        for (int i = 0; i < original.ny; ++i) {
            local.cell_kind(i, j) = original.cell_kind(i, original_j);
            local.patch_id(i, j) = original.patch_id(i, original_j);
        }
    }
    for (int j = 0; j <= local_nx; ++j) {
        const int original_j = original_offset + j;
        for (int i = 0; i <= original.ny; ++i) {
            local.x(i, j) = original.x(i, original_j);
            local.y(i, j) = original.y(i, original_j);
        }
    }

    if (left_ghost > 0) {
        local.cell_kind.leftCols(ghost_layers).setConstant(
            static_cast<int>(CellKind::Processor));
        local.patch_id.leftCols(ghost_layers).setConstant(-1);
    }
    if (right_ghost > 0) {
        local.cell_kind.rightCols(ghost_layers).setConstant(
            static_cast<int>(CellKind::Processor));
        local.patch_id.rightCols(ghost_layers).setConstant(-1);
    }

    rebuildBoundaryPatchCells(local);
    local.createInterId();
    initializeBoundaryConditions(local);
    local.initGeometry();
    local.validate(false);
    return local;
}

Mesh extractLocalMesh(const Mesh& original, int rank, int num_procs) {
    return extractLocalMesh(
        original, ParallelContext{MPI_COMM_WORLD, rank, num_procs});
}
