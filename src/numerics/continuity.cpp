#include "numerics/continuity.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>

ContinuityMetrics computeContinuityMetrics(
    const Mesh& mesh,
    const ParallelContext& parallel)
{
    parallel.validate();
    constexpr double small = 1e-30;
    std::array<double, 3> local_sum{0.0, 0.0, 0.0};
    double local_max = 0.0;

    for (int n = 0; n < mesh.internumber; ++n) {
        const int i = mesh.interi[static_cast<std::size_t>(n)];
        const int j = mesh.interj[static_cast<std::size_t>(n)];
        const double fe = mesh.u_face(i, j) * mesh.area_e(i, j);
        const double fw = mesh.u_face(i, j - 1) * mesh.area_w(i, j);
        const double fn = mesh.v_face(i - 1, j) * mesh.area_n(i, j);
        const double fs = mesh.v_face(i, j) * mesh.area_s(i, j);
        const double imbalance = fe - fw + fs - fn;

        local_sum[0] += std::abs(imbalance);
        local_sum[1] += imbalance * imbalance;
        local_sum[2] += std::abs(fe) + std::abs(fw) + std::abs(fn) + std::abs(fs);
        local_max = std::max(local_max, std::abs(imbalance));
    }

    std::array<double, 3> global_sum{};
    double global_max = 0.0;
    MPI_Allreduce(
        local_sum.data(), global_sum.data(), 3, MPI_DOUBLE, MPI_SUM,
        parallel.communicator);
    MPI_Allreduce(
        &local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX,
        parallel.communicator);

    ContinuityMetrics result;
    result.l1 = global_sum[0];
    result.l2 = std::sqrt(global_sum[1]);
    result.max_abs = global_max;
    result.relative = global_sum[0] / std::max(global_sum[2], small);
    return result;
}

ContinuityMetrics computeContinuityMetrics(const Mesh& mesh) {
    return computeContinuityMetrics(mesh, ParallelContext::world());
}
