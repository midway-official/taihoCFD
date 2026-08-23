#include "parallel/halo_exchange.h"

#include <mpi.h>

#include <stdexcept>

void exchangeColumns(Eigen::MatrixXd& matrix, int rank, int num_procs) {
    constexpr int ghost_layers = 2;
    if (num_procs == 1) {
        return;
    }
    if (rank < 0 || rank >= num_procs || matrix.cols() < 2 * ghost_layers) {
        throw std::invalid_argument("非法 MPI 分区或矩阵列数不足");
    }

    const int left = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int right = rank + 1 == num_procs ? MPI_PROC_NULL : rank + 1;
    const int begin = rank == 0 ? 0 : ghost_layers;
    const int end = static_cast<int>(matrix.cols()) -
        (rank + 1 == num_procs ? 0 : ghost_layers);
    const int count = static_cast<int>(matrix.rows()) * ghost_layers;
    double* const dummy = matrix.data();
    double* const receive_from_right =
        right == MPI_PROC_NULL ? dummy : matrix.col(end).data();
    double* const receive_from_left =
        left == MPI_PROC_NULL ? dummy : matrix.col(begin - ghost_layers).data();

    MPI_Sendrecv(
        matrix.col(begin).data(), count, MPI_DOUBLE, left, 101,
        receive_from_right, count, MPI_DOUBLE, right, 101,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(
        matrix.col(end - ghost_layers).data(), count, MPI_DOUBLE, right, 102,
        receive_from_left, count, MPI_DOUBLE, left, 102,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void vectorToMatrix(
    const Eigen::VectorXd& values,
    Eigen::MatrixXd& field,
    const Mesh& mesh)
{
    if (values.size() != mesh.internumber ||
        field.rows() != mesh.ny || field.cols() != mesh.nx) {
        throw std::invalid_argument("vectorToMatrix 尺寸不匹配");
    }
    for (int n = 0; n < mesh.internumber; ++n) {
        field(
            mesh.interi[static_cast<std::size_t>(n)],
            mesh.interj[static_cast<std::size_t>(n)]) = values[n];
    }
}

void matrixToVector(
    const Eigen::MatrixXd& field,
    Eigen::VectorXd& values,
    const Mesh& mesh)
{
    if (field.rows() != mesh.ny || field.cols() != mesh.nx) {
        throw std::invalid_argument("matrixToVector 尺寸不匹配");
    }
    values.resize(mesh.internumber);
    for (int n = 0; n < mesh.internumber; ++n) {
        values[n] = field(
            mesh.interi[static_cast<std::size_t>(n)],
            mesh.interj[static_cast<std::size_t>(n)]);
    }
}
