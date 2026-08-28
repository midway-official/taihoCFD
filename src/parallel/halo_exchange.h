#pragma once

#include "mesh/mesh.h"
#include "parallel/parallel_context.h"

void exchangeColumns(Eigen::MatrixXd& matrix, const ParallelContext& parallel);
void exchangeColumns(Eigen::MatrixXd& matrix, int rank, int num_procs);

void vectorToMatrix(
    const Eigen::VectorXd& values,
    Eigen::MatrixXd& field,
    const Mesh& mesh);

void matrixToVector(
    const Eigen::MatrixXd& field,
    Eigen::VectorXd& values,
    const Mesh& mesh);
