#pragma once

#include "mesh/mesh.h"

#include <eigen3/Eigen/Sparse>

struct Equation {
    Eigen::MatrixXd A_p;
    Eigen::MatrixXd A_e;
    Eigen::MatrixXd A_w;
    Eigen::MatrixXd A_n;
    Eigen::MatrixXd A_s;
    Eigen::VectorXd source;
    Eigen::SparseMatrix<double> A;
    Mesh& mesh;

    explicit Equation(Mesh& mesh_value);
    void reset();
    void buildMatrix();
};
