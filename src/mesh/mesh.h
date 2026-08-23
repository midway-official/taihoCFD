#pragma once

#include "mesh/boundary.h"

#include <eigen3/Eigen/Dense>

#include <cstddef>
#include <vector>

struct Mesh {
    Eigen::MatrixXd u;
    Eigen::MatrixXd u0;
    Eigen::MatrixXd u_star;
    Eigen::MatrixXd v;
    Eigen::MatrixXd v0;
    Eigen::MatrixXd v_star;

    Eigen::MatrixXd x;
    Eigen::MatrixXd y;
    Eigen::MatrixXd x_c;
    Eigen::MatrixXd y_c;

    Eigen::MatrixXd area_e;
    Eigen::MatrixXd area_w;
    Eigen::MatrixXd area_s;
    Eigen::MatrixXd area_n;
    Eigen::MatrixXd vol;

    Eigen::MatrixXd p;
    Eigen::MatrixXd p_star;
    Eigen::MatrixXd p_prime;
    Eigen::MatrixXd u_face;
    Eigen::MatrixXd v_face;

    Eigen::MatrixXi bctype;
    Eigen::MatrixXi zoneid;
    Eigen::MatrixXi interid;

    int internumber = 0;
    int nx = 0;
    int ny = 0;
    int owned_j_begin = 0;
    int owned_j_end = 0;

    std::vector<int> interi;
    std::vector<int> interj;
    std::vector<double> zoneu;
    std::vector<double> zonev;

    Mesh() = default;
    Mesh(int n_y, int n_x);

    void initializeToZero();
    void initGeometry();
    void createInterId();
    void validate(bool require_physical_outer_boundary) const;
    void setBlock(int x1, int y1, int x2, int y2, int bc_value, int zone_value);
    void setZoneUV(std::size_t zone_index, double u_value, double v_value);

    int ownedColumns() const { return owned_j_end - owned_j_begin; }
};
