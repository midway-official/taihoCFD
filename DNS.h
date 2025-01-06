// dns.h
#ifndef DNS_H
#define DNS_H

#include <iostream>
#include <iomanip>  
#include <eigen3/Eigen/Sparse>
#include <algorithm>
#include <fstream>
#include <vector>

using namespace Eigen;
using namespace std;
// 全局变量声明
extern int n_x0, n_y0;
extern double dx, dy, vx;
extern double velocity;
extern double l2_norm_x, l2_norm_y, l2_norm_p;
extern double a, b;
// Mesh 类声明
class Mesh {
public:
    MatrixXd u,u0,u_star;
    MatrixXd v,v0,v_star;
    MatrixXd p, p_star, p_prime;
    MatrixXd u_face, v_face;
    MatrixXd bctype,zoneid;
    MatrixXi interid; 
    int internumber;
    int nx, ny;  // 添加网格尺寸成员
    vector<int> interi, interj;  // 新增位置向量
    vector<double> zoneu;
    vector<double> zonev;

    Mesh() = default;  // 默认构造函数
    Mesh(int n_y, int n_x);  // 参数化构造函数

    void initializeToZero(); // 初始化所有矩阵为零
    void displayMatrix(const MatrixXd& matrix, const std::string& name) const; // 显示矩阵内容
    void displayAll() const; // 显示所有矩阵
    void createInterId(); // 创建内部点编号
    void setBlock(int x1, int y1, int x2, int y2, double bcValue, double zoneValue); // 设置区域边界条件
    void setZoneUV(int zoneIndex, double u, double v); // 设置区域速度
    void initializeBoundaryConditions();  // 初始化边界条件
};
class Equation {
public:
    MatrixXd A_p, A_e, A_w, A_n, A_s;
    VectorXd source;
    SparseMatrix<double> A;
    int n_x, n_y;
    Mesh& mesh;

    // 只保留声明
    Equation(Mesh& mesh_);

    void initializeToZero();
    void build_matrix();
};
// 函数声明

void solve(Equation& equation, double epsilon, double& l2_norm, MatrixXd& phi);
void face_velocity(Mesh &mesh,Equation &equ_u);


void pressure_function(Mesh &mesh,Equation &equ_p,Equation &equ_u);

//修正压力
void correct_pressure(Mesh &mesh,Equation &equ_u);

void correct_velocity(Mesh &mesh,Equation &equ_u);

void post_processing(Mesh &mseh,int n_x,int n_y,double a);

void show_progress_bar(int current_step, int total_steps, double elapsed_time);


//离散动量方程

void movement_function(Mesh &mesh, Equation &equ_u, Equation &equ_v,double re);

#endif // DNS_H