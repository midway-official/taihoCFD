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

// 在其他函数声明后添加
void printMatrix(const MatrixXd& matrix, const string& name, int precision = 4);
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
    Mesh(const std::string& folderPath);  // 网格文件夹构造函数
    void initializeToZero(); // 初始化所有矩阵为零
    void displayMatrix(const MatrixXd& matrix, const std::string& name) const; // 显示矩阵内容
    void displayAll() const; // 显示所有矩阵
    void createInterId(); // 创建内部点编号
    void setBlock(int x1, int y1, int x2, int y2, double bcValue, double zoneValue); // 设置区域边界条件
    void setZoneUV(int zoneIndex, double u, double v); // 设置区域速度
    void initializeBoundaryConditions();  // 初始化边界条件
    // 保存网格到文件夹
    void saveToFolder(const std::string& folderPath) const;
    
    
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


void pressure_function(Mesh &mesh, Equation &equ_p, Equation &equ_u);

//修正压力
void correct_pressure(Mesh &mesh,Equation &equ_u,double alpha_p);


void correct_velocity(Mesh &mesh,Equation &equ_u);

void post_processing(Mesh &mseh);

void show_progress_bar(int current_step, int total_steps, double elapsed_time);


//离散动量方程

void momentum_function(Mesh &mesh, Equation &equ_u, Equation &equ_v,double re,double alpha_uv);
void momentum_function_unsteady(Mesh &mesh, Equation &equ_u, Equation &equ_v,double mu,double dt,double alpha_uv);
void momentum_function_PISO(Mesh &mesh, Equation &equ_u, Equation &equ_v,double mu,double dt);
//竖向分割网格
vector<Mesh> splitMeshVertically(const Mesh& original_mesh, int n);
//合并网格
Mesh mergeMeshesWithoutInterface(const std::vector<Mesh>& sub_meshes);

void readParams(const std::string& folderPath, double& dx, double& dy);
void saveMeshData(const Mesh& mesh, int rank, const std::string& timestep_folder = "");
#endif // DNS_H