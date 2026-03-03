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
extern double dx, dy;


// 在其他函数声明后添加
void printMatrix(const MatrixXd& matrix, const string& name, int precision = 4);
// Mesh 类声明
class Mesh {
public:
    MatrixXd u,u0,u_star;
    MatrixXd v,v0,v_star;
    MatrixXd x,y;
    MatrixXd x_c,y_c;
    MatrixXd p, p_star, p_prime;
    MatrixXd u_face, v_face;
    MatrixXi bctype,zoneid;
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
    void setZoneUV(size_t zoneIndex, double u, double v); // 设置区域速度
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
    Equation(Mesh& mesh_);

    void initializeToZero();
    void build_matrix();
};
// 函数声明

void solve(Equation& equation, double epsilon, double& l2_norm, MatrixXd& phi);
void face_velocity(Mesh &mesh,Equation &equ_u);


void pressure_function(Mesh &mesh, Equation &equ_p, Equation &equ_u);

//修正压力
void correct_pressure(Mesh &mesh,double alpha_p);


void correct_velocity(Mesh &mesh,Equation &equ_u);

void post_processing(Mesh &mseh);

void show_progress_bar(int current_step, int total_steps, double elapsed_time);


//离散动量方程

void momentum_function(Mesh &mesh, Equation &equ_u, Equation &equ_v,double re,double alpha_uv);
void momentum_function_unsteady(Mesh &mesh, Equation &equ_u, Equation &equ_v,double mu,double dt);




// ============================================================================
// 并行计算相关函数
// ============================================================================

/**
 * @defgroup ParallelFunctions 并行计算函数
 * @{
 */

/**
 * @brief 垂直分割网格(用于并行计算)
 * @param original_mesh 原始完整网格
 * @param n 分割数量(进程数)
 * @return 子网格向量
 * @details
 * 区域分解策略:
 * 1. 将网格在x方向分割成n个子区域
 * 2. 尽量均匀分配宽度: widths[k] ≈ nx/n
 * 3. 在接口处添加虚拟交换层(bctype=-3)
 * 4. 每个子网格包含完整的边界信息
 * 
 * 子网格结构:
 * - 第一个子网格: 保留左边界,右侧添加交换层
 * - 中间子网格: 两侧都添加交换层
 * - 最后子网格: 左侧添加交换层,保留右边界
 * 
 * @note 分割后各子网格独立求解,通过交换层通信
 */
vector<Mesh> splitMeshVertically(const Mesh& original_mesh, int n);


/** @} */ // end of ParallelFunctions

// ============================================================================
// 文件I/O函数
// ============================================================================

/**
 * @defgroup IOFunctions 文件输入输出函数
 * @{
 */

/**
 * @brief 从参数文件读取网格步长
 * @param folderPath 文件夹路径
 * @param dx 输出x方向步长
 * @param dy 输出y方向步长
 * @details
 * 文件格式(params.txt):
 * 第一行: nx ny
 * 第二行: dx dy
 * 
 * @note 如果读取失败会在stderr输出错误信息
 */
void readParams(const std::string& folderPath, double& dx, double& dy);

/**
 * @brief 保存网格数据到文件
 * @param mesh 网格对象
 * @param rank 进程编号(用于多进程输出)
 * @param timestep_folder 时间步文件夹路径(可选,默认为空)
 * @details
 * 输出文件:
 * - u_<rank>.dat: x方向速度(u_star)
 * - v_<rank>.dat: y方向速度(v_star)
 * - p_<rank>.dat: 压力(p)
 * 
 * 同时在steady/目录下保存:
 * - u_<rank>.dat, v_<rank>.dat, p_<rank>.dat
 * - uf_<rank>.dat: u_face
 * - vf_<rank>.dat: v_face
 * 
 * @note 如果指定timestep_folder,文件保存在该目录下
 * @note steady/目录会自动创建(如果不存在)
 */
void saveMeshData(const Mesh& mesh, int rank, const std::string& timestep_folder = "");

/**
 * @brief 保存预测数据(用于非稳态计算)
 * @param mesh 网格对象
 * @param rank 进程编号
 * @param timesteps 时间步编号
 * @param timestep_folder 时间步根文件夹路径
 * @details
 * 输出文件(在timestep_folder/<timesteps>/目录下):
 * - up_<rank>.dat: u_face
 * - vp_<rank>.dat: v_face
 * - pp_<rank>.dat: p_prime
 * 
 * 用途: 保存中间预测步的数据,用于分析PISO算法收敛过程
 * 
 * @note 会自动创建时间步子目录
 */
void saveforecastData(const Mesh& mesh, int rank, int timesteps,
                     double mu);

/** @} */ // end of IOFunctions

// ==================== 辅助函数声明 ====================
void parseInputParameters(int argc, char* argv[], std::string& mesh_folder, 
                         int& timesteps, double& mu, int& n_splits);
void broadcastParameters(std::string& mesh_folder, double& dt, int& timesteps, 
                        double& mu, int& n_splits, int rank);
void verifyParameterConsistency(const std::string& mesh_folder, double dt, 
                               int timesteps, double mu, int n_splits, 
                               int rank, int num_procs);
void printSimulationSetup(const std::vector<Mesh>& sub_meshes, int n_splits);
bool checkConvergence(double norm_res_x, double norm_res_y, double norm_res_p);
// ==================== 非定常求解器辅助函数声明 ====================
// 添加到 fluid.h 文件中

// 参数解析(非定常版本)
void parseInputParameters_unsteady(int argc, char* argv[], std::string& mesh_folder, 
                                   double& dt, int& timesteps, double& mu, int& n_splits);

// 参数广播(非定常版本)
void broadcastParameters_unsteady(std::string& mesh_folder, double& dt, int& timesteps, 
                                  double& mu, int& n_splits, int rank);

// 参数一致性验证(非定常版本)
void verifyParameterConsistency_unsteady(const std::string& mesh_folder, double dt, 
                                         int timesteps, double mu, int n_splits, 
                                         int rank, int num_procs);

// 打印模拟设置(非定常版本)
void printSimulationSetup_unsteady(const std::vector<Mesh>& sub_meshes, int n_splits, 
                                   double dt, int timesteps);

// 收敛性检查
bool checkConvergence(double norm_res_x, double norm_res_y, double norm_res_p);

#endif // DNS_H