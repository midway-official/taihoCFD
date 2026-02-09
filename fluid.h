/**
 * @file fluid.h
 * @brief 不可压缩流体DNS求解器头文件
 * @details 定义了网格、方程系统和求解算法的接口
 * 
 * 主要组件:
 * - Mesh类: 网格数据结构,存储速度、压力场和边界信息
 * - Equation类: 离散方程系统,存储系数矩阵和源项
 * - 求解器函数: SIMPLE/PISO算法的各个步骤
 * - 并行计算支持: 网格分解与合并
 * 
 * @author CFD Team
 * @date 2024
 */

#ifndef FLUID_H
#define FLUID_H

// ============================================================================
// 标准库头文件
// ============================================================================
#include <iostream>
#include <iomanip>  
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>

// ============================================================================
// 第三方库头文件
// ============================================================================
#include <eigen3/Eigen/Sparse>  ///< Eigen稀疏矩阵库

// ============================================================================
// 命名空间
// ============================================================================
using namespace Eigen;
using namespace std;

// ============================================================================
// 全局变量声明
// ============================================================================

/**
 * @defgroup GlobalVars 全局变量
 * @{
 */

extern int n_x0, n_y0;          ///< 初始网格尺寸(x方向, y方向)
extern double dx, dy;           ///< 网格步长(x方向, y方向)
extern double vx;               ///< x方向速度临时变量
extern double velocity;         ///< 速度幅值临时变量

// 残差范数
extern double l2_norm_x;        ///< x方向动量方程L2残差
extern double l2_norm_y;        ///< y方向动量方程L2残差
extern double l2_norm_p;        ///< 压力修正方程L2残差

extern double a, b;             ///< 通用临时变量

/** @} */ // end of GlobalVars

// ============================================================================
// Mesh类 - 网格数据结构
// ============================================================================

/**
 * @class Mesh
 * @brief 流场网格数据结构
 * @details 
 * 存储流场变量(速度、压力)和网格信息(边界类型、区域ID)
 * 
 * 网格布局:
 * - 单元中心存储: u, v, p (速度和压力)
 * - 单元面存储: u_face, v_face (用于对流项计算)
 * - 包含虚拟边界层: 实际尺寸为(ny+2, nx+2)
 * 
 * 边界类型编码(bctype):
 * - 0: 内部计算点
 * - >0: 固壁边界(无滑移条件)
 * - -1: 压力出口(指定压力)
 * - -2: 速度入口(指定速度)
 * - -3: 并行计算交界面(用于区域分解)
 */
class Mesh {
public:
    // ------------------------------------------------------------------------
    // 流场变量 - 单元中心值
    // ------------------------------------------------------------------------
    
    MatrixXd u;                 ///< x方向速度(当前时刻)
    MatrixXd u0;                ///< x方向速度(上一时间步)
    MatrixXd u_star;            ///< x方向修正速度(SIMPLE/PISO中间值)
    
    MatrixXd v;                 ///< y方向速度(当前时刻)
    MatrixXd v0;                ///< y方向速度(上一时间步)
    MatrixXd v_star;            ///< y方向修正速度(SIMPLE/PISO中间值)
    
    MatrixXd p;                 ///< 压力场(当前时刻)
    MatrixXd p_star;            ///< 修正后压力场
    MatrixXd p_prime;           ///< 压力修正量(p' = p_star - p)
    
    // ------------------------------------------------------------------------
    // 流场变量 - 单元面值
    // ------------------------------------------------------------------------
    
    MatrixXd u_face;            ///< 垂直面(i, j±1/2)上的x方向速度, 尺寸:(ny+2, nx+1)
    MatrixXd v_face;            ///< 水平面(i±1/2, j)上的y方向速度, 尺寸:(ny+1, nx+2)
    
    // ------------------------------------------------------------------------
    // 网格拓扑信息
    // ------------------------------------------------------------------------
    
    MatrixXd bctype;            ///< 边界类型标记矩阵
    MatrixXd zoneid;            ///< 区域ID(用于指定不同区域的边界条件)
    MatrixXi interid;           ///< 内部点编号(从1开始,用于构建线性方程组)
    
    int internumber;            ///< 内部点总数
    int nx, ny;                 ///< 网格尺寸(不含边界层)
    
    vector<int> interi;         ///< 内部点i坐标列表
    vector<int> interj;         ///< 内部点j坐标列表
    
    // ------------------------------------------------------------------------
    // 边界条件数据
    // ------------------------------------------------------------------------
    
    vector<double> zoneu;       ///< 各区域x方向速度边界值
    vector<double> zonev;       ///< 各区域y方向速度边界值
    
    // ------------------------------------------------------------------------
    // 构造函数
    // ------------------------------------------------------------------------
    
    /**
     * @brief 默认构造函数
     */
    Mesh() = default;
    
    /**
     * @brief 参数化构造函数
     * @param n_y y方向内部网格数
     * @param n_x x方向内部网格数
     * @note 实际矩阵尺寸为(n_y+2, n_x+2),包含边界层
     */
    Mesh(int n_y, int n_x);
    
    /**
     * @brief 从文件夹加载网格(构造函数)
     * @param folderPath 网格数据文件夹路径
     * @details 读取以下文件:
     *          - params.txt: 网格参数(nx, ny, dx, dy)
     *          - bctype.dat: 边界类型矩阵
     *          - zoneid.dat: 区域ID矩阵
     *          - zoneuv.txt: 区域速度边界值
     * @throws std::runtime_error 如果文件不存在或格式错误
     */
    Mesh(const std::string& folderPath);
    
    // ------------------------------------------------------------------------
    // 初始化方法
    // ------------------------------------------------------------------------
    
    /**
     * @brief 将所有矩阵初始化为零
     */
    void initializeToZero();
    
    /**
     * @brief 为内部点创建编号系统
     * @details 遍历网格,为所有bctype=0的点分配连续编号(从1开始)
     *          用于构建线性方程组的行列索引
     * @post internumber被设置为内部点总数
     * @post interid矩阵存储每个点的编号
     * @post interi, interj向量存储内部点坐标
     */
    void createInterId();
    
    /**
     * @brief 初始化边界条件
     * @details 根据bctype和zoneid设置:
     *          1. 边界点的速度值(u, v, u_star, v_star)
     *          2. 单元面速度(u_face, v_face)
     * @note 必须在setBlock和setZoneUV之后调用
     */
    void initializeBoundaryConditions();
    
    // ------------------------------------------------------------------------
    // 边界条件设置
    // ------------------------------------------------------------------------
    
    /**
     * @brief 设置矩形区域的边界类型和区域ID
     * @param x1, y1 左上角坐标(网格索引,包含边界)
     * @param x2, y2 右下角坐标(网格索引,包含边界)
     * @param bcValue 边界类型值
     *                - 0: 内部点
     *                - >0: 固壁
     *                - -1: 压力出口
     *                - -2: 速度入口
     * @param zoneValue 区域ID值
     * @note 坐标会自动限制在有效范围内
     */
    void setBlock(int x1, int y1, int x2, int y2, double bcValue, double zoneValue);
    
    /**
     * @brief 设置指定区域的速度边界条件
     * @param zoneIndex 区域索引
     * @param u x方向速度
     * @param v y方向速度
     * @note 会自动扩展zoneu和zonev向量
     */
    void setZoneUV(int zoneIndex, double u, double v);
    
    // ------------------------------------------------------------------------
    // 文件I/O
    // ------------------------------------------------------------------------
    
    /**
     * @brief 保存网格配置到文件夹
     * @param folderPath 目标文件夹路径
     * @details 保存以下文件:
     *          - params.txt: 网格参数
     *          - bctype.dat: 边界类型矩阵
     *          - zoneid.dat: 区域ID矩阵
     *          - zoneuv.txt: 区域速度边界值
     * @note 如果文件夹不存在会自动创建
     */
    void saveToFolder(const std::string& folderPath) const;
    
    // ------------------------------------------------------------------------
    // 调试方法
    // ------------------------------------------------------------------------
    
    /**
     * @brief 显示单个矩阵内容(调试用)
     * @param matrix 要显示的矩阵
     * @param name 矩阵名称
     */
    void displayMatrix(const MatrixXd& matrix, const std::string& name) const;
    
    /**
     * @brief 显示所有流场矩阵(调试用)
     */
    void displayAll() const;
};

// ============================================================================
// Equation类 - 离散方程系统
// ============================================================================

/**
 * @class Equation
 * @brief 离散偏微分方程系统
 * @details
 * 存储五对角系数矩阵和源项,对应有限体积法离散形式:
 * A_p*phi_P = A_e*phi_E + A_w*phi_W + A_n*phi_N + A_s*phi_S + source
 * 
 * 系数矩阵结构:
 * - A_p: 中心点系数
 * - A_e, A_w: 东、西邻点系数
 * - A_n, A_s: 北、南邻点系数
 * - A: 稀疏矩阵形式(用于求解器)
 * 
 * 适用方程:
 * - x方向动量方程
 * - y方向动量方程
 * - 压力修正方程(Poisson方程)
 */
class Equation {
public:
    // ------------------------------------------------------------------------
    // 系数矩阵(网格存储)
    // ------------------------------------------------------------------------
    
    MatrixXd A_p;               ///< 中心点系数, 尺寸:(ny+2, nx+2)
    MatrixXd A_e;               ///< 东邻点系数(i, j+1)
    MatrixXd A_w;               ///< 西邻点系数(i, j-1)
    MatrixXd A_n;               ///< 北邻点系数(i-1, j)
    MatrixXd A_s;               ///< 南邻点系数(i+1, j)
    
    // ------------------------------------------------------------------------
    // 线性方程组(压缩存储)
    // ------------------------------------------------------------------------
    
    VectorXd source;            ///< 源项向量, 长度:internumber
    SparseMatrix<double> A;     ///< 稀疏系数矩阵, 尺寸:(internumber, internumber)
    
    // ------------------------------------------------------------------------
    // 网格信息
    // ------------------------------------------------------------------------
    
    int n_x, n_y;               ///< 网格尺寸(不含边界层)
    Mesh& mesh;                 ///< 关联的网格对象引用
    
    // ------------------------------------------------------------------------
    // 构造函数
    // ------------------------------------------------------------------------
    
    /**
     * @brief 构造函数
     * @param mesh_ 网格对象引用
     * @note 矩阵尺寸根据网格尺寸自动确定
     */
    Equation(Mesh& mesh_);
    
    // ------------------------------------------------------------------------
    // 初始化方法
    // ------------------------------------------------------------------------
    
    /**
     * @brief 将所有系数矩阵和源项初始化为零
     */
    void initializeToZero();
    
    /**
     * @brief 构建稀疏系数矩阵A
     * @details 将五对角系数矩阵(A_p, A_e, A_w, A_n, A_s)组装成Eigen稀疏矩阵
     *          方程形式: A*phi = source
     *          其中 A_ij = A_p(i,j) 对角线元素
     *               A_ij = -A_nb(i,j) 非对角线元素
     * @note 只包含内部点(bctype=0)
     * @post A矩阵被填充,可用于求解器
     */
    void build_matrix();
};

// ============================================================================
// 全局函数声明
// ============================================================================

/**
 * @defgroup SolverFunctions 求解器函数
 * @{
 */

// ----------------------------------------------------------------------------
// 工具函数
// ----------------------------------------------------------------------------

/**
 * @brief 打印矩阵到控制台(调试用)
 * @param matrix 要打印的矩阵
 * @param name 矩阵名称
 * @param precision 输出精度(小数位数),默认4位
 */
void printMatrix(const MatrixXd& matrix, const string& name, int precision = 4);

/**
 * @brief 显示进度条和预估剩余时间
 * @param current_step 当前步数
 * @param total_steps 总步数
 * @param elapsed_time 已用时间(秒)
 * @note 输出格式: [=====>    ] 50% 已用时间: 10.5秒 预计剩余时间: 10.5秒
 */
void show_progress_bar(int current_step, int total_steps, double elapsed_time);

// ----------------------------------------------------------------------------
// 线性方程组求解
// ----------------------------------------------------------------------------

/**
 * @brief 求解线性方程组 A*phi = source
 * @param equation 方程对象(包含系数矩阵和源项)
 * @param epsilon 求解器收敛容差
 * @param l2_norm 输出残差L2范数
 * @param phi 解向量(输入初值,输出解)
 * @details
 * 使用共轭梯度法(CG)求解稀疏线性方程组
 * 在SIMPLE循环中通常设置最大迭代次数=1(每次只迭代一步)
 * @note 求解前必须调用equation.build_matrix()
 */
void solve(Equation& equation, double epsilon, double& l2_norm, MatrixXd& phi);

// ----------------------------------------------------------------------------
// SIMPLE/PISO算法步骤
// ----------------------------------------------------------------------------

/**
 * @brief 计算单元面速度(Rhie-Chow插值)
 * @param mesh 网格对象
 * @param equ_u x方向动量方程(提供A_p系数)
 * @details
 * 使用压力梯度修正的动量插值,避免压力场棋盘式震荡
 * 
 * 插值公式(以u_face为例):
 * u_face(i,j) = 0.5*(u(i,j) + u(i,j+1))
 *             + [p(i,j+1) - p(i,j-1)]*dy/(4*A_p(i,j))
 *             + [p(i,j+2) - p(i,j)]*dy/(4*A_p(i,j+1))
 *             - 0.5*(1/A_p(i,j) + 1/A_p(i,j+1))*(p(i,j+1) - p(i,j))*dy
 * 
 * @note 必须在momentum_function之后调用
 * @post u_face和v_face被更新
 */
void face_velocity(Mesh &mesh, Equation &equ_u);

/**
 * @brief 构建压力修正方程(Poisson方程)
 * @param mesh 网格对象
 * @param equ_p 压力修正方程
 * @param equ_u x方向动量方程(提供A_p系数)
 * @details
 * 离散形式: sum(Ap_nb * p'_nb) = -div(u)
 * 
 * 系数: Ap_e = 0.5*(1/A_p(P) + 1/A_p(E))*dy^2
 *       Ap_w = 0.5*(1/A_p(P) + 1/A_p(W))*dy^2
 *       Ap_n = 0.5*(1/A_p(P) + 1/A_p(N))*dx^2
 *       Ap_s = 0.5*(1/A_p(P) + 1/A_p(S))*dx^2
 * 
 * 源项: source = -(u_e - u_w)*dy - (v_n - v_s)*dx
 * 
 * @note 必须在face_velocity之后调用
 * @post equ_p的系数矩阵和源项被填充
 */
void pressure_function(Mesh &mesh, Equation &equ_p, Equation &equ_u);

/**
 * @brief 修正压力场
 * @param mesh 网格对象
 * @param equ_u 动量方程(未使用,保留用于接口一致性)
 * @param alpha_p 压力松弛因子(通常0.3-0.7)
 * @details
 * 更新公式: p_star = p + alpha_p * p_prime
 * 
 * 边界点的压力修正量设为0
 * @note 必须在求解压力修正方程后调用
 * @post p_star被更新
 */
void correct_pressure(Mesh &mesh, Equation &equ_u, double alpha_p);

/**
 * @brief 修正速度场
 * @param mesh 网格对象
 * @param equ_u x方向动量方程(提供A_p系数)
 * @details
 * 修正单元中心速度:
 * u_star(i,j) = u(i,j) + 0.5*(p'(i,j-1) - p'(i,j+1))*dy/A_p(i,j)
 * v_star(i,j) = v(i,j) + 0.5*(p'(i+1,j) - p'(i-1,j))*dx/A_p(i,j)
 * 
 * 修正单元面速度:
 * u_face += 0.5*(1/A_p(P) + 1/A_p(E))*(p'(P) - p'(E))*dy
 * v_face += 0.5*(1/A_p(P) + 1/A_p(N))*(p'(N) - p'(P))*dx
 * 
 * @note 必须在correct_pressure之后调用
 * @post u_star, v_star, u_face, v_face被更新
 */
void correct_velocity(Mesh &mesh, Equation &equ_u);

// ----------------------------------------------------------------------------
// 动量方程离散化
// ----------------------------------------------------------------------------

/**
 * @brief 构建动量方程系数矩阵(稳态SIMPLE算法)
 * @param mesh 网格对象
 * @param equ_u x方向动量方程
 * @param equ_v y方向动量方程
 * @param mu 动力粘度
 * @param alpha_uv 速度松弛因子(通常0.5-0.8)
 * @details
 * 离散形式(以x方向为例):
 * A_p*u_P = A_e*u_E + A_w*u_W + A_n*u_N + A_s*u_S + S
 * 
 * 对流-扩散系数(混合格式):
 * A_e = D_e + max(0, -F_e)
 * A_w = D_w + max(0, F_w)
 * A_n = D_n + max(0, -F_n)
 * A_s = D_s + max(0, F_s)
 * A_p = sum(A_nb) + max(0, F_e) + max(0, -F_w) + max(0, F_n) + max(0, -F_s)
 * 
 * 其中:
 * D = mu*A/dx (扩散系数)
 * F = rho*u*A (对流通量)
 * 
 * 源项:
 * S = (p_W - p_E)*dy/2 + (1-alpha)*A_p*u_old
 * 
 * @note 系数应用松弛: A_nb *= alpha_uv
 * @post equ_u和equ_v的系数矩阵和源项被填充
 */
void momentum_function(Mesh &mesh, Equation &equ_u, Equation &equ_v, 
                      double mu, double alpha_uv);

/**
 * @brief 构建动量方程系数矩阵(非稳态SIMPLE算法)
 * @param mesh 网格对象
 * @param equ_u x方向动量方程
 * @param equ_v y方向动量方程
 * @param mu 动力粘度
 * @param dt 时间步长
 * @param alpha_uv 速度松弛因子
 * @details
 * 相比稳态版本,增加时间项:
 * A_p = A_p_稳态 + dx*dy/dt
 * S = S_稳态 + alpha*dx*dy*u0/dt
 * 
 * 时间离散: 隐式欧拉格式(一阶精度)
 * @note 需要mesh.u0和mesh.v0存储上一时间步的值
 * @post equ_u和equ_v的系数矩阵和源项被填充
 */
void momentum_function_unsteady(Mesh &mesh, Equation &equ_u, Equation &equ_v,
                               double mu, double dt, double alpha_uv);

/**
 * @brief 构建动量方程系数矩阵(PISO算法)
 * @param mesh 网格对象
 * @param equ_u x方向动量方程
 * @param equ_v y方向动量方程
 * @param mu 动力粘度
 * @param dt 时间步长
 * @details
 * PISO算法特点:
 * - 不使用速度松弛(相当于alpha_uv=1.0)
 * - 进行多次压力修正(通常2-3次)
 * - 时间精度更高,适合非稳态计算
 * 
 * 与非稳态SIMPLE的区别:
 * - 移除松弛项: (1-alpha)*A_p*u_old
 * - 直接使用旧时间步值: dx*dy*u0/dt
 * 
 * @note PISO算法收敛更快但单步计算量更大
 * @post equ_u和equ_v的系数矩阵和源项被填充
 */
void momentum_function_PISO(Mesh &mesh, Equation &equ_u, Equation &equ_v,
                            double mu, double dt);

// ----------------------------------------------------------------------------
// 后处理
// ----------------------------------------------------------------------------

/**
 * @brief 保存计算结果到文件
 * @param mesh 网格对象
 * @details
 * 输出文件:
 * - u.dat: x方向速度场(u_star)
 * - v.dat: y方向速度场(v_star)
 * - p.dat: 压力场(p_star)
 * 
 * 文件格式: 矩阵形式,可直接用MATLAB/Python读取
 */
void post_processing(Mesh &mesh);

/** @} */ // end of SolverFunctions

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

/**
 * @brief 合并子网格(去除接口交换层)
 * @param sub_meshes 子网格向量
 * @return 合并后的完整网格
 * @details
 * 合并策略:
 * 1. 识别并跳过交换层(bctype=-3的列)
 * 2. 按顺序拼接各子网格的内部和边界数据
 * 3. 重建完整的流场
 * 
 * @note 用于收集并行计算结果
 * @post 返回的网格与原始网格尺寸一致
 */
Mesh mergeMeshesWithoutInterface(const std::vector<Mesh>& sub_meshes);

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
void printSimulationSetup(const std::vector<Mesh>& sub_meshes, int n_splits, int rank);
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
                                   double dt, int timesteps, int rank);

// 收敛性检查
bool checkConvergence(double norm_res_x, double norm_res_y, double norm_res_p);

#endif // FLUID_H