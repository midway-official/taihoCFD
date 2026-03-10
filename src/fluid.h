/**
 * @file    fluid.h
 * @brief   不可压缩黏性流求解器 —— 核心数据结构与函数接口声明
 *
 * @details
 * 本文件定义了基于有限体积法（FVM）的二维不可压缩 Navier-Stokes 方程求解器
 * 所使用的核心类与函数接口，支持结构化四边形网格、MPI 域分解并行计算，
 * 以及稳态（SIMPLE）和非稳态（PISO）两种求解模式。
 *
 * 边界类型编码（bctype）约定：
 * |  值  | 含义                        |
 * |------|-----------------------------|
 * |  0   | 内部计算点                  |
 * |  >0  | 无滑移壁面（wall）          |
 * | -1   | 压力出口（给定压强 = 0）    |
 * | -2   | 速度入口（给定速度）        |
 * | -3   | MPI 并行接口（ghost 层）    |
 *
 * 坐标/索引约定：
 * - i 为行索引（y 方向，从上到下增大）
 * - j 为列索引（x 方向，从左到右增大）
 * - 网格节点数组大小为 (ny+1) × (nx+1)，单元中心数组大小为 ny × nx
 *
 * @author  midway
 * @version 2.0
 */

#ifndef FLUID_H
#define FLUID_H

#include <iostream>
#include <iomanip>
#include <eigen3/Eigen/Sparse>
#include <algorithm>
#include <fstream>
#include <vector>

using namespace Eigen;
using namespace std;


// ============================================================================
// 工具函数
// ============================================================================

/**
 * @brief 格式化打印 Eigen 矩阵到标准输出
 *
 * @param matrix    待打印的矩阵
 * @param name      矩阵名称（用于标题行）
 * @param precision 小数位数，默认为 4
 */
void printMatrix(const MatrixXd& matrix, const string& name, int precision = 4);


// ============================================================================
// Mesh 类 —— 网格数据容器
// ============================================================================

/**
 * @class Mesh
 * @brief 存储结构化四边形网格的几何信息、场变量和边界条件
 *
 * @details
 * 采用交错网格（Staggered Grid）思想，但将所有变量存储于单元中心。
 * 面速度（u_face / v_face）单独存储，用于 Rhie-Chow 动量插值，
 * 以避免压力-速度解耦（棋盘格不稳定性）。
 *
 * 内存布局（行优先，与 Eigen 默认 ColMajor 转置对应）：
 * - 所有 ny×nx 矩阵按 (i=行, j=列) 存储
 * - u_face 大小为 ny×(nx-1)，位于相邻单元东面
 * - v_face 大小为 (ny-1)×nx，位于相邻单元南面
 */
class Mesh {
public:
    // ── 速度场 ────────────────────────────────────────────────────────────
    MatrixXd u;       ///< x 方向速度（当前迭代步，ny×nx）
    MatrixXd u0;      ///< x 方向速度（上一时间步，用于非定常计算，ny×nx）
    MatrixXd u_star;  ///< x 方向速度（SIMPLE 修正后，ny×nx）

    MatrixXd v;       ///< y 方向速度（当前迭代步，ny×nx）
    MatrixXd v0;      ///< y 方向速度（上一时间步，ny×nx）
    MatrixXd v_star;  ///< y 方向速度（SIMPLE 修正后，ny×nx）

    // ── 网格坐标 ──────────────────────────────────────────────────────────
    MatrixXd x;    ///< 节点 x 坐标，(ny+1)×(nx+1)
    MatrixXd y;    ///< 节点 y 坐标，(ny+1)×(nx+1)
    MatrixXd x_c;  ///< 单元中心 x 坐标，ny×nx
    MatrixXd y_c;  ///< 单元中心 y 坐标，ny×nx

    // ── 几何量 ────────────────────────────────────────────────────────────
    MatrixXd area_e;  ///< 东面面积（边长），ny×nx
    MatrixXd area_w;  ///< 西面面积（边长），ny×nx
    MatrixXd area_s;  ///< 南面面积（边长），ny×nx
    MatrixXd area_n;  ///< 北面面积（边长），ny×nx
    MatrixXd vol;     ///< 单元体积（2D 中为面积），ny×nx

    // ── 压力场 ────────────────────────────────────────────────────────────
    MatrixXd p;        ///< 当前压力场，ny×nx
    MatrixXd p_star;   ///< 修正后压力场（p + α_p·p'），ny×nx
    MatrixXd p_prime;  ///< 压力修正量 p'，ny×nx

    // ── 面速度（用于 Rhie-Chow 插值） ────────────────────────────────────
    MatrixXd u_face;  ///< 单元东面 x 速度，ny×(nx-1)
    MatrixXd v_face;  ///< 单元南面 y 速度，(ny-1)×nx

    // ── 网格拓扑与边界 ────────────────────────────────────────────────────
    MatrixXi bctype;  ///< 边界类型标记，ny×nx（编码见文件头注释）
    MatrixXi zoneid;  ///< 区域编号，ny×nx（对应 zoneu/zonev 的索引）
    MatrixXi interid; ///< 内部点全局编号（用于线性方程组组装），ny×nx

    int internumber;  ///< 内部点总数（线性方程组规模）
    int nx;           ///< x 方向单元数
    int ny;           ///< y 方向单元数

    vector<int> interi;  ///< 内部点行索引列表（与 interid 配套）
    vector<int> interj;  ///< 内部点列索引列表（与 interid 配套）

    vector<double> zoneu;  ///< 各区域指定的 x 方向速度（壁面/入口条件）
    vector<double> zonev;  ///< 各区域指定的 y 方向速度（壁面/入口条件）

    // ── 构造函数 ──────────────────────────────────────────────────────────

    /** @brief 默认构造函数（允许延迟初始化） */
    Mesh() = default;

    /**
     * @brief 按尺寸构造空网格，所有矩阵分配内存但不初始化
     * @param n_y  y 方向单元数
     * @param n_x  x 方向单元数
     */
    Mesh(int n_y, int n_x);

    /**
     * @brief 从磁盘文件夹读取网格并完成初始化
     *
     * @param folderPath 网格文件夹路径，需包含以下文件：
     *   - params.txt   : 第一行为 "nx ny"
     *   - bctype.dat   : ny×nx 整数矩阵
     *   - zoneid.dat   : ny×nx 整数矩阵
     *   - x.dat        : (ny+1)×(nx+1) 浮点矩阵（节点 x 坐标）
     *   - y.dat        : (ny+1)×(nx+1) 浮点矩阵（节点 y 坐标）
     *   - zoneuv.txt   : 每行 "u v"，一行对应一个区域
     *
     * @throws std::runtime_error 若文件夹或任一文件不存在
     */
    Mesh(const std::string& folderPath);

    // ── 初始化方法 ────────────────────────────────────────────────────────

    /** @brief 将所有场变量矩阵清零 */
    void initializeToZero();

    /**
     * @brief 根据 bctype / zoneid / zoneu / zonev 初始化边界点速度，
     *        并为面速度 u_face / v_face 赋予与相邻边界一致的初值
     */
    void initializeBoundaryConditions();

    /**
     * @brief 计算单元中心坐标、各面面积及单元体积
     *
     * @details 依赖节点坐标矩阵 x / y 已经正确填充。
     *          体积采用对角线叉积公式：vol = 0.5|d1×d2|
     */
    void initGeometry();

    /**
     * @brief 遍历 bctype，为所有内部点（bctype==0）分配连续编号，
     *        并填充 interid、interi、interj、internumber
     */
    void createInterId();

    // ── 辅助设置方法 ──────────────────────────────────────────────────────

    /**
     * @brief 批量设置矩形区块的 bctype 和 zoneid
     *
     * @param x1,y1  区块左上角单元列/行索引（包含）
     * @param x2,y2  区块右下角单元列/行索引（包含）
     * @param bcValue   写入 bctype 的值
     * @param zoneValue 写入 zoneid 的值
     */
    void setBlock(int x1, int y1, int x2, int y2, double bcValue, double zoneValue);

    /**
     * @brief 设置指定区域编号对应的速度边界条件
     *
     * @param zoneIndex 区域编号（若超出 zoneu/zonev 长度则自动扩展）
     * @param u  该区域的 x 方向速度
     * @param v  该区域的 y 方向速度
     */
    void setZoneUV(size_t zoneIndex, double u_val, double v_val);

    // ── 调试输出 ──────────────────────────────────────────────────────────

    /**
     * @brief 打印指定矩阵到标准输出
     * @param matrix 待打印矩阵
     * @param name   矩阵名称
     */
    void displayMatrix(const MatrixXd& matrix, const std::string& name) const;

    /** @brief 打印所有场变量矩阵（调试用） */
    void displayAll() const;
};


// ============================================================================
// Equation 类 —— 离散线性方程组容器
// ============================================================================

/**
 * @class Equation
 * @brief 存储有限体积离散后的系数矩阵和源项向量
 *
 * @details
 * 采用五点格式（East/West/North/South/Center）存储标量离散系数，
 * 最终通过 build_matrix() 组装为 Eigen 稀疏矩阵用于求解。
 *
 * 同一套系数可同时用于 u 和 v 方程（在 momentum_function 中复制），
 * 从而避免重复组装。
 */
class Equation {
public:
    // ── 离散系数（五点格式，ny×nx） ──────────────────────────────────────
    MatrixXd A_p;  ///< 中心系数（对角项）
    MatrixXd A_e;  ///< 东向系数（离轴项）
    MatrixXd A_w;  ///< 西向系数（离轴项）
    MatrixXd A_n;  ///< 北向系数（离轴项）
    MatrixXd A_s;  ///< 南向系数（离轴项）

    VectorXd source;          ///< 右端源项向量（长度 = internumber）
    SparseMatrix<double> A;   ///< 组装后的稀疏系数矩阵（internumber × internumber）

    int n_x;  ///< 所关联网格的 nx
    int n_y;  ///< 所关联网格的 ny
    Mesh& mesh;  ///< 关联的网格对象引用

    /**
     * @brief 构造函数，按网格尺寸分配所有矩阵和向量
     * @param mesh_ 关联的 Mesh 对象（引用，生命周期须不短于本对象）
     */
    Equation(Mesh& mesh_);

    /** @brief 将所有系数矩阵和源项向量清零 */
    void initializeToZero();

    /**
     * @brief 遍历内部点，将五点系数组装为 Eigen 稀疏矩阵 A
     *
     * @details
     * 仅遍历 bctype==0 的内部点，邻居若为边界点则不添加对应列项
     * （边界贡献已在 momentum_function / pressure_function 中并入源项）。
     *
     */
    void build_matrix();
};


// ============================================================================
// 求解器函数
// ============================================================================

/**
 * @brief 单进程 CG 求解器（串行版本，仅供调试）
 *
 * @param equation  已组装好 A 和 source 的方程对象
 * @param epsilon   收敛容差
 * @param l2_norm   输出：求解完成后的残差 L2 范数
 * @param phi       输入/输出：初始猜测值及解向量（内部点更新，边界点不变）
 */
void solve(Equation& equation, double epsilon, double& l2_norm, MatrixXd& phi);


// ============================================================================
// 物理离散函数
// ============================================================================

/**
 * @brief 计算单元面速度（Rhie-Chow 动量插值）
 *
 * @details
 * 利用相邻单元的中心速度和压力梯度，通过动量插值计算面速度，
 * 消除同位网格上压力-速度解耦导致的棋盘格振荡。
 * 同时处理各类边界面（压力出口/速度入口/壁面/并行接口）。
 *
 * @param mesh   网格对象（读取 u, v, p, A_p）
 * @param equ_u  动量方程对象（提供 A_p 系数）
 */
void face_velocity(Mesh& mesh, Equation& equ_u);

/**
 * @brief 离散稳态动量方程（对流-扩散方程，一阶迎风格式）
 *
 * @details
 * 采用一阶迎风对流格式和中心差分扩散格式，同时处理全部边界类型。
 * 完成后自动将 u、v 方程的系数矩阵同步（equ_v 共享 equ_u 的系数），
 * 并调用 build_matrix() 组装稀疏矩阵。
 *
 * @param mesh      网格对象
 * @param equ_u     x 方向动量方程（输出 A_p/A_e/.../source_x）
 * @param equ_v     y 方向动量方程（输出 source_y，系数与 equ_u 共享）
 * @param mu        动力粘度 μ
 * @param alpha_uv  速度亚松弛因子（0 < α ≤ 1）
 */
void momentum_function(Mesh& mesh, Equation& equ_u, Equation& equ_v,
                       double mu, double alpha_uv);

/**
 * @brief 离散非定常动量方程（一阶隐式 Euler 时间离散）
 *
 * @details
 * 在稳态版本基础上，对中心系数增加 vol/dt 时间项，
 * 并将 u0/v0（上一时间步速度）加入源项。
 *
 * @param mesh   网格对象（u0/v0 须已更新为上一时间步速度）
 * @param equ_u  x 方向动量方程
 * @param equ_v  y 方向动量方程
 * @param mu     动力粘度 μ
 * @param dt     时间步长 Δt
 */
void momentum_function_unsteady(Mesh& mesh, Equation& equ_u, Equation& equ_v,
                                 double mu, double dt);

/**
 * @brief 离散压力修正方程（连续性方程推导）
 *
 * @details
 * 基于当前面速度 u_face/v_face 构建质量通量散度作为源项，
 * 系数由动量方程系数 A_p 和几何量共同决定（满足 Rhie-Chow 一致性）。
 * 完成后调用 build_matrix() 组装稀疏矩阵。
 *
 * @param mesh   网格对象（面速度须已由 face_velocity 更新）
 * @param equ_p  压力修正方程（输出）
 * @param equ_u  动量方程（提供 A_p 系数）
 */
void pressure_function(Mesh& mesh, Equation& equ_p, Equation& equ_u);

/**
 * @brief 用压力修正量 p' 更新压力场
 *
 * @details  p_star = p + alpha_p * p_prime
 *
 * @param mesh     网格对象（p_prime 须已求解，输出 p_star）
 * @param alpha_p  压力亚松弛因子（典型值 0.3）
 */
void correct_pressure(Mesh& mesh, double alpha_p);

/**
 * @brief 用压力修正量 p' 修正单元中心速度和面速度
 *
 * @details
 * 分别修正：
 * - u_star, v_star（单元中心速度）
 * - u_face, v_face（面速度）
 *
 * 边界面速度不参与压力修正（保持由边界条件给定的值）。
 *
 * @param mesh   网格对象
 * @param equ_u  动量方程（提供 A_p 系数）
 */
void correct_velocity(Mesh& mesh, Equation& equ_u);

/**
 * @brief 将最终计算结果写入文本文件（串行调试用）
 *
 * @details 输出 u.dat, v.dat, p.dat 到当前工作目录
 * @param mesh 网格对象（读取 u_star, v_star, p_star）
 */
void post_processing(Mesh& mesh);

/**
 * @brief 在终端打印 ASCII 进度条
 *
 * @param current_step   当前步数
 * @param total_steps    总步数
 * @param elapsed_time   已用时间（秒）
 */
void show_progress_bar(int current_step, int total_steps, double elapsed_time);


// ============================================================================
// 并行计算相关函数
// ============================================================================

/**
 * @brief 将完整网格沿 x 方向分割为 n 个子网格（MPI 域分解）
 *
 * @details
 * 分割策略：
 * 1. 尽量均匀分配：widths[k] ≈ nx / n
 * 2. 在接口处各添加 2 列 ghost 层（bctype=-3），用于通信边界插值
 * 3. 子网格完整继承原始网格的 zoneu/zonev 边界速度配置
 * 4. 各子网格独立调用 initGeometry() 和 createInterId()
 *
 * 子网格列数：real_w + left_ghost(0或2) + right_ghost(0或2)
 *
 * @param original  原始完整网格（只读）
 * @param n         分割数（通常等于 MPI 进程数）
 * @return          长度为 n 的子网格向量，sub_meshes[rank] 分配给对应进程
 */
vector<Mesh> splitMeshVertically(const Mesh& original, int n);


// ============================================================================
// 文件 I/O 函数
// ============================================================================

/**
 * @brief 保存网格计算结果到文件（支持分进程输出）
 *
 * @details
 * 输出文件（位于 timestep_folder/ 目录下，若为空则输出到当前目录）：
 * - u_<rank>.dat  : u_star 场
 * - v_<rank>.dat  : v_star 场
 * - p_<rank>.dat  : 压力 p 场
 * - xc_<rank>.dat : 单元中心 x 坐标
 * - yc_<rank>.dat : 单元中心 y 坐标
 *
 * @param mesh             网格对象（只读）
 * @param rank             当前 MPI 进程编号（用于文件命名）
 * @param timestep_folder  输出目录（默认为空，表示当前目录）
 *
 * @throws std::runtime_error 若文件无法创建
 */
void saveMeshData(const Mesh& mesh, int rank,
                  const std::string& timestep_folder = "");

/**
 * @brief 保存非定常计算中间步数据（用于过程分析）
 *
 * @details
 * 输出目录：mu_<mu>/<timestep>/
 * 输出文件：u_<rank>.dat, v_<rank>.dat, p_<rank>.dat, pp_<rank>.dat
 *
 * @param mesh      网格对象
 * @param rank      MPI 进程编号
 * @param timestep  当前时间步编号（用于子目录命名）
 * @param mu        动力粘度（用于父目录命名）
 *
 * @throws std::runtime_error 若文件无法创建
 */
void saveforecastData(const Mesh& mesh, int rank, int timestep, double mu);


// ============================================================================
// 稳态求解器辅助函数
// ============================================================================

/**
 * @brief 解析命令行参数（稳态版本）
 *
 * @details
 * 命令行格式：`./solver <mesh_folder> <timesteps> <mu>`
 * 若参数不足则切换为交互式输入。
 *
 * @param argc         main 函数 argc
 * @param argv         main 函数 argv
 * @param mesh_folder  输出：网格文件夹路径
 * @param timesteps    输出：最大 SIMPLE 迭代次数
 * @param mu           输出：动力粘度
 * @param n_splits     输出：MPI 进程数（由 MPI_Comm_size 填充，此处仅打印）
 */
void parseInputParameters(int argc, char* argv[], std::string& mesh_folder,
                          int& timesteps, double& mu, int& n_splits);

/**
 * @brief 将参数从 rank 0 广播到所有 MPI 进程（稳态版本）
 *
 * @param mesh_folder  输入/输出：网格路径字符串
 * @param dt           输入/输出：时间步长（稳态计算中为占位参数）
 * @param timesteps    输入/输出：迭代次数
 * @param mu           输入/输出：动力粘度
 * @param n_splits     输入/输出：进程数
 * @param rank         当前进程编号
 */
void broadcastParameters(std::string& mesh_folder, double& dt, int& timesteps,
                         double& mu, int& n_splits, int rank);

/**
 * @brief 验证所有进程的参数一致性（稳态版本）
 *
 * @details
 * 通过 MPI_Allreduce 比较各进程的参数最大值和最小值，
 * 若不一致则在 rank 0 打印错误信息并调用 MPI_Abort。
 *
 * @param mesh_folder  待验证的网格路径
 * @param dt           时间步长
 * @param timesteps    迭代次数
 * @param mu           动力粘度
 * @param n_splits     进程数
 * @param rank         当前进程编号
 * @param num_procs    总进程数
 */
void verifyParameterConsistency(const std::string& mesh_folder, double dt,
                                int timesteps, double mu, int n_splits,
                                int rank, int num_procs);

/**
 * @brief 打印子网格分割信息摘要（稳态版本）
 *
 * @param sub_meshes  分割后的子网格向量
 * @param n_splits    分割数
 */
void printSimulationSetup(const std::vector<Mesh>& sub_meshes, int n_splits);

/**
 * @brief 检查 SIMPLE 迭代是否达到收敛条件
 *
 * @details
 * 收敛判据：
 * - 速度残差 < 1e-25
 * - 压力残差 < 1e-21
 *
 * @param norm_res_x  x 方向速度残差
 * @param norm_res_y  y 方向速度残差
 * @param norm_res_p  压力修正残差
 * @return true 表示已收敛，false 表示未收敛
 */
bool checkConvergence(double norm_res_x, double norm_res_y, double norm_res_p);


// ============================================================================
// 非定常求解器辅助函数
// ============================================================================

/**
 * @brief 解析命令行参数（非定常版本）
 *
 * @details
 * 命令行格式：`./solver_unsteady <mesh_folder> <dt> <timesteps> <mu>`
 * 若参数不足则切换为交互式输入。
 *
 * @param argc         main 函数 argc
 * @param argv         main 函数 argv
 * @param mesh_folder  输出：网格文件夹路径
 * @param dt           输出：时间步长 Δt
 * @param timesteps    输出：总时间步数
 * @param mu           输出：动力粘度
 * @param n_splits     输出：MPI 进程数（打印用）
 */
void parseInputParameters_unsteady(int argc, char* argv[], std::string& mesh_folder,
                                   double& dt, int& timesteps, double& mu, int& n_splits);

/**
 * @brief 将参数从 rank 0 广播到所有 MPI 进程（非定常版本）
 *
 * @param mesh_folder  输入/输出：网格路径
 * @param dt           输入/输出：时间步长
 * @param timesteps    输入/输出：总时间步数
 * @param mu           输入/输出：动力粘度
 * @param n_splits     输入/输出：进程数
 * @param rank         当前进程编号
 */
void broadcastParameters_unsteady(std::string& mesh_folder, double& dt, int& timesteps,
                                  double& mu, int& n_splits, int rank);

/**
 * @brief 验证所有进程的参数一致性（非定常版本）
 *
 * @param mesh_folder  待验证的网格路径
 * @param dt           时间步长
 * @param timesteps    总时间步数
 * @param mu           动力粘度
 * @param n_splits     进程数
 * @param rank         当前进程编号
 * @param num_procs    总进程数
 */
void verifyParameterConsistency_unsteady(const std::string& mesh_folder, double dt,
                                         int timesteps, double mu, int n_splits,
                                         int rank, int num_procs);

/**
 * @brief 打印子网格分割信息及时间离散摘要（非定常版本）
 *
 * @param sub_meshes  分割后的子网格向量
 * @param n_splits    分割数
 * @param dt          时间步长
 * @param timesteps   总时间步数
 */
void printSimulationSetup_unsteady(const std::vector<Mesh>& sub_meshes, int n_splits,
                                   double dt, int timesteps);


#endif // FLUID_H