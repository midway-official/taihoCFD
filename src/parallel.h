/**
 * @file    parallel.h
 * @brief   MPI 并行求解器接口 —— 通信、向量转换与并行线性求解器声明
 *
 * @details
 * 本文件提供基于 MPI 的并行计算支持，包含三个功能层：
 *
 * 1. **数据通信层**
 *    - ghost 列交换（exchangeColumns）：在 MPI 域分解接口处同步 2 列数据
 *    - 向量/矩阵互转（vectorToMatrix / matrixToVector）：连接线性代数层与场变量层
 *
 * 2. **并行修正层**
 *    - Parallel_correction / Parallel_correction2：处理跨进程接口处的矩阵-向量乘法修正，
 *      补偿因 ghost 列存在而产生的 Ax 计算误差
 *
 * 3. **并行线性求解器层**
 *    - CG_parallel  ：无预条件共轭梯度法（MPI 并行）
 *    - PCG_parallel ：Jacobi 预条件共轭梯度法（MPI 并行，推荐使用）
 *    - solveFieldCG / solveFieldPCG：场变量级封装，直接输出到 MatrixXd
 *
 * 并行通信约定：
 * - 每个子网格左右两侧各有 2 列 ghost 层（bctype = -3）
 * - ghost 列 [0,1] 存储来自左邻进程的数据，[nx-2,nx-1] 存储来自右邻进程的数据
 * - 真实计算列范围为 [2, nx-3]（中间子网格）
 *
 * 停滞检测约定：
 * - 连续 3 步相对残差下降率 < 1e-6 时判定为停滞，提前退出
 * - 最小保护迭代次数为 5 步，避免初始阶段误判
 *
 * @note 所有求解器函数均通过 MPI_Allreduce 同步全局残差，
 *       收敛判断由 rank 0 执行后广播，确保所有进程同步退出。
 *
 * @author  midway
 * @version 2.0
 */

#ifndef PARALLEL_H
#define PARALLEL_H

#include "fluid.h"
#include <mpi.h>
#include <omp.h>


// ============================================================================
// 性能统计全局变量
// ============================================================================

extern double total_comm_time;   ///< 累计 MPI 通信耗时（秒），由 exchangeColumns 更新
extern int    total_comm_count;  ///< 累计 MPI 通信调用次数
extern int    totalcount;        ///< 通用计数器（迭代总次数统计）
extern double start_time;        ///< 计时起点（秒，由 MPI_Wtime 赋值）
extern double end_time;          ///< 计时终点（秒，由 MPI_Wtime 赋值）


// ============================================================================
// 数据通信函数
// ============================================================================

/**
 * @brief 在相邻 MPI 进程间交换 ghost 列数据（列方向域分解专用）
 *
 * @details
 * 通信模式（双向 Sendrecv）：
 * - 向左邻进程发送本进程第 [2,3] 列，接收并写入第 [0,1] 列（左 ghost）
 * - 向右邻进程发送本进程第 [nx-4,nx-3] 列，接收并写入第 [nx-2,nx-1] 列（右 ghost）
 *
 * 最左/最右进程使用 MPI_PROC_NULL 跳过不存在方向的通信。
 *
 * @param matrix     待交换的场变量矩阵（行优先，ny×nx），原地更新 ghost 列
 * @param rank       当前进程编号
 * @param num_procs  总进程数
 *
 * @note 矩阵须为 Eigen ColMajor 格式（默认），列内数据连续，可直接映射为发送缓冲区
 * @note 使用固定 tag（0 和 1），多字段并发通信时需注意 tag 冲突
 */
void exchangeColumns(MatrixXd& matrix, int rank, int num_procs);

/**
 * @brief 将线性方程组解向量写回场变量矩阵（仅更新内部点）
 *
 * @details
 * 遍历网格中所有 bctype==0 的内部点，按 interid 从解向量 x 中取值写入 phi。
 * 边界点和 ghost 列保持原值不变。
 *
 * @param x    输入：长度为 internumber 的解向量
 * @param phi  输出：ny×nx 场变量矩阵（仅内部点被修改）
 * @param mesh 网格对象（提供 bctype 和 interid）
 */
void vectorToMatrix(const VectorXd& x, MatrixXd& phi, const Mesh& mesh);

/**
 * @brief 将场变量矩阵中的内部点值打包为线性方程组初始解向量
 *
 * @details
 * 遍历网格中所有 bctype==0 的内部点，按 interid 将 phi 值写入向量 x。
 * 常用于向求解器提供初始猜测值。
 *
 * @param phi  输入：ny×nx 场变量矩阵
 * @param x    输出：长度为 internumber 的解向量
 * @param mesh 网格对象（提供 bctype 和 interid）
 */
void matrixToVector(const MatrixXd& phi, VectorXd& x, const Mesh& mesh);


// ============================================================================
// 并行接口修正函数
// ============================================================================

/**
 * @brief 从源场 phi2 中减去 ghost 列的贡献，修正目标向量 phi1（减法修正）
 *
 * @details
 * 在并行 CG 的矩阵-向量乘法 Ap 步骤中，ghost 列的值来自上一次通信，
 * 本函数将其对内部点的影响从 Ap 中减去，等价于：
 *   Ap(i,j) -= A_e(i,j) * phi2(i, j+1)   （若东邻为 ghost 列）
 *   Ap(i,j) -= A_w(i,j) * phi2(i, j-1)   （若西邻为 ghost 列）
 *
 * 与 Parallel_correction2 配合使用：初始化残差时用加法（+），
 * 迭代中计算 Ap 时用减法（-）。
 *
 * @param mesh   网格对象（提供 bctype）
 * @param equ    方程对象（提供 A_e, A_w 系数）
 * @param phi1   输入/输出：目标向量场（通常为 Ap，原地修正）
 * @param phi2   输入：源向量场（通常为搜索方向 p 的 ghost 列已交换版本）
 */
void Parallel_correction(Mesh& mesh, Equation& equ, MatrixXd& phi1, MatrixXd& phi2);

/**
 * @brief 从源场 phi2 中加上 ghost 列的贡献，修正目标向量 phi1（加法修正）
 *
 * @details
 * 用于初始化残差 r = b - Ax 时，将 ghost 列对 Ax 的贡献加回到残差中：
 *   r(i,j) += A_e(i,j) * x(i, j+1)   （若东邻为 ghost 列）
 *   r(i,j) += A_w(i,j) * x(i, j-1)   （若西邻为 ghost 列）
 *
 * @param mesh   网格对象（提供 bctype）
 * @param equ    方程对象（提供 A_e, A_w 系数）
 * @param phi1   输入/输出：目标向量场（通常为残差 r，原地修正）
 * @param phi2   输入：源向量场（通常为当前解 x 的 ghost 列已交换版本）
 */
void Parallel_correction2(Mesh& mesh, Equation& equ, MatrixXd& phi1, MatrixXd& phi2);


// ============================================================================
// 并行线性求解器（底层接口）
// ============================================================================

/**
 * @brief 无预条件并行共轭梯度法（MPI 分布式）
 *
 * @details
 * 算法流程：
 * 1. 计算初始残差 r = b - Ax，并对 ghost 列做加法修正（Parallel_correction2）
 * 2. 初始化搜索方向 p = r
 * 3. CG 主迭代：
 *    - 计算 Ap，对 ghost 列做减法修正（Parallel_correction）
 *    - 通过 MPI_Allreduce 同步全局内积 (p, Ap) 和 ‖r‖²
 *    - 更新 x, r, p
 * 4. 停滞检测：连续 max_stagnation 步残差下降率 < stagnation_tol 则提前退出
 * 5. 收敛判断由 rank 0 执行，通过 MPI_Bcast 广播退出标志
 *
 * 每次迭代的 MPI 通信量：
 * - exchangeColumns：1 次（更新 ghost 列）
 * - MPI_Allreduce：2 次（内积同步）
 * - MPI_Bcast：1 次（退出标志同步）
 *
 * @param equ        方程对象（提供稀疏矩阵 A 和系数 A_p/A_e/A_w）
 * @param mesh       网格对象（值传递，供修正函数使用）
 * @param b          右端向量（长度 internumber）
 * @param x          输入/输出：初始解猜测值，求解完成后存放解向量
 * @param epsilon    相对收敛容差（‖r‖/‖r₀‖ < epsilon 时收敛）
 * @param max_iter   最大迭代次数
 * @param rank       当前 MPI 进程编号
 * @param num_procs  总 MPI 进程数
 * @param r0         输出：最终残差范数（所有进程返回相同值）
 * @param verbose    日志级别（0=静默，1=打印收敛信息），默认为 0
 */
void CG_parallel(Equation& equ, Mesh mesh,
                 VectorXd& b, VectorXd& x,
                 double epsilon, int max_iter,
                 int rank, int num_procs,
                 double& r0, int verbose = 0);

/**
 * @brief Jacobi 预条件并行共轭梯度法（MPI 分布式，推荐使用）
 *
 * @details
 * 在 CG_parallel 基础上引入 Jacobi 预条件器 M = diag(A_p)：
 * - 预条件操作：z = M⁻¹r（逐元素除以对角系数，无通信）
 * - 搜索方向更新为 p = z + β·p（PCG 公式）
 * - 步长系数 α = (r,z)/(p,Ap)
 *
 * Jacobi 预条件的优势：
 * - 构建代价为 O(N)，无额外通信
 * - 对于扩散主导问题可显著减少迭代次数
 * - 对于对流主导问题效果有限，可考虑换用 ILU 预条件
 *
 * 每次迭代的 MPI 通信量与 CG_parallel 相同（预条件操作为纯本地运算）。
 *
 * @param equ        方程对象（提供稀疏矩阵 A 和 A_p 对角系数）
 * @param mesh       网格对象（值传递）
 * @param b          右端向量（长度 internumber）
 * @param x          输入/输出：初始解猜测值及解向量
 * @param epsilon    相对收敛容差
 * @param max_iter   最大迭代次数
 * @param rank       当前 MPI 进程编号
 * @param num_procs  总 MPI 进程数
 * @param r0         输出：最终残差范数
 * @param verbose    日志级别（0=静默，1=打印收敛信息），默认为 0
 */
void PCG_parallel(Equation& equ, Mesh mesh,
                  VectorXd& b, VectorXd& x,
                  double epsilon, int max_iter,
                  int rank, int num_procs,
                  double& r0, int verbose = 0);


// ============================================================================
// 场变量级求解器封装（高层接口）
// ============================================================================

/**
 * @brief 用 CG_parallel 求解线性方程组，结果直接写回场变量矩阵
 *
 * @details
 * 封装流程：
 * 1. 以全零向量作为初始解
 * 2. 调用 CG_parallel 求解 equ.A * x = equ.source
 * 3. 通过 vectorToMatrix 将解写回 field 的内部点
 * 4. 调用 exchangeColumns 同步 ghost 列，确保后续计算可直接访问邻居值
 *
 * @param equ        已完成离散（含 A 和 source）的方程对象
 * @param mesh       网格对象（提供内部点编号和边界信息）
 * @param field      输出：ny×nx 场变量矩阵（内部点被解更新，ghost 列被同步）
 * @param tol        相对收敛容差
 * @param max_iter   最大迭代次数
 * @param rank       当前 MPI 进程编号
 * @param num_procs  总 MPI 进程数
 * @param l2_norm    输出：求解完成后的残差范数
 * @param verbose    日志级别（0=静默，1=打印收敛信息）
 */
void solveFieldCG(Equation& equ, Mesh& mesh, MatrixXd& field,
                  double tol, int max_iter,
                  int rank, int num_procs,
                  double& l2_norm, int verbose);

/**
 * @brief 用 PCG_parallel 求解线性方程组，结果直接写回场变量矩阵（推荐使用）
 *
 * @details
 * 封装流程与 solveFieldCG 完全相同，仅底层求解器替换为 PCG_parallel。
 * 对于绝大多数 CFD 问题，PCG 收敛速度优于 CG，推荐优先使用本函数。
 *
 * @param equ        已完成离散（含 A 和 source）的方程对象
 * @param mesh       网格对象
 * @param field      输出：ny×nx 场变量矩阵（内部点被解更新，ghost 列被同步）
 * @param tol        相对收敛容差
 * @param max_iter   最大迭代次数
 * @param rank       当前 MPI 进程编号
 * @param num_procs  总 MPI 进程数
 * @param l2_norm    输出：求解完成后的残差范数
 * @param verbose    日志级别（0=静默，1=打印收敛信息）
 */
void solveFieldPCG(Equation& equ, Mesh& mesh, MatrixXd& field,
                   double tol, int max_iter,
                   int rank, int num_procs,
                   double& l2_norm, int verbose);


#endif // PARALLEL_H