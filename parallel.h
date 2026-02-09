/**
 * @file parallel.h
 * @brief 并行计算接口 - MPI通信和并行求解器
 * @details 
 * 提供基于MPI的区域分解并行计算接口,包括:
 * - 网格边界数据交换(高性能优化)
 * - 数据完整性校验
 * - 并行迭代求解器(CG, BiCGSTAB)
 * - 向量与矩阵转换工具
 * 
 * 并行计算模型:
 * - 数据并行: 网格在x方向分解
 * - 通信模式: 最近邻通信(只与左右邻居交换数据)
 * - 同步策略: 全局同步(MPI_Barrier)
 * 
 * 性能优化:
 * - Persistent通信: 重用通信模式
 * - 笛卡尔拓扑: 优化邻居通信
 * - 聚合通信: 批量发送多列数据
 * 
 * @author CFD Team
 * @date 2024
 */

#ifndef PARALLEL_H
#define PARALLEL_H

// ============================================================================
// 标准库和第三方库头文件
// ============================================================================
#include <mpi.h>                    ///< MPI并行库
#include <iostream>
#include <vector>
#include <cmath>
#include <eigen3/Eigen/Sparse>      ///< Eigen稀疏矩阵库

// ============================================================================
// 项目头文件
// ============================================================================
#include "fluid.h"                  ///< 流体求解器基础类(Mesh, Equation)

// ============================================================================
// 命名空间
// ============================================================================
using namespace std;
using namespace Eigen;

// ============================================================================
// 全局变量声明
// ============================================================================

/**
 * @defgroup ParallelStats 并行计算性能统计
 * @{
 */

extern double total_comm_time;      ///< 累计通信时间(秒)
extern int total_comm_count;        ///< 累计通信次数
extern double start_time;           ///< 计时开始时间
extern double end_time;             ///< 计时结束时间
extern int totalcount;              ///< 总计数器(通用)

/** @} */ // end of ParallelStats

// ============================================================================
// 数据完整性校验函数
// ============================================================================

/**
 * @defgroup DataIntegrity 数据完整性校验
 * @{
 */

/**
 * @brief 计算数据的哈希值(用于通信校验)
 * @param data 待计算哈希的数据向量
 * @return 哈希值
 * @details
 * 使用FNV-1a哈希算法的变体,用于检测MPI通信中的数据损坏
 * 
 * 算法特点:
 * - 计算快速,适合实时校验
 * - 取模1e18保持数值稳定性
 * - 非加密哈希,仅用于错误检测
 * 
 * @note 碰撞概率极低但非零
 */
double computeHash(const vector<double>& data);

/**
 * @brief 发送矩阵列数据(带安全校验)
 * @param src_matrix 源矩阵
 * @param src_col 要发送的列索引
 * @param send_buffer 发送缓冲区(会被修改)
 * @param target_rank 目标进程号
 * @param tag MPI消息标签
 * @details
 * 发送格式: [hash_value, data[0], data[1], ..., data[n-1]]
 * 
 * 工作流程:
 * 1. 提取矩阵列数据到缓冲区
 * 2. 计算数据哈希值
 * 3. 将哈希值置于缓冲区首位
 * 4. 使用MPI_Isend非阻塞发送
 * 5. MPI_Wait等待发送完成
 * 
 * @note 接收端必须使用recvMatrixColumnWithSafety()接收和验证
 * @see recvMatrixColumnWithSafety()
 */
void sendMatrixColumnWithSafety(const MatrixXd& src_matrix, int src_col, 
                                vector<double>& send_buffer, 
                                int target_rank, int tag);

/**
 * @brief 接收矩阵列数据(带安全校验)
 * @param recv_buffer 接收缓冲区(输出,不含哈希值)
 * @param src_rank 源进程号
 * @param tag MPI消息标签
 * @details
 * 接收格式: [hash_value, data[0], data[1], ..., data[n-1]]
 * 
 * 工作流程:
 * 1. 使用MPI_Irecv接收完整缓冲区(哈希+数据)
 * 2. MPI_Wait等待接收完成
 * 3. 分离哈希值和数据
 * 4. 重新计算接收数据的哈希值
 * 5. 比对哈希值,不匹配则中止程序
 * 
 * @throws MPI_Abort 如果哈希值不匹配(数据损坏)
 * @note 哈希值匹配容差为1e-5
 * @see sendMatrixColumnWithSafety()
 */
void recvMatrixColumnWithSafety(vector<double>& recv_buffer, int src_rank, int tag);

/** @} */ // end of DataIntegrity

// ============================================================================
// 网格边界通信
// ============================================================================

/**
 * @defgroup BoundaryComm 网格边界通信
 * @{
 */

/**
 * @brief 交换相邻进程间的网格重叠列(高性能优化版本)
 * @param matrix 网格矩阵(会被修改)
 * @param rank 当前进程号
 * @param num_procs 总进程数
 * @details
 * 网格重叠区域结构(每个进程):
 * ```
 * 列索引:  0    1  |  2   3  ...  nx-2  nx-1 | nx  nx+1
 *         虚拟层   |    内部计算区域          | 虚拟层
 *         (接收)   |    (自有数据)           | (接收)
 * ```
 * 
 * 通信模式:
 * - **向左邻居**: 发送列2,3 → 接收列0,1
 * - **向右邻居**: 发送列nx-2,nx-1 → 接收列nx,nx+1
 * 
 * 性能优化策略:
 * 1. **聚合通信**: 一次发送2列(4个数据包→1个数据包)
 * 2. **Persistent通信**: 
 *    - MPI_Send_init/Recv_init: 预初始化通信模式
 *    - MPI_Startall: 批量启动4个通信请求
 *    - 避免重复初始化开销
 * 3. **笛卡尔拓扑**: 
 *    - MPI_Cart_create: 建立1维拓扑
 *    - 优化邻居通信性能
 * 4. **通信重叠**: 
 *    - 同时启动发送和接收
 *    - MPI自动重叠通信
 * 
 * 边界处理:
 * - 第一个进程(rank=0): 无左邻居,left_rank=MPI_PROC_NULL
 * - 最后进程(rank=num_procs-1): 无右邻居,right_rank=MPI_PROC_NULL
 * - MPI_PROC_NULL: MPI会自动忽略相关通信操作
 * 
 * 资源管理:
 * - 通信完成后释放persistent请求(MPI_Request_free)
 * - 释放笛卡尔拓扑通信子(MPI_Comm_free)
 * 
 * @post 矩阵的虚拟列(0,1,nx,nx+1)被更新为相邻进程的数据
 * @note 每次调用包含4个MPI通信操作(2发送+2接收)
 * @warning 必须在所有进程上同时调用(集合通信)
 */
void exchangeColumns(MatrixXd& matrix, int rank, int num_procs);

/** @} */ // end of BoundaryComm

// ============================================================================
// 向量与矩阵转换
// ============================================================================

/**
 * @defgroup VectorMatrixConv 向量与矩阵转换
 * @{
 */

/**
 * @brief 从解向量转换为场矩阵
 * @param x 解向量(压缩存储,只含内部点)
 * @param phi 场矩阵(输出,包含边界)
 * @param mesh 网格对象
 * @details
 * 转换规则:
 * - 仅处理内部点(mesh.bctype(i,j) == 0)
 * - 使用mesh.interid(i,j)映射到向量索引
 * - 边界点保持不变
 * 
 * 用途:
 * 1. 将求解器的解向量转换回网格矩阵
 * 2. 便于后续场操作(边界交换,后处理等)
 * 
 * 存储格式对比:
 * ```
 * 向量x:  [x0, x1, x2, ..., x_{N-1}]  (N = internumber)
 * 矩阵φ:  包含边界层和边界点的完整网格
 * ```
 * 
 * @note 边界点的值不会被覆盖
 * @post phi的内部点被x的值填充
 */
void vectorToMatrix(const VectorXd& x, MatrixXd& phi, const Mesh& mesh);

/**
 * @brief 从场矩阵转换为解向量
 * @param phi 场矩阵(输入,包含边界)
 * @param x 解向量(输出,压缩存储)
 * @param mesh 网格对象
 * @details
 * 转换规则:
 * - 仅提取内部点(mesh.bctype(i,j) == 0)
 * - 使用mesh.interid(i,j)映射到向量索引
 * - 边界点不会被提取
 * 
 * 用途:
 * 1. 将网格矩阵转换为求解器所需的向量格式
 * 2. 便于线性方程组求解
 * 
 * @note x的长度必须等于mesh.internumber
 * @post x被填充为phi的内部点值
 */
void matrixToVector(const MatrixXd& phi, VectorXd& x, const Mesh& mesh);

/** @} */ // end of VectorMatrixConv

// ============================================================================
// 并行边界修正
// ============================================================================

/**
 * @defgroup ParallelCorrection 并行边界修正
 * @{
 */

/**
 * @brief 并行边界修正(减法修正)
 * @param mesh 网格对象
 * @param equ 方程对象(提供系数矩阵A_e, A_w)
 * @param phi1 被修正的场矩阵(会被修改)
 * @param phi2 用于修正的场矩阵(提供边界值)
 * @details
 * 修正公式:
 * ```
 * 对于内部点(i,j):
 *   如果东邻是并行边界(-3): phi1(i,j) -= A_e(i,j) * phi2(i,j+1)
 *   如果西邻是并行边界(-3): phi1(i,j) -= A_w(i,j) * phi2(i,j-1)
 * ```
 * 
 * 物理意义:
 * 在矩阵向量乘法 Ap = A*p 中,去除边界点的贡献,因为这些点的值
 * 已经通过exchangeColumns()从邻居进程获取并更新了。
 * 
 * 使用场景:
 * 1. CG算法中: Ap = A*p 的计算
 * 2. BiCGSTAB算法中: v = A*p 和 t = A*s 的计算
 * 
 * 算法原理:
 * ```
 * 串行: Ap = A*p (A是完整矩阵)
 * 并行: Ap_local = A_local*p_local (A_local是局部矩阵,不含边界连接)
 *       需要补偿: Ap_local -= A_boundary*p_boundary
 * ```
 * 
 * @note 必须在exchangeColumns()之后调用
 * @note 只修正东西边界(x方向分解),不修正南北边界
 * @see Parallel_correction2() 加法修正版本
 */
void Parallel_correction(Mesh mesh, Equation equ, MatrixXd &phi1, MatrixXd &phi2);

/**
 * @brief 并行边界修正(加法修正)
 * @param mesh 网格对象
 * @param equ 方程对象(提供系数矩阵A_e, A_w)
 * @param phi1 被修正的场矩阵(会被修改)
 * @param phi2 用于修正的场矩阵(提供边界值)
 * @details
 * 修正公式:
 * ```
 * 对于内部点(i,j):
 *   如果东邻是并行边界(-3): phi1(i,j) += A_e(i,j) * phi2(i,j+1)
 *   如果西邻是并行边界(-3): phi1(i,j) += A_w(i,j) * phi2(i,j-1)
 * ```
 * 
 * 使用场景:
 * 1. CG算法初始化: r = b - A*x 的残差计算
 * 2. BiCGSTAB算法初始化: r = b - A*x 的残差计算
 * 
 * 与Parallel_correction的区别:
 * - 此函数使用加法(+=),用于补偿残差
 * - Parallel_correction使用减法(-=),用于修正矩阵向量乘结果
 * 
 * @note 必须在exchangeColumns()之后调用
 * @see Parallel_correction() 减法修正版本
 */
void Parallel_correction2(Mesh mesh, Equation equ, MatrixXd &phi1, MatrixXd &phi2);

/** @} */ // end of ParallelCorrection

// ============================================================================
// 并行迭代求解器
// ============================================================================

/**
 * @defgroup ParallelSolvers 并行迭代求解器
 * @{
 */

/**
 * @brief 并行共轭梯度法(CG)求解线性方程组 Ax = b
 * @param equ 方程对象(包含系数矩阵A)
 * @param mesh 网格对象(用于并行通信)
 * @param b 右端项向量
 * @param x 解向量(输入初值,输出解)
 * @param epsilon 收敛容差(相对残差)
 * @param max_iter 最大迭代次数
 * @param rank 当前进程号
 * @param num_procs 总进程数
 * @param r0 输出最终残差范数
 * @details
 * 算法: 共轭梯度法(Conjugate Gradient)
 * 适用: 对称正定矩阵
 * 
 * 标准CG算法:
 * ```
 * r = b - A*x
 * p = r
 * while ||r|| > ε:
 *     α = (r,r) / (p,Ap)
 *     x = x + α*p
 *     r_new = r - α*Ap
 *     β = (r_new,r_new) / (r,r)
 *     p = r_new + β*p
 *     r = r_new
 * ```
 * 
 * 并行实现关键点:
 * 1. **矩阵向量乘Ap = A*p**:
 *    ```
 *    局部计算 Ap_local = A_local * p_local
 *    转换为场矩阵
 *    交换边界列(exchangeColumns)
 *    并行修正(Parallel_correction)
 *    转换回向量
 *    ```
 * 
 * 2. **全局规约操作**:
 *    - 残差范数: MPI_Allreduce(SUM, ||r||²)
 *    - 点积: MPI_Allreduce(SUM, p·Ap)
 * 
 * 3. **同步点**: 每次规约后使用MPI_Barrier确保同步
 * 
 * 性能分析(每次迭代):
 * - 矩阵向量乘: 1次
 * - 边界交换: 1次(exchangeColumns)
 * - 全局规约: 2次(MPI_Allreduce)
 * - 同步点: 3-4次(MPI_Barrier)
 * 
 * 收敛性:
 * - 理论上N步精确收敛(N=矩阵维数)
 * - 实际中通常10-100步达到工程精度
 * 
 * @note 矩阵A必须对称正定,否则可能不收敛
 * @note 如果||b||<1e-13,直接返回零解
 * @post x被更新为方程的近似解
 * @post r0为最终残差范数||r||
 * @warning 所有进程必须同时调用此函数
 */
void CG_parallel(Equation& equ, Mesh mesh,
                 VectorXd& b, VectorXd& x,
                 double epsilon, int max_iter,
                 int rank, int num_procs,
                 double& r0,
                 int verbose = 0);   // 新增：0不打印，1打印



#endif // PARALLEL_H