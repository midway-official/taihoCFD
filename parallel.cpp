/**
 * @file parallel.cpp
 * @brief 并行计算实现 - MPI通信和并行求解器
 * @details 
 * 实现基于MPI的区域分解并行计算,包括:
 * - 网格边界数据交换(高性能优化版本)
 * - 并行共轭梯度法(CG)
 * - 并行双共轭梯度稳定法(BiCGSTAB)
 * - 数据完整性校验
 * 
 * 并行策略:
 * - 区域分解: 网格在x方向分割成多个子区域
 * - 重叠区域: 相邻进程间共享2列边界数据
 * - 通信优化: 使用persistent通信和笛卡尔拓扑
 * 
 * @author CFD Team
 * @date 2024
 */

#include "parallel.h"

// ============================================================================
// 全局变量定义
// ============================================================================

double total_comm_time = 0.0;   ///< 累计通信时间(秒)
int total_comm_count = 0;       ///< 累计通信次数
double start_time, end_time;    ///< 计时器(用于性能分析)
int totalcount = 0;             ///< 总计数器(通用)

// ============================================================================
// 数据完整性校验函数
// ============================================================================

/**
 * @brief 计算数据的哈希值(用于通信校验)
 * @param data 待计算哈希的数据向量
 * @return 哈希值
 * @details
 * 使用FNV-1a哈希算法的变体:
 * - offset basis: 2166136261
 * - prime: 16777619
 * - 取模1e18保持数值稳定性
 * 
 * 用途: 检测MPI通信中的数据损坏
 * @note 这不是密码学安全的哈希,仅用于错误检测
 */
double computeHash(const vector<double>& data) {
    double hash = 2166136261.0;  // FNV offset basis
    const double prime = 16777619.0;
    
    for (double val : data) {
        hash = fmod((hash * prime), 1e18);  // 保持数值稳定
        hash += val;
    }
    
    return hash;
}

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
 * 步骤:
 * 1. 提取矩阵列数据
 * 2. 计算哈希值
 * 3. 将哈希值放在缓冲区首位
 * 4. 使用非阻塞发送
 * 
 * @note 接收端需使用recvMatrixColumnWithSafety()验证
 */
void sendMatrixColumnWithSafety(const MatrixXd& src_matrix, int src_col, 
                                vector<double>& send_buffer, 
                                int target_rank, int tag) {
    int rows = src_matrix.rows();
    send_buffer.resize(rows + 1);  // 多一个位置放hash
    
    // 填充数据(跳过第一个位置)
    for (int i = 0; i < rows; i++) {
        send_buffer[i+1] = src_matrix(i, src_col);
    }
    
    // 计算并存储哈希值
    double hash_value = computeHash(vector<double>(send_buffer.begin() + 1, send_buffer.end()));
    send_buffer[0] = hash_value;
    
    // 非阻塞发送
    MPI_Request request;
    MPI_Isend(send_buffer.data(), rows+1, MPI_DOUBLE, target_rank, tag, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
}

/**
 * @brief 接收矩阵列数据(带安全校验)
 * @param recv_buffer 接收缓冲区(不含哈希值)
 * @param src_rank 源进程号
 * @param tag MPI消息标签
 * @details
 * 接收格式: [hash_value, data[0], data[1], ..., data[n-1]]
 * 
 * 步骤:
 * 1. 接收完整缓冲区(哈希+数据)
 * 2. 分离哈希值和数据
 * 3. 重新计算哈希值
 * 4. 比对验证,不匹配则中止程序
 * 
 * @throws MPI_Abort 如果哈希值不匹配(数据损坏)
 * @note 容差设为1e-5,可根据需要调整
 */
void recvMatrixColumnWithSafety(vector<double>& recv_buffer, int src_rank, int tag) {
    int rows = recv_buffer.size();
    vector<double> full_buffer(rows + 1);  // 接收hash+数据
    
    // 非阻塞接收
    MPI_Request request;
    MPI_Irecv(full_buffer.data(), rows+1, MPI_DOUBLE, src_rank, tag, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    
    // 分离哈希值和数据
    double recv_hash = full_buffer[0];
    for (int i = 0; i < rows; i++) {
        recv_buffer[i] = full_buffer[i+1];
    }
    
    // 验证哈希值
    double computed_hash = computeHash(recv_buffer);
    if (abs(computed_hash - recv_hash) > 1e-5) {
        cerr << "数据校验失败! 哈希不匹配!" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// ============================================================================
// 网格边界数据交换 - 高性能版本
// ============================================================================

/**
 * @brief 交换相邻进程间的网格重叠列(高性能优化版本)
 * @param matrix 网格矩阵(会被修改)
 * @param rank 当前进程号
 * @param num_procs 总进程数
 * @details
 * 重叠区域结构(每个进程):
 * ```
 * [0][1] | [2][3] ... [nx-2][nx-1] | [nx][nx+1]
 *  虚拟层   内部计算区域             虚拟层
 * ```
 * 
 * 通信模式:
 * - 向左发送: 列2和列3 → 左邻进程的列nx和列nx+1
 * - 向右发送: 列nx-2和列nx-1 → 右邻进程的列0和列1
 * 
 * 性能优化:
 * 1. **聚合通信**: 一次发送2列而非2次单列发送
 * 2. **Persistent通信**: 使用MPI_Send_init/Recv_init重用通信模式
 * 3. **笛卡尔拓扑**: 使用MPI_Cart_create优化邻居通信
 * 4. **批量启动**: MPI_Startall同时启动4个通信请求
 * 
 * @note 第一个进程(rank=0)无左邻居,最后一个进程无右邻居
 * @note 使用MPI_PROC_NULL处理边界情况
 * @post 矩阵的虚拟层(列0,1,nx,nx+1)被更新为相邻进程的数据
 */
void exchangeColumns(MatrixXd& matrix, int rank, int num_procs) {
    const int rows = matrix.rows();
    const int cols = matrix.cols();

    // 确定左右邻居进程号
    int left_rank  = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int right_rank = (rank == num_procs - 1) ? MPI_PROC_NULL : rank + 1;

    // 每侧发送2列
    const int num_cols_per_side = 2;

    // 分配并清零缓冲区
    vector<double> sendbuf_left(rows * num_cols_per_side, 0.0);   // 发送给左邻
    vector<double> sendbuf_right(rows * num_cols_per_side, 0.0);  // 发送给右邻
    vector<double> recvbuf_left(rows * num_cols_per_side, 0.0);   // 从左邻接收
    vector<double> recvbuf_right(rows * num_cols_per_side, 0.0);  // 从右邻接收

    // ===== 填充发送缓冲区 =====
    // 布局: [row0_col0, row0_col1, row1_col0, row1_col1, ...]
    for (int i = 0; i < rows; i++) {
        // 向左发送列2和列3
        sendbuf_left[i * num_cols_per_side + 0] = matrix(i, 2);
        sendbuf_left[i * num_cols_per_side + 1] = matrix(i, 3);

        // 向右发送列nx-2和列nx-1
        sendbuf_right[i * num_cols_per_side + 0] = matrix(i, cols - 4);
        sendbuf_right[i * num_cols_per_side + 1] = matrix(i, cols - 3);
    }

    // ===== 建立笛卡尔拓扑 =====
    // 一维拓扑: num_procs个进程排成一行,非周期
    MPI_Comm cart_comm;
    int dims[1] = { num_procs };
    int periods[1] = { 0 };  // 非周期(第一个和最后一个不相连)
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 0, &cart_comm);

    // ===== 定义Persistent通信请求 =====
    // Persistent通信可以重复使用,避免重复初始化开销
    MPI_Request requests[4];
    
    // 请求0: 向左发送
    MPI_Send_init(sendbuf_left.data(),  rows * num_cols_per_side, MPI_DOUBLE, 
                  left_rank,  0, MPI_COMM_WORLD, &requests[0]);
    
    // 请求1: 从左接收
    MPI_Recv_init(recvbuf_left.data(),  rows * num_cols_per_side, MPI_DOUBLE, 
                  left_rank,  1, MPI_COMM_WORLD, &requests[1]);
    
    // 请求2: 向右发送
    MPI_Send_init(sendbuf_right.data(), rows * num_cols_per_side, MPI_DOUBLE, 
                  right_rank, 1, MPI_COMM_WORLD, &requests[2]);
    
    // 请求3: 从右接收
    MPI_Recv_init(recvbuf_right.data(), rows * num_cols_per_side, MPI_DOUBLE, 
                  right_rank, 0, MPI_COMM_WORLD, &requests[3]);

    // ===== 启动通信 =====
    // 同时启动4个通信,MPI可以重叠发送和接收
    MPI_Startall(4, requests);

    // ===== 等待通信完成 =====
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    // ===== 更新矩阵边界值 =====
    // 从左邻进程接收的数据写入列0和列1
    if (left_rank != MPI_PROC_NULL) {
        for (int i = 0; i < rows; i++) {
            matrix(i, 0) = recvbuf_left[i * num_cols_per_side + 0];
            matrix(i, 1) = recvbuf_left[i * num_cols_per_side + 1];
        }
    }

    // 从右邻进程接收的数据写入列nx和列nx+1
    if (right_rank != MPI_PROC_NULL) {
        for (int i = 0; i < rows; i++) {
            matrix(i, cols - 2) = recvbuf_right[i * num_cols_per_side + 0];
            matrix(i, cols - 1) = recvbuf_right[i * num_cols_per_side + 1];
        }
    }

    // ===== 清理资源 =====
    // 释放persistent请求和communicator
    for (int i = 0; i < 4; ++i) {
        MPI_Request_free(&requests[i]);
    }
    MPI_Comm_free(&cart_comm);
}

// ============================================================================
// 向量与矩阵转换函数
// ============================================================================

/**
 * @brief 从解向量转换为场矩阵
 * @param x 解向量(压缩存储,只含内部点)
 * @param phi 场矩阵(输出,包含边界)
 * @param mesh 网格对象
 * @details
 * 仅处理内部点(bctype=0):
 * - 使用interid映射找到对应的向量索引
 * - 将向量元素写入矩阵对应位置
 * 
 * 用途: 将求解器的解向量转换回网格矩阵,便于后续计算和通信
 * @note 边界点不会被修改,保持原有边界条件
 */
void vectorToMatrix(const VectorXd& x, MatrixXd& phi, const Mesh& mesh) {
    for (int i = 0; i <= mesh.ny + 1; i++) {
        for (int j = 0; j <= mesh.nx + 1; j++) {
            if (mesh.bctype(i, j) == 0) {  // 仅处理内部点
                int n = mesh.interid(i, j);  // 获取向量索引
                phi(i, j) = x[n];
            }
        }
    }
}

/**
 * @brief 从场矩阵转换为解向量
 * @param phi 场矩阵(输入,包含边界)
 * @param x 解向量(输出,压缩存储)
 * @param mesh 网格对象
 * @details
 * 仅提取内部点(bctype=0):
 * - 使用interid映射找到对应的向量索引
 * - 将矩阵元素写入向量对应位置
 * 
 * 用途: 将网格矩阵转换为求解器所需的向量格式
 * @note 边界点不会被提取,向量只包含内部点数据
 */
void matrixToVector(const MatrixXd& phi, VectorXd& x, const Mesh& mesh) {
    for (int i = 0; i <= mesh.ny + 1; i++) {
        for (int j = 0; j <= mesh.nx + 1; j++) {
            if (mesh.bctype(i, j) == 0) {  // 仅处理内部点
                int n = mesh.interid(i, j);  // 获取向量索引
                x[n] = phi(i, j);
            }
        }
    }
}

// ============================================================================
// 并行边界修正函数
// ============================================================================

/**
 * @brief 并行边界修正(减法修正)
 * @param mesh 网格对象
 * @param equ 方程对象(提供系数矩阵)
 * @param phi1 被修正的场矩阵(会被修改)
 * @param phi2 用于修正的场矩阵(提供边界值)
 * @details
 * 修正公式:
 * - 东边界: phi1(i,j) -= A_e(i,j) * phi2(i,j+1)  当bctype(i,j+1)=-3
 * - 西边界: phi1(i,j) -= A_w(i,j) * phi2(i,j-1)  当bctype(i,j-1)=-3
 * 
 * 用途:
 * 在矩阵向量乘法Ax中,去除边界点的贡献,因为这些点的值
 * 已经通过exchangeColumns()更新了
 * 
 * 使用场景:
 * 1. CG中的Ap = A*p计算
 * 2. BiCGSTAB中的v = A*p和t = A*s计算
 * 
 * @note bctype=-3表示并行交界面
 */
void Parallel_correction(Mesh mesh, Equation equ, MatrixXd &phi1, MatrixXd &phi2) {
    for (int i = 0; i <= mesh.ny + 1; i++) {
        for (int j = 0; j <= mesh.nx + 1; j++) {
            if (mesh.bctype(i, j) == 0) {  // 仅处理内部点
                
                // 东边界修正
                if (mesh.bctype(i, j+1) == -3) {
                    phi1(i, j) -= equ.A_e(i, j) * phi2(i, j+1);
                }
                
                // 西边界修正
                if (mesh.bctype(i, j-1) == -3) {
                    phi1(i, j) -= equ.A_w(i, j) * phi2(i, j-1);
                }
            }
        }
    }
}

/**
 * @brief 并行边界修正(加法修正)
 * @param mesh 网格对象
 * @param equ 方程对象(提供系数矩阵)
 * @param phi1 被修正的场矩阵(会被修改)
 * @param phi2 用于修正的场矩阵(提供边界值)
 * @details
 * 修正公式:
 * - 东边界: phi1(i,j) += A_e(i,j) * phi2(i,j+1)  当bctype(i,j+1)=-3
 * - 西边界: phi1(i,j) += A_w(i,j) * phi2(i,j-1)  当bctype(i,j-1)=-3
 * 
 * 用途:
 * 在残差计算r = b - A*x中,补偿边界点的贡献
 * 
 * 使用场景:
 * 1. CG初始化时的残差修正
 * 
 * @note 与Parallel_correction()的区别仅在于加减号
 */
void Parallel_correction2(Mesh mesh, Equation equ, MatrixXd &phi1, MatrixXd &phi2) {
    for (int i = 0; i <= mesh.ny + 1; i++) {
        for (int j = 0; j <= mesh.nx + 1; j++) {
            if (mesh.bctype(i, j) == 0) {  // 仅处理内部点
                
                // 东边界修正
                if (mesh.bctype(i, j+1) == -3) {
                    phi1(i, j) += equ.A_e(i, j) * phi2(i, j+1);
                }
                
                // 西边界修正
                if (mesh.bctype(i, j-1) == -3) {
                    phi1(i, j) += equ.A_w(i, j) * phi2(i, j-1);
                }
            }
        }
    }
}

// ============================================================================
// 并行共轭梯度法(CG)
// ============================================================================

/**
 * @brief 并行共轭梯度法求解线性方程组 Ax = b
 * @param equ 方程对象(包含系数矩阵A)
 * @param mesh 网格对象(用于并行通信)
 * @param b 右端项向量
 * @param x 解向量(输入初值,输出解)
 * @param epsilon 收敛容差
 * @param max_iter 最大迭代次数
 * @param rank 当前进程号
 * @param num_procs 总进程数
 * @param r0 输出最终残差范数
 * @details
 * 算法流程:
 * ```
 * 1. 初始化: r = b - Ax, p = r
 * 2. 迭代:
 *    a. 计算 Ap = A*p (需要并行通信)
 *    b. α = (r,r) / (p,Ap)
 *    c. x = x + α*p
 *    d. r = r - α*Ap
 *    e. β = (r_new,r_new) / (r_old,r_old)
 *    f. p = r + β*p
 * 3. 判断收敛: ||r|| < ε
 * ```
 * 
 * 并行实现要点:
 * 1. **矩阵向量乘**: 
 *    - 转换为场矩阵
 *    - 交换边界列(exchangeColumns)
 *    - 并行修正(Parallel_correction)
 *    - 转换回向量
 * 
 * 2. **全局规约**:
 *    - 残差范数: MPI_Allreduce(||r||²)
 *    - 点积: MPI_Allreduce(p·Ap)
 * 
 * 3. **同步点**: 每次规约后使用MPI_Barrier确保同步
 * 
 * @note 每次迭代包含2次MPI_Allreduce和1次exchangeColumns
 * @note 如果||b||<1e-13,直接返回零解
 * @post x被更新为方程的近似解
 * @post r0为最终残差范数
 */
void CG_parallel(Equation& equ, Mesh mesh, VectorXd& b, VectorXd& x, double epsilon, 
                 int max_iter, int rank, int num_procs, double& r0,
                 int verbose) {
    
    int n = equ.A.rows();
    SparseMatrix<double> A = equ.A;

    // ===== 初始化残差 r = b - Ax =====
    VectorXd r = VectorXd::Zero(n);
    r = b - A * x;  // 局部矩阵向量乘
    MPI_Barrier(MPI_COMM_WORLD);

    // 转换为场矩阵进行并行修正
    MatrixXd r_field = MatrixXd::Zero(mesh.ny+2, mesh.nx+2);
    MatrixXd x_field = MatrixXd::Zero(mesh.ny+2, mesh.nx+2);

    vectorToMatrix(r, r_field, mesh);
    vectorToMatrix(x, x_field, mesh);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // 交换边界并修正残差
    exchangeColumns(x_field, rank, num_procs);
    Parallel_correction2(mesh, equ, r_field, x_field);
    matrixToVector(r_field, r, mesh);
    MPI_Barrier(MPI_COMM_WORLD);

    // ===== 初始化搜索方向 p = r =====
    VectorXd p = VectorXd::Zero(n);
    p = r;
    VectorXd Ap = VectorXd::Zero(n);

    // ===== 计算初始残差范数 =====
    double r_norm = r.squaredNorm();      // 局部||r||²
    double b_norm = b.squaredNorm();      // 局部||b||²
    double local_b_norm = b_norm;
    double global_b_norm = 0.0;

    // 全局规约||b||²
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(&local_b_norm, &global_b_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // 特殊情况: 右端项接近零
    if (global_b_norm < 1e-20) {
        x.setZero();
        r0 = 0.0;
        if (rank == 0 && verbose == 1 ) {
            std::cout << "  [CG] 全局b_norm < 1e-20, 返回零解" << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }

    // 全局规约||r||²
    double tol = epsilon * epsilon;
    double global_r_norm = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(&r_norm, &global_r_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    double initial_r_norm = std::sqrt(global_r_norm);  // 初始残差
    r0 = initial_r_norm;
    
    // 用于检查残差变化的变量
    double prev_r0 = r0;
    const double stagnation_ratio = 0.999;  // 停滞判据: 如果残差下降不到0.1%则视为停滞
    int stagnation_count = 0;
    const int max_stagnation = 3;  // 连续停滞3次则终止

    // ===== CG迭代 =====
    int iter = 0;
    while (iter < max_iter) {
        
        // ----- 计算 Ap = A*p (并行矩阵向量乘) -----
        Ap.setZero();
        Ap = A * p;  // 局部乘法

        // 转换为场矩阵
        MatrixXd p_field = MatrixXd::Zero(mesh.ny+2, mesh.nx+2);
        MatrixXd Ap_field = MatrixXd::Zero(mesh.ny+2, mesh.nx+2);

        vectorToMatrix(p, p_field, mesh);
        vectorToMatrix(Ap, Ap_field, mesh);
        MPI_Barrier(MPI_COMM_WORLD);
        
        // 交换边界并修正Ap
        exchangeColumns(p_field, rank, num_procs);
        MPI_Barrier(MPI_COMM_WORLD);
        Parallel_correction(mesh, equ, Ap_field, p_field);

        matrixToVector(Ap_field, Ap, mesh);

        // ----- 计算步长 α = (r,r) / (p,Ap) -----
        double local_dot_p_Ap = p.dot(Ap);     // 局部点积
        double global_dot_p_Ap = 0.0;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&local_dot_p_Ap, &global_dot_p_Ap, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // 防止除零
        if (std::abs(global_dot_p_Ap) < 1e-20) {
            if (rank == 0&& verbose == 1) {
                std::cout << "  [CG] 警告: (p,Ap)接近零, 第" << iter << "轮提前终止" << std::endl;
            }
            break;
        }

        double alpha = global_r_norm / global_dot_p_Ap;

        // ----- 更新解和残差 -----
        x += alpha * p;
        r -= alpha * Ap;

        // ----- 计算新残差范数 -----
        double new_r_norm = r.squaredNorm();
        double global_new_r_norm = 0.0;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&new_r_norm, &global_new_r_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // ----- 计算共轭方向 β = (r_new,r_new) / (r_old,r_old) -----
        double beta = global_new_r_norm / global_r_norm;
        p = r + beta * p;
        
        // 更新残差范数
        global_r_norm = global_new_r_norm;
        r0 = std::sqrt(global_r_norm);
        
        iter++;
        
        // ===== 收敛性检查 =====
        
        // 1. 绝对收敛: ||r|| / ||r0|| < epsilon
        double relative_residual = r0 / initial_r_norm;
        if (relative_residual < epsilon) {
            if (rank == 0&& verbose == 1) {
                std::cout << "  [CG] 达到收敛: 相对残差 " << std::scientific << std::setprecision(3) 
                          << relative_residual << " < " << epsilon 
                          << " (第" << iter << "轮)" << std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            return;
        }
        
        // 2. 残差变化检查: r_current / r_previous
        if (iter > 1) {
            double residual_ratio = r0 / prev_r0;
            
            // 如果残差几乎不下降 (下降小于1%)
            if (residual_ratio > stagnation_ratio) {
                stagnation_count++;
                
                if (stagnation_count >= max_stagnation) {
                    if (rank == 0&& verbose == 1) {
                        std::cout << "  [CG] 残差停滞: 连续" << stagnation_count 
                                  << "轮下降<" << (1.0 - stagnation_ratio) * 100 << "%, 第" 
                                  << iter << "轮终止" << std::endl;
                        std::cout << "      当前残差比值: " << std::scientific << std::setprecision(3) 
                                  << residual_ratio << std::endl;
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                    return;
                }
            } else {
                // 残差有显著下降,重置停滞计数
                stagnation_count = 0;
            }
        }
        
        // 保存本轮残差用于下一轮比较
        prev_r0 = r0;
        
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // 达到最大迭代次数
    if (rank == 0&& verbose == 1) {
        std::cout << "  [CG] 达到最大迭代次数(" << max_iter << "), 最终残差: " 
                  << std::scientific << std::setprecision(3) << r0 / initial_r_norm << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}