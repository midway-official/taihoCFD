#include "parallel.h"


// 定义全局变量
double total_comm_time = 0.0; // 通信时间
int total_comm_count = 0;     // 通信次数
double start_time, end_time;
int totalcount = 0;  




//超高性能的列交换函数
void exchangeColumns(MatrixXd& matrix, int rank, int num_procs) {
    const int rows = matrix.rows();
    const int cols = matrix.cols();

    // 确定左右邻居
    int left_rank  = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int right_rank = (rank == num_procs - 1) ? MPI_PROC_NULL : rank + 1;

    // 聚合通信：打包4列 {2,3,cols-4,cols-3}
    const int num_cols_per_side = 2;

    // 分配并清零缓冲区
    vector<double> sendbuf_left(rows * num_cols_per_side, 0.0);
    vector<double> sendbuf_right(rows * num_cols_per_side, 0.0);
    vector<double> recvbuf_left(rows * num_cols_per_side, 0.0);
    vector<double> recvbuf_right(rows * num_cols_per_side, 0.0);

    // 填充发送缓冲区
    for (int i = 0; i < rows; i++) {
        sendbuf_left[i * num_cols_per_side + 0] = matrix(i, 2);
        sendbuf_left[i * num_cols_per_side + 1] = matrix(i, 3);

        sendbuf_right[i * num_cols_per_side + 0] = matrix(i, cols - 4);
        sendbuf_right[i * num_cols_per_side + 1] = matrix(i, cols - 3);
    }

    
    // 建立 persistent communicator topology（cartesian 拓扑）
    MPI_Comm cart_comm;
    int dims[1] = { num_procs };
    int periods[1] = { 0 };
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 0, &cart_comm);

    // 定义 persistent request
    MPI_Request requests[4];
    MPI_Send_init(sendbuf_left.data(),  rows * num_cols_per_side, MPI_DOUBLE, left_rank,  0, MPI_COMM_WORLD, &requests[0]);
    MPI_Recv_init(recvbuf_left.data(),  rows * num_cols_per_side, MPI_DOUBLE, left_rank,  1, MPI_COMM_WORLD, &requests[1]);
    MPI_Send_init(sendbuf_right.data(), rows * num_cols_per_side, MPI_DOUBLE, right_rank, 1, MPI_COMM_WORLD, &requests[2]);
    MPI_Recv_init(recvbuf_right.data(), rows * num_cols_per_side, MPI_DOUBLE, right_rank, 0, MPI_COMM_WORLD, &requests[3]);

    // 启动通信
    MPI_Startall(4, requests);

    

    // 等待通信完成
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

  

    // 更新矩阵边界值
    if (left_rank != MPI_PROC_NULL) {
        for (int i = 0; i < rows; i++) {
            matrix(i, 0) = recvbuf_left[i * num_cols_per_side + 0];
            matrix(i, 1) = recvbuf_left[i * num_cols_per_side + 1];
        }
    }

    if (right_rank != MPI_PROC_NULL) {
        for (int i = 0; i < rows; i++) {
            matrix(i, cols - 2) = recvbuf_right[i * num_cols_per_side + 0];
            matrix(i, cols - 1) = recvbuf_right[i * num_cols_per_side + 1];
        }
    }

    // 释放 persistent request 和 communicator
    for (int i = 0; i < 4; ++i) {
        MPI_Request_free(&requests[i]);
    }
    MPI_Comm_free(&cart_comm);
}
// 从解向量转换为场矩阵
void vectorToMatrix(const VectorXd& x, MatrixXd& phi, const Mesh& mesh) {
    for (int i = 0; i < mesh.ny ; i++) {
        for (int j = 0; j < mesh.nx ; j++) {
            if (mesh.bctype(i, j) == 0) { // 仅处理内部点
                int n = mesh.interid(i, j) ; // 获取对应的解向量索引
                phi(i, j) = x[n];
            }
        }
    }
}
// 从场矩阵转换为解向量
void matrixToVector(const MatrixXd& phi, VectorXd& x, const Mesh& mesh) {
    for (int i = 0; i < mesh.ny ; i++) {
        for (int j = 0; j < mesh.nx ; j++) {
            if (mesh.bctype(i, j) == 0) { // 仅处理内部点
                int n = mesh.interid(i, j) ; // 获取对应的解向量索引
                x[n] = phi(i, j);
            }
        }
    }
}
void Parallel_correction(Mesh mesh,Equation equ,MatrixXd &phi1,MatrixXd &phi2){
for (int i = 0; i < mesh.ny ; i++) {
        for (int j = 0; j < mesh.nx ; j++) {
            if (mesh.bctype(i, j) == 0) { // 仅处理内部点
                if (mesh.bctype(i, j+1) == -3)
                {
                     phi1(i, j)-=equ.A_e(i, j)*phi2(i, j+1);
                }
                 if (mesh.bctype(i, j-1) == -3)
                {
                     phi1(i, j)-=equ.A_w(i, j)*phi2(i, j-1);
                }
                
               
            }
        }
    }
}
void Parallel_correction2(Mesh mesh,Equation equ,MatrixXd &phi1,MatrixXd &phi2){
for (int i = 0; i < mesh.ny ; i++) {
        for (int j = 0; j < mesh.nx ; j++) {
            if (mesh.bctype(i, j) == 0) { // 仅处理内部点
                if (mesh.bctype(i, j+1) == -3)
                {
                     phi1(i, j)+=equ.A_e(i, j)*phi2(i, j+1);
                }
                 if (mesh.bctype(i, j-1) == -3)
                {
                     phi1(i, j)+=equ.A_w(i, j)*phi2(i, j-1);
                }
                
               
            }
        }
    }
}
void CG_parallel(Equation& equ, Mesh mesh, VectorXd& b, VectorXd& x, double epsilon, 
                 int max_iter, int rank, int num_procs, double& r0,
                 int verbose) {
    
    int n = equ.A.rows();
    SparseMatrix<double> A = equ.A;

    // ===== 1. 初始化残差 r = b - Ax =====
    VectorXd r = b - A * x; 

    MatrixXd r_field = MatrixXd::Zero(mesh.ny, mesh.nx);
    MatrixXd x_field = MatrixXd::Zero(mesh.ny, mesh.nx);

    vectorToMatrix(r, r_field, mesh);
    vectorToMatrix(x, x_field, mesh);
    
    exchangeColumns(x_field, rank, num_procs);
    Parallel_correction2(mesh, equ, r_field, x_field);
    matrixToVector(r_field, r, mesh);

    // ===== 2. 初始化搜索方向 p = r =====
    VectorXd p = r;
    VectorXd Ap = VectorXd::Zero(n);
    
    // ===== 3. 计算全局初始状态 =====
    double local_r_norm_sq = r.squaredNorm();
    double local_b_norm_sq = b.squaredNorm();
    double global_b_norm_sq = 0.0;
    double global_r_norm_sq = 0.0;

    MPI_Allreduce(&local_b_norm_sq, &global_b_norm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_r_norm_sq, &global_r_norm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double initial_r_norm = std::sqrt(global_r_norm_sq);
    r0 = initial_r_norm;
    if (rank == 0 && verbose == 1) {
         std::cout << "  [CG] r0 "  <<   r0   << std::endl;
    }
    // 如果初始残差已经足够小，直接返回
    if (initial_r_norm < 1e-15 || (initial_r_norm / std::sqrt(global_b_norm_sq + 1e-16) < epsilon)) {
        if (rank == 0 && verbose) std::cout << "  [CG] 初始残差已达标。" << std::endl;
        return;
    }

    // --- 停滞检测优化变量 ---
    double prev_r_norm = r0;
    int stagnation_count = 0;
    const int max_stagnation = 3;       // 稍微增加容忍度
    const double stagnation_tol = 1e-4; // 判定为“下降极其缓慢”的阈值
    const int min_iter_protect = 5;    // 至少迭代10次才允许停滞退出

    int exit_status = 0; // 0:运行, 1:收敛, 2:停滞, 3:数值失效
    int iter = 0;
    double current_global_r_norm_sq = global_r_norm_sq;

    // ===== 4. CG 迭代 =====
    while (iter < max_iter) {
        // --- 计算 Ap ---
        Ap = A * p; 
        
        MatrixXd p_field = MatrixXd::Zero(mesh.ny, mesh.nx);
        MatrixXd Ap_field = MatrixXd::Zero(mesh.ny, mesh.nx);
        
        vectorToMatrix(p, p_field, mesh);
        vectorToMatrix(Ap, Ap_field, mesh);
        
        exchangeColumns(p_field, rank, num_procs);
        Parallel_correction(mesh, equ, Ap_field, p_field);
        matrixToVector(Ap_field, Ap, mesh);
        // --- 计算步长 alpha ---
        double local_dot_p_Ap = p.dot(Ap);
        double global_dot_p_Ap = 0.0;
        MPI_Allreduce(&local_dot_p_Ap, &global_dot_p_Ap, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // 数学安全检查：如果分母过小，可能矩阵非正定或已陷入数值陷阱
        if (std::abs(global_dot_p_Ap) < 1e-18) {
            exit_status = 3;
        }
        if (exit_status == 0) {
            double alpha = current_global_r_norm_sq / global_dot_p_Ap;
            // --- 更新解和残差 ---
            x += alpha * p;
            r -= alpha * Ap;
            double local_new_r_sq = r.squaredNorm();
            double global_new_r_sq = 0.0;
            MPI_Allreduce(&local_new_r_sq, &global_new_r_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            double current_r_norm = std::sqrt(global_new_r_sq);            
            // --- 计算共轭方向 beta ---
            double beta = global_new_r_sq / current_global_r_norm_sq;
            p = r + beta * p;
            current_global_r_norm_sq = global_new_r_sq;
            r0 = current_r_norm;
            iter++;
            // ===== 5. 鲁棒性判断逻辑 (仅 Rank 0) =====
            if (rank == 0) {
                double rel_res = r0 / initial_r_norm;
                
                // A. 收敛检查
                if (rel_res < epsilon) {
                    exit_status = 1;
                } 
                // B. 停滞检查 (引入缓冲区和最小迭代保护)
                else if (iter > min_iter_protect) {
                    // 判断相对下降比例是否小于一个极小值
                    if ((prev_r_norm - r0) / prev_r_norm < stagnation_tol) {
                        stagnation_count++;
                    } else {
                        stagnation_count = 0;
                    }

                    if (stagnation_count >= max_stagnation) {
                        exit_status = 2;
                    }
                }
                prev_r_norm = r0;
            }
        }
        // 广播退出状态
        MPI_Bcast(&exit_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (exit_status != 0) break;
    }
    // ===== 6. 日志打印 =====
    if (rank == 0 && verbose == 1) {
        if (exit_status == 1) {
            std::cout << "  [CG] 收敛: 相对残差 " << std::scientific << std::setprecision(3) 
                      << r0 / initial_r_norm << " (" << iter << " iterations)" << std::endl;
        } else if (exit_status == 2) {
            std::cout << "  [CG] 停滞退出: 连续 " << max_stagnation << " 步下降率低于 " 
                      << stagnation_tol << " (Final Rel.Res: " << r0/initial_r_norm << ")"<< " (" << iter << " iterations)" << std::endl;
        } else if (exit_status == 3) {
            std::cout << "  [CG] 错误: 数学失效 (p,Ap) ≈ 0" << std::endl;
        } else {
            std::cout << "  [CG] 达到最大迭代次数. Rel.Res: " << r0/initial_r_norm << " (" << iter << " iterations)" << std::endl;
        }
    }
}
