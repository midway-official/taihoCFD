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
    const int count = rows * 2; // 每次交换2列

    // 确定邻居
    int left_rank  = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int right_rank = (rank == num_procs - 1) ? MPI_PROC_NULL : rank + 1;

    // 分配缓冲区：Eigen 默认是 ColMajor，列内数据连续
    // 如果 matrix 是 RowMajor，建议直接使用 matrix.block(...)
    VectorXd send_left = Map<VectorXd>(matrix.block(0, 2, rows, 2).data(), count);
    VectorXd send_right = Map<VectorXd>(matrix.block(0, cols - 4, rows, 2).data(), count);
    VectorXd recv_left(count), recv_right(count);

    // 使用 Sendrecv 一步到位，自动处理非阻塞逻辑，简洁且安全
    // 向左发，从左收
    MPI_Sendrecv(send_left.data(),  count, MPI_DOUBLE, left_rank,  0,
                 recv_left.data(),  count, MPI_DOUBLE, left_rank,  1, 
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // 向右发，从右收
    MPI_Sendrecv(send_right.data(), count, MPI_DOUBLE, right_rank, 1,
                 recv_right.data(), count, MPI_DOUBLE, right_rank, 0, 
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // 写回数据
    if (left_rank != MPI_PROC_NULL)
        matrix.block(0, 0, rows, 2) = Map<MatrixXd>(recv_left.data(), rows, 2);
    
    if (right_rank != MPI_PROC_NULL)
        matrix.block(0, cols - 2, rows, 2) = Map<MatrixXd>(recv_right.data(), rows, 2);
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
void Parallel_correction(Mesh& mesh,Equation& equ,MatrixXd &phi1,MatrixXd &phi2){
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
void Parallel_correction2(Mesh& mesh,Equation& equ,MatrixXd &phi1,MatrixXd &phi2){
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
    const SparseMatrix<double>& A = equ.A;  // 零拷贝

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
    MatrixXd p_field = MatrixXd::Zero(mesh.ny, mesh.nx);
    MatrixXd Ap_field = MatrixXd::Zero(mesh.ny, mesh.nx);
    // ===== 4. CG 迭代 =====
    while (iter < max_iter) {
        p_field.setZero();    
        Ap_field.setZero();   
        // --- 计算 Ap ---
        Ap = A * p; 
        

        
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

void PCG_parallel(Equation& equ, Mesh mesh, VectorXd& b, VectorXd& x, 
                 double epsilon, int max_iter, int rank, int num_procs,
                 double& r0, int verbose) {
    
    int n = equ.A.rows();
    const SparseMatrix<double>& A = equ.A;  // 零拷贝

    // ★ 1. 构建Jacobi预条件（仅一次，O(N)，无通信）
    VectorXd inv_diag(n);
    for (int i = 0; i < mesh.ny; i++) {
        for (int j = 0; j < mesh.nx; j++) {
            if (mesh.bctype(i, j) == 0) {
                int idx = mesh.interid(i, j);
                double d = equ.A_p(i, j);
                inv_diag[idx] = (std::abs(d) > 1e-14) ? 1.0/d : 1.0;
            }
        }
    }

    // ===== 初始化残差（与原来完全相同）=====
    VectorXd r = b - A * x;
    MatrixXd r_field = MatrixXd::Zero(mesh.ny, mesh.nx);
    MatrixXd x_field = MatrixXd::Zero(mesh.ny, mesh.nx);
    vectorToMatrix(r, r_field, mesh);
    vectorToMatrix(x, x_field, mesh);
    exchangeColumns(x_field, rank, num_procs);
    Parallel_correction2(mesh, equ, r_field, x_field);
    matrixToVector(r_field, r, mesh);

    // ★ 2. 初始化：p = z = M⁻¹r，而非 p = r
    VectorXd z = inv_diag.cwiseProduct(r);   // 本地操作，无通信
    VectorXd p = z;
    VectorXd Ap = VectorXd::Zero(n);

    // ★ 3. 初始内积改为 r·z（PCG用r·z，CG用r·r）
    double local_rz       = r.dot(z);
    double local_r_norm_sq = r.squaredNorm();
    double local_b_norm_sq = b.squaredNorm();

    double global_rz = 0.0, global_r_norm_sq = 0.0, global_b_norm_sq = 0.0;

    // ★ 三个Allreduce合并为一次（减少同步开销）
    double local_buf3[3]  = {local_rz, local_r_norm_sq, local_b_norm_sq};
    double global_buf3[3] = {0, 0, 0};
    MPI_Allreduce(local_buf3, global_buf3, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    global_rz        = global_buf3[0];
    global_r_norm_sq = global_buf3[1];
    global_b_norm_sq = global_buf3[2];

    double initial_r_norm = std::sqrt(global_r_norm_sq);
    r0 = initial_r_norm;

    if (rank == 0 && verbose == 1)
        std::cout << "  [PCG] r0 " << r0 << std::endl;

    if (initial_r_norm < 1e-15 || 
        initial_r_norm / std::sqrt(global_b_norm_sq + 1e-16) < epsilon) {
        if (rank == 0 && verbose) 
            std::cout << "  [PCG] 初始残差已达标。" << std::endl;
        return;
    }

    // 停滞检测（与原来相同）
    double prev_r_norm = r0;
    int stagnation_count = 0;
    const int max_stagnation = 3;
    const double stagnation_tol = 1e-4;
    const int min_iter_protect = 5;
    int exit_status = 0, iter = 0;

    double current_rz = global_rz;  // ★ 保存当前 r·z（替代原来的 r·r）
    MatrixXd p_field  = MatrixXd::Zero(mesh.ny, mesh.nx);
    MatrixXd Ap_field = MatrixXd::Zero(mesh.ny, mesh.nx);
    // ===== PCG 迭代 =====
    while (iter < max_iter) {
        p_field.setZero();    
        Ap_field.setZero();  
        // Ap计算（与原来完全相同）
        Ap = A * p;

        vectorToMatrix(p,  p_field,  mesh);
        vectorToMatrix(Ap, Ap_field, mesh);
        exchangeColumns(p_field, rank, num_procs);
        Parallel_correction(mesh, equ, Ap_field, p_field);
        matrixToVector(Ap_field, Ap, mesh);

        // ★ alpha分母是 p·Ap，分子是 r·z（不是r·r）
        double local_pAp = p.dot(Ap);
        double global_pAp = 0.0;
        MPI_Allreduce(&local_pAp, &global_pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (std::abs(global_pAp) < 1e-18) { exit_status = 3; }

        if (exit_status == 0) {
            double alpha = current_rz / global_pAp;  // ★ 分子用r·z

            x += alpha * p;
            r -= alpha * Ap;

            // ★ 每轮更新z（本地操作，无通信！）
            z = inv_diag.cwiseProduct(r);

            // ★ 合并两个Allreduce：同时算新的r·z和‖r‖²
            double local_buf2[2]  = { r.dot(z), r.squaredNorm() };
            double global_buf2[2] = { 0.0, 0.0 };
            MPI_Allreduce(local_buf2, global_buf2, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            double new_rz      = global_buf2[0];
            double new_r_norm  = std::sqrt(global_buf2[1]);

            // ★ beta用新旧r·z之比（PCG公式）
            double beta = new_rz / current_rz;
            p = z + beta * p;   // ★ 用z更新p，不用r

            current_rz = new_rz;
            r0 = new_r_norm;
            iter++;

            // 收敛/停滞判断（与原来相同）
            if (rank == 0) {
                double rel_res = r0 / initial_r_norm;
                if (rel_res < epsilon) {
                    exit_status = 1;
                } else if (iter > min_iter_protect) {
                    if ((prev_r_norm - r0) / prev_r_norm < stagnation_tol)
                        stagnation_count++;
                    else
                        stagnation_count = 0;
                    if (stagnation_count >= max_stagnation) exit_status = 2;
                }
                prev_r_norm = r0;
            }
        }

        MPI_Bcast(&exit_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (exit_status != 0) break;
    }

    // ===== 6. 日志打印 =====
    if (rank == 0 && verbose == 1) {
        if (exit_status == 1) {
            std::cout << "  [PCG] 收敛: 相对残差 " << std::scientific << std::setprecision(3) 
                      << r0 / initial_r_norm << " (" << iter << " iterations)" << std::endl;
        } else if (exit_status == 2) {
            std::cout << "  [PCG] 停滞退出: 连续 " << max_stagnation << " 步下降率低于 " 
                      << stagnation_tol << " (Final Rel.Res: " << r0/initial_r_norm << ")"<< " (" << iter << " iterations)" << std::endl;
        } else if (exit_status == 3) {
            std::cout << "  [PCG] 错误: 数学失效 (p,Ap) ≈ 0" << std::endl;
        } else {
            std::cout << "  [PCG] 达到最大迭代次数. Rel.Res: " << r0/initial_r_norm << " (" << iter << " iterations)" << std::endl;
        }
    }
}
void solveFieldCG(
    Equation& equ,
    Mesh& mesh,
    MatrixXd& field,
    double tol,
    int max_iter,
    int rank,
    int num_procs,
    double& l2_norm,
    int verbose
)
{
    VectorXd x(mesh.internumber);
    x.setZero();

    CG_parallel(equ, mesh, equ.source, x,
                tol, max_iter,
                rank, num_procs,
                l2_norm, verbose);

    vectorToMatrix(x, field, mesh);


    exchangeColumns(field, rank, num_procs);
    
}


void solveFieldPCG(
    Equation& equ,
    Mesh& mesh,
    MatrixXd& field,
    double tol,
    int max_iter,
    int rank,
    int num_procs,
    double& l2_norm,
    int verbose
)
{
    VectorXd x(mesh.internumber);
    x.setZero();

    PCG_parallel(equ, mesh, equ.source, x,
                tol, max_iter,
                rank, num_procs,
                l2_norm, verbose);

    vectorToMatrix(x, field, mesh);


    exchangeColumns(field, rank, num_procs);
    
}