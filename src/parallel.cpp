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
    VectorXd send_left = Map<VectorXd>(matrix.block(0, 2, rows, 2).data(), count);
    VectorXd send_right = Map<VectorXd>(matrix.block(0, cols - 4, rows, 2).data(), count);
    VectorXd recv_left(count), recv_right(count);

    // 使用 Sendrecv 
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
    // 注意：exchangeColumns + Parallel_correction2 是针对并行分区边界的
    // GHOST CELL列修正，必须保留，否则跨进程的 Ap 计算会有边界误差。
    VectorXd r = b - A * x;
    MatrixXd r_field = MatrixXd::Zero(mesh.ny, mesh.nx);
    MatrixXd x_field = MatrixXd::Zero(mesh.ny, mesh.nx);
    vectorToMatrix(r, r_field, mesh);
    vectorToMatrix(x, x_field, mesh);
    exchangeColumns(x_field, rank, num_procs);
    Parallel_correction2(mesh, equ, r_field, x_field);
    matrixToVector(r_field, r, mesh);

    // ===== 2. 初始化搜索方向 p = r =====
    VectorXd p  = r;
    VectorXd Ap = VectorXd::Zero(n);

    // ===== 3. 计算全局初始状态（两个 Allreduce 合并为一次）=====
    double local_buf2[2]  = { r.squaredNorm(), b.squaredNorm() };
    double global_buf2[2] = { 0.0, 0.0 };
    MPI_Allreduce(local_buf2, global_buf2, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double current_r_sq   = global_buf2[0];
    double initial_r_norm = std::sqrt(current_r_sq);  // 只写一次，全程不变
    double b_norm         = std::sqrt(global_buf2[1]);

    // 迭代中用 current_r_norm 追踪残差，所有进程同步持有
    double current_r_norm = initial_r_norm;

    if (rank == 0 && verbose == 1)
        std::cout << "  [CG] r0 = " << initial_r_norm << std::endl;

    if (initial_r_norm < 1e-15 ||
        initial_r_norm / (b_norm + 1e-16) < epsilon) {
        if (rank == 0 && verbose)
            std::cout << "  [CG] 初始残差已达标。" << std::endl;
        r0 = initial_r_norm;  // ★ FIX2：输出初始残差
        return;
    }

    // 停滞检测
    double prev_r_norm   = current_r_norm;
    int stagnation_count = 0;
    const int    max_stagnation   = 3;
    const double stagnation_tol   = 1e-6;
    const int    min_iter_protect = 5;

    int exit_status = 0, iter = 0;
    MatrixXd p_field  = MatrixXd::Zero(mesh.ny, mesh.nx);
    MatrixXd Ap_field = MatrixXd::Zero(mesh.ny, mesh.nx);

    // ===== 4. CG 迭代 =====
    while (iter < max_iter) {

        // ── Ap 计算 ──────────────────────────────────────────────
        Ap = A * p;
        vectorToMatrix(p,  p_field,  mesh);
        vectorToMatrix(Ap, Ap_field, mesh);
        exchangeColumns(p_field, rank, num_procs);
        Parallel_correction(mesh, equ, Ap_field, p_field);
        matrixToVector(Ap_field, Ap, mesh);

        // ── 检测 (p, Ap) ≈ 0 ──────────────────────────────────────
        double local_pAp  = p.dot(Ap);
        double global_pAp = 0.0;
        MPI_Allreduce(&local_pAp, &global_pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // 检测到数学失效后立即广播并退出，不再绕行
        if (std::abs(global_pAp) < 1e-35) {
            exit_status = 3;
            MPI_Bcast(&exit_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
            break;
        }

        // ── 更新 x, r ─────────────────────────────────────────────
        double alpha = current_r_sq / global_pAp;
        x += alpha * p;
        r -= alpha * Ap;

        // ── Allreduce：新 ‖r‖² ───────────────────────────────────
        double local_new_r_sq  = r.squaredNorm();
        double global_new_r_sq = 0.0;
        MPI_Allreduce(&local_new_r_sq, &global_new_r_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // current_r_norm 由 Allreduce 结果赋值，所有进程同步持有
        current_r_norm = std::sqrt(global_new_r_sq);

        // ── 更新 p（CG 公式）──────────────────────────────────────
        double beta = global_new_r_sq / current_r_sq;
        p = r + beta * p;
        current_r_sq = global_new_r_sq;
        iter++;

        // ── 收敛 / 停滞判断（rank0 负责逻辑，结果广播）─────────────
        //判断基于所有进程同步后的 current_r_norm，语义明确
        if (rank == 0) {
            double rel_res = current_r_norm / initial_r_norm;
            if (rel_res < epsilon) {
                exit_status = 1;
            } else if (iter > min_iter_protect) {
                double drop_rate = (prev_r_norm - current_r_norm) / prev_r_norm;
                stagnation_count = (drop_rate < stagnation_tol)
                                   ? stagnation_count + 1 : 0;
                if (stagnation_count >= max_stagnation)
                    exit_status = 2;
            }
            prev_r_norm = current_r_norm;
        }

        MPI_Bcast(&exit_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (exit_status != 0) break;
    }

    //函数结束时统一写回输出参数（最终残差），只写这一次
    r0 = current_r_norm;

    // ===== 5. 日志打印 =====
    if (rank == 0 && verbose == 1) {
        double rel_res = r0 / initial_r_norm;
        if (exit_status == 1) {
            std::cout << "  [CG] 收敛: 相对残差 " << std::scientific
                      << std::setprecision(3) << rel_res
                      << " (" << iter << " iterations)" << std::endl;
        } else if (exit_status == 2) {
            std::cout << "  [CG] 停滞退出: 连续 " << max_stagnation
                      << " 步下降率低于 " << stagnation_tol
                      << " (Final Rel.Res: " << rel_res << ")"
                      << " (" << iter << " iterations)" << std::endl;
        } else if (exit_status == 3) {
            std::cout << "  [CG] 错误: 数学失效 (p,Ap) ≈ 0" << std::endl;
        } else {
            std::cout << "  [CG] 达到最大迭代次数. Rel.Res: " << rel_res
                      << " (" << iter << " iterations)" << std::endl;
        }
    }
}

void PCG_parallel(Equation& equ, Mesh mesh, VectorXd& b, VectorXd& x,
                 double epsilon, int max_iter, int rank, int num_procs,
                 double& r0, int verbose) {

    int n = equ.A.rows();
    const SparseMatrix<double>& A = equ.A;

    // 构建Jacobi预条件（仅一次，O(N)，无通信）
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

    // ===== 初始化残差 =====
    // 注意：exchangeColumns + Parallel_correction2 是针对并行分区边界的
    // GHOST CELL列修正，必须保留，否则跨进程的 Ap 计算会有边界误差。
    VectorXd r = b - A * x;
    MatrixXd r_field = MatrixXd::Zero(mesh.ny, mesh.nx);
    MatrixXd x_field = MatrixXd::Zero(mesh.ny, mesh.nx);
    vectorToMatrix(r, r_field, mesh);
    vectorToMatrix(x, x_field, mesh);
    exchangeColumns(x_field, rank, num_procs);
    Parallel_correction2(mesh, equ, r_field, x_field);
    matrixToVector(r_field, r, mesh);

    // 初始化：p = z = M⁻¹r
    VectorXd z  = inv_diag.cwiseProduct(r);
    VectorXd p  = z;
    VectorXd Ap = VectorXd::Zero(n);

    // 初始内积（三个Allreduce合并为一次）
    double local_buf3[3]  = { r.dot(z), r.squaredNorm(), b.squaredNorm() };
    double global_buf3[3] = { 0.0, 0.0, 0.0 };
    MPI_Allreduce(local_buf3, global_buf3, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double current_rz    = global_buf3[0];
    double initial_r_norm = std::sqrt(global_buf3[1]);   // 只写这一次，不再修改
    double b_norm         = std::sqrt(global_buf3[2]);

    // r0 仅作输出，记录初始残差；迭代过程用独立变量 current_r_norm
    double current_r_norm = initial_r_norm;  // 所有进程均持有，随迭代同步更新

    if (rank == 0 && verbose == 1)
        std::cout << "  [PCG] r0 = " << initial_r_norm << std::endl;

    if (initial_r_norm < 1e-15 ||
        initial_r_norm / (b_norm + 1e-16) < epsilon) {
        if (rank == 0 && verbose)
            std::cout << "  [PCG] 初始残差已达标。" << std::endl;
        r0 = initial_r_norm;   // 输出初始残差
        return;
    }

    // 停滞检测
    double prev_r_norm    = current_r_norm;
    int stagnation_count  = 0;
    const int    max_stagnation  = 3;
    const double stagnation_tol  = 1e-6;
    const int    min_iter_protect = 5;
    int exit_status = 0, iter = 0;

    MatrixXd p_field  = MatrixXd::Zero(mesh.ny, mesh.nx);
    MatrixXd Ap_field = MatrixXd::Zero(mesh.ny, mesh.nx);

    // ===== PCG 迭代 =====
    while (iter < max_iter) {

        // ── Ap 计算 ──────────────────────────────────────────────
        Ap = A * p;
        vectorToMatrix(p,  p_field,  mesh);
        vectorToMatrix(Ap, Ap_field, mesh);
        exchangeColumns(p_field, rank, num_procs);
        Parallel_correction(mesh, equ, Ap_field, p_field);
        matrixToVector(Ap_field, Ap, mesh);

        // ── 检测 (p, Ap) ≈ 0 ──────────────────────────────────────
        double local_pAp = p.dot(Ap);
        double global_pAp = 0.0;
        MPI_Allreduce(&local_pAp, &global_pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // 检测到数学失效后立即广播并退出，不再绕行
        if (std::abs(global_pAp) < 1e-35) {
            exit_status = 3;
            MPI_Bcast(&exit_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
            break;
        }

        // ── 更新 x, r, z ──────────────────────────────────────────
        double alpha = current_rz / global_pAp;
        x += alpha * p;
        r -= alpha * Ap;
        z  = inv_diag.cwiseProduct(r);   // 本地操作，无通信

        // ── 合并 Allreduce：新 r·z 和 ‖r‖² ──────────────────────
        double local_buf2[2]  = { r.dot(z), r.squaredNorm() };
        double global_buf2[2] = { 0.0, 0.0 };
        MPI_Allreduce(local_buf2, global_buf2, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double new_rz = global_buf2[0];
        //current_r_norm 由所有进程同步更新，rank0 不再独享
        current_r_norm = std::sqrt(global_buf2[1]);

        // ── 更新 p（PCG 公式）─────────────────────────────────────
        double beta = new_rz / current_rz;
        p = z + beta * p;
        current_rz = new_rz;
        iter++;

        // ── 收敛 / 停滞判断（rank0 负责逻辑，结果广播）─────────────
        //判断基于所有进程同步后的 current_r_norm，语义明确
        if (rank == 0) {
            double rel_res = current_r_norm / initial_r_norm;
            if (rel_res < epsilon) {
                exit_status = 1;
            } else if (iter > min_iter_protect) {
                double drop_rate = (prev_r_norm - current_r_norm) / prev_r_norm;
                stagnation_count = (drop_rate < stagnation_tol)
                                   ? stagnation_count + 1 : 0;
                if (stagnation_count >= max_stagnation)
                    exit_status = 2;
            }
            prev_r_norm = current_r_norm;
        }

        MPI_Bcast(&exit_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (exit_status != 0) break;
    }

    // 函数结束时统一写回输出参数，语义清晰（最终残差）
    r0 = current_r_norm;

    // ===== 日志打印 =====
    if (rank == 0 && verbose == 1) {
        double rel_res = r0 / initial_r_norm;
        if (exit_status == 1) {
            std::cout << "  [PCG] 收敛: 相对残差 " << std::scientific
                      << std::setprecision(3) << rel_res
                      << " (" << iter << " iterations)" << std::endl;
        } else if (exit_status == 2) {
            std::cout << "  [PCG] 停滞退出: 连续 " << max_stagnation
                      << " 步下降率低于 " << stagnation_tol
                      << " (Final Rel.Res: " << rel_res << ")"
                      << " (" << iter << " iterations)" << std::endl;
        } else if (exit_status == 3) {
            std::cout << "  [PCG] 错误: 数学失效 (p,Ap) ≈ 0" << std::endl;
        } else {
            std::cout << "  [PCG] 达到最大迭代次数. Rel.Res: " << rel_res
                      << " (" << iter << " iterations)" << std::endl;
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