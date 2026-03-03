#include "fluid.h"
#include <filesystem>
#include <chrono>
#include "parallel.h"
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/Dense>

namespace fs = std::filesystem;


// ==================== 主函数 ====================
int main(int argc, char* argv[]) 
{    
    // 初始化MPI环境
    MPI_Init(&argc, &argv);
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    // -------------------- 参数设置 --------------------
    std::string mesh_folder;
    double dt = 0.1;
    double mu;
    int timesteps, n_splits;
    MPI_Comm_size(MPI_COMM_WORLD, &n_splits);  // 获取 MPI 总进程数
    // 读取输入参数(仅rank 0)
    if (rank == 0) {
        parseInputParameters(argc, argv, mesh_folder, timesteps, mu, n_splits);
    }
    
    // 广播参数到所有进程
    broadcastParameters(mesh_folder, dt, timesteps, mu, n_splits, rank);
    // 验证参数一致性
    verifyParameterConsistency(mesh_folder, dt, timesteps, mu, n_splits, rank, num_procs);
    
    // -------------------- 网格分割 --------------------
    Mesh original_mesh(mesh_folder);
    
    std::vector<Mesh> sub_meshes = splitMeshVertically(original_mesh, n_splits);
    if (rank == 0) {
        printSimulationSetup(sub_meshes, n_splits);
    }
    
    // 检查进程数匹配
    if (num_procs != n_splits) {
        if (rank == 0) {
            std::cerr << "错误: MPI进程数(" << num_procs 
                      << ") 与并行线程数(" << n_splits << ")不匹配" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // 每个进程获取对应子网格
    Mesh mesh = sub_meshes[rank];
    // -------------------- 初始化场变量 --------------------
    mesh.u0.setZero();
    mesh.v0.setZero();
    mesh.u_star.setZero();
    mesh.v_star.setZero();
    mesh.u_face.setZero();
    mesh.v_face.setZero(); 
    mesh.u.setZero();
    mesh.v.setZero();
    mesh.p.setZero();
    mesh.p_prime.setZero();
    mesh.p_star.setZero();
   
    // -------------------- 建立方程系统 --------------------
    Equation equ_u(mesh);
    Equation equ_v(mesh);
    Equation equ_p(mesh);
    
    // -------------------- 求解参数设置 --------------------
    const double alpha_p = 0.1;   // 压力松弛因子
    const double alpha_uv = 0.3;  // 动量松弛因子
    const double tol_uv = 1e-7;   // 速度求解精度
    const double tol_p = 1e-7;    // 压力求解精度
    const int max_iter_uv = 25;   // 速度最大迭代次数
    const int max_iter_p = 200;   // 压力最大迭代次数
    const double stagnation_tol = 1e-4;   // 0.01% 停滞阈值

    double l2_norm_x, l2_norm_y, l2_norm_p;
    double prev_l2_u = -1.0;
    double prev_l2_v = -1.0;
    double prev_l2_p = -1.0;
    auto start_time = std::chrono::steady_clock::now();
    
    if (rank == 0) {
        std::cout << "\n==================== 开始SIMPLE迭代 ====================" << std::endl;
    }
    
    // ==================== SIMPLE算法主循环 ====================
    for (int n = 1; n <= timesteps; n++) {
        
        // -------------------- 步骤1: 求解动量方程 --------------------
        // 离散动量方程
        momentum_function(mesh, equ_u, equ_v, mu, alpha_uv);

        //解速度场
        solveFieldPCG(equ_u, mesh, mesh.u,
             tol_uv, max_iter_uv,
             rank, num_procs,
             l2_norm_x,1);

        solveFieldPCG(equ_v, mesh, mesh.v,
             tol_uv, max_iter_uv,
             rank, num_procs,
             l2_norm_y, 1);
        //交换Ap 用于动量插值
        exchangeColumns(equ_u.A_p, rank, num_procs);
        

        
        // -------------------- 步骤2: 速度插值到面 --------------------
        face_velocity(mesh, equ_u);

        
        // -------------------- 步骤3: 求解压力修正方程 --------------------
        
        pressure_function(mesh, equ_p, equ_u);
        
        
        solveFieldPCG(equ_p, mesh, mesh.p_prime,
             tol_p, max_iter_p,
             rank, num_procs,
             l2_norm_p, 1);

        
        // -------------------- 步骤4: 修正压力和速度 --------------------
        correct_pressure(mesh, alpha_p);
        correct_velocity(mesh, equ_u);
        
        // 更新压力场,并交换
        mesh.p = mesh.p_star;
        exchangeColumns(mesh.p, rank, num_procs);

        // -------------------- 步骤5: 收敛性检查 --------------------
 
        // 仅rank 0打印残差信息
        if (rank == 0) {
            std::cout << std::scientific 
                      << "迭代 " << std::setw(4) << n 
                      << " | 全局残差 → "
                      << " u: " << std::setprecision(4) << l2_norm_x
                      << " v: " << std::setprecision(4) << l2_norm_y
                      << " p: " << std::setprecision(4) << l2_norm_p
                      << std::endl;
        }
            // ==================== 时间步内停滞判断 ====================
            int local_stagnated = 0;

            if (n > 1) {
            double du = std::abs(l2_norm_x - prev_l2_u) / (prev_l2_u + 1e-20);
            double dv = std::abs(l2_norm_y - prev_l2_v) / (prev_l2_v + 1e-20);
            double dp = std::abs(l2_norm_p - prev_l2_p) / (prev_l2_p + 1e-20);

            double max_change = std::max({du, dv, dp});

            if (max_change < stagnation_tol) {
            local_stagnated = 1;
            }
            }

            // 记录本次残差，供下一次比较
            prev_l2_u = l2_norm_x;
            prev_l2_v = l2_norm_y;
            prev_l2_p = l2_norm_p;

            // 全局同步判断
            int global_stagnated = 0;
            MPI_Allreduce(&local_stagnated, &global_stagnated,
              1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

            if (global_stagnated) {
                if (rank == 0) {
                  std::cout << "  SIMPLE 停滞退出：残差变化 < "
                  << stagnation_tol * 100.0 << "%，迭代步 = "
                  << n << std::endl;
            }
            break;
            }        
        // 检查全局收敛
        int local_converged = checkConvergence(l2_norm_x, l2_norm_y, l2_norm_p);
        int global_converged;
        MPI_Allreduce(&local_converged, &global_converged, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        
        if (global_converged) {
            if (rank == 0) {
                std::cout << "\n✓ 所有进程达到收敛条件 (迭代次数: " << n << ")" << std::endl;
            }
            break;
        }
        
        // -------------------- 步骤6: 数据保存 --------------------
        if (n % 5 == 0) {
            saveMeshData(mesh, rank);
        }
    }
    
    // ==================== 计算完成 ====================
    saveMeshData(mesh, rank);
    
    auto total_elapsed_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();
    
    if (rank == 0) {
        std::cout << "\n==================== 计算完成 ====================" << std::endl;
        std::cout << "总耗时: " << total_elapsed_time << " 秒" << std::endl;
        std::cout << "===================================================\n" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}
