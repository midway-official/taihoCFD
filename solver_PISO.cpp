#include "fluid.h"
#include <filesystem>
#include <chrono>
#include "parallel.h"
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseLU>

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
    double dt;
    double mu;
    int timesteps, n_splits;
    
    // 读取输入参数(仅rank 0)
    if (rank == 0) {
        parseInputParameters_unsteady(argc, argv, mesh_folder, dt, timesteps, mu, n_splits);
    }
    
    // 广播参数到所有进程
    broadcastParameters_unsteady(mesh_folder, dt, timesteps, mu, n_splits, rank);
    
    // 读取并同步网格参数
    readParams(mesh_folder, dx, dy);
    MPI_Bcast(&dx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // 验证参数一致性
    verifyParameterConsistency_unsteady(mesh_folder, dt, timesteps, mu, n_splits, rank, num_procs);
    
    // -------------------- 网格分割 --------------------
    Mesh original_mesh(mesh_folder);
    std::vector<Mesh> sub_meshes = splitMeshVertically(original_mesh, n_splits);
    
    if (rank == 0) {
        printSimulationSetup_unsteady(sub_meshes, n_splits, dt, timesteps, rank);
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
    const double alpha_p = 0.3;   // PISO压力松弛因子
    const double tol_uv = 1e-2;   // 速度求解精度
    const double tol_p = 1e-2;    // 压力求解精度
    const int max_iter_uv = 35;   // 速度最大迭代次数
    const int max_iter_p = 140;   // 压力最大迭代次数
    const int max_piso_correctors = 4;  // PISO压力修正次数
    
    auto start_time = std::chrono::steady_clock::now();
    
    if (rank == 0) {
        std::cout << "\n==================== 开始PISO计算 ====================" << std::endl;
        std::cout << "算法: PISO (Pressure Implicit with Splitting of Operators)" << std::endl;
        std::cout << "压力修正次数: " << max_piso_correctors << std::endl;
        std::cout << "======================================================\n" << std::endl;
    }
    
    // ==================== 时间推进主循环 ====================
    for (int time_step = 0; time_step <= timesteps; ++time_step) {
        
        if (rank == 0) {
            std::cout << "\n-------------------- 时间步: " << time_step 
                      << " / " << timesteps << " --------------------" << std::endl;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        double init_l2_norm_x = -1.0;
        double init_l2_norm_y = -1.0;
        double init_l2_norm_p = -1.0;
        
        double l2_norm_x, l2_norm_y, l2_norm_p;
        
        // ==================== 步骤1: 求解动量预测方程 ====================
        equ_u.initializeToZero();
        equ_v.initializeToZero();
        
        // 离散PISO动量方程(不含压力梯度项)
        momentum_function_PISO(mesh, equ_u, equ_v, mu, dt);
        
        // 组装动量方程
        equ_u.build_matrix();
        equ_v.build_matrix();
        
        // 求解u和v速度场
        VectorXd x_v(mesh.internumber), y_v(mesh.internumber);
        x_v.setZero();
        y_v.setZero();
        
        CG_parallel(equ_u, mesh, equ_u.source, x_v, tol_uv, max_iter_uv, 
                    rank, num_procs, l2_norm_x);
        CG_parallel(equ_v, mesh, equ_v.source, y_v, tol_uv, max_iter_uv, 
                    rank, num_procs, l2_norm_y);
        
        vectorToMatrix(x_v, mesh.u, mesh);
        vectorToMatrix(y_v, mesh.v, mesh);
        
        // 交换边界数据
        exchangeColumns(mesh.u, rank, num_procs);
        exchangeColumns(mesh.v, rank, num_procs);
        exchangeColumns(equ_u.A_p, rank, num_procs);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // ==================== PISO压力修正循环 ====================
        for (int corrector = 1; corrector <= max_piso_correctors; corrector++) {
            
            // -------------------- 步骤2: 速度插值到面 --------------------
            face_velocity(mesh, equ_u);
            MPI_Barrier(MPI_COMM_WORLD);
            
            // -------------------- 步骤3: 求解压力修正方程 --------------------
            equ_p.initializeToZero();
            pressure_function(mesh, equ_p, equ_u);
            equ_p.build_matrix();
            
            mesh.p_prime.setZero();
            VectorXd p_v(mesh.internumber);
            p_v.setZero();
            
            CG_parallel(equ_p, mesh, equ_p.source, p_v, tol_p, max_iter_p, 
                        rank, num_procs, l2_norm_p);
            
            vectorToMatrix(p_v, mesh.p_prime, mesh);
            
            MPI_Barrier(MPI_COMM_WORLD);
            exchangeColumns(mesh.p_prime, rank, num_procs);
            MPI_Barrier(MPI_COMM_WORLD);
            
            // -------------------- 步骤4: 修正压力和速度 --------------------
            correct_pressure(mesh, equ_u, alpha_p);
            correct_velocity(mesh, equ_u);
            
            // 更新压力和速度场
            mesh.p = mesh.p_star;
            mesh.u = mesh.u_star;
            mesh.v = mesh.v_star;
            
            // 交换更新后的场数据
            exchangeColumns(mesh.u, rank, num_procs);
            exchangeColumns(mesh.v, rank, num_procs);
            exchangeColumns(mesh.p, rank, num_procs);
            exchangeColumns(mesh.u_face, rank, num_procs);
            exchangeColumns(mesh.v_face, rank, num_procs);
            
            MPI_Barrier(MPI_COMM_WORLD);
            
            // -------------------- 步骤5: 残差监控 --------------------

            // 仅rank 0打印残差信息
            if (rank == 0) {
                std::cout << std::scientific 
                          << "  修正 " << std::setw(1) << corrector 

                          << " | 全局 → "
                          << " u: " << std::setprecision(3) << l2_norm_x
                          << " v: " << std::setprecision(3) << l2_norm_y
                          << " p: " << std::setprecision(3) << l2_norm_p
                          << std::endl;
            }
            
            // 检查收敛(PISO通常不需要提前退出,执行固定次数修正)
            int local_converged = checkConvergence(l2_norm_x, l2_norm_y, l2_norm_p);
            int global_converged;
            MPI_Allreduce(&local_converged, &global_converged, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            
            if (global_converged) {
                if (rank == 0) {
                    std::cout << "  ✓ 时间步 " << time_step 
                              << " 收敛 (PISO修正: " << corrector << ")" << std::endl;
                }
                break;
            }
            
            MPI_Barrier(MPI_COMM_WORLD);
        }
        
        // -------------------- 步骤6: 时间推进 --------------------
        // 保存当前时间步数据
        saveMeshData(mesh, rank);
        
        // 更新上一时间步速度场
        mesh.u0 = mesh.u;
        mesh.v0 = mesh.v;
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // ==================== 计算完成 ====================
    saveMeshData(mesh, rank);
    
    auto total_elapsed_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();
    
    if (rank == 0) {
        std::cout << "\n==================== 计算完成 ====================" << std::endl;
        std::cout << "总时间步数: " << timesteps + 1 << std::endl;
        std::cout << "总耗时: " << total_elapsed_time << " 秒" << std::endl;
        std::cout << "平均每步: " << total_elapsed_time / (timesteps + 1) << " 秒" << std::endl;
        std::cout << "===================================================\n" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}