#include "fluid.h"
#include <filesystem>
#include <chrono>
#include "parallel.h"
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/Dense>
namespace fs = std::filesystem;






int main(int argc, char* argv[]) 
{    
    // 获取输入参数
std::string mesh_folder;
double dt;
int timesteps;
    int n_splits;  // 并行计算线程数
double mu;

    if(argc == 6) {
        // 命令行参数输入
        mesh_folder = argv[1];
        dt = std::stod(argv[2]);
        timesteps = std::stoi(argv[3]);
        mu= std::stod(argv[4]);
        n_splits = std::stoi(argv[5]);
        
    }
    else {
        // 手动输入
        std::cout << "网格文件夹路径:";
        std::cin >> mesh_folder;
        std::cout << "时间步长:";
        std::cin >> dt;
        std::cout << "时间步长数:";
        std::cin >> timesteps;
        std::cout << "并行线程数:";
        std::cin >> n_splits;
        std::cout << "粘度:";
        std::cin >> mu;
    }

    

    // 初始化MPI环境
     MPI_Init(&argc, &argv);
     MPI_Barrier(MPI_COMM_WORLD);
     readParams(mesh_folder, dx, dy);

             // 同步 dx 和 dy 给所有进程
     MPI_Bcast(&dx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
     MPI_Bcast(&dy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
     MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
     MPI_Bcast(&mu, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 加载原始网格
    Mesh original_mesh(mesh_folder);
    // 获取进程信息
    int rank, num_procs;
    // 垂直分割网格
    std::vector<Mesh> sub_meshes = splitMeshVertically(original_mesh, n_splits);
    if (rank==0)
    {
         std::cout << "模式:从命令行读取参数:" << std::endl;
        std::cout << "网格文件夹: " << mesh_folder << std::endl;
        std::cout << "时间步长: " << dt << std::endl;
        std::cout << "时间步数: " << timesteps << std::endl;
        std::cout << "并行进程数: " << n_splits << std::endl;
        std::cout << "粘度: " << mu << std::endl;
        // 打印分割信息
    std::cout << "网格已分割为 " << n_splits << " 个子网格:" << std::endl;
    for(int i = 0; i < sub_meshes.size(); i++) {
        std::cout << "子网格 " << i << " 尺寸: " 
                  << sub_meshes[i].nx << "x" << sub_meshes[i].ny << std::endl;
    }
    
    }
    
   
   
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // 检查进程数是否匹配
    if(num_procs != n_splits) {
        if(rank == 0) {
            std::cerr << "错误: MPI进程数 (" << num_procs 
                      << ") 与指定的并行线程数 (" << n_splits 
                      << ") 不匹配" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }


    // 每个进程获取对应的子网格
    Mesh mesh = sub_meshes[rank];
    //初始化
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

    


    

   
   int nx0,ny0;
   nx0=mesh.nx;
   ny0=mesh.ny;
    //建立u v p的方程
    Equation equ_u(mesh);
    Equation equ_v(mesh);
    Equation equ_p(mesh);
    
   double l2x = 0.0, l2y = 0.0, l2p = 0.0;
    
   auto start_time0 = chrono::steady_clock::now();  // 开始计时


   double alpha_p = 0.1; // 压力松弛因子
   double alpha_uv = 0.3; // 动量松弛因子
   int inter=0;


    for (int i = 0; i <= timesteps; ++i) { 
        
       if(rank==0){ cout<<"时间步长 "<< i <<std::endl;}
        // 切换到当前编号文件夹
      
       //记录上一个时间步长的u v
       inter++;
       int max_outer_iterations=300;
           //simple算法迭代
  
        MPI_Barrier(MPI_COMM_WORLD);
        double init_l2_norm_x = -1.0;
       double init_l2_norm_y = -1.0;
       double init_l2_norm_p = -1.0;
    for(int n=1;n<=max_outer_iterations;n++) {
        MPI_Barrier(MPI_COMM_WORLD);
        //1离散动量方程 
        double l2_norm_x, l2_norm_y;
        mesh.u.setZero();
        mesh.v.setZero();
        equ_v.initializeToZero();
        equ_u.initializeToZero();
        momentum_function_unsteady(mesh,equ_u,equ_v,mu,dt,alpha_uv);
        equ_u.build_matrix();
        equ_v.build_matrix();


    
         //3求解线性方程组
        double epsilon_uv=0.01;
       
             // 同步 dx 和 dy 给所有进程
    
        VectorXd x_v(mesh.internumber),y_v(mesh.internumber);
        x_v.setZero();
        y_v.setZero();
     
       CG_parallel(equ_u,mesh,equ_u.source,x_v,1e-5,25,rank,num_procs,l2_norm_x);
        
        
        CG_parallel(equ_v,mesh,equ_v.source,y_v,1e-5,25,rank,num_procs,l2_norm_y);
      
        vectorToMatrix(x_v,mesh.u,mesh);
        vectorToMatrix(y_v,mesh.v,mesh);
       
        


       

      
        
         

       
        exchangeColumns(mesh.u, rank, num_procs);
    
        exchangeColumns(mesh.v, rank, num_procs);

        exchangeColumns(equ_u.A_p, rank, num_procs);
         
       
        //cell中心速度插值到面 动量插值
        face_velocity(mesh ,equ_u);
        
        MPI_Barrier(MPI_COMM_WORLD);

        double epsilon_p=1e-5;

        //初始化压力修正方程
        equ_p.initializeToZero();

        //离散压力修正方程
        pressure_function(mesh, equ_p, equ_u);
       
        // 组装压力修正方程
        equ_p.build_matrix();
        //求解压力修正方程
        VectorXd p_v(mesh.internumber);

        //初始压力修正场
        mesh.p_prime.setZero();


        p_v.setZero();
      

        //求解压力修正方程
        CG_parallel(equ_p,mesh,equ_p.source,p_v,1e-2,140,rank,num_procs,l2_norm_p);
     
        vectorToMatrix(p_v,mesh.p_prime,mesh);
       
        
         MPI_Barrier(MPI_COMM_WORLD);
         exchangeColumns(mesh.p_prime, rank, num_procs); 
         MPI_Barrier(MPI_COMM_WORLD);
        //压力修正
         correct_pressure(mesh,equ_u,alpha_p);
        
        
        //速度修正
         correct_velocity(mesh,equ_u);
       
        
        
        //更新压力 速度 并交换数值
        mesh.p = mesh.p_star;
        
        
        MPI_Barrier(MPI_COMM_WORLD);
        exchangeColumns(mesh.p, rank, num_procs);
        MPI_Barrier(MPI_COMM_WORLD);
       // 记录初始残差（仅第一次迭代）
if (n == 1) {
    init_l2_norm_x = l2_norm_x;
    init_l2_norm_y = l2_norm_y;
    init_l2_norm_p = l2_norm_p;
}

// 避免除以 0（数值健壮性）
double norm_res_x = (init_l2_norm_x > 1e-200) ? l2_norm_x / init_l2_norm_x : 0.0;
double norm_res_y = (init_l2_norm_y > 1e-200) ? l2_norm_y / init_l2_norm_y : 0.0;
double norm_res_p = (init_l2_norm_p > 1e-200) ? l2_norm_p / init_l2_norm_p : 0.0;
   
// 只在主进程(rank=0)打印残差信息
if(rank == 0) {
    std::cout << scientific 
              << "时间步: " << i 
              << " 迭代轮数: " << n 
              <<"  归一化残差："
              << " u: " << setprecision(4) << norm_res_x
              << " v: " << setprecision(4) << norm_res_y
              << " p " << setprecision(4) << norm_res_p
              <<"  全局残差："
              << " u: " << setprecision(4) << l2_norm_x
              << " v: " << setprecision(4) << l2_norm_y
              << " p " << setprecision(4) << l2_norm_p
              << std::endl;
}

// 检查收敛性
int local_converged = (norm_res_x < 1e-1) && 
                      (norm_res_y < 1e-1) && 
                      (norm_res_p < 1e-3);

// 同步所有进程的收敛状态
int global_converged;
MPI_Allreduce(&local_converged, &global_converged, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

if(global_converged) {
    if(rank == 0) {
        std::cout << "所有进程达到收敛条件" << std::endl;
    }
    break;
}
    MPI_Barrier(MPI_COMM_WORLD);
    }
    //时间推进
    saveMeshData(mesh,rank);
    mesh.u0 = mesh.u_star;
    mesh.v0 = mesh.v_star;
    }
    
   
    // 最后显示实际计算总耗时
    auto total_elapsed_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time0).count();
    std::cout << "\n计算完成 总耗时: " << total_elapsed_time << "秒" << std::endl;
  
    saveMeshData(mesh,rank);
    MPI_Finalize();
    
     
    return 0;
}