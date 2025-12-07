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
double dt=0.1;
int timesteps;
    int n_splits;  // 并行计算线程数
double mu;

    if(argc == 5) {
        // 命令行参数输入
        mesh_folder = argv[1];
        
        timesteps = std::stoi(argv[2]);
        mu= std::stod(argv[3]);
        n_splits = std::stoi(argv[4]);
        std::cout << "从命令行读取参数:" << std::endl;
        std::cout << "网格文件夹: " << mesh_folder << std::endl;
        std::cout << "步数: " << timesteps << std::endl;
        std::cout << "并行线程数: " << n_splits << std::endl;
        std::cout << "粘度: " << mu << std::endl;

    }
    else {
        // 手动输入
        std::cout << "网格文件夹路径:";
        std::cin >> mesh_folder;
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
       // 获取进程信息
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
     readParams(mesh_folder, dx, dy);

             // 同步 dx 和 dy 给所有进程
     MPI_Bcast(&dx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
     MPI_Bcast(&dy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    // 同步字符串长度
int folder_length;
if (rank == 0) folder_length = mesh_folder.size();
MPI_Bcast(&folder_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

// 同步字符串内容
char* folder_cstr = new char[folder_length + 1];
if (rank == 0) strcpy(folder_cstr, mesh_folder.c_str());
MPI_Bcast(folder_cstr, folder_length + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
if (rank != 0) mesh_folder = std::string(folder_cstr);
delete[] folder_cstr;

// 广播参数
MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Bcast(&timesteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(&mu, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Bcast(&n_splits, 1, MPI_INT, 0, MPI_COMM_WORLD);




// 检查浮点变量是否一致（dx, dy, dt, mu）
double local_vals[4] = { dx, dy, dt, mu };
double global_max[4], global_min[4];

MPI_Allreduce(local_vals, global_max, 4, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
MPI_Allreduce(local_vals, global_min, 4, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

// 检查整型变量是否一致（timesteps, n_splits）
int local_ints[2] = { timesteps, n_splits };
int global_int_max[2], global_int_min[2];

MPI_Allreduce(local_ints, global_int_max, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
MPI_Allreduce(local_ints, global_int_min, 2, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

// 字符串比较（mesh_folder）
int folder_match = 1;
std::string expected_folder = mesh_folder;
int folder_length_check = expected_folder.size();
char* local_str = new char[folder_length_check + 1];
strcpy(local_str, expected_folder.c_str());

char* all_strings = new char[(folder_length_check + 1) * num_procs];
MPI_Allgather(local_str, folder_length_check + 1, MPI_CHAR,
              all_strings, folder_length_check + 1, MPI_CHAR,
              MPI_COMM_WORLD);

for (int i = 0; i < num_procs; ++i) {
    if (strcmp(local_str, &all_strings[i * (folder_length_check + 1)]) != 0) {
        folder_match = 0;
        break;
    }
}

int global_folder_match;
MPI_Allreduce(&folder_match, &global_folder_match, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

// 打印检查结果（仅0号进程）
if (rank == 0) {
    bool float_consistent = true;
    for (int i = 0; i < 4; ++i)
        if (fabs(global_max[i] - global_min[i]) > 1e-12) float_consistent = false;

    bool int_consistent = (global_int_max[0] == global_int_min[0]) && (global_int_max[1] == global_int_min[1]);

    if (!float_consistent || !int_consistent || global_folder_match == 0) {
        std::cerr << " MPI同步变量不一致！终止运行。\n";
        if (!float_consistent)
            std::cerr << "  → 某些浮点参数不同步 (dx/dy/dt/mu)\n";
        if (!int_consistent)
            std::cerr << "  → timesteps 或 n_splits 不一致\n";
        if (global_folder_match == 0)
            std::cerr << "  → mesh_folder 不一致\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    } else {
        std::cout << " 所有进程同步变量一致，当前值为：\n";
        std::cout << "  dx = " << dx << ", dy = " << dy << "\n";
        std::cout << "  dt = " << dt << ", mu = " << mu << "\n";
        std::cout << "  timesteps = " << timesteps << ", n_splits = " << n_splits << "\n";
        std::cout << "  mesh_folder = " << mesh_folder << "\n";
    }
}

delete[] local_str;
delete[] all_strings;
    // 加载原始网格
    Mesh original_mesh(mesh_folder);
   
    // 垂直分割网格
    std::vector<Mesh> sub_meshes = splitMeshVertically(original_mesh, n_splits);
    if (rank==0)
    {
        // 打印分割信息
    std::cout << "网格已分割为 " << n_splits << " 个子网格:" << std::endl;
    for(int i = 0; i < sub_meshes.size(); i++) {
        std::cout << "子网格 " << i << " 尺寸: " 
                  << sub_meshes[i].nx << "x" << sub_meshes[i].ny << std::endl;
    }
    
    }
    
   
   
   

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

   
    //test
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
       //记录上一个时间步长的u v
      
       int max_outer_iterations=timesteps;
           //simple算法迭代
  
        MPI_Barrier(MPI_COMM_WORLD);
        double init_l2_norm_x = -1.0;
       double init_l2_norm_y = -1.0;
       double init_l2_norm_p = -1.0;
    for(int n=1;n<=max_outer_iterations;n++) {

         //alpha_p=computePressureRelaxationFactor(l2_norm_p);
        
        inter++;
         MPI_Barrier(MPI_COMM_WORLD);
        //1离散动量方程 
        double l2_norm_x, l2_norm_y;
        mesh.u.setZero();
        mesh.v.setZero();
        equ_v.initializeToZero();
        equ_u.initializeToZero();

        
        momentum_function(mesh,equ_u,equ_v,mu,alpha_uv);
        
        equ_u.build_matrix();
        equ_v.build_matrix();


    
         //3求解线性方程组
        double epsilon_uv=0.01;
       
    
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
         
        MPI_Barrier(MPI_COMM_WORLD);
        //cell中心速度插值到面 动量插值
        face_velocity(mesh ,equ_u);
        
        MPI_Barrier(MPI_COMM_WORLD);

        double epsilon_p=1e-5;

        //初始化压力修正方程
        equ_p.initializeToZero();
 
      
        pressure_function(mesh, equ_p, equ_u);
        MPI_Barrier(MPI_COMM_WORLD);
        // 组装压力修正方程
        
        equ_p.build_matrix();
        MPI_Barrier(MPI_COMM_WORLD);
        //求解压力修正方程
        VectorXd p_v(mesh.internumber);

        //初始压力修正场

        mesh.p_prime.setZero();


        p_v.setZero();
      

        



        MPI_Barrier(MPI_COMM_WORLD);
        //求解压力修正方程
        CG_parallel(equ_p,mesh,equ_p.source,p_v,1e-2,200,rank,num_procs,l2_norm_p);
     
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
        double init_l2_norm_p = -1.0;
      
    
        
        //交换数值
       
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
             
              << " 迭代轮数: " << n 
              
              <<"  RMSE残差："
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
    
     
     if (n % 5 == 0) {
        saveMeshData(mesh, rank);
     }
    MPI_Barrier(MPI_COMM_WORLD);
    }
    
    saveMeshData(mesh,rank);

    
    
   
    // 最后显示实际计算总耗时
    auto total_elapsed_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time0).count();
    std::cout << "\n计算完成 总耗时: " << total_elapsed_time << "秒" << std::endl;
  
    saveMeshData(mesh,rank);
    MPI_Finalize();
    
     
    return 0;
}