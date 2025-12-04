#include "DNS.h"
#include <filesystem>
#include <chrono>
#include "parallel.h"
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseLU>
namespace fs = std::filesystem;



// 读取矩阵数据的辅助函数
bool loadMatrixFromFile(const std::string& filename, Eigen::MatrixXd& mat) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return false;
    }

    std::vector<double> values;
    int rows = 0, cols = -1;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> row_values;
        double val;
        while (iss >> val) {
            row_values.push_back(val);
        }

        if (cols == -1) {
            cols = static_cast<int>(row_values.size());
        } else if (static_cast<int>(row_values.size()) != cols) {
            std::cerr << "行列数不一致，文件: " << filename << std::endl;
            return false;
        }

        values.insert(values.end(), row_values.begin(), row_values.end());
        rows++;
    }

    mat = Eigen::Map<Eigen::MatrixXd>(values.data(), cols, rows).transpose(); // 行优先，转置回来
    return true;
}

void loadMeshDataFromSteady(Mesh& mesh, int rank) {
    std::string steady_folder = "steady";

    // 构建文件名
    std::string u_filename  = steady_folder + "/u_"  + std::to_string(rank) + ".dat";
    std::string v_filename  = steady_folder + "/v_"  + std::to_string(rank) + ".dat";
    std::string p_filename  = steady_folder + "/p_"  + std::to_string(rank) + ".dat";
    std::string uf_filename = steady_folder + "/uf_" + std::to_string(rank) + ".dat";
    std::string vf_filename = steady_folder + "/vf_" + std::to_string(rank) + ".dat";

    // 读取到 mesh 对应物理量
    if (!loadMatrixFromFile(u_filename, mesh.u0)) {
        std::cerr << "读取 u0 失败" << std::endl;
    }
    if (!loadMatrixFromFile(u_filename, mesh.u)) {
        std::cerr << "读取 u0 失败" << std::endl;
    }
    if (!loadMatrixFromFile(u_filename, mesh.u_star)) {
        std::cerr << "读取 u0 失败" << std::endl;
    }
    if (!loadMatrixFromFile(v_filename, mesh.v0)) {
        std::cerr << "读取 v0 失败" << std::endl;
    }
    if (!loadMatrixFromFile(v_filename, mesh.v)) {
        std::cerr << "读取 v0 失败" << std::endl;
    }
    if (!loadMatrixFromFile(v_filename, mesh.v_star)) {
        std::cerr << "读取 v0 失败" << std::endl;
    }
    if (!loadMatrixFromFile(p_filename, mesh.p)) {
        std::cerr << "读取 p 失败" << std::endl;
    }
    if (!loadMatrixFromFile(uf_filename, mesh.u_face)) {
        std::cerr << "读取 u_face 失败" << std::endl;
    }
    if (!loadMatrixFromFile(vf_filename, mesh.v_face)) {
        std::cerr << "读取 v_face 失败" << std::endl;
    }
}


// 稀疏矩阵条件数估算函数
double estimateConditionNumber(const Eigen::SparseMatrix<double>& A) {
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::VectorXd Vec;

    // 先求矩阵范数：这里用最大行和范数（infinity norm）
    double normA = 0.0;
    for (int i = 0; i < A.rows(); ++i) {
        double rowSum = 0.0;
        for (SpMat::InnerIterator it(A, i); it; ++it) {
            rowSum += std::abs(it.value());
        }
        normA = std::max(normA, rowSum);
    }

    // 用SparseLU求逆矩阵Ax=b的解，来近似求A逆的范数
    Eigen::SparseLU<SpMat> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        std::cerr << "矩阵分解失败，无法估算条件数" << std::endl;
        return -1.0;
    }

    // b为全1向量
    Vec b = Vec::Ones(A.rows());
    Vec x = solver.solve(b);
    if (solver.info() != Eigen::Success) {
        std::cerr << "求解失败，无法估算条件数" << std::endl;
        return -1.0;
    }

    // 求A逆范数（用解向量x的无穷范数近似）
    double normInvA = x.lpNorm<Eigen::Infinity>();

    // 条件数估算
    double cond = normA * normInvA;
    return cond;
}

void saveMeshData(const Mesh& mesh, int rank, const std::string& timestep_folder = "") {
    // 创建文件名
    std::string u_filename = "u_" + std::to_string(rank) + ".dat";
    std::string v_filename = "v_" + std::to_string(rank) + ".dat";
    std::string p_filename = "p_" + std::to_string(rank) + ".dat";
    
    // 如果提供了时间步文件夹，添加到路径中
    if(!timestep_folder.empty()) {
        if (!fs::exists(timestep_folder)) {
            fs::create_directory(timestep_folder);
        }
        u_filename = timestep_folder + "/" + u_filename;
        v_filename = timestep_folder + "/" + v_filename;
        p_filename = timestep_folder + "/" + p_filename;
    }

    try {
        // 保存u场
        std::ofstream u_file(u_filename);
        if(!u_file) {
            throw std::runtime_error("无法创建文件: " + u_filename);
        }
        u_file << mesh.u_star;
        u_file.close();

        // 保存v场
        std::ofstream v_file(v_filename);
        if(!v_file) {
            throw std::runtime_error("无法创建文件: " + v_filename);
        }
        v_file << mesh.v_star;
        v_file.close();

        // 保存p场
        std::ofstream p_file(p_filename);
        if(!p_file) {
            throw std::runtime_error("无法创建文件: " + p_filename);
        }
        p_file << mesh.p;
        p_file.close();

        //std::cout << "进程 " << rank << " 的数据已保存到文件" << std::endl;
    }
    catch(const std::exception& e) {
        std::cerr << "保存数据时出错: " << e.what() << std::endl;
    }
}








int main(int argc, char* argv[]) 
{    
    MPI_Init(&argc, &argv);
     // 获取进程信息
     int rank, num_procs;
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   // 获取输入参数，只在 rank 0 上执行
std::string mesh_folder;
double dt;
int timesteps;
int n_splits;
double mu;

if (rank == 0) {
    if (argc == 6) {
        // 命令行参数输入
        mesh_folder = argv[1];
        dt = std::stod(argv[2]);
        timesteps = std::stoi(argv[3]);
        mu = std::stod(argv[4]);
        n_splits = std::stoi(argv[5]);

        std::cout << "从命令行读取参数:" << std::endl;
        std::cout << "网格文件夹: " << mesh_folder << std::endl;
        std::cout << "时间步长: " << dt << std::endl;
        std::cout << "时间步数: " << timesteps << std::endl;
        std::cout << "并行线程数: " << n_splits << std::endl;
        std::cout << "粘度: " << mu << std::endl;
    } else {
        // 手动输入
        std::cout << "网格文件夹路径: ";
        std::cin >> mesh_folder;
        std::cout << "时间步长: ";
        std::cin >> dt;
        std::cout << "时间步数: ";
        std::cin >> timesteps;
        std::cout << "并行线程数: ";
        std::cin >> n_splits;
        std::cout << "粘度: ";
        std::cin >> mu;
    }
}

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



    // 加载原始网格
    Mesh original_mesh(mesh_folder);
    readParams(mesh_folder, dx, dy);

    // 同步 dx 和 dy 给所有进程
    MPI_Bcast(&dx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);  
    MPI_Bcast(&dy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // 垂直分割网格
    std::vector<Mesh> sub_meshes = splitMeshVertically(original_mesh, n_splits);
    MPI_Barrier(MPI_COMM_WORLD);
    // 打印分割信息
    if (rank==0)
    {
        std::cout << "网格已分割为 " << n_splits << " 个子网格:" << std::endl;
    for(int i = 0; i < sub_meshes.size(); i++) {
        std::cout << "子网格 " << i << " 尺寸: " 
                  << sub_meshes[i].nx << "x" << sub_meshes[i].ny << std::endl;
    }
    }
    
    
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
   

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
        //初始化 默认初始化为全0场
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

   //设置网格参数
   int nx0,ny0;
   nx0=mesh.nx;
   ny0=mesh.ny;


    //初始化建立u v p的方程
    Equation equ_u(mesh);
    Equation equ_v(mesh);
    Equation equ_p(mesh);

    //残差初始化
   double l2x = 0.0, l2y = 0.0, l2p = 0.0;
  
   auto start_time0 = chrono::steady_clock::now();  // 开始计时
   


    //piso算法外循环
    for (int i = 0; i <= timesteps; ++i) { 
        
       if(rank==0){ cout<<"时间步长 "<< i <<std::endl;}
        // 切换到当前编号文件夹
      
        //piso内循环轮数 矫正2次压力
       int max_outer_iterations=4;
          
  
        
        //归一化初始残差
        double init_l2_norm_x = -1.0;
       double init_l2_norm_y = -1.0;
       
      
        // 同步 dx 和 dy 给所有进程
    MPI_Bcast(&dx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
       double l2_norm_x, l2_norm_y;
       //初始化u v方向的方程
       equ_v.initializeToZero();
       equ_u.initializeToZero();

       //离散动量方程
       momentum_function_PISO(mesh,equ_u,equ_v,mu,dt);
      


       //组装动量方程
       equ_u.build_matrix();
       equ_v.build_matrix();

  
        
       //3求解线性方程组
       
      
    
       //生成并初始化解向量
       VectorXd x_v(mesh.internumber),y_v(mesh.internumber);
       x_v.setZero();
       y_v.setZero();

       //求解u的动量方程 cg求解器
      CG_parallel(equ_u,mesh,equ_u.source,x_v,1e-2,35,rank,num_procs,l2_norm_x);
      
       //求解v的动量方程 cg求解器
       CG_parallel(equ_v,mesh,equ_v.source,y_v,1e-2,35,rank,num_procs,l2_norm_y);
     

       //将解向量写回矩阵场
       vectorToMatrix(x_v,mesh.u,mesh);
       
       vectorToMatrix(y_v,mesh.v,mesh);
    
       
       
       

     
       
        //数据交换 交换u v 和矩阵主对角元系数ap
       exchangeColumns(mesh.u, rank, num_procs);
       
       exchangeColumns(mesh.v, rank, num_procs);
       
       exchangeColumns(equ_u.A_p, rank, num_procs);
        


        //piso算法压力修正循环
    for(int n=1;n<=max_outer_iterations;n++) {
        
        
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
       
        
         

        //压力修正
        correct_pressure(mesh,equ_u,0.3);
        exchangeColumns(mesh.p_prime, rank, num_procs); 
        
        //速度修正
        correct_velocity(mesh,equ_u);
       
        
        
        //更新压力 速度 并交换数值
        mesh.p = mesh.p_star;
        mesh.u = mesh.u_star;
        mesh.v = mesh.v_star;
        
  
        
        exchangeColumns(mesh.u, rank, num_procs);
       

        exchangeColumns(mesh.v, rank, num_procs);
       
        exchangeColumns(mesh.p, rank, num_procs);
        
        double init_l2_norm_p = -1.0;
      
        exchangeColumns(mesh.u_face, rank, num_procs);
        exchangeColumns(mesh.v_face, rank, num_procs);
        
       
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
    mesh.u0 = mesh.u;
    mesh.v0 = mesh.v;
    }
    
   
    // 最后显示实际计算总耗时
    auto total_elapsed_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time0).count();
    if (rank==0)
    {
        std::cout << "\n计算完成 总耗时: " << total_elapsed_time << "秒" << std::endl;
  
    }
    
    
    saveMeshData(mesh,rank);
    MPI_Finalize();
    
     
    return 0;
}