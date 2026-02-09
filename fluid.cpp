/**
 * @file dns.cpp
 * @brief 基于有限体积法的不可压缩流体直接数值模拟(DNS)求解器
 * @details 使用SIMPLE/PISO算法求解二维不可压缩Navier-Stokes方程
 * 
 * 主要功能:
 * - 网格管理与边界条件处理
 * - 动量方程离散化(对流-扩散项)
 * - 压力修正方程求解
 * - 速度-压力耦合(SIMPLE/PISO算法)
 * - 稳态/非稳态求解
 * - 并行计算支持(区域分解)
 */

#include "fluid.h"
#include <filesystem>
#include <iomanip>
#include <sstream>
#include "parallel.h"

namespace fs = std::filesystem;

// ============================================================================
// 全局变量定义
// ============================================================================

double dx;              ///< 网格x方向步长
double dy;              ///< 网格y方向步长
double vx;              ///< x方向速度(临时变量)
double velocity;        ///< 速度幅值(临时变量)

// 残差范数
double l2_norm_x = 0.0; ///< x方向动量方程L2范数
double l2_norm_y = 0.0; ///< y方向动量方程L2范数
double l2_norm_p = 0.0; ///< 压力修正方程L2范数

double a, b;            ///< 通用临时变量

// ============================================================================
// 工具函数
// ============================================================================

/**
 * @brief 打印矩阵到控制台(用于调试)
 * @param matrix 要打印的Eigen矩阵
 * @param name 矩阵名称
 * @param precision 输出精度(小数位数)
 */
void printMatrix(const MatrixXd& matrix, const string& name, int precision) {
    // 设置输出格式: 精度、无填充、逗号分隔、每行换行
    IOFormat fmt(precision, 0, ", ", "\n", "[", "]");
    
    // 打印矩阵名称和尺寸
    cout << "\n====== " << name << " (" 
         << matrix.rows() << "x" << matrix.cols() << ") ======\n";
              
    // 打印矩阵内容
    cout << matrix.format(fmt) << endl;
    
    // 打印分隔线
    cout << string(40, '=') << endl;
}

// ============================================================================
// Mesh类实现 - 网格数据结构
// ============================================================================

/**
 * @brief 构造函数 - 初始化网格矩阵
 * @param n_y y方向内部网格数
 * @param n_x x方向内部网格数
 * @note 所有矩阵尺寸为(n_y+2, n_x+2),包含虚拟边界层
 */
Mesh::Mesh(int n_y, int n_x)
    : u(n_y + 2, n_x + 2),        ///< x方向速度(单元中心)
      u_star(n_y + 2, n_x + 2),   ///< x方向修正后速度
      u0(n_y + 2, n_x + 2),       ///< x方向上一时间步速度
      v(n_y + 2, n_x + 2),        ///< y方向速度(单元中心)
      v_star(n_y + 2, n_x + 2),   ///< y方向修正后速度
      v0(n_y + 2, n_x + 2),       ///< y方向上一时间步速度
      p(n_y + 2, n_x + 2),        ///< 压力场
      p_star(n_y + 2, n_x + 2),   ///< 修正后压力场
      p_prime(n_y + 2, n_x + 2),  ///< 压力修正量
      u_face(n_y + 2, n_x + 1),   ///< 垂直面上的x方向速度
      v_face(n_y + 1, n_x + 2),   ///< 水平面上的y方向速度
      bctype(n_y + 2, n_x + 2),   ///< 边界类型标记
      zoneid(n_y + 2, n_x + 2),   ///< 区域ID
      interid(n_y + 2, n_x + 2),  ///< 内部点编号
      nx(n_x), 
      ny(n_y) {}

/**
 * @brief 初始化所有矩阵为零
 */
void Mesh::initializeToZero() {
    u.setZero();
    u_star.setZero();
    v.setZero();
    v_star.setZero();
    p.setZero();
    p_star.setZero();
    p_prime.setZero();
    u_face.setZero();
    v_face.setZero();
    u0.setZero();   
    v0.setZero();
    bctype.setZero();
    zoneid.setZero();
    interid.setZero();
}

/**
 * @brief 显示单个矩阵内容(调试用)
 * @param matrix 矩阵引用
 * @param name 矩阵名称
 */
void Mesh::displayMatrix(const MatrixXd& matrix, const string& name) const {
    cout << name << ":\n" << matrix << "\n";
}

/**
 * @brief 显示所有流场矩阵
 */
void Mesh::displayAll() const {
    displayMatrix(u, "u");
    displayMatrix(u_star, "u_star");
    displayMatrix(v, "v");
    displayMatrix(v_star, "v_star");
    displayMatrix(p, "p");
    displayMatrix(p_star, "p_star");
    displayMatrix(p_prime, "p_prime");
    displayMatrix(u_face, "u_face");
    displayMatrix(v_face, "v_face");
}

/**
 * @brief 为内部点创建编号系统
 * @details 从上到下、从左到右遍历网格,为所有内部点(bctype=0)分配唯一编号
 *          用于构建线性方程组的行号
 */
void Mesh::createInterId() {
    interid = MatrixXi::Zero(bctype.rows(), bctype.cols());
    interi.clear();
    interj.clear();
    internumber = 0;
    int count = 0;
    
    // 从上到下,从左到右遍历
    for(int i = 0; i < bctype.rows(); i++) {
        for(int j = 0; j < bctype.cols(); j++) {
            if(bctype(i,j) == 0) {  // 只对内部点编号
                interid(i,j) = count;
                interi.push_back(i);
                interj.push_back(j);
                count++;
                internumber++;
            }
        }
    }
}

/**
 * @brief 设置矩形区域的边界类型和区域ID
 * @param x1, y1 左上角坐标(网格索引)
 * @param x2, y2 右下角坐标(网格索引)
 * @param bcValue 边界类型值
 * @param zoneValue 区域ID值
 */
void Mesh::setBlock(int x1, int y1, int x2, int y2, double bcValue, double zoneValue) {
    // 确保坐标范围合法
    x1 = std::max(0, std::min(x1, nx + 1));
    x2 = std::max(0, std::min(x2, nx + 1));
    y1 = std::max(0, std::min(y1, ny + 1));
    y2 = std::max(0, std::min(y2, ny + 1));
    
    // 确保 x1 <= x2 且 y1 <= y2
    if(x1 > x2) std::swap(x1, x2);
    if(y1 > y2) std::swap(y1, y2);
    
    // 修改指定区域的 bctype 和 zoneid
    bctype.block(y1, x1, y2-y1+1, x2-x1+1).setConstant(bcValue);
    zoneid.block(y1, x1, y2-y1+1, x2-x1+1).setConstant(zoneValue);
}

/**
 * @brief 设置指定区域的速度边界条件
 * @param zoneIndex 区域索引
 * @param u x方向速度
 * @param v y方向速度
 */
void Mesh::setZoneUV(int zoneIndex, double u, double v) {
    // 确保向量足够长
    while(zoneu.size() <= zoneIndex) {
        zoneu.push_back(0.0);
        zonev.push_back(0.0);
    }
    
    zoneu[zoneIndex] = u;
    zonev[zoneIndex] = v;
}

/**
 * @brief 初始化边界条件
 * @details 根据bctype和zoneid设置:
 *          1. 单元中心速度(u, v, u_star, v_star)
 *          2. 单元面速度(u_face, v_face)
 * 
 * 边界类型定义:
 * - 0: 内部点
 * - >0: 固壁边界
 * - -1: 压力出口
 * - -2: 速度入口
 * - -3: 并行交界面
 */
void Mesh::initializeBoundaryConditions() {
    // 1. 初始化单元中心速度
    for(int i = 0; i <= ny + 1; i++) {
        for(int j = 0; j <= nx + 1; j++) {
            if(bctype(i,j) != 0) {  // 非内部点
                int zone = zoneid(i,j);
                u(i,j) = zoneu[zone];
                u_star(i,j) = zoneu[zone];
                v(i,j) = zonev[zone];
                v_star(i,j) = zonev[zone];
            }
        }
    }

    // 2. 初始化u_face(垂直面上的x方向速度)
    for(int i = 0; i <= ny + 1; i++) {
        for(int j = 0; j <= nx; j++) {
            bool left_is_internal = (bctype(i,j) == 0);
            bool right_is_internal = (bctype(i,j+1) == 0);

            if(left_is_internal && !right_is_internal) {
                // 右侧是边界,使用右侧单元格的速度
                u_face(i,j) = zoneu[zoneid(i,j+1)];
            }
            else if(!left_is_internal && right_is_internal) {
                // 左侧是边界,使用左侧单元格的速度
                u_face(i,j) = zoneu[zoneid(i,j)];
            }
            else if(!left_is_internal && !right_is_internal) {
                // 两侧都是边界,取均值
                u_face(i,j) = 0.5 * (zoneu[zoneid(i,j)] + zoneu[zoneid(i,j+1)]);
            }
        }
    }

    // 3. 初始化v_face(水平面上的y方向速度)
    for(int i = 0; i <= ny; i++) {
        for(int j = 0; j <= nx + 1; j++) {
            bool top_is_internal = (bctype(i,j) == 0);
            bool bottom_is_internal = (bctype(i+1,j) == 0);

            if(top_is_internal && !bottom_is_internal) {
                // 下侧是边界,使用下侧单元格的速度
                v_face(i,j) = zonev[zoneid(i+1,j)];
            }
            else if(!top_is_internal && bottom_is_internal) {
                // 上侧是边界,使用上侧单元格的速度
                v_face(i,j) = zonev[zoneid(i,j)];
            }
            else if(!top_is_internal && !bottom_is_internal) {
                // 上下都是边界,取均值
                v_face(i,j) = 0.5 * (zonev[zoneid(i,j)] + zonev[zoneid(i+1,j)]);
            }
        }
    }
}

/**
 * @brief 保存网格数据到文件夹
 * @param folderPath 目标文件夹路径
 */
void Mesh::saveToFolder(const std::string& folderPath) const {
    // 创建文件夹
    if (!fs::exists(folderPath)) {
        fs::create_directory(folderPath);
    }

    // 保存网格参数
    std::ofstream paramFile(folderPath + "/params.txt");
    paramFile << nx << " " << ny << "\n";
    paramFile << dx << " " << dy << "\n";
    paramFile.close();

    // 保存边界类型矩阵
    std::ofstream bcFile(folderPath + "/bctype.dat");
    bcFile << bctype;
    bcFile.close();

    // 保存区域ID矩阵
    std::ofstream zoneFile(folderPath + "/zoneid.dat");
    zoneFile << zoneid;
    zoneFile.close();

    // 保存区域速度
    std::ofstream zoneuvFile(folderPath + "/zoneuv.txt");
    for(size_t i = 0; i < zoneu.size(); i++) {
        zoneuvFile << zoneu[i] << " " << zonev[i] << "\n";
    }
    zoneuvFile.close();
}

/**
 * @brief 从文件夹加载网格数据(构造函数)
 * @param folderPath 网格数据文件夹路径
 * @throws std::runtime_error 如果文件不存在或读取失败
 */
Mesh::Mesh(const std::string& folderPath) {
    if (!fs::exists(folderPath)) {
        throw std::runtime_error("网格文件夹不存在!");
    }

    // 读取网格参数
    std::ifstream paramFile(folderPath + "/params.txt");
    if (!paramFile) {
        throw std::runtime_error("无法打开参数文件!");
    }
    
    paramFile >> nx >> ny >> ::dx >> ::dy;
    paramFile.close();

    // 初始化所有矩阵(包含边界层,尺寸为n+2)
    u.resize(ny + 2, nx + 2);
    u_star.resize(ny + 2, nx + 2);
    u0.resize(ny + 2, nx + 2);
    v.resize(ny + 2, nx + 2);
    v_star.resize(ny + 2, nx + 2);
    v0.resize(ny + 2, nx + 2);
    p.resize(ny + 2, nx + 2);
    p_star.resize(ny + 2, nx + 2);
    p_prime.resize(ny + 2, nx + 2);
    u_face.resize(ny + 2, nx + 1);
    v_face.resize(ny + 1, nx + 2);
    bctype.resize(ny + 2, nx + 2);
    zoneid.resize(ny + 2, nx + 2);
    interid.resize(ny + 2, nx + 2);

    initializeToZero();

    // 读取边界类型
    std::ifstream bcFile(folderPath + "/bctype.dat");
    if (!bcFile) {
        throw std::runtime_error("无法打开边界类型文件!");
    }
    for(int i = 0; i < bctype.rows(); i++) {
        for(int j = 0; j < bctype.cols(); j++) {
            bcFile >> bctype(i,j);
        }
    }
    bcFile.close();

    // 读取区域ID
    std::ifstream zoneFile(folderPath + "/zoneid.dat");
    if (!zoneFile) {
        throw std::runtime_error("无法打开区域ID文件!");
    }
    for(int i = 0; i < zoneid.rows(); i++) {
        for(int j = 0; j < zoneid.cols(); j++) {
            zoneFile >> zoneid(i,j);
        }
    }
    zoneFile.close();

    // 读取区域速度
    std::ifstream zoneuvFile(folderPath + "/zoneuv.txt");
    if (!zoneuvFile) {
        throw std::runtime_error("无法打开区域速度文件!");
    }
    double u, v;
    while(zoneuvFile >> u >> v) {
        zoneu.push_back(u);
        zonev.push_back(v);
    }
    zoneuvFile.close();

    // 创建内部点编号
    createInterId();
    
    // 初始化边界条件
    initializeBoundaryConditions();
}

// ============================================================================
// Equation类实现 - 离散方程系统
// ============================================================================

/**
 * @brief 构造函数 - 初始化方程系数矩阵
 * @param mesh_ 网格对象引用
 */
Equation::Equation(Mesh& mesh_)
    : A_p(mesh_.ny + 2, mesh_.nx + 2),  ///< 中心系数
      A_e(mesh_.ny + 2, mesh_.nx + 2),  ///< 东邻系数
      A_w(mesh_.ny + 2, mesh_.nx + 2),  ///< 西邻系数
      A_n(mesh_.ny + 2, mesh_.nx + 2),  ///< 北邻系数
      A_s(mesh_.ny + 2, mesh_.nx + 2),  ///< 南邻系数
      source(mesh_.internumber),         ///< 源项向量
      A(mesh_.internumber, mesh_.internumber),  ///< 稀疏系数矩阵
      n_x(mesh_.nx), 
      n_y(mesh_.ny),
      mesh(mesh_) {}

/**
 * @brief 初始化方程系数为零
 */
void Equation::initializeToZero() {
    A_p.setZero();
    A_e.setZero();
    A_w.setZero();
    A_n.setZero();
    A_s.setZero();
    source.setZero();
    A.setZero();
}

/**
 * @brief 构建稀疏系数矩阵
 * @details 将五对角系数矩阵(A_p, A_e, A_w, A_n, A_s)组装成Eigen稀疏矩阵
 *          方程形式: A_p*phi_P = A_e*phi_E + A_w*phi_W + A_n*phi_N + A_s*phi_S + source
 */
void Equation::build_matrix() {
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;

    // 遍历所有内部点
    for(int i = 1; i <= n_y; i++) {
        for(int j = 1; j <= n_x; j++) {
            if(mesh.bctype(i,j) == 0) {  // 只处理内部点
                int current_id = mesh.interid(i,j) ;  // 当前点在方程组中的行号
                
                // 添加中心点系数(对角元)
                tripletList.emplace_back(current_id, current_id, A_p(i,j));
                
                // 检查东邻点
                if(mesh.bctype(i,j+1) == 0) {
                    int east_id = mesh.interid(i,j+1) ;
                    tripletList.emplace_back(current_id, east_id, -A_e(i,j));
                }
                
                // 检查西邻点
                if(mesh.bctype(i,j-1) == 0) {
                    int west_id = mesh.interid(i,j-1) ;
                    tripletList.emplace_back(current_id, west_id, -A_w(i,j));
                }
                
                // 检查北邻点
                if(mesh.bctype(i-1,j) == 0) {
                    int north_id = mesh.interid(i-1,j) ;
                    tripletList.emplace_back(current_id, north_id, -A_n(i,j));
                }
                
                // 检查南邻点
                if(mesh.bctype(i+1,j) == 0) {
                    int south_id = mesh.interid(i+1,j) ;
                    tripletList.emplace_back(current_id, south_id, -A_s(i,j));
                }
            }
        }
    }
    
    // 组装稀疏矩阵
    A.resize(mesh.internumber, mesh.internumber);
    A.setFromTriplets(tripletList.begin(), tripletList.end());
}

// ============================================================================
// 线性方程组求解器
// ============================================================================

/**
 * @brief 求解线性方程组 A*phi = source
 * @param equation 方程对象
 * @param epsilon 求解器收敛容差
 * @param l2_norm 输出残差L2范数
 * @param phi 解向量(输入初值,输出解)
 * @note 使用共轭梯度法(CG),最大迭代次数=1(作为SIMPLE循环的一步)
 */
void solve(Equation& equation, double epsilon, double& l2_norm, MatrixXd& phi) {
    // 创建解向量
    VectorXd x(equation.mesh.internumber);

    // 从网格提取初值
    for(int i = 0; i <= equation.n_y + 1; i++) {
        for(int j = 0; j <= equation.n_x + 1; j++) {
            if(equation.mesh.bctype(i,j) == 0) {
                int n = equation.mesh.interid(i,j) ;
                x[n] = phi(i,j);
            }
        }
    }

    // 计算初始残差
    l2_norm = (equation.A * x - equation.source).norm();

    // 共轭梯度法求解
    ConjugateGradient<SparseMatrix<double>> solver;
    solver.compute(equation.A);
    solver.setTolerance(epsilon); 
    solver.setMaxIterations(1);  // SIMPLE循环中每次只迭代一步
    x = solver.solve(equation.source);

    // 将解写回网格
    for(int i = 0; i <= equation.n_y + 1; i++) {
        for(int j = 0; j <= equation.n_x + 1; j++) {
            if(equation.mesh.bctype(i,j) == 0) {
                int n = equation.mesh.interid(i,j) ;
                phi(i,j) = x[n];
            }
        }
    }
}

// ============================================================================
// 速度场计算
// ============================================================================

/**
 * @brief 计算单元面上的速度(Rhie-Chow插值)
 * @param mesh 网格对象
 * @param equ_u x方向动量方程
 * @details 
 * 使用压力梯度修正的动量插值,避免压力场棋盘式震荡
 * 
 * 垂直面速度: u_face(i,j) = 0.5*(u_P + u_E) + 压力修正项
 * 水平面速度: v_face(i,j) = 0.5*(v_P + v_N) + 压力修正项
 */
void face_velocity(Mesh& mesh, Equation& equ_u) {
    MatrixXd& u_face = mesh.u_face;
    MatrixXd& v_face = mesh.v_face;
    MatrixXd& bctype = mesh.bctype;
    MatrixXd& u = mesh.u;
    MatrixXd& v = mesh.v;
    MatrixXd& p = mesh.p;
    MatrixXd& A_p = equ_u.A_p;

    // 计算u_face(垂直面上的x方向速度)
    for(int i = 0; i <= mesh.ny + 1; i++) {
        for(int j = 0; j <= mesh.nx; j++) {
            // 情况1: 内部面或并行交界面
            if ((bctype(i,j) == 0 && bctype(i,j+1) == 0) || 
                (bctype(i,j) == 0 && bctype(i,j+1) == -3) ||
                (bctype(i,j) == -3 && bctype(i,j+1) == 0)) {

                // 处理压力边界条件
                if (bctype(i,j+2) == -2) p(i,j+2) = p(i,j+1);
                else if (bctype(i,j-1) == -2) p(i,j-1) = p(i,j);
                else if (bctype(i,j+2) == -1) p(i,j+2) = 0;
                else if (bctype(i,j-1) == -1) p(i,j-1) = 0;

                // Rhie-Chow插值公式
                u_face(i,j) = 0.5*(u(i,j) + u(i,j+1))
                            + 0.25*(p(i,j+1) - p(i,j-1)) * dy / A_p(i,j)
                            + 0.25*(p(i,j+2) - p(i,j)) * dy / A_p(i,j+1)
                            - 0.5*(1.0/A_p(i,j) + 1.0/A_p(i,j+1)) * (p(i,j+1) - p(i,j)) * dy;
            }
            // 情况2: 压力出口边界
            else if (bctype(i,j) == 0 && bctype(i,j+1) == -1) {
                u_face(i,j) = u(i,j);
            }
            else if (bctype(i,j) == -1 && bctype(i,j+1) == 0) {
                u_face(i,j) = u(i,j+1);
            }
            // 情况3: 速度入口边界
            else if (bctype(i,j) == 0 && bctype(i,j+1) == -2) {
                u_face(i,j) = mesh.zoneu[mesh.zoneid(i,j+1)];
            }
            else if (bctype(i,j) == -2 && bctype(i,j+1) == 0) {
                u_face(i,j) = mesh.zoneu[mesh.zoneid(i,j)];
            }
            else {
                u_face(i,j) = 0.0;
            }

            // NaN检查
            if (std::isnan(u_face(i,j))) u_face(i,j) = 0.0;
        }
    }

    // 计算v_face(水平面上的y方向速度)
    for(int i = 0; i <= mesh.ny; i++) {
        for(int j = 0; j <= mesh.nx + 1; j++) {
            if (bctype(i,j) == 0 && bctype(i+1,j) == 0) {
                // 处理压力边界条件
                if (bctype(i+2,j) == -2) p(i+2,j) = p(i+1,j);
                else if (bctype(i-1,j) == -2) p(i-1,j) = p(i,j);
                else if (bctype(i+2,j) == -1) p(i+2,j) = 0;
                else if (bctype(i-1,j) == -1) p(i-1,j) = 0;

                // Rhie-Chow插值公式
                v_face(i,j) = 0.5*(v(i+1,j) + v(i,j))
                            + 0.25*(p(i,j) - p(i+2,j)) * dx / A_p(i+1,j)
                            + 0.25*(p(i-1,j) - p(i+1,j)) * dx / A_p(i,j)
                            - 0.5*(1.0/A_p(i+1,j) + 1.0/A_p(i,j)) * (p(i,j) - p(i+1,j)) * dx;
            }
            else if (bctype(i,j) == 0 && bctype(i+1,j) == -2) {
                v_face(i,j) = mesh.zonev[mesh.zoneid(i+1,j)];
            }
            else if (bctype(i,j) == -2 && bctype(i+1,j) == 0) {
                v_face(i,j) = mesh.zonev[mesh.zoneid(i,j)];
            }
            else if (bctype(i,j) == 0 && bctype(i+1,j) == -1) {
                v_face(i,j) = v(i,j);
            }
            else if (bctype(i,j) == -1 && bctype(i+1,j) == 0) {
                v_face(i,j) = v(i+1,j);
            }
            else {
                v_face(i,j) = 0.0;
            }

            // NaN检查
            if (std::isnan(v_face(i,j))) v_face(i,j) = 0.0;
        }
    }
}

// ============================================================================
// 压力修正方程
// ============================================================================

/**
 * @brief 构建压力修正方程(Poisson方程)
 * @param mesh 网格对象
 * @param equ_p 压力修正方程
 * @param equ_u x方向动量方程(提供A_p系数)
 * @details
 * 离散形式: sum(Ap_nb * p'_nb) = sum(rho * u_face * A)
 * 其中 Ap_nb = 0.5*(1/A_p(P) + 1/A_p(nb)) * dy^2 (东西面)
 *      Ap_nb = 0.5*(1/A_p(P) + 1/A_p(nb)) * dx^2 (南北面)
 * 源项为速度散度: -(u_e - u_w)*dy - (v_n - v_s)*dx
 */
void pressure_function(Mesh &mesh, Equation &equ_p, Equation &equ_u) {
    MatrixXd &u_face = mesh.u_face;
    MatrixXd &v_face = mesh.v_face;
    MatrixXd &bctype = mesh.bctype;
    MatrixXd &A_p = equ_u.A_p;
    MatrixXd &Ap_p = equ_p.A_p;
    MatrixXd &Ap_e = equ_p.A_e;
    MatrixXd &Ap_w = equ_p.A_w;
    MatrixXd &Ap_n = equ_p.A_n;
    MatrixXd &Ap_s = equ_p.A_s;
    VectorXd &source_p = equ_p.source;

    // 遍历所有内部点
    for(int i = 0; i <= equ_p.n_y+1; i++) {
        for(int j = 0; j <= equ_p.n_x+1; j++) {
            if(bctype(i,j) == 0) {  // 内部点
                int n = mesh.interid(i,j) ;
                double Ap_temp = 0;
                
                // 计算东面系数
                if(bctype(i,j+1) == 0 || bctype(i,j+1) == -3) {
                    Ap_e(i,j) = 0.5*(1/A_p(i,j) + 1/A_p(i,j+1))*(dy*dy);
                    Ap_temp += Ap_e(i,j);
                } else {
                    Ap_e(i,j) = 0;
                }

                // 计算西面系数
                if(bctype(i,j-1) == 0 || bctype(i,j-1) == -3) {
                    Ap_w(i,j) = 0.5*(1/A_p(i,j) + 1/A_p(i,j-1))*(dy*dy);
                    Ap_temp += Ap_w(i,j);
                } else {
                    Ap_w(i,j) = 0;
                }

                // 计算北面系数
                if(bctype(i-1,j) == 0) {
                    Ap_n(i,j) = 0.5*(1/A_p(i,j) + 1/A_p(i-1,j))*(dx*dx);
                    Ap_temp += Ap_n(i,j);
                } else {
                    Ap_n(i,j) = 0;
                }

                // 计算南面系数
                if(bctype(i+1,j) == 0) {
                    Ap_s(i,j) = 0.5*(1/A_p(i,j) + 1/A_p(i+1,j))*(dx*dx);
                    Ap_temp += Ap_s(i,j);
                } else {
                    Ap_s(i,j) = 0;
                }

                // 设置中心系数
                Ap_p(i,j) = Ap_temp;
                
                // 计算源项(负的速度散度)
                source_p[n] = -(u_face(i,j) - u_face(i,j-1))*dy 
                            - (v_face(i-1,j) - v_face(i,j))*dx;         
            }
        }
    }
}

// ============================================================================
// 压力和速度修正
// ============================================================================

/**
 * @brief 修正压力场
 * @param mesh 网格对象
 * @param equ_u 动量方程(未使用)
 * @param alpha_p 压力松弛因子
 * @details p_star = p + alpha_p * p_prime
 */
void correct_pressure(Mesh &mesh, Equation &equ_u, double alpha_p) {
    MatrixXd &p = mesh.p;
    MatrixXd &p_star = mesh.p_star;
    MatrixXd &p_prime = mesh.p_prime;
    MatrixXd &bctype = mesh.bctype;
    int n_x = mesh.nx;
    int n_y = mesh.ny;

    // 边界点的压力修正量设为0
    for(int i = 0; i <= n_y + 1; i++) {
        for(int j = 0; j <= n_x + 1; j++) {
            if(bctype(i,j) != 0) {
                p_prime(i,j) = 0;
                p(i,j) = 0;
                p_star(i,j) = 0;
            }
        }
    }

    // 更新压力场(带松弛)
    p_star = p + alpha_p * p_prime;
}

/**
 * @brief 修正速度场
 * @param mesh 网格对象
 * @param equ_u x方向动量方程
 * @details
 * 1. 修正单元中心速度: u_star = u + 0.5*(p'_W - p'_E)*dy/A_p
 * 2. 修正单元面速度: u_face += 0.5*(1/A_p(P) + 1/A_p(E))*(p'_P - p'_E)*dy
 */
void correct_velocity(Mesh &mesh, Equation &equ_u) {
    MatrixXd &u = mesh.u;
    MatrixXd &v = mesh.v;
    MatrixXd &u_face = mesh.u_face;
    MatrixXd &v_face = mesh.v_face;
    MatrixXd &p_prime = mesh.p_prime;
    MatrixXd &u_star = mesh.u_star;
    MatrixXd &v_star = mesh.v_star;
    MatrixXd &bctype = mesh.bctype;
    MatrixXd &A_p = equ_u.A_p;
    int n_x = mesh.nx;
    int n_y = mesh.ny;

    // 修正单元中心x方向速度
    for (int i = 0; i <= n_y+1; i++) {
        for (int j = 0; j <= n_x+1; j++) {
            if (bctype(i,j) == 0) {
                double p_west, p_east;

                // 确定西面压力修正量
                if (bctype(i,j-1) == 0 || bctype(i,j-1) == -3)
                    p_west = p_prime(i,j-1);
                else
                    p_west = p_prime(i,j);

                // 确定东面压力修正量
                if (bctype(i,j+1) == 0 || bctype(i,j+1) == -3)
                    p_east = p_prime(i,j+1);
                else
                    p_east = p_prime(i,j);

                u_star(i,j) = u(i,j) + 0.5 * (p_west - p_east) * dy / A_p(i,j);
            }
        }
    }

    // 修正单元中心y方向速度
    for (int i = 0; i <= n_y+1; i++) {
        for (int j = 0; j <= n_x+1; j++) {
            if (bctype(i,j) == 0) {
                double p_north, p_south;

                // 确定北面压力修正量
                if (bctype(i-1,j) == 0 || bctype(i-1,j) == -3)
                    p_north = p_prime(i-1,j);
                else
                    p_north = p_prime(i,j);

                // 确定南面压力修正量
                if (bctype(i+1,j) == 0 || bctype(i+1,j) == -3)
                    p_south = p_prime(i+1,j);
                else
                    p_south = p_prime(i,j);

                v_star(i,j) = v(i,j) + 0.5 * (p_south - p_north) * dx / A_p(i,j);
            }
        }
    }

    // 修正垂直面上的x方向速度
    for (int i = 0; i <= n_y+1; i++) {
        for (int j = 0; j <= n_x; j++) {
            if ((bctype(i,j) == 0 && bctype(i,j+1) == 0) ||
                (bctype(i,j) == 0 && bctype(i,j+1) == -3) ||
                (bctype(i,j) == -3 && bctype(i,j+1) == 0)) {
                // 内部面或并行交界面
                u_face(i,j) += 0.5 * (1/A_p(i,j) + 1/A_p(i,j+1)) * 
                              (p_prime(i,j) - p_prime(i,j+1)) * dy;
            }
            else if (bctype(i,j) == 0 && bctype(i,j+1) == -1) {
                // 压力边界
                u_face(i,j) = u_star(i,j);
            }
            else if (bctype(i,j) == -1 && bctype(i,j+1) == 0) {
                u_face(i,j) = u_star(i,j+1);
            }
        }
    }

    // 修正水平面上的y方向速度
    for (int i = 0; i <= n_y; i++) {
        for (int j = 0; j <= n_x+1; j++) {
            if ((bctype(i,j) == 0 && bctype(i+1,j) == 0) ||
                (bctype(i,j) == 0 && bctype(i+1,j) == -3) ||
                (bctype(i,j) == -3 && bctype(i+1,j) == 0)) {
                // 内部面或并行交界面
                v_face(i,j) += 0.5 * (1/A_p(i,j) + 1/A_p(i+1,j)) * 
                              (p_prime(i+1,j) - p_prime(i,j)) * dx;
            }
            else if (bctype(i,j) == 0 && bctype(i+1,j) == -1) {
                v_face(i,j) = v_star(i,j);
            }
            else if (bctype(i,j) == -1 && bctype(i+1,j) == 0) {
                v_face(i,j) = v_star(i+1,j);
            }
        }
    }
}

// ============================================================================
// 后处理
// ============================================================================

/**
 * @brief 保存计算结果到文件
 * @param mesh 网格对象
 */
void post_processing(Mesh &mesh) {   
    std::ofstream outFile;
    
    outFile.open("u.dat");
    outFile << mesh.u_star;
    outFile.close();

    outFile.open("v.dat");
    outFile << mesh.v_star;
    outFile.close();

    outFile.open("p.dat");
    outFile << mesh.p_star;
    outFile.close();
}

// ============================================================================
// 进度条显示
// ============================================================================

/**
 * @brief 显示进度条和预估剩余时间
 * @param current_step 当前步数
 * @param total_steps 总步数
 * @param elapsed_time 已用时间(秒)
 */
void show_progress_bar(int current_step, int total_steps, double elapsed_time) {
    double progress = static_cast<double>(current_step) / total_steps;
    int bar_width = 50;
    int pos = static_cast<int>(bar_width * progress);
    double remaining_time = (elapsed_time / current_step) * (total_steps - current_step);
    
    std::cout << "[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    
    std::cout << "] " 
              << std::fixed << std::setprecision(2) << progress * 100 << "% "
              << "已用时间: " << elapsed_time << "秒 "
              << "预计剩余时间: " << remaining_time << "秒\r";
    std::cout.flush();
}

// ============================================================================
// 动量方程离散化 - 稳态SIMPLE算法
// ============================================================================

/**
 * @brief 构建动量方程系数矩阵(稳态SIMPLE)
 * @param mesh 网格对象
 * @param equ_u x方向动量方程
 * @param equ_v y方向动量方程
 * @param mu 动力粘度
 * @param alpha_uv 速度松弛因子
 * @details
 * 离散形式: A_p*u_P = A_e*u_E + A_w*u_W + A_n*u_N + A_s*u_S + S
 * 其中: A_nb = D_nb + max(0, ±F_nb)  (混合格式)
 *       D = mu*A/dx  (扩散系数)
 *       F = rho*u*A  (对流通量)
 *       S = (p_W - p_E)*dy/2 + (1-alpha)*A_p*u_old
 */
void momentum_function(Mesh &mesh, Equation &equ_u, Equation &equ_v, 
                      double mu, double alpha_uv) {
    int n_x = equ_u.n_x;
    int n_y = equ_u.n_y;
    
    // 计算扩散系数
    double D_e = dy * mu / dx;  // 东面扩散系数
    double D_w = dy * mu / dx;  // 西面扩散系数
    double D_n = dx * mu / dy;  // 北面扩散系数
    double D_s = dx * mu / dy;  // 南面扩散系数
    
    // 引用网格变量(简化代码)
    MatrixXd &zoneid = mesh.zoneid;
    MatrixXd &bctype = mesh.bctype;
    MatrixXd &u = mesh.u;
    MatrixXd &v = mesh.v;
    MatrixXd &u_face = mesh.u_face;
    MatrixXd &v_face = mesh.v_face;
    MatrixXd &p = mesh.p;
    MatrixXd &u_star = mesh.u_star;
    MatrixXd &v_star = mesh.v_star;
    MatrixXd &A_p = equ_u.A_p;
    MatrixXd &A_e = equ_u.A_e;
    MatrixXd &A_w = equ_u.A_w;
    MatrixXd &A_n = equ_u.A_n;
    MatrixXd &A_s = equ_u.A_s;
    VectorXd &source_x = equ_u.source;
    VectorXd &source_y = equ_v.source;
    vector<double> zoneu = mesh.zoneu;
    vector<double> zonev = mesh.zonev;
    
    // 遍历所有内部点
    for(int i = 0; i <= n_y+1; i++) {
        for(int j = 0; j <= n_x+1; j++) {
            if(bctype(i,j) != 0) continue;  // 跳过边界点
            
            int n = mesh.interid(i,j) ;  // 方程编号
            
            // 计算各面的对流通量
            double F_e = dy * u_face(i,j);      // 东面通量
            double F_w = dy * u_face(i,j-1);    // 西面通量
            double F_n = dx * v_face(i-1,j);    // 北面通量
            double F_s = dx * v_face(i,j);      // 南面通量
            
            double Ap_temp = 0;
            double source_x_temp = 0, source_y_temp = 0;
            
            // ===== 计算压力源项 =====
            // x方向压力梯度
            if((bctype(i,j-1) == 0 || bctype(i,j-1) == -3) && 
               (bctype(i,j+1) == 0 || bctype(i,j+1) == -3)) {
                // 两侧都是内部点,使用中心差分
                source_x_temp = 0.5 * alpha_uv * (p(i,j-1) - p(i,j+1)) * dy;
            } else if(bctype(i,j-1) == -1) {
                // 左侧是压力出口(p=0)
                source_x_temp = 0.5 * alpha_uv * (-p(i,j+1)) * dy;
            } else if(bctype(i,j+1) == -1) {
                // 右侧是压力出口
                source_x_temp = 0.5 * alpha_uv * p(i,j-1) * dy;
            } else if(bctype(i,j-1) == -2) {
                // 左侧是速度入口
                source_x_temp = 0.5 * alpha_uv * (p(i,j) - p(i,j+1)) * dy;
            } else if(bctype(i,j+1) == -2) {
                // 右侧是速度入口
                source_x_temp = 0.5 * alpha_uv * (p(i,j-1) - p(i,j)) * dy;
            } else if(bctype(i,j-1) != 0 && bctype(i,j+1) == 0) {
                // 左边界,右内部
                source_x_temp = 0.5 * alpha_uv * (p(i,j) - p(i,j+1)) * dy;
            } else if(bctype(i,j-1) == 0 && bctype(i,j+1) != 0) {
                // 左内部,右边界
                source_x_temp = 0.5 * alpha_uv * (p(i,j-1) - p(i,j)) * dy;
            } else {
                source_x_temp = 0.0;
            }
            
            // y方向压力梯度(类似处理)
            if((bctype(i-1,j) == 0 || bctype(i-1,j) == -3) && 
               (bctype(i+1,j) == 0 || bctype(i+1,j) == -3)) {
                source_y_temp = 0.5 * alpha_uv * (p(i+1,j) - p(i-1,j)) * dx;
            } else if(bctype(i-1,j) == -1) {
                source_y_temp = 0.5 * alpha_uv * p(i+1,j) * dx;
            } else if(bctype(i+1,j) == -1) {
                source_y_temp = 0.5 * alpha_uv * (-p(i-1,j)) * dx;
            } else if(bctype(i-1,j) == -2) {
                source_y_temp = 0.5 * alpha_uv * (p(i+1,j) - p(i,j)) * dx;
            } else if(bctype(i+1,j) == -2) {
                source_y_temp = 0.5 * alpha_uv * (p(i,j) - p(i-1,j)) * dx;
            } else if(bctype(i-1,j) != 0 && bctype(i+1,j) == 0) {
                source_y_temp = 0.5 * alpha_uv * (p(i+1,j) - p(i,j)) * dx;
            } else if(bctype(i-1,j) == 0 && bctype(i+1,j) != 0) {
                source_y_temp = 0.5 * alpha_uv * (p(i,j) - p(i-1,j)) * dx;
            } else {
                source_y_temp = 0.0;
            }
            
            // ===== 计算东面系数 =====
            if(bctype(i,j+1) == 0 || bctype(i,j+1) == -3) {
                // 东邻是内部点或并行接口
                A_e(i,j) = D_e + max(0.0, -F_e);  // 混合格式
                Ap_temp += D_e + max(0.0, F_e);
            } else if(bctype(i,j+1) > 0) {
                // 东邻是固壁(无滑移边界)
                A_e(i,j) = 0;
                Ap_temp += 2*D_e + max(0.0, F_e);
                source_x_temp += alpha_uv * zoneu[zoneid(i,j+1)] * (2*D_e + max(0.0, -F_e));
                source_y_temp += alpha_uv * zonev[zoneid(i,j+1)] * (2*D_e + max(0.0, -F_e));
            } else if(bctype(i,j+1) == -1) {
                // 东邻是压力出口
                A_e(i,j) = 0;
                Ap_temp += D_e + max(0.0, F_e);
                source_x_temp += alpha_uv * u_star(i,j) * (D_e + max(0.0, -F_e));
                source_y_temp += alpha_uv * v_star(i,j) * (D_e + max(0.0, -F_e));
            } else if(bctype(i,j+1) > -10) {
                // 东邻是其他边界(如速度入口)
                A_e(i,j) = 0;
                Ap_temp += D_e + max(0.0, F_e);
                source_x_temp += alpha_uv * zoneu[zoneid(i,j+1)] * (D_e + max(0.0, -F_e));
                source_y_temp += alpha_uv * zonev[zoneid(i,j+1)] * (D_e + max(0.0, -F_e));
            }
            
            // ===== 计算西面系数 =====
            if(bctype(i,j-1) == 0 || bctype(i,j-1) == -3) {
                A_w(i,j) = D_w + max(0.0, F_w);
                Ap_temp += D_w + max(0.0, -F_w);
            } else if(bctype(i,j-1) == -1) {
                A_w(i,j) = 0;
                Ap_temp += D_w + max(0.0, -F_w);
                source_x_temp += alpha_uv * u_star(i,j) * (D_w + max(0.0, F_w));
                source_y_temp += alpha_uv * v_star(i,j) * (D_w + max(0.0, F_w));
            } else if(bctype(i,j-1) > 0) {
                A_w(i,j) = 0;
                Ap_temp += 2*D_w + max(0.0, -F_w);
                source_x_temp += alpha_uv * zoneu[zoneid(i,j-1)] * (2*D_w + max(0.0, F_w));
                source_y_temp += alpha_uv * zonev[zoneid(i,j-1)] * (2*D_w + max(0.0, F_w));
            } else if(bctype(i,j-1) > -10) {
                A_w(i,j) = 0;
                Ap_temp += D_w + max(0.0, -F_w);
                source_x_temp += alpha_uv * zoneu[zoneid(i,j-1)] * (D_w + max(0.0, F_w));
                source_y_temp += alpha_uv * zonev[zoneid(i,j-1)] * (D_w + max(0.0, F_w));
            }
            
            // ===== 计算北面系数 =====
            if(bctype(i-1,j) == 0) {
                A_n(i,j) = D_n + max(0.0, -F_n);
                Ap_temp += D_n + max(0.0, F_n);
            } else if(bctype(i-1,j) == -1) {
                A_n(i,j) = 0;
                Ap_temp += D_n + max(0.0, F_n);
                source_x_temp += alpha_uv * u_star(i-1,j) * (D_n + max(0.0, -F_n));
                source_y_temp += alpha_uv * v_star(i-1,j) * (D_n + max(0.0, -F_n));
            } else if(bctype(i-1,j) > 0) {
                A_n(i,j) = 0;
                Ap_temp += 2*D_n + max(0.0, F_n);
                source_x_temp += alpha_uv * zoneu[zoneid(i-1,j)] * (2*D_n + max(0.0, -F_n));
                source_y_temp += alpha_uv * zonev[zoneid(i-1,j)] * (2*D_n + max(0.0, -F_n));
            } else if(bctype(i-1,j) > -10) {
                A_n(i,j) = 0;
                Ap_temp += D_n + max(0.0, F_n);
                source_x_temp += alpha_uv * zoneu[zoneid(i-1,j)] * (D_n + max(0.0, -F_n));
                source_y_temp += alpha_uv * zonev[zoneid(i-1,j)] * (D_n + max(0.0, -F_n));
            }
            
            // ===== 计算南面系数 =====
            if(bctype(i+1,j) == 0) {
                A_s(i,j) = D_s + max(0.0, F_s);
                Ap_temp += D_s + max(0.0, -F_s);
            } else if(bctype(i+1,j) == -1) {
                A_s(i,j) = 0;
                Ap_temp += D_s + max(0.0, -F_s);
                source_x_temp += alpha_uv * u(i+1,j) * (D_s + max(0.0, F_s));
                source_y_temp += alpha_uv * v(i+1,j) * (D_s + max(0.0, F_s));
            } else if(bctype(i+1,j) > 0) {
                A_s(i,j) = 0;
                Ap_temp += 2*D_s + max(0.0, -F_s);
                source_x_temp += alpha_uv * zoneu[zoneid(i+1,j)] * (2*D_s + max(0.0, F_s));
                source_y_temp += alpha_uv * zonev[zoneid(i+1,j)] * (2*D_s + max(0.0, F_s));
            } else if(bctype(i+1,j) > -10) {
                A_s(i,j) = 0;
                Ap_temp += D_s + max(0.0, -F_s);
                source_x_temp += alpha_uv * zoneu[zoneid(i+1,j)] * (D_s + max(0.0, F_s));
                source_y_temp += alpha_uv * zonev[zoneid(i+1,j)] * (D_s + max(0.0, F_s));
            }
            
            // 设置中心系数
            A_p(i,j) = Ap_temp;
            
            // 添加松弛项: (1-alpha)*A_p*u_old
            source_x_temp += (1 - alpha_uv) * A_p(i,j) * u_star(i,j);
            source_y_temp += (1 - alpha_uv) * A_p(i,j) * v_star(i,j);
            
            // 设置源项
            source_x[n] = source_x_temp;
            source_y[n] = source_y_temp;
        }
    }

    // 应用松弛因子到邻接系数
    A_e = alpha_uv * A_e;
    A_w = alpha_uv * A_w;
    A_n = alpha_uv * A_n;
    A_s = alpha_uv * A_s;
    
    // 复制系数到y方向动量方程
    equ_v.A_p = equ_u.A_p;
    equ_v.A_w = equ_u.A_w;
    equ_v.A_e = equ_u.A_e;
    equ_v.A_n = equ_u.A_n;
    equ_v.A_s = equ_u.A_s;
}

// ============================================================================
// 动量方程离散化 - 非稳态SIMPLE算法
// ============================================================================

/**
 * @brief 构建动量方程系数矩阵(非稳态SIMPLE)
 * @param mesh 网格对象
 * @param equ_u x方向动量方程
 * @param equ_v y方向动量方程
 * @param mu 动力粘度
 * @param dt 时间步长
 * @param alpha_uv 速度松弛因子
 * @details
 * 增加时间项: A_p = A_p_稳态 + dx*dy/dt
 * 源项增加: S = S_稳态 + alpha*dx*dy*u0/dt
 */
void momentum_function_unsteady(Mesh &mesh, Equation &equ_u, Equation &equ_v,
                               double mu, double dt, double alpha_uv) {
    int n_x = equ_u.n_x;
    int n_y = equ_u.n_y;
    
    // 计算扩散系数
    double D_e = dy * mu / dx;  // 东面扩散系数
    double D_w = dy * mu / dx;  // 西面扩散系数
    double D_n = dx * mu / dy;  // 北面扩散系数
    double D_s = dx * mu / dy;  // 南面扩散系数
    
    // 引用网格变量(简化代码)
    MatrixXd &zoneid = mesh.zoneid;
    MatrixXd &bctype = mesh.bctype;
    MatrixXd &u = mesh.u;
    MatrixXd &v = mesh.v;
    MatrixXd &u_face = mesh.u_face;
    MatrixXd &v_face = mesh.v_face;
    MatrixXd &p = mesh.p;
    MatrixXd &u_star = mesh.u_star;
    MatrixXd &v_star = mesh.v_star;
    MatrixXd &A_p = equ_u.A_p;
    MatrixXd &A_e = equ_u.A_e;
    MatrixXd &A_w = equ_u.A_w;
    MatrixXd &A_n = equ_u.A_n;
    MatrixXd &A_s = equ_u.A_s;
    VectorXd &source_x = equ_u.source;
    VectorXd &source_y = equ_v.source;
    vector<double> zoneu = mesh.zoneu;
    vector<double> zonev = mesh.zonev;
    
    // 遍历所有内部点
    for(int i = 0; i <= n_y+1; i++) {
        for(int j = 0; j <= n_x+1; j++) {
            if(bctype(i,j) != 0) continue;  // 跳过边界点
            
            int n = mesh.interid(i,j) ;  // 方程编号
            
            // 计算各面的对流通量
            double F_e = dy * u_face(i,j);      // 东面通量
            double F_w = dy * u_face(i,j-1);    // 西面通量
            double F_n = dx * v_face(i-1,j);    // 北面通量
            double F_s = dx * v_face(i,j);      // 南面通量
            
            double Ap_temp = 0;
            double source_x_temp = 0, source_y_temp = 0;
            
            // ===== 计算压力源项 =====
            // x方向压力梯度
            if((bctype(i,j-1) == 0 || bctype(i,j-1) == -3) && 
               (bctype(i,j+1) == 0 || bctype(i,j+1) == -3)) {
                // 两侧都是内部点,使用中心差分
                source_x_temp = 0.5  * (p(i,j-1) - p(i,j+1)) * dy;
            } else if(bctype(i,j-1) == -1) {
                // 左侧是压力出口(p=0)
                source_x_temp = 0.5  * (-p(i,j+1)) * dy;
            } else if(bctype(i,j+1) == -1) {
                // 右侧是压力出口
                source_x_temp = 0.5  * p(i,j-1) * dy;
            } else if(bctype(i,j-1) == -2) {
                // 左侧是速度入口
                source_x_temp = 0.5  * (p(i,j) - p(i,j+1)) * dy;
            } else if(bctype(i,j+1) == -2) {
                // 右侧是速度入口
                source_x_temp = 0.5  * (p(i,j-1) - p(i,j)) * dy;
            } else if(bctype(i,j-1) != 0 && bctype(i,j+1) == 0) {
                // 左边界,右内部
                source_x_temp = 0.5 * (p(i,j) - p(i,j+1)) * dy;
            } else if(bctype(i,j-1) == 0 && bctype(i,j+1) != 0) {
                // 左内部,右边界
                source_x_temp = 0.5 * (p(i,j-1) - p(i,j)) * dy;
            } else {
                source_x_temp = 0.0;
            }
            
            // y方向压力梯度(类似处理)
            if((bctype(i-1,j) == 0 || bctype(i-1,j) == -3) && 
               (bctype(i+1,j) == 0 || bctype(i+1,j) == -3)) {
                source_y_temp = 0.5  * (p(i+1,j) - p(i-1,j)) * dx;
            } else if(bctype(i-1,j) == -1) {
                source_y_temp = 0.5 * p(i+1,j) * dx;
            } else if(bctype(i+1,j) == -1) {
                source_y_temp = 0.5  * (-p(i-1,j)) * dx;
            } else if(bctype(i-1,j) == -2) {
                source_y_temp = 0.5  * (p(i+1,j) - p(i,j)) * dx;
            } else if(bctype(i+1,j) == -2) {
                source_y_temp = 0.5  * (p(i,j) - p(i-1,j)) * dx;
            } else if(bctype(i-1,j) != 0 && bctype(i+1,j) == 0) {
                source_y_temp = 0.5  * (p(i+1,j) - p(i,j)) * dx;
            } else if(bctype(i-1,j) == 0 && bctype(i+1,j) != 0) {
                source_y_temp = 0.5  * (p(i,j) - p(i-1,j)) * dx;
            } else {
                source_y_temp = 0.0;
            }
            
            // ===== 计算东面系数 =====
            if(bctype(i,j+1) == 0 || bctype(i,j+1) == -3) {
                // 东邻是内部点或并行接口
                A_e(i,j) = D_e + max(0.0, -F_e);  // 混合格式
                Ap_temp += D_e + max(0.0, F_e);
            } else if(bctype(i,j+1) > 0) {
                // 东邻是固壁(无滑移边界)
                A_e(i,j) = 0;
                Ap_temp += 2*D_e + max(0.0, F_e);
                source_x_temp += zoneu[zoneid(i,j+1)] * (2*D_e + max(0.0, -F_e));
                source_y_temp +=  zonev[zoneid(i,j+1)] * (2*D_e + max(0.0, -F_e));
            } else if(bctype(i,j+1) == -1) {
                // 东邻是压力出口
                A_e(i,j) = 0;
                Ap_temp += D_e + max(0.0, F_e);
                source_x_temp += u_star(i,j) * (D_e + max(0.0, -F_e));
                source_y_temp +=  v_star(i,j) * (D_e + max(0.0, -F_e));
            } else if(bctype(i,j+1) > -10) {
                // 东邻是其他边界(如速度入口)
                A_e(i,j) = 0;
                Ap_temp += D_e + max(0.0, F_e);
                source_x_temp +=  zoneu[zoneid(i,j+1)] * (D_e + max(0.0, -F_e));
                source_y_temp +=  zonev[zoneid(i,j+1)] * (D_e + max(0.0, -F_e));
            }
            
            // ===== 计算西面系数 =====
            if(bctype(i,j-1) == 0 || bctype(i,j-1) == -3) {
                A_w(i,j) = D_w + max(0.0, F_w);
                Ap_temp += D_w + max(0.0, -F_w);
            } else if(bctype(i,j-1) == -1) {
                A_w(i,j) = 0;
                Ap_temp += D_w + max(0.0, -F_w);
                source_x_temp +=  u_star(i,j) * (D_w + max(0.0, F_w));
                source_y_temp +=  v_star(i,j) * (D_w + max(0.0, F_w));
            } else if(bctype(i,j-1) > 0) {
                A_w(i,j) = 0;
                Ap_temp += 2*D_w + max(0.0, -F_w);
                source_x_temp +=  zoneu[zoneid(i,j-1)] * (2*D_w + max(0.0, F_w));
                source_y_temp +=  zonev[zoneid(i,j-1)] * (2*D_w + max(0.0, F_w));
            } else if(bctype(i,j-1) > -10) {
                A_w(i,j) = 0;
                Ap_temp += D_w + max(0.0, -F_w);
                source_x_temp +=  zoneu[zoneid(i,j-1)] * (D_w + max(0.0, F_w));
                source_y_temp +=  zonev[zoneid(i,j-1)] * (D_w + max(0.0, F_w));
            }
            
            // ===== 计算北面系数 =====
            if(bctype(i-1,j) == 0) {
                A_n(i,j) = D_n + max(0.0, -F_n);
                Ap_temp += D_n + max(0.0, F_n);
            } else if(bctype(i-1,j) == -1) {
                A_n(i,j) = 0;
                Ap_temp += D_n + max(0.0, F_n);
                source_x_temp +=  u_star(i-1,j) * (D_n + max(0.0, -F_n));
                source_y_temp +=  v_star(i-1,j) * (D_n + max(0.0, -F_n));
            } else if(bctype(i-1,j) > 0) {
                A_n(i,j) = 0;
                Ap_temp += 2*D_n + max(0.0, F_n);
                source_x_temp +=  zoneu[zoneid(i-1,j)] * (2*D_n + max(0.0, -F_n));
                source_y_temp +=  zonev[zoneid(i-1,j)] * (2*D_n + max(0.0, -F_n));
            } else if(bctype(i-1,j) > -10) {
                A_n(i,j) = 0;
                Ap_temp += D_n + max(0.0, F_n);
                source_x_temp +=  zoneu[zoneid(i-1,j)] * (D_n + max(0.0, -F_n));
                source_y_temp +=  zonev[zoneid(i-1,j)] * (D_n + max(0.0, -F_n));
            }
            
            // ===== 计算南面系数 =====
            if(bctype(i+1,j) == 0) {
                A_s(i,j) = D_s + max(0.0, F_s);
                Ap_temp += D_s + max(0.0, -F_s);
            } else if(bctype(i+1,j) == -1) {
                A_s(i,j) = 0;
                Ap_temp += D_s + max(0.0, -F_s);
                source_x_temp +=  u(i+1,j) * (D_s + max(0.0, F_s));
                source_y_temp +=  v(i+1,j) * (D_s + max(0.0, F_s));
            } else if(bctype(i+1,j) > 0) {
                A_s(i,j) = 0;
                Ap_temp += 2*D_s + max(0.0, -F_s);
                source_x_temp +=  zoneu[zoneid(i+1,j)] * (2*D_s + max(0.0, F_s));
                source_y_temp +=  zonev[zoneid(i+1,j)] * (2*D_s + max(0.0, F_s));
            } else if(bctype(i+1,j) > -10) {
                A_s(i,j) = 0;
                Ap_temp += D_s + max(0.0, -F_s);
                source_x_temp +=  zoneu[zoneid(i+1,j)] * (D_s + max(0.0, F_s));
                source_y_temp +=  zonev[zoneid(i+1,j)] * (D_s + max(0.0, F_s));
            }
            
            // 关键区别: 添加时间项到中心系数
            A_p(i,j) = Ap_temp + dx*dy/dt;
            
            // 关键区别: 源项添加旧时间步项
            source_x_temp +=  dx*dy * mesh.u0(i,j) / dt;
            source_y_temp +=  dx*dy * mesh.v0(i,j) / dt;
            
            // 设置源项
            source_x[n] = source_x_temp;
            source_y[n] = source_y_temp;
        }
    }

    // 应用松弛因子到邻接系数
    A_e = A_e;
    A_w = A_w;
    A_n = A_n;
    A_s = A_s;
    
    // 复制系数到y方向动量方程
    equ_v.A_p = equ_u.A_p;
    equ_v.A_w = equ_u.A_w;
    equ_v.A_e = equ_u.A_e;
    equ_v.A_n = equ_u.A_n;
    equ_v.A_s = equ_u.A_s;
   
}

// ============================================================================
// 动量方程离散化 - PISO算法
// ============================================================================

/**
 * @brief 构建动量方程系数矩阵(PISO算法)
 * @param mesh 网格对象
 * @param equ_u x方向动量方程
 * @param equ_v y方向动量方程
 * @param mu 动力粘度
 * @param dt 时间步长
 * @details
 * PISO算法特点: 不使用速度松弛(alpha_uv=1.0)
 * 时间离散: 隐式欧拉格式
 */
void momentum_function_PISO(Mesh &mesh, Equation &equ_u, Equation &equ_v,
                            double mu, double dt) {
    int n_x = equ_u.n_x;
    int n_y = equ_u.n_y;
    
    // 计算扩散系数
    double D_e = dy * mu / dx;  // 东面扩散系数
    double D_w = dy * mu / dx;  // 西面扩散系数
    double D_n = dx * mu / dy;  // 北面扩散系数
    double D_s = dx * mu / dy;  // 南面扩散系数
    
    // 引用网格变量(简化代码)
    MatrixXd &zoneid = mesh.zoneid;
    MatrixXd &bctype = mesh.bctype;
    MatrixXd &u = mesh.u;
    MatrixXd &v = mesh.v;
    MatrixXd &u_face = mesh.u_face;
    MatrixXd &v_face = mesh.v_face;
    MatrixXd &p = mesh.p;
    MatrixXd &u_star = mesh.u_star;
    MatrixXd &v_star = mesh.v_star;
    MatrixXd &A_p = equ_u.A_p;
    MatrixXd &A_e = equ_u.A_e;
    MatrixXd &A_w = equ_u.A_w;
    MatrixXd &A_n = equ_u.A_n;
    MatrixXd &A_s = equ_u.A_s;
    VectorXd &source_x = equ_u.source;
    VectorXd &source_y = equ_v.source;
    vector<double> zoneu = mesh.zoneu;
    vector<double> zonev = mesh.zonev;
    
    // 遍历所有内部点
    for(int i = 0; i <= n_y+1; i++) {
        for(int j = 0; j <= n_x+1; j++) {
            if(bctype(i,j) != 0) continue;  // 跳过边界点
            
            int n = mesh.interid(i,j) ;  // 方程编号
            
            // 计算各面的对流通量
            double F_e = dy * u_face(i,j);      // 东面通量
            double F_w = dy * u_face(i,j-1);    // 西面通量
            double F_n = dx * v_face(i-1,j);    // 北面通量
            double F_s = dx * v_face(i,j);      // 南面通量
            
            double Ap_temp = 0;
            double source_x_temp = 0, source_y_temp = 0;
            
            // ===== 计算压力源项 =====
            // x方向压力梯度
            if((bctype(i,j-1) == 0 || bctype(i,j-1) == -3) && 
               (bctype(i,j+1) == 0 || bctype(i,j+1) == -3)) {
                // 两侧都是内部点,使用中心差分
                source_x_temp = 0.5  * (p(i,j-1) - p(i,j+1)) * dy;
            } else if(bctype(i,j-1) == -1) {
                // 左侧是压力出口(p=0)
                source_x_temp = 0.5  * (-p(i,j+1)) * dy;
            } else if(bctype(i,j+1) == -1) {
                // 右侧是压力出口
                source_x_temp = 0.5  * p(i,j-1) * dy;
            } else if(bctype(i,j-1) == -2) {
                // 左侧是速度入口
                source_x_temp = 0.5  * (p(i,j) - p(i,j+1)) * dy;
            } else if(bctype(i,j+1) == -2) {
                // 右侧是速度入口
                source_x_temp = 0.5  * (p(i,j-1) - p(i,j)) * dy;
            } else if(bctype(i,j-1) != 0 && bctype(i,j+1) == 0) {
                // 左边界,右内部
                source_x_temp = 0.5 * (p(i,j) - p(i,j+1)) * dy;
            } else if(bctype(i,j-1) == 0 && bctype(i,j+1) != 0) {
                // 左内部,右边界
                source_x_temp = 0.5 * (p(i,j-1) - p(i,j)) * dy;
            } else {
                source_x_temp = 0.0;
            }
            
            // y方向压力梯度(类似处理)
            if((bctype(i-1,j) == 0 || bctype(i-1,j) == -3) && 
               (bctype(i+1,j) == 0 || bctype(i+1,j) == -3)) {
                source_y_temp = 0.5  * (p(i+1,j) - p(i-1,j)) * dx;
            } else if(bctype(i-1,j) == -1) {
                source_y_temp = 0.5 * p(i+1,j) * dx;
            } else if(bctype(i+1,j) == -1) {
                source_y_temp = 0.5  * (-p(i-1,j)) * dx;
            } else if(bctype(i-1,j) == -2) {
                source_y_temp = 0.5  * (p(i+1,j) - p(i,j)) * dx;
            } else if(bctype(i+1,j) == -2) {
                source_y_temp = 0.5  * (p(i,j) - p(i-1,j)) * dx;
            } else if(bctype(i-1,j) != 0 && bctype(i+1,j) == 0) {
                source_y_temp = 0.5  * (p(i+1,j) - p(i,j)) * dx;
            } else if(bctype(i-1,j) == 0 && bctype(i+1,j) != 0) {
                source_y_temp = 0.5  * (p(i,j) - p(i-1,j)) * dx;
            } else {
                source_y_temp = 0.0;
            }
            
            // ===== 计算东面系数 =====
            if(bctype(i,j+1) == 0 || bctype(i,j+1) == -3) {
                // 东邻是内部点或并行接口
                A_e(i,j) = D_e + max(0.0, -F_e);  // 混合格式
                Ap_temp += D_e + max(0.0, F_e);
            } else if(bctype(i,j+1) > 0) {
                // 东邻是固壁(无滑移边界)
                A_e(i,j) = 0;
                Ap_temp += 2*D_e + max(0.0, F_e);
                source_x_temp += zoneu[zoneid(i,j+1)] * (2*D_e + max(0.0, -F_e));
                source_y_temp +=  zonev[zoneid(i,j+1)] * (2*D_e + max(0.0, -F_e));
            } else if(bctype(i,j+1) == -1) {
                // 东邻是压力出口
                A_e(i,j) = 0;
                Ap_temp += D_e + max(0.0, F_e);
                source_x_temp += u_star(i,j) * (D_e + max(0.0, -F_e));
                source_y_temp +=  v_star(i,j) * (D_e + max(0.0, -F_e));
            } else if(bctype(i,j+1) > -10) {
                // 东邻是其他边界(如速度入口)
                A_e(i,j) = 0;
                Ap_temp += D_e + max(0.0, F_e);
                source_x_temp +=  zoneu[zoneid(i,j+1)] * (D_e + max(0.0, -F_e));
                source_y_temp +=  zonev[zoneid(i,j+1)] * (D_e + max(0.0, -F_e));
            }
            
            // ===== 计算西面系数 =====
            if(bctype(i,j-1) == 0 || bctype(i,j-1) == -3) {
                A_w(i,j) = D_w + max(0.0, F_w);
                Ap_temp += D_w + max(0.0, -F_w);
            } else if(bctype(i,j-1) == -1) {
                A_w(i,j) = 0;
                Ap_temp += D_w + max(0.0, -F_w);
                source_x_temp +=  u_star(i,j) * (D_w + max(0.0, F_w));
                source_y_temp +=  v_star(i,j) * (D_w + max(0.0, F_w));
            } else if(bctype(i,j-1) > 0) {
                A_w(i,j) = 0;
                Ap_temp += 2*D_w + max(0.0, -F_w);
                source_x_temp +=  zoneu[zoneid(i,j-1)] * (2*D_w + max(0.0, F_w));
                source_y_temp +=  zonev[zoneid(i,j-1)] * (2*D_w + max(0.0, F_w));
            } else if(bctype(i,j-1) > -10) {
                A_w(i,j) = 0;
                Ap_temp += D_w + max(0.0, -F_w);
                source_x_temp +=  zoneu[zoneid(i,j-1)] * (D_w + max(0.0, F_w));
                source_y_temp +=  zonev[zoneid(i,j-1)] * (D_w + max(0.0, F_w));
            }
            
            // ===== 计算北面系数 =====
            if(bctype(i-1,j) == 0) {
                A_n(i,j) = D_n + max(0.0, -F_n);
                Ap_temp += D_n + max(0.0, F_n);
            } else if(bctype(i-1,j) == -1) {
                A_n(i,j) = 0;
                Ap_temp += D_n + max(0.0, F_n);
                source_x_temp +=  u_star(i-1,j) * (D_n + max(0.0, -F_n));
                source_y_temp +=  v_star(i-1,j) * (D_n + max(0.0, -F_n));
            } else if(bctype(i-1,j) > 0) {
                A_n(i,j) = 0;
                Ap_temp += 2*D_n + max(0.0, F_n);
                source_x_temp +=  zoneu[zoneid(i-1,j)] * (2*D_n + max(0.0, -F_n));
                source_y_temp +=  zonev[zoneid(i-1,j)] * (2*D_n + max(0.0, -F_n));
            } else if(bctype(i-1,j) > -10) {
                A_n(i,j) = 0;
                Ap_temp += D_n + max(0.0, F_n);
                source_x_temp +=  zoneu[zoneid(i-1,j)] * (D_n + max(0.0, -F_n));
                source_y_temp +=  zonev[zoneid(i-1,j)] * (D_n + max(0.0, -F_n));
            }
            
            // ===== 计算南面系数 =====
            if(bctype(i+1,j) == 0) {
                A_s(i,j) = D_s + max(0.0, F_s);
                Ap_temp += D_s + max(0.0, -F_s);
            } else if(bctype(i+1,j) == -1) {
                A_s(i,j) = 0;
                Ap_temp += D_s + max(0.0, -F_s);
                source_x_temp +=  u(i+1,j) * (D_s + max(0.0, F_s));
                source_y_temp +=  v(i+1,j) * (D_s + max(0.0, F_s));
            } else if(bctype(i+1,j) > 0) {
                A_s(i,j) = 0;
                Ap_temp += 2*D_s + max(0.0, -F_s);
                source_x_temp +=  zoneu[zoneid(i+1,j)] * (2*D_s + max(0.0, F_s));
                source_y_temp +=  zonev[zoneid(i+1,j)] * (2*D_s + max(0.0, F_s));
            } else if(bctype(i+1,j) > -10) {
                A_s(i,j) = 0;
                Ap_temp += D_s + max(0.0, -F_s);
                source_x_temp +=  zoneu[zoneid(i+1,j)] * (D_s + max(0.0, F_s));
                source_y_temp +=  zonev[zoneid(i+1,j)] * (D_s + max(0.0, F_s));
            }
            
            // 关键区别: 添加时间项到中心系数
            A_p(i,j) = Ap_temp + dx*dy/dt;
            
            // 关键区别: 源项添加旧时间步项
            source_x_temp +=  dx*dy * mesh.u0(i,j) / dt;
            source_y_temp +=  dx*dy * mesh.v0(i,j) / dt;
            
            // 设置源项
            source_x[n] = source_x_temp;
            source_y[n] = source_y_temp;
        }
    }

    // 应用松弛因子到邻接系数
    A_e =  A_e;
    A_w =  A_w;
    A_n =  A_n;
    A_s =  A_s;
    
    // 复制系数到y方向动量方程
    equ_v.A_p = equ_u.A_p;
    equ_v.A_w = equ_u.A_w;
    equ_v.A_e = equ_u.A_e;
    equ_v.A_n = equ_u.A_n;
    equ_v.A_s = equ_u.A_s;
}

// ============================================================================
// 并行计算相关函数
// ============================================================================

/**
 * @brief 垂直分割网格(用于并行计算)
 * @param original_mesh 原始网格
 * @param n 分割数量
 * @return 子网格向量
 * @details
 * 1. 将网格在x方向分割成n个子区域
 * 2. 在接口处添加虚拟层(bctype=-3)
 * 3. 每个子网格包含完整的边界信息
 */
vector<Mesh> splitMeshVertically(const Mesh& original_mesh, int n) {
    std::vector<Mesh> sub_meshes;
    int original_nx = original_mesh.nx;
    int original_ny = original_mesh.ny;
    
    // 计算每个子网格的宽度
    vector<int> widths(n);
    int remaining = original_nx;
    for(int k = 0; k < n; k++) {
        widths[k] = (remaining + (n-k-1)) / (n-k);
        remaining -= widths[k];
    }

    int start_idx = 0;
    for(int k = 0; k < n; k++) {
        Mesh sub_mesh(original_ny, widths[k]);
        sub_mesh.initializeToZero();
        sub_mesh.zoneu = original_mesh.zoneu;
        sub_mesh.zonev = original_mesh.zonev;

        // 复制网格数据
        for(int i = 0; i <= original_ny + 1; i++) {
            for(int j = 0; j <= widths[k] + 1; j++) {
                int orig_j = start_idx + j;

                // 处理边界类型
                if(k == 0 && j == 0) {
                    // 第一个子网格的左边界
                    sub_mesh.bctype(i,j) = original_mesh.bctype(i,0);
                } else if(k == n-1 && j == widths[k] + 1) {
                    // 最后一个子网格的右边界
                    sub_mesh.bctype(i,j) = original_mesh.bctype(i,original_nx+1);
                } else if(j == 0 || j == widths[k] + 1) {
                    // 内部接口边界(标记为-3)
                    sub_mesh.bctype(i,j) = -3;
                } else {
                    sub_mesh.bctype(i,j) = original_mesh.bctype(i,orig_j);
                }

                // 复制其他数据
                if(orig_j <= original_nx + 1) {
                    sub_mesh.zoneid(i,j) = original_mesh.zoneid(i,orig_j);
                    sub_mesh.u(i,j) = original_mesh.u(i,orig_j);
                    sub_mesh.v(i,j) = original_mesh.v(i,orig_j);
                    sub_mesh.p(i,j) = original_mesh.p(i,orig_j);
                }
            }
        }

        // 检查是否需要额外的交换层
        bool left_is_interface = true;
        bool right_is_interface = true;

        for(int i = 0; i <= original_ny + 1; i++) {
            if(sub_mesh.bctype(i,0) != -3) left_is_interface = false;
            if(sub_mesh.bctype(i,widths[k]+1) != -3) right_is_interface = false;
        }

        // 如果有接口,创建扩展网格
        if(left_is_interface || right_is_interface) {
            int extra_cols = (left_is_interface ? 1 : 0) + (right_is_interface ? 1 : 0);
            Mesh sub_mesh2(original_ny, widths[k] + extra_cols);
            sub_mesh2.initializeToZero();
            sub_mesh2.zoneu = sub_mesh.zoneu;
            sub_mesh2.zonev = sub_mesh.zonev;

            int offset = left_is_interface ? 1 : 0;
            
            // 复制并扩展数据
            for(int i = 0; i <= original_ny + 1; i++) {
                for(int j = 0; j <= widths[k] + 1; j++) {
                    int new_j = j + offset;
                    sub_mesh2.bctype(i,new_j) = sub_mesh.bctype(i,j);
                    sub_mesh2.zoneid(i,new_j) = sub_mesh.zoneid(i,j);
                    sub_mesh2.u(i,new_j) = sub_mesh.u(i,j);
                    sub_mesh2.v(i,new_j) = sub_mesh.v(i,j);
                    sub_mesh2.p(i,new_j) = sub_mesh.p(i,j);
                }

                // 设置额外的接口列
                if(left_is_interface) {
                    sub_mesh2.bctype(i,0) = -3;
                    sub_mesh2.zoneid(i,0) = 0;
                    sub_mesh2.u(i,0) = 0.0;
                    sub_mesh2.v(i,0) = 0.0;
                }

                if(right_is_interface) {
                    int last_col = widths[k] + offset + 2;
                    sub_mesh2.bctype(i,last_col) = -3;
                    sub_mesh2.zoneid(i,last_col) = 0;
                    sub_mesh2.u(i,last_col) = 0.0;
                    sub_mesh2.v(i,last_col) = 0.0;
                    sub_mesh2.p(i,last_col) = 0.0;
                }
            }
            
            sub_mesh2.u_face.setZero();
            sub_mesh2.v_face.setZero();
            sub_mesh2.p.setZero();
            
            sub_mesh2.initializeBoundaryConditions();
            sub_mesh2.createInterId();
            sub_meshes.push_back(sub_mesh2);
        } else {
            sub_mesh.initializeBoundaryConditions();
            sub_mesh.createInterId();
            sub_meshes.push_back(sub_mesh);
        }

        start_idx += widths[k];
    }

    return sub_meshes;
}

/**
 * @brief 合并子网格(去除接口列)
 * @param sub_meshes 子网格向量
 * @return 合并后的完整网格
 */
Mesh mergeMeshesWithoutInterface(const std::vector<Mesh>& sub_meshes) {
    // 计算合并后的总宽度(不包括接口列)
    int total_nx = 0;
    int ny = sub_meshes[0].ny;
    
    for(size_t i = 0; i < sub_meshes.size(); i++) {
        for(int j = 0; j <= sub_meshes[i].nx + 1; j++) {
            if(sub_meshes[i].bctype(0,j) != -3) {
                total_nx++;
            }
        }
    }
    total_nx = total_nx;  // 调整

    Mesh merged_mesh(ny, total_nx);
    merged_mesh.zoneu = sub_meshes[0].zoneu;
    merged_mesh.zonev = sub_meshes[0].zonev;
    
    int current_col = 0;
    
    // 逐个合并子网格
    for(size_t k = 0; k < sub_meshes.size(); k++) {
        const Mesh& sub_mesh = sub_meshes[k];
        
        for(int i = 0; i <= ny + 1; i++) {
            for(int j = 0; j <= sub_mesh.nx + 1; j++) {
                // 跳过接口列
                if(sub_mesh.bctype(i,j) == -3) continue;
                
                merged_mesh.u(i, current_col) = sub_mesh.u(i,j);
                merged_mesh.v(i, current_col) = sub_mesh.v(i,j);
                merged_mesh.p(i, current_col) = sub_mesh.p(i,j);
                
                current_col++;
            }
        }
    }
    
    return merged_mesh;
}

// ============================================================================
// 文件I/O函数
// ============================================================================

/**
 * @brief 从参数文件读取网格步长
 * @param folderPath 文件夹路径
 * @param dx 输出x方向步长
 * @param dy 输出y方向步长
 */
void readParams(const std::string& folderPath, double& dx, double& dy) {
    std::string filePath = folderPath + "/params.txt";
    std::ifstream file(filePath);

    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filePath << std::endl;
        return;
    }

    std::string line;
    std::getline(file, line);  // 跳过第一行

    if (std::getline(file, line)) {
        std::istringstream iss(line);
        if (!(iss >> dx >> dy)) {
            std::cerr << "读取 dx 和 dy 失败" << std::endl;
        }
    }

    file.close();
}

/**
 * @brief 保存网格数据到文件
 * @param mesh 网格对象
 * @param rank 进程编号
 * @param timestep_folder 时间步文件夹(可选)
 */
void saveMeshData(
    const Mesh& mesh,
    int rank,
    const std::string& timestep_folder)
{
    try {
        fs::path dir;

        if (!timestep_folder.empty()) {
            dir = fs::path(timestep_folder);
            fs::create_directories(dir);   // MPI-safe 幂等
        }

        auto write = [&](const std::string& name, const auto& field) {
            fs::path p = dir.empty()
                ? fs::path(name + "_" + std::to_string(rank) + ".dat")
                : dir / (name + "_" + std::to_string(rank) + ".dat");

            std::ofstream f(p);
            if (!f) {
                throw std::runtime_error("无法创建文件: " + p.string());
            }
            f << field;
        };

        write("u", mesh.u_star);
        write("v", mesh.v_star);
        write("p", mesh.p);

    } catch (const std::exception& e) {
        std::cerr << "[Rank " << rank << "] 保存 Mesh 数据失败: "
                  << e.what() << std::endl;
        throw;   // 关键 
    }
}


/**
 * @brief 保存预测数据(用于非稳态计算)
 * @param mesh 网格对象
 * @param rank 进程编号
 * @param timesteps 时间步编号
 * @param timestep_folder 时间步文件夹
 */
void saveforecastData(
    const Mesh& mesh,
    int rank,
    int timestep,
    double mu)
{
    try {
        std::ostringstream ss;
        ss << "mu_" << std::fixed << std::setprecision(3) << mu;

        fs::path step_dir = fs::path(ss.str()) / std::to_string(timestep);
        fs::create_directories(step_dir); // MPI-safe 幂等

        auto write = [&](const std::string& name, const auto& field) {
            fs::path p = step_dir / (name + "_" + std::to_string(rank) + ".dat");
            std::ofstream f(p);
            if (!f) throw std::runtime_error(p.string());
            f << field;
        };

        write("up", mesh.u_face);
        write("vp", mesh.v_face);
        write("p" , mesh.p);
        write("pp", mesh.p_prime);

    } catch (const std::exception& e) {
        std::cerr << "[Rank " << rank << "] IO失败: " << e.what() << std::endl;
        throw;
    }
}

// ==================== 辅助函数实现 ====================

void parseInputParameters(int argc, char* argv[], std::string& mesh_folder, 
                         int& timesteps, double& mu, int& n_splits) 
{
    if (argc == 5) {
        mesh_folder = argv[1];
        timesteps = std::stoi(argv[2]);
        mu = std::stod(argv[3]);
        n_splits = std::stoi(argv[4]);
        
        std::cout << "==================== 参数设置 ====================" << std::endl;
        std::cout << "网格文件夹: " << mesh_folder << std::endl;
        std::cout << "时间步数: " << timesteps << std::endl;
        std::cout << "粘度系数: " << mu << std::endl;
        std::cout << "并行线程数: " << n_splits << std::endl;
        std::cout << "==================================================\n" << std::endl;
    } else {
        std::cout << "==================== 参数输入 ====================" << std::endl;
        std::cout << "网格文件夹路径: ";
        std::cin >> mesh_folder;
        std::cout << "时间步数: ";
        std::cin >> timesteps;
        std::cout << "粘度系数: ";
        std::cin >> mu;
        std::cout << "并行线程数: ";
        std::cin >> n_splits;
        std::cout << "==================================================\n" << std::endl;
    }
}

void broadcastParameters(std::string& mesh_folder, double& dt, int& timesteps, 
                        double& mu, int& n_splits, int rank) 
{
    // 同步字符串
    int folder_length;
    if (rank == 0) folder_length = mesh_folder.size();
    MPI_Bcast(&folder_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    char* folder_cstr = new char[folder_length + 1];
    if (rank == 0) strcpy(folder_cstr, mesh_folder.c_str());
    MPI_Bcast(folder_cstr, folder_length + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank != 0) mesh_folder = std::string(folder_cstr);
    delete[] folder_cstr;
    
    // 广播数值参数
    MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&timesteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mu, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_splits, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void verifyParameterConsistency(const std::string& mesh_folder, double dt, 
                               int timesteps, double mu, int n_splits, 
                               int rank, int num_procs) 
{
    // 检查浮点变量一致性
    double local_vals[4] = {dx, dy, dt, mu};
    double global_max[4], global_min[4];
    MPI_Allreduce(local_vals, global_max, 4, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(local_vals, global_min, 4, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    
    // 检查整型变量一致性
    int local_ints[2] = {timesteps, n_splits};
    int global_int_max[2], global_int_min[2];
    MPI_Allreduce(local_ints, global_int_max, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(local_ints, global_int_min, 2, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    
    // 检查字符串一致性
    int folder_length = mesh_folder.size();
    char* local_str = new char[folder_length + 1];
    strcpy(local_str, mesh_folder.c_str());
    
    char* all_strings = new char[(folder_length + 1) * num_procs];
    MPI_Allgather(local_str, folder_length + 1, MPI_CHAR,
                  all_strings, folder_length + 1, MPI_CHAR, MPI_COMM_WORLD);
    
    int folder_match = 1;
    for (int i = 0; i < num_procs; ++i) {
        if (strcmp(local_str, &all_strings[i * (folder_length + 1)]) != 0) {
            folder_match = 0;
            break;
        }
    }
    
    int global_folder_match;
    MPI_Allreduce(&folder_match, &global_folder_match, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    
    // 仅rank 0打印验证结果
    if (rank == 0) {
        bool float_consistent = true;
        for (int i = 0; i < 4; ++i) {
            if (fabs(global_max[i] - global_min[i]) > 1e-12) {
                float_consistent = false;
            }
        }
        
        bool int_consistent = (global_int_max[0] == global_int_min[0]) && 
                             (global_int_max[1] == global_int_min[1]);
        
        if (!float_consistent || !int_consistent || global_folder_match == 0) {
            std::cerr << "\n✗ MPI参数同步失败!" << std::endl;
            if (!float_consistent) 
                std::cerr << "  → 浮点参数不一致 (dx/dy/dt/mu)" << std::endl;
            if (!int_consistent) 
                std::cerr << "  → 整型参数不一致 (timesteps/n_splits)" << std::endl;
            if (global_folder_match == 0) 
                std::cerr << "  → 路径字符串不一致" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        } else {
            std::cout << "==================== 参数同步验证 ====================" << std::endl;
            std::cout << "✓ 所有进程参数同步成功" << std::endl;
            std::cout << "  dx = " << dx << ", dy = " << dy << std::endl;
            std::cout << "  dt = " << dt << ", mu = " << mu << std::endl;
            std::cout << "  timesteps = " << timesteps << ", n_splits = " << n_splits << std::endl;
            std::cout << "  mesh_folder = " << mesh_folder << std::endl;
            std::cout << "======================================================\n" << std::endl;
        }
    }
    
    delete[] local_str;
    delete[] all_strings;
}

void printSimulationSetup(const std::vector<Mesh>& sub_meshes, int n_splits, int rank) 
{
    std::cout << "==================== 网格分割信息 ====================" << std::endl;
    std::cout << "总分割数: " << n_splits << " 个子网格" << std::endl;
    for (int i = 0; i < sub_meshes.size(); i++) {
        std::cout << "  子网格 " << i << " 尺寸: " 
                  << sub_meshes[i].nx << " × " << sub_meshes[i].ny << std::endl;
    }
    std::cout << "======================================================\n" << std::endl;
}

bool checkConvergence(double norm_res_x, double norm_res_y, double norm_res_p) 
{
    const double tol_uv = 1e-15;  // 速度收敛容差
    const double tol_p = 1e-11;   // 压力收敛容差
    
    return (norm_res_x < tol_uv) && (norm_res_y < tol_uv) && (norm_res_p < tol_p);
}

// -------------------- 参数解析(非定常版本) --------------------
void parseInputParameters_unsteady(int argc, char* argv[], std::string& mesh_folder, 
                                   double& dt, int& timesteps, double& mu, int& n_splits) 
{
    if (argc == 6) {
        // 命令行参数: mesh_folder dt timesteps mu n_splits
        mesh_folder = argv[1];
        dt = std::stod(argv[2]);
        timesteps = std::stoi(argv[3]);
        mu = std::stod(argv[4]);
        n_splits = std::stoi(argv[5]);
        
        std::cout << "==================== 参数设置 ====================" << std::endl;
        std::cout << "计算模式: 非定常 (Unsteady)" << std::endl;
        std::cout << "网格文件夹: " << mesh_folder << std::endl;
        std::cout << "时间步长 dt: " << dt << std::endl;
        std::cout << "时间步数: " << timesteps << std::endl;
        std::cout << "粘度系数 μ: " << mu << std::endl;
        std::cout << "并行线程数: " << n_splits << std::endl;
        std::cout << "==================================================\n" << std::endl;
    } else {
        // 交互式输入
        std::cout << "==================== 参数输入 ====================" << std::endl;
        std::cout << "计算模式: 非定常 (Unsteady)" << std::endl;
        std::cout << "网格文件夹路径: ";
        std::cin >> mesh_folder;
        std::cout << "时间步长 dt: ";
        std::cin >> dt;
        std::cout << "时间步数: ";
        std::cin >> timesteps;
        std::cout << "粘度系数 μ: ";
        std::cin >> mu;
        std::cout << "并行线程数: ";
        std::cin >> n_splits;
        std::cout << "==================================================\n" << std::endl;
    }
}

// -------------------- 参数广播(非定常版本) --------------------
void broadcastParameters_unsteady(std::string& mesh_folder, double& dt, int& timesteps, 
                                  double& mu, int& n_splits, int rank) 
{
    // 同步字符串长度和内容
    int folder_length;
    if (rank == 0) folder_length = mesh_folder.size();
    MPI_Bcast(&folder_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    char* folder_cstr = new char[folder_length + 1];
    if (rank == 0) strcpy(folder_cstr, mesh_folder.c_str());
    MPI_Bcast(folder_cstr, folder_length + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank != 0) mesh_folder = std::string(folder_cstr);
    delete[] folder_cstr;
    
    // 广播数值参数
    MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&timesteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mu, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_splits, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

// -------------------- 参数一致性验证(非定常版本) --------------------
void verifyParameterConsistency_unsteady(const std::string& mesh_folder, double dt, 
                                         int timesteps, double mu, int n_splits, 
                                         int rank, int num_procs) 
{
    // 检查浮点变量一致性 (dx, dy, dt, mu)
    double local_vals[4] = {dx, dy, dt, mu};
    double global_max[4], global_min[4];
    MPI_Allreduce(local_vals, global_max, 4, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(local_vals, global_min, 4, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    
    // 检查整型变量一致性 (timesteps, n_splits)
    int local_ints[2] = {timesteps, n_splits};
    int global_int_max[2], global_int_min[2];
    MPI_Allreduce(local_ints, global_int_max, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(local_ints, global_int_min, 2, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    
    // 检查字符串一致性
    int folder_length = mesh_folder.size();
    char* local_str = new char[folder_length + 1];
    strcpy(local_str, mesh_folder.c_str());
    
    char* all_strings = new char[(folder_length + 1) * num_procs];
    MPI_Allgather(local_str, folder_length + 1, MPI_CHAR,
                  all_strings, folder_length + 1, MPI_CHAR, MPI_COMM_WORLD);
    
    int folder_match = 1;
    for (int i = 0; i < num_procs; ++i) {
        if (strcmp(local_str, &all_strings[i * (folder_length + 1)]) != 0) {
            folder_match = 0;
            break;
        }
    }
    
    int global_folder_match;
    MPI_Allreduce(&folder_match, &global_folder_match, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    
    // 仅rank 0打印验证结果
    if (rank == 0) {
        bool float_consistent = true;
        for (int i = 0; i < 4; ++i) {
            if (fabs(global_max[i] - global_min[i]) > 1e-12) {
                float_consistent = false;
            }
        }
        
        bool int_consistent = (global_int_max[0] == global_int_min[0]) && 
                             (global_int_max[1] == global_int_min[1]);
        
        if (!float_consistent || !int_consistent || global_folder_match == 0) {
            std::cerr << "\n✗ MPI参数同步失败!" << std::endl;
            if (!float_consistent) 
                std::cerr << "  → 浮点参数不一致 (dx/dy/dt/mu)" << std::endl;
            if (!int_consistent) 
                std::cerr << "  → 整型参数不一致 (timesteps/n_splits)" << std::endl;
            if (global_folder_match == 0) 
                std::cerr << "  → 路径字符串不一致" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        } else {
            std::cout << "==================== 参数同步验证 ====================" << std::endl;
            std::cout << "✓ 所有进程参数同步成功" << std::endl;
            std::cout << "  dx = " << dx << ", dy = " << dy << std::endl;
            std::cout << "  dt = " << dt << ", μ = " << mu << std::endl;
            std::cout << "  timesteps = " << timesteps << ", n_splits = " << n_splits << std::endl;
            std::cout << "  mesh_folder = " << mesh_folder << std::endl;
            std::cout << "======================================================\n" << std::endl;
        }
    }
    
    delete[] local_str;
    delete[] all_strings;
}

// -------------------- 打印模拟设置(非定常版本) --------------------
void printSimulationSetup_unsteady(const std::vector<Mesh>& sub_meshes, int n_splits, 
                                   double dt, int timesteps, int rank) 
{
    std::cout << "==================== 网格分割信息 ====================" << std::endl;
    std::cout << "总分割数: " << n_splits << " 个子网格" << std::endl;
    for (int i = 0; i < sub_meshes.size(); i++) {
        std::cout << "  子网格 " << i << " 尺寸: " 
                  << sub_meshes[i].nx << " × " << sub_meshes[i].ny << std::endl;
    }
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "时间离散信息:" << std::endl;
    std::cout << "  时间步长 dt: " << dt << std::endl;
    std::cout << "  总时间步数: " << timesteps + 1 << " (0 → " << timesteps << ")" << std::endl;
    std::cout << "  总模拟时间: " << dt * timesteps << std::endl;
    std::cout << "======================================================\n" << std::endl;
}
