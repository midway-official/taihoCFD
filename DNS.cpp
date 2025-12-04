// dns.cpp
#include "DNS.h"
#include <filesystem>
#include "parallel.h"
namespace fs = std::filesystem;
// 全局变量定义

double dx, dy, vx;
double velocity;
double l2_norm_x = 0.0, l2_norm_y = 0.0, l2_norm_p = 0.0;
double a, b;
 
void printMatrix(const MatrixXd& matrix, const string& name, int precision ) {
    // 设置输出格式
    IOFormat fmt(precision, 0, ", ", "\n", "[", "]");
    
    // 打印矩阵名称和尺寸
    cout << "\n====== " << name << " (" 
         << matrix.rows() << "x" << matrix.cols() << ") ======\n";
              
    // 打印矩阵内容
    cout << matrix.format(fmt) << endl;
    
    // 打印分隔线
    cout << string(40, '=') << endl;
}
// Mesh 类的构造函数
Mesh::Mesh(int n_y, int n_x)
    : u(n_y + 2, n_x + 2), u_star(n_y + 2, n_x + 2),u0(n_y + 2, n_x + 2),
      v(n_y + 2, n_x + 2), v_star(n_y + 2, n_x + 2),v0(n_y + 2, n_x + 2), 
      p(n_y + 2, n_x + 2), p_star(n_y + 2, n_x + 2), p_prime(n_y + 2, n_x + 2),
      u_face(n_y + 2, n_x + 1), v_face(n_y + 1, n_x + 2),bctype(n_y + 2, n_x + 2),zoneid(n_y + 2, n_x + 2) ,interid(n_y + 2, n_x + 2),nx(n_x), ny(n_y){}

// 初始化所有矩阵为零
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

// 显示矩阵内容
void Mesh::displayMatrix(const MatrixXd& matrix, const string& name) const {
    cout << name << ":\n" << matrix << "\n";
}

// 显示所有矩阵
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

void Mesh::createInterId() {
    interid = MatrixXi::Zero(bctype.rows(), bctype.cols());
    interi.clear();
    interj.clear();
    internumber = 0;
    int count = 1;
    // 从上到下，从左到右遍历
    for(int i = 0; i < bctype.rows(); i++) {
        for(int j = 0; j < bctype.cols(); j++) {
            if(bctype(i,j) == 0) {
                interid(i,j) = count;
                interi.push_back(i);
                interj.push_back(j);
                count++;
                internumber++;
            }
        }
    }
}
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
void Mesh::setZoneUV(int zoneIndex, double u, double v) {
    // 确保 zoneu 和 zonev 向量足够长
    while(zoneu.size() <= zoneIndex) {
        zoneu.push_back(0.0);
        zonev.push_back(0.0);
    }
    
    // 设置指定索引的值
    zoneu[zoneIndex] = u;
    zonev[zoneIndex] = v;
}
void Mesh::initializeBoundaryConditions() 
{
    // 遍历所有网格点，处理非内部点的速度
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

    // 处理 u_face
for(int i = 0; i <= ny + 1; i++) {
    for(int j = 0; j <= nx; j++) {
        // 检查面两侧的单元格
        bool left_is_internal = (bctype(i,j) == 0);
        bool right_is_internal = (bctype(i,j+1) == 0);

        if(left_is_internal && !right_is_internal) {
            // 右侧是边界，使用右侧单元格的速度
            u_face(i,j) = zoneu[zoneid(i,j+1)];
        }
        else if(!left_is_internal && right_is_internal) {
            // 左侧是边界，使用左侧单元格的速度
            u_face(i,j) = zoneu[zoneid(i,j)];
        }
        else if(!left_is_internal && !right_is_internal) {
            // 两侧都是边界，取均值
            u_face(i,j) = 0.5 * (zoneu[zoneid(i,j)] + zoneu[zoneid(i,j+1)]);
        }
    }
}

// 处理 v_face
for(int i = 0; i <= ny; i++) {
    for(int j = 0; j <= nx + 1; j++) {
        // 检查面上下的单元格
        bool top_is_internal = (bctype(i,j) == 0);
        bool bottom_is_internal = (bctype(i+1,j) == 0);

        if(top_is_internal && !bottom_is_internal) {
            // 下侧是边界，使用下侧单元格的速度
            v_face(i,j) = zonev[zoneid(i+1,j)];
        }
        else if(!top_is_internal && bottom_is_internal) {
            // 上侧是边界，使用上侧单元格的速度
            v_face(i,j) = zonev[zoneid(i,j)];
        }
        else if(!top_is_internal && !bottom_is_internal) {
            // 上下都是边界，取均值
            v_face(i,j) = 0.5 * (zonev[zoneid(i,j)] + zonev[zoneid(i+1,j)]);
        }
    }
}
}


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

    // 保存矩阵数据
    std::ofstream bcFile(folderPath + "/bctype.dat");
    bcFile << bctype;
    bcFile.close();

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

// 从文件夹构造网格
Mesh::Mesh(const std::string& folderPath) {
    if (!fs::exists(folderPath)) {
        throw std::runtime_error("网格文件夹不存在!");
    }

    // 读取网格参数
    std::ifstream paramFile(folderPath + "/params.txt");
    if (!paramFile) {
        throw std::runtime_error("无法打开参数文件!");
    }
    
    // 读取基本参数
    paramFile >> nx >> ny >> ::dx >> ::dy;
    paramFile.close();

    // 初始化所有矩阵
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

    // 初始化为零
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
// Equation 类的构造函数
Equation::Equation(Mesh& mesh_)
    : A_p(mesh_.ny + 2, mesh_.nx + 2),
      A_e(mesh_.ny + 2, mesh_.nx + 2),
      A_w(mesh_.ny + 2, mesh_.nx + 2),
      A_n(mesh_.ny + 2, mesh_.nx + 2),
      A_s(mesh_.ny + 2, mesh_.nx + 2),
      source(mesh_.internumber),
      A(mesh_.internumber, mesh_.internumber),
      n_x(mesh_.nx), 
      n_y(mesh_.ny),
      mesh(mesh_)
{}
// 初始化矩阵和源向量为零
void Equation::initializeToZero() {
    A_p.setZero();
    A_e.setZero();
    A_w.setZero();
    A_n.setZero();
    A_s.setZero();
    source.setZero();
    A.setZero();
}
void Equation::build_matrix() {
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;

    // 遍历所有网格点
    for(int i = 1; i <=n_y ; i++) {
        for(int j = 1; j <= n_x; j++) {
            // 只处理内部点（bctype为0的点）
            if(mesh.bctype(i,j) == 0) {
                int current_id = mesh.interid(i,j) - 1;  // 当前点在方程组中的编号
                
                // 添加中心点系数
                tripletList.emplace_back(current_id, current_id, A_p(i,j));
                
                // 检查东邻接单元
                if(mesh.bctype(i,j+1) == 0) {
                    int east_id = mesh.interid(i,j+1) - 1;
                    tripletList.emplace_back(current_id, east_id, -A_e(i,j));
                }
                
                // 检查西邻接单元
                if(mesh.bctype(i,j-1) == 0) {
                    int west_id = mesh.interid(i,j-1) - 1;
                    tripletList.emplace_back(current_id, west_id, -A_w(i,j));
                }
                
                // 检查北邻接单元
                if(mesh.bctype(i-1,j) == 0) {
                    int north_id = mesh.interid(i-1,j) - 1;
                    tripletList.emplace_back(current_id, north_id, -A_n(i,j));
                }
                
                // 检查南邻接单元
                if(mesh.bctype(i+1,j) == 0) {
                    int south_id = mesh.interid(i+1,j) - 1;
                    tripletList.emplace_back(current_id, south_id, -A_s(i,j));
                }
            }
        }
    }
    
    // 设置稀疏矩阵大小为内部点数量
    A.resize(mesh.internumber, mesh.internumber);
    A.setFromTriplets(tripletList.begin(), tripletList.end());
}
void solve(Equation& equation, double epsilon, double& l2_norm, MatrixXd& phi){
    // 创建解向量，长度为内部点数量
    VectorXd x(equation.mesh.internumber);

    // 根据 interid 构建初始解向量，遍历整个网格
    for(int i = 0; i <= equation.n_y + 1; i++) {
        for(int j = 0; j <= equation.n_x + 1; j++) {
            if(equation.mesh.bctype(i,j) == 0) {
                int n = equation.mesh.interid(i,j) - 1;
                x[n] = phi(i,j);
            }
        }
    }

    // 计算残差
    l2_norm = (equation.A * x - equation.source).norm();

    // 求解线性方程组
   // BiCGSTAB<SparseMatrix<double>> solver;
   ConjugateGradient<SparseMatrix<double>> solver;
    solver.compute(equation.A);
    solver.setTolerance(epsilon); 
    solver.setMaxIterations(1);     // 设置最大迭代次数
    x = solver.solve(equation.source);

    // 将结果写回网格，同样遍历整个网格
    for(int i = 0; i <= equation.n_y + 1; i++) {
        for(int j = 0; j <= equation.n_x + 1; j++) {
            if(equation.mesh.bctype(i,j) == 0) {
                int n = equation.mesh.interid(i,j) - 1;
                phi(i,j) = x[n];
            }
        }
    }
}

void face_velocity(Mesh& mesh, Equation& equ_u) {
    MatrixXd& u_face = mesh.u_face;
    MatrixXd& v_face = mesh.v_face;
    MatrixXd& bctype = mesh.bctype;
    MatrixXd& u = mesh.u;
    MatrixXd& v = mesh.v;
    MatrixXd& p = mesh.p;
    MatrixXd& A_p = equ_u.A_p;

    for(int i = 0; i <= mesh.ny + 1; i++) {
        for(int j = 0; j <= mesh.nx; j++) {
            if ((bctype(i,j) == 0 && bctype(i,j+1) == 0) || 
                (bctype(i,j) == 0 && bctype(i,j+1) == -3) ||
                (bctype(i,j) == -3 && bctype(i,j+1) == 0)) {

                if (bctype(i,j+2) == -2) p(i,j+2) = p(i,j+1);
                else if (bctype(i,j-1) == -2) p(i,j-1) = p(i,j);
                else if (bctype(i,j+2) == -1) p(i,j+2) = 0;
                else if (bctype(i,j-1) == -1) p(i,j-1) = 0;

                u_face(i,j) = 0.5*(u(i,j) + u(i,j+1))
                            + 0.25*(p(i,j+1) - p(i,j-1)) * dy / A_p(i,j)
                            + 0.25*(p(i,j+2) - p(i,j)) * dy / A_p(i,j+1)
                            - 0.5*(1.0/A_p(i,j) + 1.0/A_p(i,j+1)) * (p(i,j+1) - p(i,j)) * dy;
            }
            else if (bctype(i,j) == 0 && bctype(i,j+1) == -1) {
                u_face(i,j) = u(i,j);
            }
            else if (bctype(i,j) == -1 && bctype(i,j+1) == 0) {
                u_face(i,j) = u(i,j+1);
            }
            else if (bctype(i,j) == 0 && bctype(i,j+1) == -2) {
                u_face(i,j) = mesh.zoneu[mesh.zoneid(i,j+1)];
            }
            else if (bctype(i,j) == -2 && bctype(i,j+1) == 0) {
                u_face(i,j) = mesh.zoneu[mesh.zoneid(i,j)];
            }
            else {
                u_face(i,j) = 0.0;
            }

            // NaN 检查
            if (std::isnan(u_face(i,j))) u_face(i,j) = 0.0;
        }
    }

    for(int i = 0; i <= mesh.ny; i++) {
        for(int j = 0; j <= mesh.nx + 1; j++) {
            if (bctype(i,j) == 0 && bctype(i+1,j) == 0) {
                if (bctype(i+2,j) == -2) p(i+2,j) = p(i+1,j);
                else if (bctype(i-1,j) == -2) p(i-1,j) = p(i,j);
                else if (bctype(i+2,j) == -1) p(i+2,j) = 0;
                else if (bctype(i-1,j) == -1) p(i-1,j) = 0;

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

            // NaN 检查
            if (std::isnan(v_face(i,j))) v_face(i,j) = 0.0;
        }
    }
}






void pressure_function(Mesh &mesh, Equation &equ_p, Equation &equ_u)
{
    
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

    // 遍历网格点
    for(int i = 0; i <= equ_p.n_y+1; i++) {
        for(int j = 0; j <= equ_p.n_x+1; j++) {
            if(bctype(i,j) == 0) {  // 内部点
                int n = mesh.interid(i,j) - 1;
                double Ap_temp = 0;
                double source_temp = 0;
                // 检查东面
                if(bctype(i,j+1) == 0) {
                    Ap_e(i,j) = 0.5*(1/A_p(i,j) +1/A_p(i,j+1))*(dy*dy);
                    Ap_temp += Ap_e(i,j);
                }  
                else if (bctype(i,j+1) == -3)
                {
                    Ap_e(i,j) = 0.5*(1/A_p(i,j) +1/A_p(i,j+1))*(dy*dy);
                    Ap_temp += Ap_e(i,j);
                }
                
              
                else{
                    Ap_e(i,j) = 0;
                }

                // 检查西面
                if(bctype(i,j-1) == 0) {
                    Ap_w(i,j) = 0.5*(1/A_p(i,j) + 1/A_p(i,j-1))*(dy*dy);
                    Ap_temp += Ap_w(i,j);
                }
                else if (bctype(i,j-1) == -3)
                {
                    Ap_w(i,j) = 0.5*(1/A_p(i,j) + 1/A_p(i,j-1))*(dy*dy);
                    Ap_temp += Ap_w(i,j);
                }
                
                 else {
                    Ap_w(i,j) = 0;
                }

                // 检查北面
                if(bctype(i-1,j) == 0) {
                    Ap_n(i,j) = 0.5*(1/A_p(i,j) + 1/A_p(i-1,j))*(dx*dx);
                    Ap_temp += Ap_n(i,j);
                }
               
                 else {
                    Ap_n(i,j) = 0;
                }

                // 检查南面
                if(bctype(i+1,j) == 0) {
                    Ap_s(i,j) = 0.5*(1/A_p(i,j) + 1/A_p(i+1,j))*(dx*dx);
                    Ap_temp += Ap_s(i,j);
                } 
                
                else {
                    Ap_s(i,j) = 0;
                }

                // 设置中心系数和源项
                Ap_p(i,j) = Ap_temp;
                source_p[n]=0;
                
                source_p[n] +=   -(u_face(i,j) - u_face(i,j-1))*dy 
                - (v_face(i-1,j) - v_face(i,j))*dx;         
            }
        }
    }
}


void correct_pressure(Mesh &mesh, Equation &equ_u,double alpha_p)
{
    MatrixXd &p = mesh.p;
    MatrixXd &p_star = mesh.p_star;
    MatrixXd &p_prime = mesh.p_prime;
    MatrixXd &bctype = mesh.bctype;
    int n_x = mesh.nx;
    int n_y = mesh.ny;

 
    for(int i = 0; i <= n_y + 1; i++) {
        for(int j = 0; j <= n_x + 1; j++) {
            if(bctype(i,j) > 0) {  // 边界点
                
                    p_prime(i,j) = 0;
                }
            }
        }
    

    // 更新压力场
    
      // 压力松弛因子
p_star =p + alpha_p *p_prime;
}

void correct_velocity(Mesh &mesh, Equation &equ_u)
{
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

    

    // 修正 u_star
    for (int i = 0; i <= n_y+1; i++) {
        for (int j = 0; j <= n_x+1; j++) {
            if (bctype(i,j) == 0) {
                double p_west, p_east;

                // 西面
                if (bctype(i,j-1) == 0 || bctype(i,j-1) == -3)
                    p_west = p_prime(i,j-1);
                else
                    p_west = p_prime(i,j);

                // 东面
                if (bctype(i,j+1) == 0 || bctype(i,j+1) == -3)
                    p_east = p_prime(i,j+1);
                else
                    p_east = p_prime(i,j);

                u_star(i,j) = u(i,j) + 0.5 * (p_west - p_east) * dy / A_p(i,j);
            }
        }
    }

    // 修正 v_star
    for (int i = 0; i <= n_y+1; i++) {
        for (int j = 0; j <= n_x+1; j++) {
            if (bctype(i,j) == 0) {
                double p_north, p_south;

                // 北面
                if (bctype(i-1,j) == 0 || bctype(i-1,j) == -3)
                    p_north = p_prime(i-1,j);
                else
                    p_north = p_prime(i,j);

                // 南面
                if (bctype(i+1,j) == 0 || bctype(i+1,j) == -3)
                    p_south = p_prime(i+1,j);
                else
                    p_south = p_prime(i,j);

                v_star(i,j) = v(i,j) + 0.5 * (p_south - p_north) * dx / A_p(i,j);
            }
        }
    }

    // 修正 u_face (互斥判断)
    for (int i = 0; i <= n_y+1; i++) {
        for (int j = 0; j <= n_x; j++) {
            if ((bctype(i,j) == 0 && bctype(i,j+1) == 0) ||
                (bctype(i,j) == 0 && bctype(i,j+1) == -3) ||
                (bctype(i,j) == -3 && bctype(i,j+1) == 0)) {
                // 情况1：内部或特殊内部 → 做压力修正
                u_face(i,j) = u_face(i,j) + 
                              0.5 * (1/A_p(i,j) + 1/A_p(i,j+1)) * 
                              (p_prime(i,j) - p_prime(i,j+1)) * dy;
            }
            else if (bctype(i,j) == 0 && bctype(i,j+1) == -1) {
                // 情况2：内部-压力边界 → 用 u_star(i,j)
                u_face(i,j) = u_star(i,j);
            }
            else if (bctype(i,j) == -1 && bctype(i,j+1) == 0) {
                // 情况3：压力边界-内部 → 用 u_star(i,j+1)
                u_face(i,j) = u_star(i,j+1);
            }
            else {
                // 其它情况，保持不变或由边界条件处理
            }
        }
    }

    // 修正 v_face (互斥判断)
    for (int i = 0; i <= n_y; i++) {
        for (int j = 0; j <= n_x+1; j++) {
            if ((bctype(i,j) == 0 && bctype(i+1,j) == 0) ||
                (bctype(i,j) == 0 && bctype(i+1,j) == -3) ||
                (bctype(i,j) == -3 && bctype(i+1,j) == 0)) {
                // 情况1：内部或特殊内部 → 做压力修正
                v_face(i,j) = v_face(i,j) + 
                              0.5 * (1/A_p(i,j) + 1/A_p(i+1,j)) * 
                              (p_prime(i+1,j) - p_prime(i,j)) * dx;
            }
            else if (bctype(i,j) == 0 && bctype(i+1,j) == -1) {
                // 情况2：内部-压力边界 → 用 v_star(i,j)
                v_face(i,j) = v_star(i,j);
            }
            else if (bctype(i,j) == -1 && bctype(i+1,j) == 0) {
                // 情况3：压力边界-内部 → 用 v_star(i+1,j)
                v_face(i,j) = v_star(i+1,j);
            }
            else {
                // 其它情况，保持不变或由边界条件处理
            }
        }
    }
}

void post_processing(Mesh &mseh)
{   
   

    //保存计算结果
     std::ofstream outFile;
     outFile.open("u.dat");
     outFile << mseh.u_star;
     outFile.close();

     outFile.open("v.dat");
     outFile << mseh.v_star;
     outFile.close();

     outFile.open("p.dat");
     outFile << mseh.p_star;
     outFile.close();






}

void show_progress_bar(int current_step, int total_steps, double elapsed_time) {
    // 计算进度百分比
    double progress = static_cast<double>(current_step) / total_steps;
    
    // 设置进度条的宽度
    int bar_width = 50;
    
    // 计算进度条中"="的数量
    int pos = static_cast<int>(bar_width * progress);
    
    // 计算预计剩余时间
    double remaining_time = (elapsed_time / current_step) * (total_steps - current_step);
    
    // 打印进度条和相关信息
    std::cout << "[";
    
    // 绘制进度条
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) {
            std::cout << "=";  // 已完成的部分
        } else if (i == pos) {
            std::cout << ">";  // 当前进度的位置
        } else {
            std::cout << " ";  // 未完成的部分
        }
    }
    
    // 显示进度条，已用时间和预计剩余时间
    std::cout << "] " 
              << std::fixed << std::setprecision(2) << progress * 100 << "% "  // 显示进度百分比
              << "已用时间: " << std::fixed << std::setprecision(2) << elapsed_time << "秒 "  // 显示已用时间
              << "预计剩余时间: " << std::fixed << std::setprecision(2) << remaining_time << "秒\r";  // 显示预计剩余时间
    
    // 刷新输出，确保实时更新
    std::cout.flush();
}


void momentum_function(Mesh &mesh, Equation &equ_u, Equation &equ_v,double mu,double alpha_uv)
{   
    //-1 压力出口(给定压强)
    //-2 固定速度
    //-3 并行交界面

    int n,i,j;
    int n_x=equ_u.n_x;
    int n_y=equ_u.n_y;
    double D_e,D_w,D_n,D_s,F_e,F_n,F_s,F_w;

    
    D_e=dy*mu/(dx);
    D_w=dy*mu/(dx);
    D_n=dx*mu/(dy);
    D_s=dx*mu/(dy);
    // 引用网格变量
    MatrixXd &zoneid = mesh.zoneid;
    MatrixXd &bctype = mesh.bctype;
    MatrixXd &u= mesh.u;
    MatrixXd &v= mesh.v;
    MatrixXd &u_face= mesh.u_face;
    MatrixXd &v_face= mesh.v_face;
    MatrixXd &p= mesh.p;
    MatrixXd &p_star= mesh.p_star;
    MatrixXd &p_prime= mesh.p_prime;
    MatrixXd &u_star= mesh.u_star;
    MatrixXd &v_star= mesh.v_star;
    MatrixXd &A_p=equ_u.A_p;
    MatrixXd &A_e=equ_u.A_e;
    MatrixXd &A_w=equ_u.A_w;
    MatrixXd &A_n=equ_u.A_n;
    MatrixXd &A_s=equ_u.A_s;
    VectorXd &source_x=equ_u.source;
    VectorXd &source_y=equ_v.source;
    vector<double> zoneu=mesh.zoneu;
    vector<double> zonev=mesh.zonev;
    
    // 遍历网格
    for(i=0; i<=n_y+1; i++) {
        for(j=0; j<=n_x+1; j++) {
            if(bctype(i,j) == 0) {  // 内部面
                n = mesh.interid(i,j) - 1;
                
                // 计算面上流量
                F_e = dy*u_face(i,j);
                F_w = dy*u_face(i,j-1);
                F_n = dx*v_face(i-1,j);
                F_s = dx*v_face(i,j);
                
                double Ap_temp = 0;
               // 初始化源项
               double source_x_temp, source_y_temp;
    
                           // 处理 x 方向源项
            if((bctype(i,j-1) == 0 ||  bctype(i,j-1) == -3) && 
               (bctype(i,j+1) == 0 ||  bctype(i,j+1) == -3)) {
                // 两侧都是内部点或滑移边界，使用中心差分
                source_x_temp = 0.5*alpha_uv*(p(i,j-1)-p(i,j+1))*dy;
                
            } else if(bctype(i,j-1) == -1) {
                // 左边是压力为0的边界
                source_x_temp = 0.5*alpha_uv*(-p(i,j+1))*dy;
            } else if(bctype(i,j+1) == -1) {
                // 右边是压力为0的边界
                source_x_temp = 0.5*alpha_uv*(p(i,j-1))*dy;
            } else if(bctype(i,j-1) == -2) {
                // 左边是速度入口
                source_x_temp = 0.5*alpha_uv*(p(i,j)-p(i,j+1))*dy;
            } else if(bctype(i,j+1) == -2) {
                // 右边是速度入口
                source_x_temp = 0.5*alpha_uv*(p(i,j-1)-p(i,j))*dy;
            } else if(bctype(i,j-1) != 0 && bctype(i,j+1) == 0) {
                // 左边是其他边界，右边是内部点
                source_x_temp = 0.5*alpha_uv*(p(i,j)-p(i,j+1))*dy;
            } else if(bctype(i,j-1) == 0 && bctype(i,j+1) != 0) {
                // 左边是内部点，右边是其他边界
                source_x_temp = 0.5*alpha_uv*(p(i,j-1)-p(i,j))*dy;
            } else {
                // 两边都是固定边界或其他情况
                source_x_temp = 0.0; 
            }
            
            // 处理 y 方向源项
            if((bctype(i-1,j) == 0 ||  bctype(i-1,j) == -3) && 
               (bctype(i+1,j) == 0 ||  bctype(i+1,j) == -3)) {
                // 上下都是内部点或滑移边界，使用中心差分
                source_y_temp = 0.5*alpha_uv*(p(i+1,j)-p(i-1,j))*dx;
                
            } else if(bctype(i-1,j) == -1) {
                // 上边是压力为0的边界
                source_y_temp = 0.5*alpha_uv*(p(i+1,j))*dx;
                
            } else if(bctype(i+1,j) == -1) {
                // 下边是压力为0的边界  压力出口
                source_y_temp = 0.5*alpha_uv*(-p(i-1,j))*dx;
            } else if(bctype(i-1,j) == -2) {
                // 上边是压力为0的边界
                source_y_temp = 0.5*alpha_uv*(p(i+1,j)-p(i,j))*dx;
                
            } else if(bctype(i+1,j) == -2) {
                // 下边是压力为0的边界
                source_y_temp = 0.5*alpha_uv*(p(i,j)-p(i-1,j))*dx;
            } else if(bctype(i-1,j) != 0 && bctype(i+1,j) == 0) {
                // 上边是其他边界，下边是内部点
                source_y_temp = 0.5*alpha_uv*(p(i+1,j)-p(i,j))*dx;
            } else if(bctype(i-1,j) == 0 && bctype(i+1,j) != 0) {
                // 上边是内部点，下边是其他边界
                source_y_temp = 0.5*alpha_uv*(p(i,j)-p(i-1,j))*dx;
            } else {
                // 上下都是固定边界或其他情况
                source_y_temp = 0.0;
            }
              
                // 检查东面
                if(bctype(i,j+1) == 0) {  // 内部点
                    A_e(i,j) = D_e + max(0.0,-F_e);
                    Ap_temp += D_e + max(0.0,F_e);
                    
                } 
                else if(bctype(i,j+1) ==-3) {  // 其他边界
                    A_e(i,j) = D_e + max(0.0,-F_e);
                    Ap_temp += D_e + max(0.0,F_e);
                }
                
                else if(bctype(i,j+1) > 0) {  // wall边界
                    A_e(i,j) = 0;
                    Ap_temp += 2*D_e + max(0.0,F_e);
                    source_x_temp += alpha_uv*zoneu[zoneid(i,j+1)]*(2*D_e + max(0.0,-F_e));
                    source_y_temp += alpha_uv*zonev[zoneid(i,j+1)]*(2*D_e + max(0.0,-F_e));
                } 
                else if(bctype(i,j+1) ==-1 ) {  // 其他边界
                    A_e(i,j) = 0;
                    Ap_temp += D_e + max(0.0,F_e);
                    source_x_temp += alpha_uv*u_star(i,j)*(D_e + max(0.0,-F_e));  // 移除系数2
                    source_y_temp += alpha_uv*v_star(i,j)*(D_e + max(0.0,-F_e));  // 移除系数2
                }
                else if(bctype(i,j+1) > -10) {  // 其他边界
                    A_e(i,j) = 0;
                    Ap_temp += D_e + max(0.0,F_e);  // 移除系数2
                    source_x_temp += alpha_uv*zoneu[zoneid(i,j+1)]*(D_e + max(0.0,-F_e));  // 移除系数2
                    source_y_temp += alpha_uv*zonev[zoneid(i,j+1)]*(D_e + max(0.0,-F_e));  // 移除系数2
                }
                 
                // 检查西面
                if(bctype(i,j-1) == 0) {  // 内部点
                    A_w(i,j) = D_w + max(0.0,F_w);
                    Ap_temp += D_w + max(0.0,-F_w);
                } 
                else if(bctype(i,j-1) ==-3) {  // 其他边界
                    A_w(i,j) = D_w + max(0.0,F_w);
                    Ap_temp += D_w + max(0.0,-F_w);
                }
                else if(bctype(i,j-1) ==-1) {  // 其他边界
                    A_w(i,j) = 0;
                    Ap_temp += D_w + max(0.0,-F_w);
                    source_x_temp += alpha_uv*u_star(i,j)*(D_w + max(0.0,F_w));  // 移除系数2
                    source_y_temp += alpha_uv*v_star(i,j)*(D_w + max(0.0,F_w));  // 移除系数2
                }
                else if(bctype(i,j-1) > 0) {  //wall边界
                    A_w(i,j) = 0;
                    Ap_temp += 2*D_w + max(0.0,-F_w);
                    source_x_temp += alpha_uv*zoneu[zoneid(i,j-1)]*(2*D_w + max(0.0,F_w));
                    source_y_temp += alpha_uv*zonev[zoneid(i,j-1)]*(2*D_w + max(0.0,F_w));
                } else if(bctype(i,j-1) > -10) {  //其他
                    A_w(i,j) = 0;
                    Ap_temp += D_w + max(0.0,-F_w);  // 移除系数2
                    source_x_temp += alpha_uv*zoneu[zoneid(i,j-1)]*(D_w + max(0.0,F_w));  // 移除系数2
                    source_y_temp += alpha_uv*zonev[zoneid(i,j-1)]*(D_w + max(0.0,F_w));  // 移除系数2
                }
                
                // 检查北面
                if(bctype(i-1,j) == 0) {  // 内部点
                    A_n(i,j) = D_n + max(0.0,-F_n);
                    Ap_temp += D_n + max(0.0,F_n);
                } 
                else if(bctype(i-1,j) == -1) {  // 压力出口
                    A_n(i,j) = 0;
                    Ap_temp += D_n + max(0.0,F_n);
                    source_x_temp += alpha_uv*u_star(i-1,j)*(D_n + max(0.0,-F_n));
                    source_y_temp += alpha_uv*v_star(i-1,j)*(D_n + max(0.0,-F_n));
                } 
               
                else if(bctype(i-1,j) > 0) {  // wall边界
                    A_n(i,j) = 0;
                    Ap_temp += 2*D_n + max(0.0,F_n);
                    source_x_temp += alpha_uv*zoneu[zoneid(i-1,j)]*(2*D_n + max(0.0,-F_n));
                    source_y_temp += alpha_uv*zonev[zoneid(i-1,j)]*(2*D_n + max(0.0,-F_n));
                } else if(bctype(i-1,j) > -10) {  // 其他边界
                    A_n(i,j) = 0;
                    Ap_temp += D_n + max(0.0,F_n);  // 移除系数2
                    source_x_temp += alpha_uv*zoneu[zoneid(i-1,j)]*(D_n + max(0.0,-F_n));  // 移除系数2
                    source_y_temp += alpha_uv*zonev[zoneid(i-1,j)]*(D_n + max(0.0,-F_n));  // 移除系数2
                }
                
                // 检查南面
                if(bctype(i+1,j) == 0) {  // 内部点
                    A_s(i,j) = D_s + max(0.0,F_s);
                    Ap_temp += D_s + max(0.0,-F_s);
                }
                else if(bctype(i+1,j) == -1) {  // 压力出口
                    A_s(i,j) =0;
                    Ap_temp += D_s + max(0.0,-F_s);  
                    source_x_temp += alpha_uv*u(i+1,j)*(D_s + max(0.0,F_s));  
                    source_y_temp += alpha_uv*v(i+1,j)*(D_s + max(0.0,F_s));  
                }
                 else if(bctype(i+1,j) > 0) {  // wall边界
                    A_s(i,j) = 0;
                    Ap_temp += 2*D_s + max(0.0,-F_s);
                    source_x_temp += alpha_uv*zoneu[zoneid(i+1,j)]*(2*D_s + max(0.0,F_s));
                    source_y_temp += alpha_uv*zonev[zoneid(i+1,j)]*(2*D_s + max(0.0,F_s));
                } else if(bctype(i+1,j) > -10) {  // 其他边界
                    A_s(i,j) = 0;
                    Ap_temp += D_s + max(0.0,-F_s);  // 移除系数2
                    source_x_temp += alpha_uv*zoneu[zoneid(i+1,j)]*(D_s + max(0.0,F_s));  // 移除系数2
                    source_y_temp += alpha_uv*zonev[zoneid(i+1,j)]*(D_s + max(0.0,F_s));  // 移除系数2
                }
                
                A_p(i,j) = Ap_temp;
                
                source_x_temp += (1-alpha_uv)*A_p(i,j)*u_star(i,j);
                source_y_temp += (1-alpha_uv)*A_p(i,j)*v_star(i,j);
                // 设置源项
                source_x[n] = source_x_temp;
                source_y[n] = source_y_temp;
            }
        }
    }

    A_e = alpha_uv*A_e;
    A_w = alpha_uv*A_w;
    A_n = alpha_uv*A_n;
    A_s = alpha_uv*A_s;
    
    // 将系数复制到v方程
    equ_v.A_p = equ_u.A_p;
    equ_v.A_w = equ_u.A_w;
    equ_v.A_e = equ_u.A_e;
    equ_v.A_n = equ_u.A_n;
    equ_v.A_s = equ_u.A_s;
    
}

void momentum_function_unsteady(Mesh &mesh, Equation &equ_u, Equation &equ_v,double mu,double dt,double alpha_uv)
{   
     //-1 压力出口(给定压强)
    //-2 固定速度
    //-3 并行交界面

    int n,i,j;
    int n_x=equ_u.n_x;
    int n_y=equ_u.n_y;
    double D_e,D_w,D_n,D_s,F_e,F_n,F_s,F_w;
    
    
    D_e=dy*mu/(dx);
    D_w=dy*mu/(dx);
    D_n=dx*mu/(dy);
    D_s=dx*mu/(dy);
    // 引用网格变量
    MatrixXd &zoneid = mesh.zoneid;
    MatrixXd &bctype = mesh.bctype;
    MatrixXd &u= mesh.u;
    MatrixXd &v= mesh.v;
    MatrixXd &u_face= mesh.u_face;
    MatrixXd &v_face= mesh.v_face;
    MatrixXd &p= mesh.p;
    MatrixXd &p_star= mesh.p_star;
    MatrixXd &p_prime= mesh.p_prime;
    MatrixXd &u_star= mesh.u_star;
    MatrixXd &v_star= mesh.v_star;
    MatrixXd &A_p=equ_u.A_p;
    MatrixXd &A_e=equ_u.A_e;
    MatrixXd &A_w=equ_u.A_w;
    MatrixXd &A_n=equ_u.A_n;
    MatrixXd &A_s=equ_u.A_s;
    VectorXd &source_x=equ_u.source;
    VectorXd &source_y=equ_v.source;
    vector<double> zoneu=mesh.zoneu;
    vector<double> zonev=mesh.zonev;
    
    // 遍历网格
    for(i=0; i<=n_y+1; i++) {
        for(j=0; j<=n_x+1; j++) {
            if(bctype(i,j) == 0) {  // 内部面
                n = mesh.interid(i,j) - 1;
                
                // 计算面上流量
                F_e = dy*u_face(i,j);
                F_w = dy*u_face(i,j-1);
                F_n = dx*v_face(i-1,j);
                F_s = dx*v_face(i,j);
                
                double Ap_temp = 0;
               // 初始化源项
               double source_x_temp, source_y_temp;
    
                           // 处理 x 方向源项
            if((bctype(i,j-1) == 0 ||  bctype(i,j-1) == -3) && 
               (bctype(i,j+1) == 0 ||  bctype(i,j+1) == -3)) {
                // 两侧都是内部点或滑移边界，使用中心差分
                source_x_temp = 0.5*alpha_uv*(p(i,j-1)-p(i,j+1))*dy;
                
            } else if(bctype(i,j-1) == -1) {
                // 左边是压力为0的边界
                source_x_temp = 0.5*alpha_uv*(-p(i,j+1))*dy;
            } else if(bctype(i,j+1) == -1) {
                // 右边是压力为0的边界
                source_x_temp = 0.5*alpha_uv*(p(i,j-1))*dy;
            } else if(bctype(i,j-1) == -2) {
                // 左边是速度入口
                source_x_temp = 0.5*alpha_uv*(p(i,j)-p(i,j+1))*dy;
            } else if(bctype(i,j+1) == -2) {
                // 右边是速度入口
                source_x_temp = 0.5*alpha_uv*(p(i,j-1)-p(i,j))*dy;
            } else if(bctype(i,j-1) != 0 && bctype(i,j+1) == 0) {
                // 左边是其他边界，右边是内部点
                source_x_temp = 0.5*alpha_uv*(p(i,j)-p(i,j+1))*dy;
            } else if(bctype(i,j-1) == 0 && bctype(i,j+1) != 0) {
                // 左边是内部点，右边是其他边界
                source_x_temp = 0.5*alpha_uv*(p(i,j-1)-p(i,j))*dy;
            } else {
                // 两边都是固定边界或其他情况
                source_x_temp = 0.0; 
            }
            
            // 处理 y 方向源项
            if((bctype(i-1,j) == 0 ||  bctype(i-1,j) == -3) && 
               (bctype(i+1,j) == 0 ||  bctype(i+1,j) == -3)) {
                // 上下都是内部点或滑移边界，使用中心差分
                source_y_temp = 0.5*alpha_uv*(p(i+1,j)-p(i-1,j))*dx;
                
            } else if(bctype(i-1,j) == -1) {
                // 上边是压力为0的边界
                source_y_temp = 0.5*alpha_uv*(p(i+1,j))*dx;
                
            } else if(bctype(i+1,j) == -1) {
                // 下边是压力为0的边界  压力出口
                source_y_temp = 0.5*alpha_uv*(-p(i-1,j))*dx;
            } else if(bctype(i-1,j) == -2) {
                // 上边是压力为0的边界
                source_y_temp = 0.5*alpha_uv*(p(i+1,j)-p(i,j))*dx;
                
            } else if(bctype(i+1,j) == -2) {
                // 下边是压力为0的边界
                source_y_temp = 0.5*alpha_uv*(p(i,j)-p(i-1,j))*dx;
            } else if(bctype(i-1,j) != 0 && bctype(i+1,j) == 0) {
                // 上边是其他边界，下边是内部点
                source_y_temp = 0.5*alpha_uv*(p(i+1,j)-p(i,j))*dx;
            } else if(bctype(i-1,j) == 0 && bctype(i+1,j) != 0) {
                // 上边是内部点，下边是其他边界
                source_y_temp = 0.5*alpha_uv*(p(i,j)-p(i-1,j))*dx;
            } else {
                // 上下都是固定边界或其他情况
                source_y_temp = 0.0;
            }
              
                // 检查东面
                if(bctype(i,j+1) == 0) {  // 内部点
                    A_e(i,j) = D_e + max(0.0,-F_e);
                    Ap_temp += D_e + max(0.0,F_e);
                    
                } 
                else if(bctype(i,j+1) ==-3) {  // 其他边界
                    A_e(i,j) = D_e + max(0.0,-F_e);
                    Ap_temp += D_e + max(0.0,F_e);
                }
                
                else if(bctype(i,j+1) > 0) {  // wall边界
                    A_e(i,j) = 0;
                    Ap_temp += 2*D_e + max(0.0,F_e);
                    source_x_temp += alpha_uv*zoneu[zoneid(i,j+1)]*(2*D_e + max(0.0,-F_e));
                    source_y_temp += alpha_uv*zonev[zoneid(i,j+1)]*(2*D_e + max(0.0,-F_e));
                } 
                else if(bctype(i,j+1) ==-1 ) {  // 其他边界
                    A_e(i,j) = 0;
                    Ap_temp += D_e + max(0.0,F_e);
                    source_x_temp += alpha_uv*u_star(i,j)*(D_e + max(0.0,-F_e));  // 移除系数2
                    source_y_temp += alpha_uv*v_star(i,j)*(D_e + max(0.0,-F_e));  // 移除系数2
                }
                else if(bctype(i,j+1) > -10) {  // 其他边界
                    A_e(i,j) = 0;
                    Ap_temp += D_e + max(0.0,F_e);  // 移除系数2
                    source_x_temp += alpha_uv*zoneu[zoneid(i,j+1)]*(D_e + max(0.0,-F_e));  // 移除系数2
                    source_y_temp += alpha_uv*zonev[zoneid(i,j+1)]*(D_e + max(0.0,-F_e));  // 移除系数2
                }
                 
                // 检查西面
                if(bctype(i,j-1) == 0) {  // 内部点
                    A_w(i,j) = D_w + max(0.0,F_w);
                    Ap_temp += D_w + max(0.0,-F_w);
                } 
                else if(bctype(i,j-1) ==-3) {  // 其他边界
                    A_w(i,j) = D_w + max(0.0,F_w);
                    Ap_temp += D_w + max(0.0,-F_w);
                }
                else if(bctype(i,j-1) ==-1) {  // 其他边界
                    A_w(i,j) = 0;
                    Ap_temp += D_w + max(0.0,-F_w);
                    source_x_temp += alpha_uv*u_star(i,j)*(D_w + max(0.0,F_w));  // 移除系数2
                    source_y_temp += alpha_uv*v_star(i,j)*(D_w + max(0.0,F_w));  // 移除系数2
                }
                else if(bctype(i,j-1) > 0) {  //wall边界
                    A_w(i,j) = 0;
                    Ap_temp += 2*D_w + max(0.0,-F_w);
                    source_x_temp += alpha_uv*zoneu[zoneid(i,j-1)]*(2*D_w + max(0.0,F_w));
                    source_y_temp += alpha_uv*zonev[zoneid(i,j-1)]*(2*D_w + max(0.0,F_w));
                } else if(bctype(i,j-1) > -10) {  //其他
                    A_w(i,j) = 0;
                    Ap_temp += D_w + max(0.0,-F_w);  // 移除系数2
                    source_x_temp += alpha_uv*zoneu[zoneid(i,j-1)]*(D_w + max(0.0,F_w));  // 移除系数2
                    source_y_temp += alpha_uv*zonev[zoneid(i,j-1)]*(D_w + max(0.0,F_w));  // 移除系数2
                }
                
                // 检查北面
                if(bctype(i-1,j) == 0) {  // 内部点
                    A_n(i,j) = D_n + max(0.0,-F_n);
                    Ap_temp += D_n + max(0.0,F_n);
                } 
                else if(bctype(i-1,j) == -1) {  // 压力出口
                    A_n(i,j) = 0;
                    Ap_temp += D_n + max(0.0,F_n);
                    source_x_temp += alpha_uv*u_star(i-1,j)*(D_n + max(0.0,-F_n));
                    source_y_temp += alpha_uv*v_star(i-1,j)*(D_n + max(0.0,-F_n));
                } 
               
                else if(bctype(i-1,j) > 0) {  // wall边界
                    A_n(i,j) = 0;
                    Ap_temp += 2*D_n + max(0.0,F_n);
                    source_x_temp += alpha_uv*zoneu[zoneid(i-1,j)]*(2*D_n + max(0.0,-F_n));
                    source_y_temp += alpha_uv*zonev[zoneid(i-1,j)]*(2*D_n + max(0.0,-F_n));
                } else if(bctype(i-1,j) > -10) {  // 其他边界
                    A_n(i,j) = 0;
                    Ap_temp += D_n + max(0.0,F_n);  // 移除系数2
                    source_x_temp += alpha_uv*zoneu[zoneid(i-1,j)]*(D_n + max(0.0,-F_n));  // 移除系数2
                    source_y_temp += alpha_uv*zonev[zoneid(i-1,j)]*(D_n + max(0.0,-F_n));  // 移除系数2
                }
                
                // 检查南面
                if(bctype(i+1,j) == 0) {  // 内部点
                    A_s(i,j) = D_s + max(0.0,F_s);
                    Ap_temp += D_s + max(0.0,-F_s);
                }
                else if(bctype(i+1,j) == -1) {  // 压力出口
                    A_s(i,j) =0;
                    Ap_temp += D_s + max(0.0,-F_s);  
                    source_x_temp += alpha_uv*u(i+1,j)*(D_s + max(0.0,F_s));  
                    source_y_temp += alpha_uv*v(i+1,j)*(D_s + max(0.0,F_s));  
                }
                 else if(bctype(i+1,j) > 0) {  // wall边界
                    A_s(i,j) = 0;
                    Ap_temp += 2*D_s + max(0.0,-F_s);
                    source_x_temp += alpha_uv*zoneu[zoneid(i+1,j)]*(2*D_s + max(0.0,F_s));
                    source_y_temp += alpha_uv*zonev[zoneid(i+1,j)]*(2*D_s + max(0.0,F_s));
                } else if(bctype(i+1,j) > -10) {  // 其他边界
                    A_s(i,j) = 0;
                    Ap_temp += D_s + max(0.0,-F_s);  // 移除系数2
                    source_x_temp += alpha_uv*zoneu[zoneid(i+1,j)]*(D_s + max(0.0,F_s));  // 移除系数2
                    source_y_temp += alpha_uv*zonev[zoneid(i+1,j)]*(D_s + max(0.0,F_s));  // 移除系数2
                }
                
                A_p(i,j) = Ap_temp+dx*dy/dt;
                
                source_x_temp += (1-alpha_uv)*A_p(i,j)*u_star(i,j)+alpha_uv* dx*dy*mesh.u0(i,j)/dt;
                source_y_temp += (1-alpha_uv)*A_p(i,j)*v_star(i,j)+alpha_uv* dx*dy*mesh.v0(i,j)/dt;
                // 设置源项
                source_x[n] = source_x_temp;
                source_y[n] = source_y_temp;
            }
        }
    }

    A_e = alpha_uv*A_e;
    A_w = alpha_uv*A_w;
    A_n = alpha_uv*A_n;
    A_s = alpha_uv*A_s;
    
    // 将系数复制到v方程
    equ_v.A_p = equ_u.A_p;
    equ_v.A_w = equ_u.A_w;
    equ_v.A_e = equ_u.A_e;
    equ_v.A_n = equ_u.A_n;
    equ_v.A_s = equ_u.A_s;
    
}

void momentum_function_PISO(Mesh &mesh, Equation &equ_u, Equation &equ_v,double mu,double dt)
{   
    //-1 压力出口(给定压强)
    //-2 固定速度
    //-3 并行交界面

    int n,i,j;
    int n_x=equ_u.n_x;
    int n_y=equ_u.n_y;
    double D_e,D_w,D_n,D_s,F_e,F_n,F_s,F_w;

    
    D_e=dy*mu/(dx);
    D_w=dy*mu/(dx);
    D_n=dx*mu/(dy);
    D_s=dx*mu/(dy);
    
    // 引用网格变量
    MatrixXd &zoneid = mesh.zoneid;
    MatrixXd &bctype = mesh.bctype;
    MatrixXd &u= mesh.u;
    MatrixXd &v= mesh.v;
    MatrixXd &u_face= mesh.u_face;
    MatrixXd &v_face= mesh.v_face;
    MatrixXd &p= mesh.p;
    MatrixXd &p_star= mesh.p_star;
    MatrixXd &p_prime= mesh.p_prime;
    MatrixXd &u_star= mesh.u_star;
    MatrixXd &v_star= mesh.v_star;
    MatrixXd &A_p=equ_u.A_p;
    MatrixXd &A_e=equ_u.A_e;
    MatrixXd &A_w=equ_u.A_w;
    MatrixXd &A_n=equ_u.A_n;
    MatrixXd &A_s=equ_u.A_s;
    VectorXd &source_x=equ_u.source;
    VectorXd &source_y=equ_v.source;
    vector<double> zoneu=mesh.zoneu;
    vector<double> zonev=mesh.zonev;
    
    // 遍历网格
    for(i=0; i<=n_y+1; i++) {
        for(j=0; j<=n_x+1; j++) {
            if(bctype(i,j) == 0) {  // 内部面
                n = mesh.interid(i,j) - 1;
                
                // 计算面上流量
                F_e = dy*u_face(i,j);
                F_w = dy*u_face(i,j-1);
                F_n = dx*v_face(i-1,j);
                F_s = dx*v_face(i,j);
                
                double Ap_temp = 0;
               // 初始化源项
               double source_x_temp, source_y_temp;
    
                           // 处理 x 方向源项
            if((bctype(i,j-1) == 0 ||  bctype(i,j-1) == -3) && 
               (bctype(i,j+1) == 0 ||  bctype(i,j+1) == -3)) {
                // 两侧都是内部点或滑移边界，使用中心差分
                source_x_temp = 0.5*(p(i,j-1)-p(i,j+1))*dy;
                
            } else if(bctype(i,j-1) == -1) {
                // 左边是压力为0的边界
                source_x_temp = 0.5*(-p(i,j+1))*dy;
            } else if(bctype(i,j+1) == -1) {
                // 右边是压力为0的边界
                source_x_temp = 0.5*(p(i,j-1))*dy;
            } else if(bctype(i,j-1) == -2) {
                // 左边是速度入口
                source_x_temp = 0.5*(p(i,j)-p(i,j+1))*dy;
            } else if(bctype(i,j+1) == -2) {
                // 右边是速度入口
                source_x_temp = 0.5*(p(i,j-1)-p(i,j))*dy;
            } else if(bctype(i,j-1) != 0 && bctype(i,j+1) == 0) {
                // 左边是其他边界，右边是内部点
                source_x_temp = 0.5*(p(i,j)-p(i,j+1))*dy;
            } else if(bctype(i,j-1) == 0 && bctype(i,j+1) != 0) {
                // 左边是内部点，右边是其他边界
                source_x_temp = 0.5*(p(i,j-1)-p(i,j))*dy;
            } else {
                // 两边都是固定边界或其他情况
                source_x_temp = 0.0; 
            }
            
            // 处理 y 方向源项
            if((bctype(i-1,j) == 0 ||  bctype(i-1,j) == -3) && 
               (bctype(i+1,j) == 0 ||  bctype(i+1,j) == -3)) {
                // 上下都是内部点或滑移边界，使用中心差分
                source_y_temp = 0.5*(p(i+1,j)-p(i-1,j))*dx;
                
            } else if(bctype(i-1,j) == -1) {
                // 上边是压力为0的边界
                source_y_temp = 0.5*(p(i+1,j))*dx;
                
            } else if(bctype(i+1,j) == -1) {
                // 下边是压力为0的边界  压力出口
                source_y_temp = 0.5*(-p(i-1,j))*dx;
            } else if(bctype(i-1,j) == -2) {
                // 上边是压力为0的边界
                source_y_temp = 0.5*(p(i+1,j)-p(i,j))*dx;
                
            } else if(bctype(i+1,j) == -2) {
                // 下边是压力为0的边界
                source_y_temp = 0.5*(p(i,j)-p(i-1,j))*dx;
            } else if(bctype(i-1,j) != 0 && bctype(i+1,j) == 0) {
                // 上边是其他边界，下边是内部点
                source_y_temp = 0.5*(p(i+1,j)-p(i,j))*dx;
            } else if(bctype(i-1,j) == 0 && bctype(i+1,j) != 0) {
                // 上边是内部点，下边是其他边界
                source_y_temp = 0.5*(p(i,j)-p(i-1,j))*dx;
            } else {
                // 上下都是固定边界或其他情况
                source_y_temp = 0.0;
            }
              
                // 检查东面
                if(bctype(i,j+1) == 0) {  // 内部点
                    A_e(i,j) = D_e + max(0.0,-F_e);
                    Ap_temp += D_e + max(0.0,F_e);
                    
                } 
                else if(bctype(i,j+1) ==-3) {  // 其他边界
                    A_e(i,j) = D_e + max(0.0,-F_e);
                    Ap_temp += D_e + max(0.0,F_e);
                }
                
                else if(bctype(i,j+1) > 0) {  // wall边界
                    A_e(i,j) = 0;
                    Ap_temp += 2*D_e + max(0.0,F_e);
                    source_x_temp += zoneu[zoneid(i,j+1)]*(2*D_e + max(0.0,-F_e));
                    source_y_temp += zonev[zoneid(i,j+1)]*(2*D_e + max(0.0,-F_e));
                } 
                else if(bctype(i,j+1) ==-1 ) {  // 其他边界
                    A_e(i,j) = 0;
                    Ap_temp += D_e + max(0.0,F_e);
                    source_x_temp += u_star(i,j)*(D_e + max(0.0,-F_e));  // 移除系数2
                    source_y_temp += v_star(i,j)*(D_e + max(0.0,-F_e));  // 移除系数2
                }
                else if(bctype(i,j+1) > -10) {  // 其他边界
                    A_e(i,j) = 0;
                    Ap_temp += D_e + max(0.0,F_e);  // 移除系数2
                    source_x_temp += zoneu[zoneid(i,j+1)]*(D_e + max(0.0,-F_e));  // 移除系数2
                    source_y_temp += zonev[zoneid(i,j+1)]*(D_e + max(0.0,-F_e));  // 移除系数2
                }
                 
                // 检查西面
                if(bctype(i,j-1) == 0) {  // 内部点
                    A_w(i,j) = D_w + max(0.0,F_w);
                    Ap_temp += D_w + max(0.0,-F_w);
                } 
                else if(bctype(i,j-1) ==-3) {  // 其他边界
                    A_w(i,j) = D_w + max(0.0,F_w);
                    Ap_temp += D_w + max(0.0,-F_w);
                }
                else if(bctype(i,j-1) ==-1) {  // 其他边界
                    A_w(i,j) = 0;
                    Ap_temp += D_w + max(0.0,-F_w);
                    source_x_temp += u_star(i,j)*(D_w + max(0.0,F_w));  // 移除系数2
                    source_y_temp += v_star(i,j)*(D_w + max(0.0,F_w));  // 移除系数2
                }
                else if(bctype(i,j-1) > 0) {  //wall边界
                    A_w(i,j) = 0;
                    Ap_temp += 2*D_w + max(0.0,-F_w);
                    source_x_temp += zoneu[zoneid(i,j-1)]*(2*D_w + max(0.0,F_w));
                    source_y_temp += zonev[zoneid(i,j-1)]*(2*D_w + max(0.0,F_w));
                } else if(bctype(i,j-1) > -10) {  //其他
                    A_w(i,j) = 0;
                    Ap_temp += D_w + max(0.0,-F_w);  // 移除系数2
                    source_x_temp += zoneu[zoneid(i,j-1)]*(D_w + max(0.0,F_w));  // 移除系数2
                    source_y_temp += zonev[zoneid(i,j-1)]*(D_w + max(0.0,F_w));  // 移除系数2
                }
                
                // 检查北面
                if(bctype(i-1,j) == 0) {  // 内部点
                    A_n(i,j) = D_n + max(0.0,-F_n);
                    Ap_temp += D_n + max(0.0,F_n);
                } 
                else if(bctype(i-1,j) == -1) {  // 压力出口
                    A_n(i,j) = 0;
                    Ap_temp += D_n + max(0.0,F_n);
                    source_x_temp += u_star(i-1,j)*(D_n + max(0.0,-F_n));
                    source_y_temp += v_star(i-1,j)*(D_n + max(0.0,-F_n));
                } 
               
                else if(bctype(i-1,j) > 0) {  // wall边界
                    A_n(i,j) = 0;
                    Ap_temp += 2*D_n + max(0.0,F_n);
                    source_x_temp += zoneu[zoneid(i-1,j)]*(2*D_n + max(0.0,-F_n));
                    source_y_temp += zonev[zoneid(i-1,j)]*(2*D_n + max(0.0,-F_n));
                } else if(bctype(i-1,j) > -10) {  // 其他边界
                    A_n(i,j) = 0;
                    Ap_temp += D_n + max(0.0,F_n);  // 移除系数2
                    source_x_temp += zoneu[zoneid(i-1,j)]*(D_n + max(0.0,-F_n));  // 移除系数2
                    source_y_temp += zonev[zoneid(i-1,j)]*(D_n + max(0.0,-F_n));  // 移除系数2
                }
                
                // 检查南面
                if(bctype(i+1,j) == 0) {  // 内部点
                    A_s(i,j) = D_s + max(0.0,F_s);
                    Ap_temp += D_s + max(0.0,-F_s);
                }
                else if(bctype(i+1,j) == -1) {  // 压力出口
                    A_s(i,j) =0;
                    Ap_temp += D_s + max(0.0,-F_s);  
                    source_x_temp += u(i+1,j)*(D_s + max(0.0,F_s));  
                    source_y_temp += v(i+1,j)*(D_s + max(0.0,F_s));  
                }
                 else if(bctype(i+1,j) > 0) {  // wall边界
                    A_s(i,j) = 0;
                    Ap_temp += 2*D_s + max(0.0,-F_s);
                    source_x_temp += zoneu[zoneid(i+1,j)]*(2*D_s + max(0.0,F_s));
                    source_y_temp += zonev[zoneid(i+1,j)]*(2*D_s + max(0.0,F_s));
                } else if(bctype(i+1,j) > -10) {  // 其他边界
                    A_s(i,j) = 0;
                    Ap_temp += D_s + max(0.0,-F_s);  // 移除系数2
                    source_x_temp += zoneu[zoneid(i+1,j)]*(D_s + max(0.0,F_s));  // 移除系数2
                    source_y_temp += zonev[zoneid(i+1,j)]*(D_s + max(0.0,F_s));  // 移除系数2
                }
                
                A_p(i,j) = Ap_temp+dx*dy/dt;
                
                source_x_temp += dx*dy*mesh.u0(i,j)/dt;
                source_y_temp += dx*dy*mesh.v0(i,j)/dt;
                // 设置源项
                source_x[n] = source_x_temp;
                source_y[n] = source_y_temp;
            }
        }
    }

    A_e = A_e;
    A_w = A_w;
    A_n = A_n;
    A_s = A_s;
    
    // 将系数复制到v方程
    equ_v.A_p = equ_u.A_p;
    equ_v.A_w = equ_u.A_w;
    equ_v.A_e = equ_u.A_e;
    equ_v.A_n = equ_u.A_n;
    equ_v.A_s = equ_u.A_s;
    
}
//并行相关
vector<Mesh> splitMeshVertically(const Mesh& original_mesh, int n) {
    std::vector<Mesh> sub_meshes;

    // 原始网格尺寸和子网格宽度计算保持不变
    int original_nx = original_mesh.nx;
    int original_ny = original_mesh.ny;
    vector<int> widths(n);
    int remaining = original_nx;
    for(int k = 0; k < n; k++) {
        widths[k] = (remaining + (n-k-1)) / (n-k);
        remaining -= widths[k];
    }

    int start_idx = 0;
    for(int k = 0; k < n; k++) {
        // 创建子网格
        Mesh sub_mesh(original_ny, widths[k]);
         sub_mesh.initializeToZero();
        // 复制属性
        sub_mesh.zoneu = original_mesh.zoneu;
        sub_mesh.zonev = original_mesh.zonev;

        // 复制网格数据
        for(int i = 0; i <= original_ny + 1; i++) {
            for(int j = 0; j <= widths[k] + 1; j++) {
                int orig_j = start_idx + j;

                // 复制bctype
                if(k == 0 && j == 0) {
                    // 第一个子网格的左边界
                    sub_mesh.bctype(i,j) = original_mesh.bctype(i,0);
                } else if(k == n-1 && j == widths[k] + 1) {
                    // 最后一个子网格的右边界
                    sub_mesh.bctype(i,j) = original_mesh.bctype(i,original_nx+1);
                } else if(j == 0 || j == widths[k] + 1) {
                    // 内部接口边界
                    sub_mesh.bctype(i,j) = -3;
                } else {
                    // 内部点
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

        // 检查是否需要创建扩展网格
        bool left_is_interface = true;
        bool right_is_interface = true;

        // 检查左边界是否全为-3
        for(int i = 0; i <= original_ny + 1; i++) {
            if(sub_mesh.bctype(i,0) != -3) {
                left_is_interface = false;
                break;
            }
        }

        // 检查右边界是否全为-3
        for(int i = 0; i <= original_ny + 1; i++) {
            if(sub_mesh.bctype(i,widths[k]+1) != -3) {
                right_is_interface = false;
                break;
            }
        }

        // 创建扩展网格
        if(left_is_interface || right_is_interface) {
            int extra_cols = (left_is_interface ? 1 : 0) + (right_is_interface ? 1 : 0);
            Mesh sub_mesh2(original_ny, widths[k] + extra_cols);
            sub_mesh2.initializeToZero();
            // 复制基本属性
            sub_mesh2.zoneu = sub_mesh.zoneu;
            sub_mesh2.zonev = sub_mesh.zonev;

            // 复制并扩展网格数据
            int offset = left_is_interface ? 1 : 0;
            for(int i = 0; i <= original_ny + 1; i++) {
                // 复制原有数据
                for(int j = 0; j <= widths[k] + 1; j++) {
                    int new_j = j + offset;
                    sub_mesh2.bctype(i,new_j) = sub_mesh.bctype(i,j);
                    sub_mesh2.zoneid(i,new_j) = sub_mesh.zoneid(i,j);
                    sub_mesh2.u(i,new_j) = sub_mesh.u(i,j);
                    sub_mesh2.v(i,new_j) = sub_mesh.v(i,j);
                    sub_mesh2.p(i,new_j) = sub_mesh.p(i,j);
                    
                }

                // 设置左侧额外列
                if(left_is_interface) {
                    sub_mesh2.bctype(i,0) = -3;
                    sub_mesh2.zoneid(i,0) = 0;
                    sub_mesh2.u(i,0) = 0.0;
                    sub_mesh2.v(i,0) = 0.0;
                    
                    
                }

                // 设置右侧额外列
                if(right_is_interface) {
                    int last_col = widths[k] + offset + 2;  // 修改这里
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
            
            // 初始化扩展网格
            sub_mesh2.initializeBoundaryConditions();
            sub_mesh2.createInterId();

            // 将扩展网格添加到结果中
            sub_meshes.push_back(sub_mesh2);
        } else {
            // 初始化原始子网格
            sub_mesh.initializeBoundaryConditions();
            sub_mesh.createInterId();
            sub_meshes.push_back(sub_mesh);
        }

        start_idx += widths[k];
    }

    return sub_meshes;
}


Mesh mergeMeshesWithoutInterface(const std::vector<Mesh>& sub_meshes) {
    // 计算合并后的网格尺寸
    int total_nx = 0;
    int ny = sub_meshes[0].ny;
    
    // 计算总宽度（不包括接口列）
    for(size_t i = 0; i < sub_meshes.size(); i++) {
        for(int j = 0; j <= sub_meshes[i].nx + 1; j++) {
            // 只计算非接口列
            if(sub_meshes[i].bctype(0,j) != -3) {
                total_nx++;
            }
        }
    }
    total_nx = total_nx ; // 调整重复计算的列

    // 创建合并后的网格
    Mesh merged_mesh(ny, total_nx);
    merged_mesh.zoneu = sub_meshes[0].zoneu;
    merged_mesh.zonev = sub_meshes[0].zonev;
    
    // 记录当前写入位置
    int current_col = 0;
    
    // 逐个合并子网格
    for(size_t k = 0; k < sub_meshes.size(); k++) {
        const Mesh& sub_mesh = sub_meshes[k];
        
        // 复制非接口列的数据
        for(int i = 0; i <= ny + 1; i++) {
            for(int j = 0; j <= sub_mesh.nx + 1; j++) {
                // 跳过接口列
                if(sub_mesh.bctype(i,j) == -3) {
                    continue;
                }
                
                // 复制数据到合并网格
                
                merged_mesh.u(i, current_col) = sub_mesh.u(i,j);
                merged_mesh.v(i, current_col) = sub_mesh.v(i,j);
                merged_mesh.p(i, current_col) = sub_mesh.p(i,j);
             
                
                
                current_col++;
            }
        }
        
        
    }
    
   
    return merged_mesh;
}

void readParams(const std::string& folderPath, double& dx, double& dy) {
    std::string filePath = folderPath + "/params.txt";
    std::ifstream file(filePath);

    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filePath << std::endl;
        return;
    }

    std::string line;
    // 先读掉第一行
    std::getline(file, line);

    // 再读第二行
    if (std::getline(file, line)) {
        std::istringstream iss(line);
        if (!(iss >> dx >> dy)) {
            std::cerr << "读取 dx 和 dy 失败，检查第二行格式" << std::endl;
        }
    } else {
        std::cerr << "文件内容不足，缺少第二行" << std::endl;
    }

    file.close();
}


void saveMeshData(const Mesh& mesh, int rank, const std::string& timestep_folder) {
    // 创建基础文件名
    std::string u_filename = "u_" + std::to_string(rank) + ".dat";
    std::string v_filename = "v_" + std::to_string(rank) + ".dat";
    std::string p_filename = "p_" + std::to_string(rank) + ".dat";

    // 如果提供了时间步文件夹，添加路径
    if(!timestep_folder.empty()) {
        if (!fs::exists(timestep_folder)) {
            fs::create_directory(timestep_folder);
        }
        u_filename = timestep_folder + "/" + u_filename;
        v_filename = timestep_folder + "/" + v_filename;
        p_filename = timestep_folder + "/" + p_filename;
    }

    try {
        // 保存 u_star
        std::ofstream u_file(u_filename);
        if(!u_file) throw std::runtime_error("无法创建文件: " + u_filename);
        u_file << mesh.u_star;
        u_file.close();

        // 保存 v_star
        std::ofstream v_file(v_filename);
        if(!v_file) throw std::runtime_error("无法创建文件: " + v_filename);
        v_file << mesh.v_star;
        v_file.close();

        // 保存 p_star
        std::ofstream p_file(p_filename);
        if(!p_file) throw std::runtime_error("无法创建文件: " + p_filename);
        p_file << mesh.p;
        p_file.close();

        // 新建 steady 文件夹（如果不存在）
        std::string steady_folder = "steady";
        if (!fs::exists(steady_folder)) {
            fs::create_directory(steady_folder);
        }

        // steady 目录下的文件名
        std::string su_filename = steady_folder + "/u_" + std::to_string(rank) + ".dat";
        std::string sv_filename = steady_folder + "/v_" + std::to_string(rank) + ".dat";
        std::string sp_filename = steady_folder + "/p_" + std::to_string(rank) + ".dat";
        std::string suf_filename = steady_folder + "/uf_" + std::to_string(rank) + ".dat";
        std::string svf_filename = steady_folder + "/vf_" + std::to_string(rank) + ".dat";

        // 保存 steady/u_star
        std::ofstream su_file(su_filename);
        if(!su_file) throw std::runtime_error("无法创建文件: " + su_filename);
        su_file << mesh.u_star;
        su_file.close();

        // 保存 steady/v_star
        std::ofstream sv_file(sv_filename);
        if(!sv_file) throw std::runtime_error("无法创建文件: " + sv_filename);
        sv_file << mesh.v_star;
        sv_file.close();

        // 保存 steady/p_star
        std::ofstream sp_file(sp_filename);
        if(!sp_file) throw std::runtime_error("无法创建文件: " + sp_filename);
        sp_file << mesh.p_star;
        sp_file.close();

        // 保存 steady/u_face -> uf_rank
        std::ofstream suf_file(suf_filename);
        if(!suf_file) throw std::runtime_error("无法创建文件: " + suf_filename);
        suf_file << mesh.u_face;
        suf_file.close();

        // 保存 steady/v_face -> vf_rank
        std::ofstream svf_file(svf_filename);
        if(!svf_file) throw std::runtime_error("无法创建文件: " + svf_filename);
        svf_file << mesh.v_face;
        svf_file.close();

    } catch(const std::exception& e) {
        std::cerr << "保存数据时出错: " << e.what() << std::endl;
    }
}


