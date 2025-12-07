#include "fluid.h"
#include <filesystem>
#include <chrono>
namespace fs = std::filesystem;

int main ()
{

    double L;

    int n_x0,n_y0;
    std::cout << "x方向上划分个数:";
    std::cin >> n_x0;
    
    std::cout << "y方向上划分个数:";
    std::cin >> n_y0;
    //a，b为网格边长

    std::cout << "长度:";
    std::cin >> L;
    
    dx=a/n_x0;
    dy=a/n_y0;
    
    std::cout << "inlet速度:";
    std::cin >> vx;

    
    
    //创建网格
    Mesh mesh(n_y0,n_x0);
    // 设置四周边界条件 bctype=1
    mesh.setBlock(0, 0, mesh.nx+1, 0, 1, 0);      // 上边界
    mesh.setBlock(0, mesh.ny+1, mesh.nx+1, mesh.ny+1, 1, 0);  // 下
    mesh.setBlock(0, 0, 0, mesh.ny+1, -2, 2);      // 左边界
    mesh.setBlock(mesh.nx+1, 0, mesh.nx+1, mesh.ny+1, -1, 0);  // 右边界

    mesh.setBlock(30, (mesh.ny/2)-4, L+30, (mesh.ny/2), 1, 1);


    // 设置各区域速度
    mesh.setZoneUV(0, 0.0, 0.0);  // 默认值
    mesh.setZoneUV(1, 0.0, 0.0);  // 固定壁面
    mesh.setZoneUV(2, vx, 0.0);   // 顶盖速度
    
    // 初始化边界条件
    mesh.initializeBoundaryConditions();
    //初始化网格id
    mesh.createInterId();
    mesh.saveToFolder("test2" );

}