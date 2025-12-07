# Taiho 流体力学求解器套件

## 项目简介

Taiho 是一个基于有限体积法的并行流体力学计算流体动力学(CFD)求解器套件,采用 C++ 开发,支持 MPI 并行计算。该套件实现了多种经典求解算法,适用于不可压缩流动问题的数值模拟。

### 主要功能

- **网格生成**: 支持结构化直角网格生成,可灵活设置边界条件和障碍物
- **多种求解器**:
  - **PISO 求解器**: 压力隐式分裂算法(Pressure-Implicit with Splitting of Operators),适用于非定常流动
  - **SIMPLE 稳态求解器**: 半隐式压力连接方程算法(Semi-Implicit Method for Pressure-Linked Equations),用于稳态问题
  - **SIMPLE 非定常求解器**: SIMPLE 算法的非定常版本
- **并行计算**: 基于 MPI 的区域分解并行,支持垂直方向网格分割
- **灵活的边界条件**: 支持固壁、入口、出口等多种边界条件设置

## 系统依赖

### 必需的库和工具

1. **编译器**: 支持 C++17 标准的编译器(如 g++ 7.0+)
2. **MPI 库**: MPICH 或 OpenMPI
   ```bash
   sudo apt-get install mpich libmpich-dev
   ```
3. **Eigen3 库**: 线性代数库
   ```bash
   sudo apt-get install libeigen3-dev
   ```
4. **标准库**: C++17 filesystem 支持

### 安装验证

```bash
# 检查 MPI 安装
mpic++ --version

# 检查 Eigen3 安装
ls /usr/include/eigen3/
```

## 编译方法

### 编译所有程序

```bash
make all
```

### 编译单个程序

```bash
make mesh_generation        # 仅编译网格生成程序
make solver_PISO           # 仅编译 PISO 求解器
make solver_simple_steady   # 仅编译 SIMPLE 稳态求解器
make solver_simple_unsteady # 仅编译 SIMPLE 非定常求解器
```

### 清理编译文件

```bash
make clean
```

## 使用说明

### 1. 网格生成

#### 运行方式

```bash
./mesh_generation
```

#### 交互式输入

程序会提示输入以下参数:
- **x方向划分个数**: 网格在 x 方向的单元数
- **y方向划分个数**: 网格在 y 方向的单元数
- **长度**: 计算域的特征长度
- **inlet速度**: 入口速度

#### 修改网格生成逻辑

编辑 `mesh_generation.cpp` 中的 `main` 函数,关键代码说明:

```cpp
// 创建网格
Mesh mesh(n_y0, n_x0);  // 参数: (y方向单元数, x方向单元数)

// 设置边界条件
// setBlock(x1, y1, x2, y2, bcValue, zoneValue)
// bcValue: 边界类型 (-2=入口, -1=出口, 1=固壁)
// zoneValue: 区域ID,用于设置速度
mesh.setBlock(0, 0, mesh.nx+1, 0, 1, 0);  // 设置上边界为固壁

// 设置障碍物示例
mesh.setBlock(30, (mesh.ny/2)-4, L+30, (mesh.ny/2), 1, 1);

// 设置区域速度
// setZoneUV(zoneID, u速度, v速度)
mesh.setZoneUV(0, 0.0, 0.0);  // 区域0默认速度
mesh.setZoneUV(2, vx, 0.0);   // 区域2入口速度

// 保存网格
mesh.saveToFolder("网格文件夹名称");
```

### 2. PISO 求解器(非定常)

#### 命令行参数模式

```bash
./solver_PISO <网格文件夹> <时间步长> <时间步数> <粘度> <并行进程数>
```

**示例**:
```bash
mpirun -np 4 ./solver_PISO test2 0.001 1000 0.01 4
```

#### 交互式模式

```bash
./solver_PISO
```
然后按提示输入各参数。

#### 参数说明

- **网格文件夹**: 网格数据存储路径(由 mesh_generation 生成)
- **时间步长**: 每个时间步的大小(如 0.001)
- **时间步数**: 总共计算的时间步数
- **粘度**: 流体动力粘度系数
- **并行进程数**: MPI 并行进程数,应与 `mpirun -np` 的数值一致

### 3. SIMPLE 稳态求解器

#### 命令行参数模式

```bash
./solver_simple_steady <网格文件夹> <迭代步数> <粘度> <并行进程数>
```

**示例**:
```bash
mpirun -np 4 ./solver_simple_steady test2 500 0.01 4
```

#### 参数说明

- **网格文件夹**: 网格数据存储路径
- **迭代步数**: 稳态求解的迭代次数
- **粘度**: 流体动力粘度系数
- **并行进程数**: MPI 并行进程数

**注意**: 稳态求解器不需要输入时间步长参数。

### 4. SIMPLE 非定常求解器

#### 命令行参数模式

```bash
./solver_simple_unsteady <网格文件夹> <时间步长> <时间步数> <粘度> <并行进程数>
```

**示例**:
```bash
mpirun -np 4 ./solver_simple_unsteady test2 0.001 1000 0.01 4
```

参数说明同 PISO 求解器。

## 典型工作流程

```bash
# 1. 生成网格
./mesh_generation
# 按提示输入: x划分=100, y划分=50, 长度=1.0, 入口速度=1.0

# 2. 运行求解器(选择一个)
# 非定常问题用 PISO
mpirun -np 4 ./solver_PISO test2 0.001 1000 0.01 4

# 稳态问题用 SIMPLE
mpirun -np 4 ./solver_simple_steady test2 500 0.01 4
```

## 并行计算说明

- 程序采用垂直方向(y方向)区域分解
- 并行进程数应能整除网格的 y 方向单元数,以获得均匀分割
- 使用 `mpirun -np N` 启动,其中 N 应与命令行参数中的并行进程数一致

## 输出文件

求解器运行后会在指定目录生成:
- 速度场数据
- 压力场数据
- 计算残差信息

## 常见问题

1. **编译错误: 找不到 Eigen3**
   - 确认已安装 libeigen3-dev
   - 检查 `/usr/include/eigen3/` 是否存在

2. **MPI 错误**
   - 确保 `mpirun -np N` 的 N 值与程序参数中的并行进程数一致
   - 检查 MPICH 是否正确安装

3. **网格生成失败**
   - 确保 x、y 方向划分数为正整数
   - 检查边界条件设置是否合理

## 技术支持

如需修改求解算法或添加新功能,请参考:
- `fluid.h` / `fluid.cpp`: 核心网格和求解器类
- `parallel.h` / `parallel.cpp`: 并行通信接口
- 各求解器源文件: 具体算法实现