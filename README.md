# Taiho-CFD

Taiho-CFD 是一个基于 C++17、Eigen 和 MPI 的二维不可压缩 Navier–Stokes
有限体积求解器。当前实现采用 SIMPLE 压力-速度耦合，使用常量密度 `rho` 与
动力粘度 `mu`，提供定常和一阶 Backward Euler 非定常计算，并针对结构化正交网格进行验证。

当前版本的核心接口已经统一为 `SolverContext` + `ParallelContext` +
`FlowSolver`：应用入口只负责读取算例、构造上下文和驱动时间/迭代循环；网格、
离散算子、线性求解和 MPI 通信分别位于独立模块。求解器本身不从命令行读取物理、
离散或收敛参数，所有求解设置必须在算例目录的 `case.cfg` 中显式给出。

阅读顺序建议：先看[整体对象和数据布局](#核心对象与数据结构)，再看[一次
SIMPLE 迭代](#一次-simple-迭代的精确状态转换)和[`casecfg` 配置映射](#casecfg-到-c-对象的映射)，
最后按[二次开发约定](#可扩展性约定)添加算法或物理量。

## 当前实现范围

- 二维、结构化、轴对齐正交网格；支持 x/y 方向非均匀拉伸。
- SIMPLE 迭代，沿 x 方向 MPI 域分解，每个子域使用两层 ghost 单元。
- 速度方程：一阶 Upwind 对流、Orthogonal 两点扩散、Central 压力梯度。
- 面速度：线性插值和 Rhie–Chow 修正。
- 动量方程使用 BiCGSTAB + ILUT，压力修正方程使用 PCG + incomplete Cholesky。
- 壁面、速度入口、压力出口和 MPI Processor 接口。
- 压力出口将 Dirichlet 条件正确加入压力方程对角项；全封闭区域使用单个压力参考单元。

当前不包含非正交修正、复杂几何边界、湍流模型和多组分物理。`docs/thesis/`
中的内容是后续研究规划，不代表当前代码已经实现的功能。

## 目录结构

```text
.
├── src/
│   ├── apps/                 # case.cfg 解析与单一应用入口
│   ├── io/                   # 网格读取、MPI 结果写出
│   ├── mesh/                 # 网格几何、CellKind、BoundaryPatch
│   ├── numerics/             # 离散算子、动量、压力修正、Rhie–Chow
│   ├── parallel/             # ParallelContext、x 方向域分解和 halo 交换
│   └── solvers/              # 上下文、结果/配置类型、FlowSolver 和求解器
├── tests/                    # 压力、动量和边界模型回归测试
├── tools/
│   ├── generate_mesh.py      # 网格生成和输入校验
│   └── postprocess.py        # MPI 结果拼接、VTK/Tecplot/PNG 输出
├── examples/
│   ├── cases/                # 稳态/瞬态 case.cfg 示例
│   ├── meshes/               # 可直接读取的示例网格
│   └── notebooks/            # 网格、后处理和网格检查 notebook
├── docs/thesis/              # 论文目录和后续规划
├── Makefile
└── LICENSE
```

编译生成的 `build/`、`report/`、求解器可执行文件和运行结果不属于源代码，
由 `.gitignore` 排除；运行结果建议放在单独的 case 目录中。

## 核心对象与数据结构

下面的说明以 `src/mesh/`、`src/numerics/` 和 `src/solvers/` 中的实际 C++
类型为准。求解器没有把每个单元封装成一个 C++ 对象，而是采用“规则网格上的
Eigen 矩阵 + interior 单元压缩向量”的数据布局：几何和场量保留二维索引，
线性系统只对真正参与求解的 interior 单元编号。这样既便于有限体积离散，
也避免把物理边界和 MPI ghost 单元错误地放进线性系统。

### 1. 索引、尺寸和存储约定

所有矩阵使用 Eigen 的 `(行, 列)` 索引 `(i, j)`：`i` 沿 y 方向，`j` 沿 x
方向。对全局网格而言，`i=0` 是底部 `y=0`，`i=ny-1` 是顶部 `y=Ly`；
`j=0` 是左侧，`j=nx-1` 是右侧。一个局部 MPI 网格的 `nx` 包括 ghost 列，
而不是全局真实列数。

| 数据 | Eigen 尺寸 | 说明 |
|---|---:|---|
| 单元中心场 `u,v,p` 及其系数矩阵 | `ny × nx` | cell-centered 数据；包含物理边界和 MPI ghost 列 |
| 节点坐标 `x,y` | `(ny+1) × (nx+1)` | 网格顶点坐标 |
| 单元中心坐标 `x_c,y_c` | `ny × nx` | 四个顶点的平均值 |
| 面面积 `area_e/w/n/s`、体积 `vol` | `ny × nx` | 轴对齐网格上的几何度量 |
| x 法向面速度 `u_face` | `ny × (nx-1)` | 相邻 x 单元之间的面速度 |
| y 法向面速度 `v_face` | `(ny-1) × nx` | 相邻 y 单元之间的面速度 |
| `interi/interj` | 长度 `internumber` | 第 k 个 interior 方程对应的二维 `(i,j)` |
| `interid` | `ny × nx` 整数矩阵 | interior 单元到压缩方程编号的反向映射；非 interior 为 `-1` |

`ny` 和 `nx` 是当前 `Mesh` 对象的局部尺寸。串行时
`owned_j_begin=0, owned_j_end=nx`；MPI 分解后，真实列区间为
`[owned_j_begin, owned_j_end)`，左、右最多各有两列 ghost。代码约定右端点
是开区间，因此 owned 列数始终为 `owned_j_end-owned_j_begin`。

### 2. `Mesh`：所有场、几何和拓扑的所有者

`src/mesh/mesh.h` 中的 `Mesh` 是求解器的核心状态容器。它自身不负责求解
线性方程，而是向离散算子、边界模型和 `SimpleSolver` 提供统一的数据视图。

```text
Mesh
├── 速度场
│   ├── u, v          当前线性方程求得的速度预测量
│   ├── u_star,v_star 压力修正后的速度（用于下一次 SIMPLE 迭代/输出）
│   └── u0, v0       非定常时间项使用的上一时间层速度
├── 压力和面通量
│   ├── p             当前压力
│   ├── p_prime       压力修正方程的未知量
│   └── u_face,v_face Rhie–Chow 面速度
├── 几何
│   ├── x,y           节点坐标
│   ├── x_c,y_c       单元中心坐标
│   ├── area_e/w/n/s  四个面的长度/面积
│   └── vol           单元体积（二维中为面积）
├── 拓扑和边界
│   ├── cell_kind     Interior、PhysicalBoundary 或 Processor
│   ├── patch_id      单元所属 BoundaryPatch 的索引，未绑定时为 -1
│   ├── interid       二维单元到压缩编号的映射
│   └── boundary_patches 结构化边界条件列表
└── 尺寸与 MPI
    ├── nx,ny         当前局部矩阵尺寸
    ├── internumber   interior 单元数
    └── owned_j_begin/end 真实列范围
```

其中 `u/v` 和 `u_star/v_star` 要区分：动量方程先以当前 `u_star/v_star`
为初值装配并求得 `u/v` 预测量，压力修正和速度修正随后更新 `u_star/v_star`。
`u0/v0` 只在非定常计算中跨时间步保存；稳态计算通过
`TimeTerm::none()` 禁用它们的时间项。`p_prime` 是每轮 SIMPLE 的压力修正，
不是要直接输出的绝对压力；`p` 才是物理压力场。

网格构造流程为：分配矩阵 → `initializeToZero()` 初始化场和拓扑默认值 →
`initGeometry()` 计算中心、面面积和体积 → `createInterId()` 只给
`CellKind::Interior` 单元连续编号。`validate()` 会检查尺寸、坐标单调性、
有限且为正的体积，以及要求开启时的物理外边界完整性。

### 3. `CellKind`、`BoundaryPatch` 和字段边界条件

`CellKind` 描述单元在离散拓扑中的角色，而 `BoundaryPatch` 描述边界的物理
条件，两者有意分离：Processor 单元是 MPI 通信拓扑，不是物理边界；一个物理
patch 可以包含多个边界单元。

```cpp
enum class CellKind { Interior, PhysicalBoundary, Processor };
enum class PatchKind { Generic, Wall };
enum class BoundaryConditionType {
    FixedValue, ZeroGradient, InletOutlet, NoSlip
};

struct BoundaryPatch {
    std::string name;
    PatchKind kind;
    VectorBoundaryCondition velocity;
    ScalarBoundaryCondition pressure;
    std::vector<CellIndex> cells;  // 该 patch 的所有单元坐标
};
```

`VectorBoundaryCondition` 保存速度类型、固定值和入口值；
`ScalarBoundaryCondition` 保存压力对应的标量值。当前 legacy reader 将
`bctype.dat` 转换成这些类型：wall 为速度固定/压力零梯度，velocity inlet
为速度固定/压力零梯度，pressure outlet 为速度零梯度/压力固定。转换后，
离散代码通过 `patch_id` 和 `BoundaryPatch` 查询条件，不再在主循环中直接
比较 `0、-1、-2、-3` 等旧整数编码。

### 4. `Equation`：二维系数矩阵和压缩稀疏矩阵

`src/numerics/equation.h` 的 `Equation` 不拥有独立的网格副本，而是保存一个
`Mesh& mesh` 引用，并在该网格上维护一个五点有限体积方程：

```text
Equation
├── A_p, A_e, A_w, A_n, A_s : ny × nx 的面/中心系数
├── source                  : 长度 internumber 的 interior 源项
├── A                       : internumber × internumber 的 Eigen 稀疏矩阵
└── mesh&                   : 关联的 Mesh
```

对一个 interior 单元 P，离散形式约定为：

```text
A_p(P) φ_P - A_e(P) φ_E - A_w(P) φ_W
       - A_n(P) φ_N - A_s(P) φ_S = source(P)
```

`A_*` 保留二维位置，便于按面装配、松弛和 Rhie–Chow 使用；
`buildMatrix()` 再根据 `interid` 把它们压缩成稀疏矩阵 `A`。只有邻居也是
interior 时才写入 `-A_e/-A_w/-A_n/-A_s` 非对角项；物理边界的 Dirichlet
贡献已经在装配阶段并入对角项和 `source`，Processor 邻居则先通过 halo
交换取得系数/场值。`reset()` 清零本轮系数和源项，但不改变网格拓扑。

### 5. 离散策略与 `TimeTerm`

`NumericalSchemes` 是无状态的离散策略集合，当前字段为：

```text
time                  : SteadyState 或 BackwardEuler
velocity_convection   : Upwind
pressure_gradient     : Central
velocity_laplacian    : Orthogonal
face_interpolation    : Linear
```

默认构造的 `NumericalSchemes` 各字段均为 `Unset`，必须由配置解析或显式工厂构造
填充后才能通过 `validate()`；因此运行时不会悄悄回退到某个离散格式。

定常和非定常不再维护两套主循环或两套动量装配，而是通过
`TimeTerm` 注入时间项：

```cpp
struct TimeTerm {
    double dt;
    const Eigen::MatrixXd* u_previous;
    const Eigen::MatrixXd* v_previous;
};
```

`TimeTerm::none()` 表示无时间项；`TimeTerm::backwardEuler(dt, u0, v0)`
使动量方程增加 `rho*V/dt` 到 `A_p`，并增加
`rho*V*u0/dt`、`rho*V*v0/dt` 到源项。两个指针是非 owning 指针，只在一次迭代期间
借用调用者的矩阵，因此 `u0/v0` 必须在 `solveIteration()` 返回前保持有效。

### 6. 线性求解和 SIMPLE 控制对象

`SolutionConfig` 将线性求解器参数与外层 SIMPLE 收敛参数分开：

```text
SolutionConfig
├── velocity : LinearSolverConfig
│   └── BiCGSTAB + ILUT（示例算例显式配置）
├── pressure : LinearSolverConfig
│   └── PCG + IncompleteCholesky（示例算例显式配置）
└── simple   : SimpleControl
    ├── max_iterations
    ├── pressure_relaxation
    ├── velocity_relaxation
    └── residual { continuity, velocity_change }
```

`LinearSolverConfig` 还包含绝对/相对容差、最大线性迭代次数和显式的
`warm_start` 状态；未设置 `warm_start` 的配置不能通过 `validate()`。结果类型集中
在 `solver_result.h`，配置类型集中在 `solver_config.h`，旧的
`solution_config.h` 仅作为兼容包含入口。

### 7. `SimpleSolver` 的引用关系和一次迭代

`SolverContext` 将一次求解所需的网格、物性、离散格式、线性/SIMPLE 控制和
`ParallelContext` 集中起来；`SimpleSolver` 持有它，并在构造时创建两个都指向该网格的
`Equation`：

```text
SimpleSolver
├── context_ : SolverContext
│   ├── mesh&、fluid、schemes、solution
│   └── parallel : ParallelContext
├── momentum_ : Equation(context_.mesh)
├── pressure_ : Equation(context_.mesh)
├── source_v_ : v 方程压缩源项
├── previous_u_/previous_v_ : 预测速度压缩向量
└── （Equation 和压缩向量均复用 context_.mesh）
```

`solveIteration(const TimeTerm&)` 的数据流固定为：

```text
u_star/v_star
    │
    ├─ 动量装配（ddt、对流、扩散、梯度、松弛）
    ├─ u/v 线性求解 ──┐
    └─ Rhie–Chow 面速度 │
                       ▼
                 压力修正方程
                       │
                 p_prime 线性求解
                       │
          p、u_star、v_star 修正
                       │
                连续性和变化量检查
```

`SimpleSolver` 实现统一的 `FlowSolver` 对象接口，应用层通过
`createFlowSolver(algorithm, const SolverContext&)` 工厂按 `case.cfg` 的
`solution.algorithm` 创建算法。旧的多参数工厂重载仍保留用于兼容已有调用方。
新增压力-速度耦合算法时，只需实现 `solveIteration()` 并在工厂注册。

返回值 `SolverIterationResult` 同时携带 u、v、pressure 三个
`LinearSolverResult`，以及 `ContinuityMetrics`、相对速度变化、相对压力修正、
`healthy` 和 `converged` 标志。`converged` 只有在线性子问题没有失败、连续性
残差和速度变化都满足 `SimpleControl` 容差时才为真；这使调用者无需解析日志
来判断一轮 SIMPLE 是否真正收敛。

### 8. MPI 局部网格、压缩向量和结果写出

MPI 运行时先由 `ParallelContext::world()` 获取通信器、rank 和规模，读取一个全局
`Mesh`，再由 `extractLocalMesh()` 沿 x 方向切分：

```text
ParallelContext
    └─ extractLocalMesh(global Mesh, ghost_layers=2)
          └─ local Mesh (owned columns + Processor ghost columns)
                ├─ createInterId()
                ├─ matrixToVector()/vectorToMatrix()
                └─ exchangeColumns(field/coefficient)
```

`matrixToVector()` 按 `interi/interj` 顺序只提取 interior 单元，供 Eigen 稀疏
线性系统使用；`vectorToMatrix()` 将求解结果写回对应二维位置。每轮需要邻域
数据时，`exchangeColumns()` 使用 `ParallelContext` 中的通信器通过 `MPI_Sendrecv`
交换两层列；串行运行时为空操作。旧的 `(rank, num_procs)` 重载仍保留，兼容已有
调用代码。
因此 `Equation::A` 的行数永远是本地 `internumber`，不会包含 Processor ghost
单元或已经被边界条件消元的单元。

`saveMeshData()` 写出 `u_star、v_star、p、x_c、y_c`，并默认只写
`[owned_j_begin, owned_j_end)` 区间。后处理脚本再按 rank 拼回全局场；ghost
列从不写入结果文件，所以不同 MPI 规模的结果不能直接混合拼接。

## 依赖和编译

需要：

- MPI（OpenMPI 或 MPICH）
- Eigen 3
- 支持 C++17 的 C++ 编译器
- Python 3.8+、NumPy；生成 PNG 时还需要 Matplotlib

Ubuntu/Debian 示例：

```bash
sudo apt install libopenmpi-dev libeigen3-dev
python3 -m pip install numpy matplotlib
```

构建主程序和测试程序：

```bash
make                 # 单一可执行文件 taiho_cfd
make test            # 全部回归测试
make test-pressure   # 压力出口/压力参考测试
make test-momentum   # 稳态/非稳态动量装配测试
make test-boundary   # patch、字段边界和 MPI 拓扑测试
make debug           # ASan/UBSan 调试构建
make clean           # 删除 build/ 和本地可执行文件
make distclean       # 额外删除编译报告
```

## 网格生成

当前 C++ reader 使用以下 legacy 文本输入，但读取后立即转换为
`BoundaryPatch` 和逐字段边界条件：

```bash
python3 tools/generate_mesh.py cavity \
  --nx 64 --ny 64 --alpha-x 4 --alpha-y 4 \
  --output-dir examples/meshes/ldc_stretched

python3 tools/generate_mesh.py poiseuille \
  --nx 200 --ny 80 --lx 3 --ly 0.2 --alpha-y 4 \
  --output-dir examples/meshes/poiseuille
```

生成器会检查尺寸、有限坐标、严格单调性、正体积假设、外边界标记和 zone
引用。坐标约定为：数组第 0 行是 `y=0` 底部，最后一行是 `y=Ly` 顶部；
列从 `x=0` 左侧到 `x=Lx` 右侧。因此方腔顶盖速度使用最后一行
`zoneid[-1, :]`。

网格输入文件：

| 文件 | 尺寸 | 含义 |
|---|---:|---|
| `params.txt` | 1 行 | `nx ny` |
| `x.dat`、`y.dat` | `(ny+1)×(nx+1)` | 节点坐标 |
| `bctype.dat` | `ny×nx` | legacy 单元边界编码 |
| `zoneid.dat` | `ny×nx` | `zoneuv.txt` 行号 |
| `zoneuv.txt` | `nzone×2` | 每个 zone 的 `(u,v)` |

`bctype.dat` 编码：

| 值 | 类型 | 当前字段条件 |
|---:|---|---|
| `0` | interior | 参与方程求解 |
| `>0` | wall | `U=fixedValue/noSlip`，`p=zeroGradient` |
| `-2` | velocity inlet | `U=fixedValue`，`p=zeroGradient` |
| `-1` | pressure outlet | `U=zeroGradient`，`p=fixedValue` |
| `-3` | processor | 仅由 MPI 分解内部生成，用户网格不要设置 |

当前 solver 要求 `nx,ny≥5`，四个物理外边界不能是 interior。MPI 运行时还要
满足每个子域至少有两列真实单元，即全局 `nx` 应足够大。

## 运行

求解器不接收运行参数。进入包含 `case.cfg` 的算例目录后直接执行：

```bash
cd examples/cases/ldc_uniform
mpirun -np 4 ../../../taiho_cfd
```

稳态、瞬态、物性、时间推进、离散格式、线性求解器和 SIMPLE 控制均只从
`case.cfg` 读取。没有默认补值：缺少必填项、重复项、未知项、非法类型或不支持
的选项都会终止运行。路径相对当前算例目录解析。

`case.cfg` 使用 `key = value`，`#` 后为注释。配置按接口分组：

| 分组 | 必填内容 |
|---|---|
| `mesh.*`, `output.*` | 网格目录、结果目录 |
| `fluid.*` | `rho`、动力粘度 `mu` |
| `schemes.*` | ddt、对流、梯度、Laplacian、面插值 |
| `time.*` | Backward Euler 的 `delta_t`、`steps`；稳态不使用 |
| `solution.algorithm` | 压力-速度耦合算法，当前为 `SIMPLE` |
| `solution.velocity.*` | 速度线性求解器、预条件器、容差、迭代上限、warm start |
| `solution.pressure.*` | 压力线性求解器、预条件器、容差、迭代上限、warm start |
| `solution.simple.*` | SIMPLE 循环上限、松弛因子、连续性和速度变化收敛条件 |

`fluid.mu` 是动力粘度，`fluid.rho` 是常量密度，运动黏度为
`nu=fluid.mu/fluid.rho`。`schemes.ddt=steadyState` 表示稳态；设置为
`backwardEuler` 时还必须提供 `time.delta_t` 和 `time.steps`。完整配置示例见
`examples/cases/ldc_uniform/case.cfg` 和 `examples/cases/ldc_unsteady/case.cfg`。

解析后的 `CaseConfig::time` 是仅在 Backward Euler 下存在的显式时间控制对象，
稳态算例不会携带一个隐藏的零步长或零步数。

每个进程把结果写入 `output.path` 指定目录中的 `u/v/p/xc/yc_<rank>.dat`。
结果只包含 owned columns，不包含 ghost 列；同一次计算必须使用同一个 MPI
规模进行后处理。

## 离散和代码接口

稳态和非稳态共享 `SimpleSolver::solveIteration`；区别只由
`TimeTerm::none()` 或 `TimeTerm::backwardEuler(dt, u0, v0)` 提供。动量方程
装配位于 `src/numerics/momentum.cpp`，依次使用：

1. `addDdt`：稳态无时间项，非稳态加入 `rho*V/dt` 和旧速度源项；
2. `addConvection`：基于 `rho*U·S` 的一阶 Upwind 对流；
3. `addLaplacian`：使用动力粘度 `mu` 的 Orthogonal 两点扩散；
4. `addGradient`：Central 单元中心梯度，目前用于压力梯度；
5. `applyVelocityEquationRelaxation`：统一施加速度松弛。

`addGradient` 是通用梯度接口，避免把离散算子名称绑定到某一个物理量；
后续增加温度或其他标量场时可以复用同一命名和接口约定。

压力修正方程对有固定压力的边界把 `p_prime=0` 的 Dirichlet 贡献加入
对角项；封闭区域没有固定压力时固定一个参考压力单元。边界代码先把 legacy
编码转换为 patch，再由离散装配统一处理，不在主循环中散落判断整数编码。

## 后处理

```bash
cd examples/cases/ldc_uniform
python3 ../../../tools/postprocess.py --data-dir result --ranks 4

# 只导出拼接数据、VTK 和 Tecplot，不生成 PNG
python3 ../../../tools/postprocess.py --data-dir result --ranks 4 --no-plots
```

`--ranks` 是 MPI 进程数，不是“进程数减一”；省略时自动发现从 0 开始的
连续 rank 文件。输出到 `result/postprocess/`：

- `u/v/p/xc/yc_combined.dat`：完整全局场；
- `result.vtk`：保留非均匀 cell-center 坐标的 VTK structured grid；
- `result.plt`：Tecplot ASCII POINT 数据；
- `velocity_magnitude.png`、`pressure.png`、`streamlines.png`：可选图像。

后处理不会裁剪边界单元，也不会翻转 y 坐标。由于 Matplotlib 的 streamplot
要求等间距轴，流线图仅在显示阶段插值；数据文件和 VTK/Tecplot 使用原始坐标。

## 验证

在仓库根目录执行：

```bash
make test
```

测试覆盖：

- 压力出口 Dirichlet 对角项、闭域压力参考和压力矩阵性质；
- 稳态/非稳态动量时间项以及四个离散算子组合；
- legacy 边界转换、非零固定压力、物理边界和 MPI Processor 拓扑。

生成器和后处理也可以独立检查：

```bash
python3 tools/generate_mesh.py cavity --nx 16 --ny 12 --output-dir /tmp/taihocfd_mesh
mpirun -np 2 ./boundary_model_test /tmp/taihocfd_mesh closed
```

## Notebook

Notebook 仅作为交互式入口，实际逻辑位于 `tools/`：

- `examples/notebooks/gen.ipynb`：调用统一网格生成器；
- `examples/notebooks/plot.ipynb`：调用统一后处理器；
- `examples/notebooks/mesh_plt.ipynb`：检查网格方向和拉伸。

命令行脚本是批处理和复现实验的推荐入口。

## 源码级 API 速查

本节把当前公开头文件按依赖方向列出，便于阅读源码或从现有模块派生新功能。
实现文件中的匿名命名空间类型（例如 `LinearSolveWorkspace` 和
`DistributedOperator`）属于内部实现，不应作为外部接口依赖。

### 数据层

| 头文件 | 主要类型/函数 | 职责 |
|---|---|---|
| `mesh/mesh.h` | `Mesh`、`Mesh::initGeometry()`、`Mesh::createInterId()` | 场量、几何、拓扑和压缩编号 |
| `mesh/boundary.h` | `CellKind`、`BoundaryPatch`、`evaluate*Boundary()` | 物理边界模型和拓扑查询 |
| `numerics/fluid_properties.h` | `FluidProperties` | `rho` 和动力粘度 `mu` |
| `numerics/schemes.h` | `NumericalSchemes` | 时间、对流、梯度、扩散、面插值选择 |
| `numerics/time_term.h` | `TimeTerm` | 一次迭代使用的时间离散状态 |

### 数值层

| 头文件 | 主要函数 | 调用关系 |
|---|---|---|
| `numerics/equation.h` | `Equation::reset()`、`buildMatrix()` | 管理二维系数和压缩稀疏矩阵 |
| `numerics/operators.h` | `addDdt()`、`addConvection()`、`addLaplacian()`、`addGradient()` | 对单个速度方程逐项累加 |
| `numerics/momentum.h` | `assembleMomentum()` | 按固定顺序组合上述算子 |
| `numerics/rhie_chow.h` | `interpolateFaceVelocity()` | 中心速度、压力梯度和 mobility 的面插值 |
| `numerics/pressure_correction.h` | `assemblePressureCorrection()`、`correctPressure()`、`correctVelocity()` | 压力修正矩阵和 SIMPLE 校正 |
| `numerics/continuity.h` | `computeContinuityMetrics()` | 全局连续性 L1/L2/max/relative |

### 并行和求解器层

| 头文件 | 主要接口 | 说明 |
|---|---|---|
| `parallel/parallel_context.h` | `ParallelContext::world()` | MPI communicator、rank、size 的唯一值对象 |
| `parallel/domain_decomposition.h` | `extractLocalMesh()` | 全局网格到 x 方向局部网格 |
| `parallel/halo_exchange.h` | `exchangeColumns()`、`matrixToVector()`、`vectorToMatrix()` | 双 ghost 列和压缩向量映射 |
| `solvers/linear_solver.h` | `solveField()` | PCG/BiCGSTAB 统一入口 |
| `solvers/solver_context.h` | `SolverContext` | 一个算法实例的全部依赖 |
| `solvers/solver_config.h` | `LinearSolverConfig`、`SimpleControl`、`SolutionConfig` | 配置值和合法性检查 |
| `solvers/solver_result.h` | `LinearSolverResult`、`SolverIterationResult` | 线性和外层迭代结果 |
| `solvers/flow_solver.h` | `FlowSolver`、`createFlowSolver()` | 可插拔压力-速度算法接口 |
| `solvers/simple_solver.h` | `SimpleSolver` | 当前 SIMPLE 算法实现 |

## 一次 SIMPLE 迭代的精确状态转换

以下顺序对应 `SimpleSolver::solveIteration()` 的实际实现，任何新算法如果复用现有
数值模块都应明确它读取和写回哪些字段：

```text
输入：Mesh.u_star, Mesh.v_star, Mesh.p, TimeTerm, SolverContext

1. matrixToVector(u_star/v_star, previous_u/previous_v)
2. assembleMomentum()
   ├─ Equation::reset()
   ├─ addDdt(rho, TimeTerm)
   ├─ addConvection(rho, u_face/v_face)
   ├─ addLaplacian(mu)
   ├─ addGradient(p)
   ├─ applyVelocityEquationRelaxation()
   └─ Equation::buildMatrix()
3. Mesh.u/v <- Mesh.u_star/v_star，作为速度线性求解的初值
4. solveField(momentum, momentum.source, u)
5. solveField(momentum, source_v, v)
6. exchangeColumns(momentum.A_p)
7. interpolateFaceVelocity(mesh, momentum, schemes.face_interpolation)
8. assemblePressureCorrection(mesh, pressure, momentum, parallel)
9. solveField(pressure, pressure.source, p_prime)
10. correctPressure(mesh, pressure_relaxation)
11. correctVelocity(mesh, momentum)
12. exchangeColumns(mesh.p)
13. 全局归约速度变化、压力修正和连续性指标

输出：Mesh.u_star, Mesh.v_star, Mesh.p, SolverIterationResult
```

`momentum_.source` 专门对应 u 方程，`source_v_` 对应 v 方程；两者共享同一组
动量系数和稀疏矩阵。压力方程使用独立的 `pressure_` 和 `mesh.p_prime`。

### 收敛判定

`SolverIterationResult::healthy` 和 `converged` 具有不同语义：

```text
healthy = 三个线性结果没有 Breakdown 且报告残差/外层指标均为有限数
converged = healthy
            且 u、v、p 三个 LinearSolverResult 都是 Converged
            且 continuity.relative <= simple.residual.continuity
            且 relative_velocity_change <= simple.residual.velocity_change
```

`MaxIterations` 是正常的未收敛终止状态，不会被转换成 `Converged`。应用层应优先
读取返回值，而不是通过正则表达式解析日志文本。

## 线性系统的压缩和 MPI 语义

### 从二维场到 Eigen 向量

```cpp
Eigen::VectorXd values(mesh.internumber);
matrixToVector(mesh.u_star, values, mesh);
// values[n] 对应 (mesh.interi[n], mesh.interj[n])

vectorToMatrix(values, mesh.u, mesh);
```

`interid(i,j)` 是反向查找编号。物理边界和 Processor ghost 永远不会成为线性系统
的行；它们只参与边界源项、邻居面值或 halo 通信。

### 分布式矩阵向量乘

Eigen 的 `Equation::A` 只包含本 rank 的 interior 行。`DistributedOperator` 先
执行本地稀疏乘法，再将输入向量放回二维临时矩阵并调用 `exchangeColumns()`，最后
补上跨 rank 东/西邻居对本地行的贡献。因此 PCG/BiCGSTAB 的点积、范数和收敛目标
必须通过 `ParallelContext::communicator` 归约，不能改回隐式的 `MPI_COMM_WORLD`。

### owned/ghost 写出规则

局部 `Mesh` 的 ghost 列只用于计算，不属于结果所有权。默认的
`saveMeshData(..., owned_only=true)` 写出：

```text
for (j = owned_j_begin; j < owned_j_end; ++j)
    write(field(:, j))
```

后处理器直接按 rank 顺序拼接列，不做 ghost 裁剪。若调用方确实需要调试完整局部
矩阵，可传 `owned_only=false`，但这种输出不能送入常规全局拼接流程。

## `case.cfg` 到 C++ 对象的映射

解析器 `readCaseConfig()` 先消费所有键，再执行 `CaseConfig::validate()`。映射关系
如下：

| 配置键 | C++ 成员 | 值域 |
|---|---|---|
| `mesh.path` | `CaseConfig::mesh_path` | 相对 `case.cfg` 的目录或绝对路径 |
| `output.path` | `CaseConfig::output_path` | 结果目录 |
| `fluid.rho` | `CaseConfig::fluid.rho` | 正有限数 |
| `fluid.mu` | `CaseConfig::fluid.mu` | 正有限动力粘度 |
| `schemes.ddt` | `schemes.time` | `steadyState`/`backwardEuler` |
| `schemes.div_u` | `schemes.velocity_convection` | 当前为 `upwind` |
| `schemes.grad_p` | `schemes.pressure_gradient` | 当前为 `central` |
| `schemes.laplacian_u` | `schemes.velocity_laplacian` | 当前为 `orthogonal` |
| `schemes.interpolation` | `schemes.face_interpolation` | 当前为 `linear` |
| `time.delta_t` | `CaseConfig::time->delta_t` | 仅瞬态，正有限数 |
| `time.steps` | `CaseConfig::time->steps` | 仅瞬态，正整数 |
| `solution.algorithm` | `CaseConfig::algorithm` | 当前为 `SIMPLE` |
| `solution.velocity.*` | `solution.velocity` | BiCGSTAB/ILUT 及其控制量 |
| `solution.pressure.*` | `solution.pressure` | PCG/Incomplete Cholesky 及其控制量 |
| `solution.simple.*` | `solution.simple` | SIMPLE 上限、松弛和外层容差 |

`warm_start` 使用 `std::optional<bool>` 是有意设计：结构体默认构造后它为空，
`validate()` 会拒绝空值；只有 parser 或调用方明确写入 `true/false` 才能运行。
同理，`NumericalSchemes` 和求解器枚举以 `Unset` 起步，防止增加新功能时出现
静默默认行为。

## 手工构造和单元测试时的推荐模式

不通过 `case.cfg` 使用库接口时，也应遵循相同的显式初始化顺序：

```cpp
FluidProperties fluid{1.0, 0.01};
NumericalSchemes schemes = NumericalSchemes::steady();

LinearSolverConfig velocity{
    LinearSolverType::BiCGSTAB,
    PreconditionerType::ILUT,
    1e-14, 1e-7, 200, true};
LinearSolverConfig pressure{
    LinearSolverType::PCG,
    PreconditionerType::IncompleteCholesky,
    1e-14, 1e-7, 1000, false};
SolutionConfig solution{
    velocity, pressure,
    SimpleControl{1000, 0.3, 0.5, {1e-7, 1e-6}}};

fluid.validate();
schemes.validate();
solution.validate();
```

手工构造 `LinearSolverConfig` 时，最后一个初始化值必须是 `bool`，会自动构造成
`std::optional<bool>`；更推荐使用成员赋值，以便在代码审查时看出每个设置均已提供。
测试应优先断言系数、源项、稀疏矩阵性质和结果状态，只有在验证应用生命周期时才
启动完整 MPI 算例。

## 可扩展性约定

### 增加新的压力-速度算法

新增类只需依赖 `SolverContext`，不应重新保存 rank、size、rho、粘度或配置副本：

```cpp
class MySolver final : public FlowSolver {
public:
    explicit MySolver(const SolverContext& context);
    SolverIterationResult solveIteration(const TimeTerm& time) override;
};
```

然后在 `createFlowSolver()` 中注册字符串。应用层的 `case.cfg` 读取、MPI 初始化、
结果写出和后处理均不需要改变。

### 增加新的离散格式

完整路径是“枚举 → `toString` → `validate` → case.cfg choice → 数值实现 → 回归”。
不应只在 `operators.cpp` 增加分支而不更新 parser，否则配置无法显式选择该格式。

### 增加新的场量

推荐沿现有模式增加：

1. 在 `Mesh` 中放置二维场和必要的 face/历史状态；
2. 用 `Equation` 保存系数，使用 `interid` 建立压缩系统；
3. 在 `operators` 中实现可复用的项；
4. 通过 `SolverContext` 或专用物性对象传递设置；
5. 复用 `matrixToVector`、`vectorToMatrix`、`exchangeColumns` 和结果状态；
6. 在 `make test` 之外增加该场量的数值和 MPI 范数回归。

## 验收命令和结果解释

提交前建议执行：

```bash
git diff --check
make clean
make -j2 all
make -j2 test

cd examples/cases/ldc_uniform
mpirun -np 2 ../../../taiho_cfd
python3 ../../../tools/postprocess.py --data-dir result --ranks 2 --no-plots

cd ../ldc_unsteady
mpirun -np 2 ../../../taiho_cfd
python3 ../../../tools/postprocess.py --data-dir result --ranks 2 --no-plots
```

检查点应包括：

- 构建和 `make test` 返回码为 0；
- 稳态/瞬态日志没有 `solver breakdown`；
- 后处理显示预期的 `(ny, nx)`，且所有合并数组有限；
- 重复运行的范数或逐点差在允许的浮点误差内；
- 稳态若达到 `solution.simple.max_iterations`，必须标记为 iteration-limit，
  不能写成“已收敛”；
- MPI 的 `CANNOT CREATE FIFO ... errno 95` 若仅为 OpenMPI 调试 FIFO 警告且作业
  返回码为 0，不应覆盖实际数值测试结论。
