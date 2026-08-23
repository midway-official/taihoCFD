# Taiho-CFD

Taiho-CFD 是一个基于 C++17、Eigen 和 MPI 的二维不可压缩 Navier–Stokes
有限体积求解器。当前实现采用 SIMPLE 压力-速度耦合，提供定常和一阶
Backward Euler 非定常计算，并针对结构化正交网格进行验证。

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
│   ├── apps/                 # 定常/非定常入口和共用运行循环
│   ├── io/                   # 网格读取、MPI 结果写出
│   ├── mesh/                 # 网格几何、CellKind、BoundaryPatch
│   ├── numerics/             # 离散算子、动量、压力修正、Rhie–Chow
│   ├── parallel/             # x 方向域分解和 halo 交换
│   └── solvers/              # 线性求解器和 SIMPLE 求解器
├── tests/                    # 压力、动量和边界模型回归测试
├── tools/
│   ├── generate_mesh.py      # 网格生成和输入校验
│   └── postprocess.py        # MPI 结果拼接、VTK/Tecplot/PNG 输出
├── examples/
│   ├── meshes/               # 可直接读取的示例网格
│   └── notebooks/            # 网格、后处理和网格检查 notebook
├── docs/thesis/              # 论文目录和后续规划
├── Makefile
└── LICENSE
```

编译生成的 `build/`、`report/`、求解器可执行文件和运行结果不属于源代码，
由 `.gitignore` 排除；运行结果建议放在单独的 case 目录中。

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
make                 # solver_simple_steady、solver_simple_unsteady
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

定常程序参数为：

```text
solver_simple_steady <mesh_folder> <max_simple_iterations> <viscosity>
```

非定常程序参数为：

```text
solver_simple_unsteady <mesh_folder> <dt> <time_steps> <viscosity>
```

推荐把结果写到独立目录，避免不同 MPI 规模的 rank 文件混在一起：

```bash
mkdir -p runs/ldc_re100
cd runs/ldc_re100

mpirun -np 4 ../../solver_simple_steady \
  ../../examples/meshes/ldc_stretched 1000 0.01

# dt=0.01，推进 200 个时间步
mpirun -np 4 ../../solver_simple_unsteady \
  ../../examples/meshes/ldc_stretched 0.01 200 0.01
```

程序从 MPI rank 0 读取参数并广播到所有进程。每个进程将结果写到当前工作
目录的 `result/`：`u_<rank>.dat`、`v_<rank>.dat`、`p_<rank>.dat`、
`xc_<rank>.dat`、`yc_<rank>.dat`。结果文件只包含 owned columns，不包含 ghost
列；同一次计算必须使用同一个 MPI 规模进行后处理。

主要默认控制参数在 `src/apps/application.cpp` 中集中设置：

| 参数 | 定常 | 非定常 |
|---|---:|---:|
| pressure relaxation | 0.3 | 0.3 |
| velocity relaxation | 0.5 | 0.7 |
| velocity relative tolerance | 1e-7 | 1e-7 |
| pressure relative tolerance | 1e-7 | 1e-7 |
| SIMPLE continuity tolerance | 1e-7 | 1e-7 |
| velocity-change tolerance | 1e-6 | 1e-4 |
| velocity max linear iterations | 200 | 200 |
| pressure max linear iterations | 1000 | 1000 |

## 离散和代码接口

稳态和非稳态共享 `SimpleSolver::solveIteration`；区别只由
`TimeTerm::none()` 或 `TimeTerm::backwardEuler(dt, u0, v0)` 提供。动量方程
装配位于 `src/numerics/momentum.cpp`，依次使用：

1. `addDdt`：稳态无时间项，非稳态加入 `V/dt` 和旧速度源项；
2. `addConvection`：一阶 Upwind 对流；
3. `addLaplacian`：Orthogonal 两点扩散；
4. `addGradient`：Central 单元中心梯度，目前用于压力梯度；
5. `applyVelocityEquationRelaxation`：统一施加速度松弛。

`addGradient` 是通用梯度接口，避免把离散算子名称绑定到某一个物理量；
后续增加温度或其他标量场时可以复用同一命名和接口约定。

压力修正方程对有固定压力的边界直接施加 Dirichlet 行，并把边界贡献放入
对角项；封闭区域没有固定压力时固定一个参考压力单元。边界代码先把 legacy
编码转换为 patch，再由离散装配统一处理，不在主循环中散落判断整数编码。

## 后处理

```bash
cd runs/ldc_re100
python3 ../../tools/postprocess.py --data-dir result --ranks 4

# 只导出拼接数据、VTK 和 Tecplot，不生成 PNG
python3 ../../tools/postprocess.py --data-dir result --ranks 4 --no-plots
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
