# ========= 基本设置 =========
MPICXX      = mpic++
CXXFLAGS    = -std=c++17 -O3 -Wall -Wextra
INCLUDES    = -I/usr/lib/x86_64-linux-gnu/mpich/include

# ========= 目标 =========
TARGETS     = mesh_generation solver_PISO solver_simple_steady solver_simple_unsteady

# ========= 通用依赖 =========
COMMON_SRCS = fluid.cpp parallel.cpp
COMMON_OBJS = $(COMMON_SRCS:.cpp=.o)
COMMON_DEPS = $(COMMON_SRCS:.cpp=.d)

# ========= 各自独立源文件 =========
MESH_GEN_SRCS       = mesh_generation.cpp
MESH_GEN_OBJS       = $(MESH_GEN_SRCS:.cpp=.o)
MESH_GEN_DEPS       = $(MESH_GEN_SRCS:.cpp=.d)

PISO_SRCS           = solver_PISO.cpp
PISO_OBJS           = $(PISO_SRCS:.cpp=.o)
PISO_DEPS           = $(PISO_SRCS:.cpp=.d)

SIMPLE_STEADY_SRCS  = solver_simple_steady.cpp
SIMPLE_STEADY_OBJS  = $(SIMPLE_STEADY_SRCS:.cpp=.o)
SIMPLE_STEADY_DEPS  = $(SIMPLE_STEADY_SRCS:.cpp=.d)

SIMPLE_UNSTEADY_SRCS = solver_simple_unsteady.cpp
SIMPLE_UNSTEADY_OBJS = $(SIMPLE_UNSTEADY_SRCS:.cpp=.o)
SIMPLE_UNSTEADY_DEPS = $(SIMPLE_UNSTEADY_SRCS:.cpp=.d)

# ========= 默认目标 =========
all: $(TARGETS)

# ========= 构建 mesh_generation =========
mesh_generation: $(COMMON_OBJS) $(MESH_GEN_OBJS)
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

# ========= 构建 solver_PISO =========
solver_PISO: $(COMMON_OBJS) $(PISO_OBJS)
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

# ========= 构建 solver_simple_steady =========
solver_simple_steady: $(COMMON_OBJS) $(SIMPLE_STEADY_OBJS)
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

# ========= 构建 solver_simple_unsteady =========
solver_simple_unsteady: $(COMMON_OBJS) $(SIMPLE_UNSTEADY_OBJS)
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

# ========= 通用编译（自动依赖） =========
%.o: %.cpp
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

# ========= 清理 =========
clean:
	rm -f *.o *.d $(TARGETS)

.PHONY: all clean

# ========= 自动依赖处理 =========
-include $(COMMON_DEPS) $(MESH_GEN_DEPS) $(PISO_DEPS) $(SIMPLE_STEADY_DEPS) $(SIMPLE_UNSTEADY_DEPS)