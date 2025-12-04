# ========= 基本设置 =========
MPICXX      = mpic++
CXXFLAGS    = -std=c++17 -O2 -Wall -Wextra
INCLUDES    = -I/usr/lib/x86_64-linux-gnu/mpich/include

# ========= 目标 =========
TARGETS     = SOLVER_SIMPLE t2

# ========= 通用依赖（所有程序都需要这些） =========
COMMON_SRCS = DNS.cpp parallel.cpp
COMMON_OBJS = $(COMMON_SRCS:.cpp=.o)
COMMON_DEPS = $(COMMON_SRCS:.cpp=.d)

# ========= 各自独立源文件 =========
SOLVER_SRCS = solver_simple.cpp
SOLVER_OBJS = $(SOLVER_SRCS:.cpp=.o)
SOLVER_DEPS = $(SOLVER_SRCS:.cpp=.d)

T2_SRCS     = t2.cpp
T2_OBJS     = $(T2_SRCS:.cpp=.o)
T2_DEPS     = $(T2_SRCS:.cpp=.d)

# ========= 默认目标 =========
all: $(TARGETS)

# ========= 构建 SOLVER_SIMPLE =========
SOLVER_SIMPLE: $(COMMON_OBJS) $(SOLVER_OBJS)
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

# ========= 构建 t2 =========
t2: $(COMMON_OBJS) $(T2_OBJS)
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

# ========= 通用编译（自动依赖） =========
%.o: %.cpp
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

# ========= 清理 =========
clean:
	rm -f *.o *.d $(TARGETS)

.PHONY: all clean

# ========= 自动依赖处理 =========
-include $(COMMON_DEPS) $(SOLVER_DEPS) $(T2_DEPS)
