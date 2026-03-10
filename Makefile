# ==========================================================
# 编译器与选项
# ==========================================================

MPICXX   := mpic++
CXX_STD  := -std=c++17

# 基础优化
OPT_FLAGS := -O3 -march=native -mtune=native -funroll-loops -ffast-math \
             -fomit-frame-pointer -fprefetch-loop-arrays

# 警告
WARN_FLAGS := -Wall -Wextra -Wpedantic -Wshadow \
              -Wno-unused-parameter

# 完整编译标志
CXXFLAGS := $(CXX_STD) $(OPT_FLAGS) $(WARN_FLAGS)
INCLUDES := -Isrc

# ==========================================================
# 目录
# ==========================================================

SRC_DIR    := src
BUILD_DIR  := build
REPORT_DIR := report

# ==========================================================
# 目标程序
# ==========================================================

SOLVERS := solver_simple_steady solver_simple_unsteady
TARGETS := $(SOLVERS)

# ==========================================================
# 源文件
# ==========================================================

COMMON_SRCS  := $(SRC_DIR)/fluid.cpp \
                $(SRC_DIR)/parallel.cpp

STEADY_SRC   := $(SRC_DIR)/solver_simple_steady.cpp
UNSTEADY_SRC := $(SRC_DIR)/solver_simple_unsteady.cpp

ALL_SRCS := $(COMMON_SRCS) $(STEADY_SRC) $(UNSTEADY_SRC)

# ==========================================================
# 目标文件映射到 build 目录
# ==========================================================

COMMON_OBJS  := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(COMMON_SRCS))
STEADY_OBJ   := $(BUILD_DIR)/solver_simple_steady.o
UNSTEADY_OBJ := $(BUILD_DIR)/solver_simple_unsteady.o

ALL_OBJS := $(COMMON_OBJS) $(STEADY_OBJ) $(UNSTEADY_OBJ)

# 依赖文件
DEPS := $(ALL_OBJS:.o=.d)

# ==========================================================
# 颜色输出
# ==========================================================

COLOR_RESET  := \033[0m
COLOR_GREEN  := \033[32m
COLOR_YELLOW := \033[33m
COLOR_CYAN   := \033[36m
COLOR_BOLD   := \033[1m

LOG  = @printf "$(COLOR_CYAN)$(COLOR_BOLD)[%-8s]$(COLOR_RESET) %s\n"
LOGC = @printf "$(COLOR_CYAN)$(COLOR_BOLD)[%-8s]$(COLOR_RESET) $(COLOR_GREEN)%s$(COLOR_RESET)\n"

# ==========================================================
# 默认目标
# ==========================================================

.DEFAULT_GOAL := all

all: $(REPORT_DIR) $(TARGETS)
	$(LOG) "DONE" "所有目标构建完成"

# ==========================================================
# 构建规则
# ==========================================================

solver_simple_steady: $(COMMON_OBJS) $(STEADY_OBJ)
	$(LOG) "LINK" "$@"
	@$(MPICXX) $(CXXFLAGS) $^ -o $@
	$(LOGC) "OK" "solver_simple_steady 链接成功"

solver_simple_unsteady: $(COMMON_OBJS) $(UNSTEADY_OBJ)
	$(LOG) "LINK" "$@"
	@$(MPICXX) $(CXXFLAGS) $^ -o $@
	$(LOGC) "OK" "solver_simple_unsteady 链接成功"

# ==========================================================
# 创建目录
# ==========================================================

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(REPORT_DIR):
	@mkdir -p $(REPORT_DIR)

# ==========================================================
# 编译规则（带自动依赖 + 向量化报告）
# ==========================================================

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR) $(REPORT_DIR)
	$(LOG) "CXX" "$<"
	@$(MPICXX) $(CXXFLAGS) $(INCLUDES) \
	    -MMD -MP \
	    -fopt-info-vec-optimized-missed=$(REPORT_DIR)/$(*F)_vec.log \
	    -c $< -o $@

# ==========================================================
# 编译报告目标
# ==========================================================

## report-vec：汇总向量化报告
report-vec: all
	$(LOG) "REPORT" "$(COLOR_YELLOW)向量化报告$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_BOLD)=== 向量化报告（优化成功 / 未能优化）===$(COLOR_RESET)"
	@cat $(REPORT_DIR)/*_vec.log 2>/dev/null || echo "  (无日志，请先 make all)"
	@echo ""

## report-asm：生成所有源文件的汇编输出
report-asm: $(REPORT_DIR)
	$(LOG) "ASM" "生成汇编文件到 $(REPORT_DIR)/"
	@for src in $(ALL_SRCS); do \
	    base=$$(basename $$src .cpp); \
	    $(MPICXX) $(CXXFLAGS) $(INCLUDES) \
	        -S -masm=intel \
	        -fverbose-asm \
	        $$src -o $(REPORT_DIR)/$$base.asm; \
	    printf "  生成: $(REPORT_DIR)/$$base.asm\n"; \
	done

## report-pp：生成预处理后的源文件（查看宏展开）
report-pp: $(REPORT_DIR)
	$(LOG) "PP" "生成预处理文件到 $(REPORT_DIR)/"
	@for src in $(ALL_SRCS); do \
	    base=$$(basename $$src .cpp); \
	    $(MPICXX) $(CXXFLAGS) $(INCLUDES) \
	        -E $$src -o $(REPORT_DIR)/$$base.ii; \
	    printf "  生成: $(REPORT_DIR)/$$base.ii\n"; \
	done

## report-simd：检查二进制中 SIMD 指令使用情况（需先 make all）
report-simd: all
	$(LOG) "SIMD" "$(COLOR_YELLOW)SIMD 指令统计$(COLOR_RESET)"
	@for target in $(TARGETS); do \
	    echo ""; \
	    echo "$(COLOR_BOLD)--- $$target ---$(COLOR_RESET)"; \
	    echo "  AVX-512 (zmm): $$(objdump -d $$target 2>/dev/null | grep -c 'zmm' || echo 0) 条指令"; \
	    echo "  AVX2    (ymm): $$(objdump -d $$target 2>/dev/null | grep -c 'ymm' || echo 0) 条指令"; \
	    echo "  SSE     (xmm): $$(objdump -d $$target 2>/dev/null | grep -c 'xmm' || echo 0) 条指令"; \
	done
	@echo ""

## report-size：查看二进制大小分布
report-size: all
	$(LOG) "SIZE" "$(COLOR_YELLOW)二进制段大小$(COLOR_RESET)"
	@for target in $(TARGETS); do \
	    echo ""; \
	    echo "$(COLOR_BOLD)--- $$target ---$(COLOR_RESET)"; \
	    size $$target 2>/dev/null || echo "  (size 工具不可用)"; \
	done
	@echo ""

## report-flags：显示实际编译标志
report-flags:
	$(LOG) "FLAGS" "$(COLOR_YELLOW)当前编译配置$(COLOR_RESET)"
	@echo ""
	@echo "  编译器  : $(MPICXX)"
	@echo "  标准    : $(CXX_STD)"
	@echo "  优化    : $(OPT_FLAGS)"
	@echo "  警告    : $(WARN_FLAGS)"
	@echo "  Include : $(INCLUDES)"
	@echo ""
	@echo "$(COLOR_BOLD)  完整 CXXFLAGS:$(COLOR_RESET)"
	@echo "  $(CXXFLAGS)"
	@echo ""
	@echo "$(COLOR_BOLD)  MPI 实际调用:$(COLOR_RESET)"
	@$(MPICXX) --showme 2>/dev/null || echo "  (--showme 不可用)"
	@echo ""

## report-all：生成所有报告
report-all: report-flags report-vec report-simd report-size
	$(LOG) "DONE" "$(COLOR_GREEN)所有报告已生成，详细日志在 $(REPORT_DIR)/$(COLOR_RESET)"

# ==========================================================
# PGO（Profile-Guided Optimization）
# ==========================================================

pgo-generate: CXXFLAGS += -fprofile-generate=$(BUILD_DIR)/pgo
pgo-generate: clean all
	$(LOG) "PGO" "插桩编译完成，请运行程序以收集 profile"

pgo-use: CXXFLAGS += -fprofile-use=$(BUILD_DIR)/pgo -fprofile-correction
pgo-use: all
	$(LOG) "PGO" "$(COLOR_GREEN)PGO 优化编译完成$(COLOR_RESET)"

# ==========================================================
# Debug 构建
# ==========================================================

debug: CXXFLAGS := $(CXX_STD) -O0 -g3 -Wall -Wextra -DDEBUG \
                   -fsanitize=address,undefined
debug: clean all
	$(LOG) "DEBUG" "Debug 构建完成（含 ASan + UBSan）"

# ==========================================================
# 清理
# ==========================================================

clean:
	$(LOG) "CLEAN" "清理构建产物"
	@rm -rf $(BUILD_DIR) $(TARGETS)

clean-report:
	$(LOG) "CLEAN" "清理报告"
	@rm -rf $(REPORT_DIR)

distclean: clean clean-report
	$(LOG) "CLEAN" "完全清理"

# ==========================================================
# 帮助
# ==========================================================

help:
	@echo ""
	@echo "$(COLOR_BOLD)用法:$(COLOR_RESET) make [目标]"
	@echo ""
	@echo "$(COLOR_BOLD)构建目标:$(COLOR_RESET)"
	@echo "  all              默认构建所有程序"
	@echo "  debug            Debug 构建（ASan + UBSan）"
	@echo "  pgo-generate     PGO 第一步：插桩编译"
	@echo "  pgo-use          PGO 第二步：优化编译"
	@echo ""
	@echo "$(COLOR_BOLD)报告目标:$(COLOR_RESET)"
	@echo "  report-flags     显示编译标志"
	@echo "  report-vec       向量化优化报告"
	@echo "  report-asm       生成汇编文件"
	@echo "  report-pp        生成预处理文件"
	@echo "  report-simd      SIMD 指令统计"
	@echo "  report-size      二进制段大小"
	@echo "  report-all       生成所有报告"
	@echo ""
	@echo "$(COLOR_BOLD)清理目标:$(COLOR_RESET)"
	@echo "  clean            清理构建产物"
	@echo "  clean-report     清理报告文件"
	@echo "  distclean        完全清理"
	@echo ""

# ==========================================================
# 伪目标声明
# ==========================================================

.PHONY: all clean clean-report distclean debug help \
        report-vec report-asm report-pp report-simd \
        report-size report-flags report-all \
        pgo-generate pgo-use

# ==========================================================
# 自动依赖
# ==========================================================

-include $(DEPS)