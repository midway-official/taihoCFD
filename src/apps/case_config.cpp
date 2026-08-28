#include "apps/case_config.h"

#include <cmath>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace {

std::string trim(std::string value) {
    constexpr std::string_view whitespace = " \t\r\n";
    const auto begin = value.find_first_not_of(whitespace);
    if (begin == std::string::npos) {
        return {};
    }
    return value.substr(begin, value.find_last_not_of(whitespace) - begin + 1);
}

class Dictionary {
public:
    explicit Dictionary(const std::filesystem::path& path) {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("无法打开算例参数文件: " + path.string());
        }
        std::string line;
        int line_number = 0;
        while (std::getline(input, line)) {
            ++line_number;
            line = trim(line.substr(0, line.find('#')));
            if (line.empty()) {
                continue;
            }
            const auto separator = line.find('=');
            if (separator == std::string::npos) {
                throw std::runtime_error(
                    "case.cfg 第 " + std::to_string(line_number) + " 行缺少 '='");
            }
            const std::string key = trim(line.substr(0, separator));
            const std::string value = trim(line.substr(separator + 1));
            if (key.empty() || value.empty() || !entries_.emplace(key, value).second) {
                throw std::runtime_error(
                    "case.cfg 第 " + std::to_string(line_number) + " 行为空或键重复");
            }
        }
    }

    std::string take(const std::string& key) {
        const auto position = entries_.find(key);
        if (position == entries_.end()) {
            throw std::runtime_error("case.cfg 缺少必填项: " + key);
        }
        std::string value = std::move(position->second);
        entries_.erase(position);
        return value;
    }

    template <typename T>
    T scalar(const std::string& key) {
        const std::string text = take(key);
        std::istringstream stream(text);
        T value{};
        if (!(stream >> value) || !(stream >> std::ws).eof()) {
            throw std::runtime_error("case.cfg 参数格式错误: " + key + '=' + text);
        }
        return value;
    }

    bool boolean(const std::string& key) {
        const std::string value = take(key);
        if (value == "true" || value == "false") {
            return value == "true";
        }
        throw std::runtime_error("case.cfg 布尔值必须是 true/false: " + key);
    }

    template <typename T>
    T choice(
        const std::string& key,
        std::initializer_list<std::pair<std::string_view, T>> choices)
    {
        const std::string value = take(key);
        for (const auto& [name, choice_value] : choices) {
            if (value == name) {
                return choice_value;
            }
        }
        throw std::runtime_error("case.cfg 不支持的选项: " + key + '=' + value);
    }

    void requireEmpty() const {
        if (!entries_.empty()) {
            throw std::runtime_error("case.cfg 包含未知项: " + entries_.begin()->first);
        }
    }

private:
    std::unordered_map<std::string, std::string> entries_;
};

std::filesystem::path resolvePath(
    const std::filesystem::path& base,
    const std::string& value)
{
    const std::filesystem::path path(value);
    return (path.is_absolute() ? path : base / path).lexically_normal();
}

LinearSolverConfig readLinearSolver(Dictionary& dictionary, const char* prefix) {
    const std::string key(prefix);
    LinearSolverConfig config;
    config.solver = dictionary.choice<LinearSolverType>(key + ".solver", {
        {"BiCGSTAB", LinearSolverType::BiCGSTAB},
        {"PCG", LinearSolverType::PCG},
    });
    config.preconditioner = dictionary.choice<PreconditionerType>(
        key + ".preconditioner", {
            {"ILUT", PreconditionerType::ILUT},
            {"incompleteCholesky", PreconditionerType::IncompleteCholesky},
        });
    config.absolute_tolerance = dictionary.scalar<double>(key + ".absolute_tolerance");
    config.relative_tolerance = dictionary.scalar<double>(key + ".relative_tolerance");
    config.max_iterations = dictionary.scalar<int>(key + ".max_iterations");
    config.warm_start = dictionary.boolean(key + ".warm_start");
    return config;
}

}  // namespace

CaseConfig readCaseConfig(const std::filesystem::path& path) {
    Dictionary dictionary(path);
    CaseConfig config;
    const auto base = path.parent_path();
    config.mesh_path = resolvePath(base, dictionary.take("mesh.path"));
    config.output_path = resolvePath(base, dictionary.take("output.path"));
    config.fluid.rho = dictionary.scalar<double>("fluid.rho");
    config.fluid.mu = dictionary.scalar<double>("fluid.mu");
    config.schemes.time = dictionary.choice<TimeScheme>("schemes.ddt", {
        {"steadyState", TimeScheme::SteadyState},
        {"backwardEuler", TimeScheme::BackwardEuler},
    });
    config.schemes.velocity_convection =
        dictionary.choice<ConvectionScheme>("schemes.div_u", {
            {"upwind", ConvectionScheme::Upwind},
        });
    config.schemes.pressure_gradient =
        dictionary.choice<GradientScheme>("schemes.grad_p", {
            {"central", GradientScheme::Central},
        });
    config.schemes.velocity_laplacian =
        dictionary.choice<LaplacianScheme>("schemes.laplacian_u", {
            {"orthogonal", LaplacianScheme::Orthogonal},
        });
    config.schemes.face_interpolation =
        dictionary.choice<InterpolationScheme>("schemes.interpolation", {
            {"linear", InterpolationScheme::Linear},
    });
    if (config.transient()) {
        config.time = TimeControl{
            dictionary.scalar<double>("time.delta_t"),
            dictionary.scalar<int>("time.steps"),
        };
    }
    config.solution.velocity = readLinearSolver(dictionary, "solution.velocity");
    config.solution.pressure = readLinearSolver(dictionary, "solution.pressure");
    config.algorithm = dictionary.take("solution.algorithm");
    config.solution.simple.max_iterations =
        dictionary.scalar<int>("solution.simple.max_iterations");
    config.solution.simple.pressure_relaxation =
        dictionary.scalar<double>("solution.simple.pressure_relaxation");
    config.solution.simple.velocity_relaxation =
        dictionary.scalar<double>("solution.simple.velocity_relaxation");
    config.solution.simple.residual.continuity =
        dictionary.scalar<double>("solution.simple.continuity_tolerance");
    config.solution.simple.residual.velocity_change =
        dictionary.scalar<double>("solution.simple.velocity_change_tolerance");
    dictionary.requireEmpty();

    config.validate();
    return config;
}

void CaseConfig::validate() const {
    if (mesh_path.empty() || output_path.empty() || algorithm.empty()) {
        throw std::invalid_argument(
            "算例必须提供 mesh.path、output.path 和 solution.algorithm");
    }
    fluid.validate();
    schemes.validate();
    solution.validate();
    if (transient() != time.has_value()) {
        throw std::invalid_argument(
            "time.delta_t/time.steps 只允许在 backwardEuler 下提供");
    }
    if (transient() &&
        (!(time->delta_t > 0.0) || !std::isfinite(time->delta_t) ||
         time->steps <= 0)) {
        throw std::invalid_argument("time.delta_t 和 time.steps 必须为正数");
    }
}
