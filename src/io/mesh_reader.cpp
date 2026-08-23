#include "io/mesh_reader.h"

#include "mesh/boundary.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

template <typename Matrix>
void readMatrixFile(const fs::path& path, Matrix& matrix) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("无法打开文件: " + path.string());
    }
    for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
        for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
            if (!(input >> matrix(i, j))) {
                std::ostringstream message;
                message << "文件数据不足或格式错误: " << path
                        << " at (" << i << ", " << j << ')';
                throw std::runtime_error(message.str());
            }
        }
    }
}

}  // namespace

Mesh readMesh(const std::string& folder_path) {
    const fs::path folder(folder_path);
    if (!fs::is_directory(folder)) {
        throw std::runtime_error("网格文件夹不存在: " + folder_path);
    }

    std::ifstream params(folder / "params.txt");
    int nx = 0;
    int ny = 0;
    if (!(params >> nx >> ny)) {
        throw std::runtime_error("params.txt 必须包含 nx ny");
    }
    if (nx < 5 || ny < 5) {
        throw std::runtime_error("网格每个方向至少需要 5 个单元");
    }

    Mesh mesh(ny, nx);
    readMatrixFile(folder / "bctype.dat", mesh.bctype);
    readMatrixFile(folder / "zoneid.dat", mesh.zoneid);
    readMatrixFile(folder / "x.dat", mesh.x);
    readMatrixFile(folder / "y.dat", mesh.y);

    std::ifstream zone_file(folder / "zoneuv.txt");
    if (!zone_file) {
        throw std::runtime_error("无法打开 zoneuv.txt");
    }
    double u_value = 0.0;
    double v_value = 0.0;
    while (zone_file >> u_value >> v_value) {
        mesh.zoneu.push_back(u_value);
        mesh.zonev.push_back(v_value);
    }
    if (mesh.zoneu.empty() || mesh.zoneu.size() != mesh.zonev.size()) {
        throw std::runtime_error("zoneuv.txt 为空或格式错误");
    }

    mesh.createInterId();
    initializeBoundaryConditions(mesh);
    mesh.initGeometry();
    mesh.validate(true);
    return mesh;
}
