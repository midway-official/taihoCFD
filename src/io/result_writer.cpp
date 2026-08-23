#include "io/result_writer.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace fs = std::filesystem;

void saveMeshData(
    const Mesh& mesh,
    int rank,
    const std::string& output_folder,
    bool owned_only)
{
    const fs::path directory(output_folder);
    fs::create_directories(directory);

    const int begin = owned_only ? mesh.owned_j_begin : 0;
    const int columns = owned_only ? mesh.ownedColumns() : mesh.nx;

    const auto write = [&](const std::string& name, const Eigen::MatrixXd& field) {
        const fs::path path =
            directory / (name + "_" + std::to_string(rank) + ".dat");
        std::ofstream output(path);
        if (!output) {
            throw std::runtime_error("无法创建结果文件: " + path.string());
        }
        output << std::setprecision(17)
               << field.block(0, begin, field.rows(), columns);
    };

    write("u", mesh.u_star);
    write("v", mesh.v_star);
    write("p", mesh.p);
    write("xc", mesh.x_c);
    write("yc", mesh.y_c);
}
