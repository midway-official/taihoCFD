#pragma once

#include <eigen3/Eigen/Core>

#include <string>
#include <vector>

struct Mesh;

enum class CellKind : int {
    Interior,
    PhysicalBoundary,
    Processor,
};

enum class PatchKind {
    Generic,
    Wall,
};

enum class BoundaryConditionType {
    FixedValue,
    ZeroGradient,
    InletOutlet,
    NoSlip,
};

struct CellIndex {
    int i = 0;
    int j = 0;
};

struct VectorBoundaryCondition {
    BoundaryConditionType type = BoundaryConditionType::ZeroGradient;
    Eigen::Vector2d value = Eigen::Vector2d::Zero();
    Eigen::Vector2d inlet_value = Eigen::Vector2d::Zero();
};

struct ScalarBoundaryCondition {
    BoundaryConditionType type = BoundaryConditionType::ZeroGradient;
    double value = 0.0;
    double inlet_value = 0.0;
};

struct BoundaryPatch {
    std::string name;
    PatchKind kind = PatchKind::Generic;
    VectorBoundaryCondition velocity;
    ScalarBoundaryCondition pressure;
    std::vector<CellIndex> cells;
};

bool isInteriorCell(const Mesh& mesh, int i, int j);
bool isPhysicalBoundaryCell(const Mesh& mesh, int i, int j);
bool isProcessorCell(const Mesh& mesh, int i, int j);
bool isCoupledCell(const Mesh& mesh, int i, int j);
bool isFixedPressureBoundaryCell(const Mesh& mesh, int i, int j);

const BoundaryPatch& boundaryPatch(const Mesh& mesh, int i, int j);
BoundaryPatch& boundaryPatch(Mesh& mesh, int i, int j);

bool fixesValue(BoundaryConditionType type);
bool hasFixedPressure(const BoundaryPatch& patch);
Eigen::Vector2d evaluateVelocityBoundary(
    const BoundaryPatch& patch,
    const Eigen::Vector2d& owner_value,
    double outward_flux);
double evaluatePressureBoundary(
    const BoundaryPatch& patch,
    double owner_value,
    double outward_flux = 0.0);

void importLegacyBoundaryData(
    Mesh& mesh,
    const Eigen::MatrixXi& legacy_type,
    const Eigen::MatrixXi& legacy_zone,
    const std::vector<double>& zone_u,
    const std::vector<double>& zone_v);
void rebuildBoundaryPatchCells(Mesh& mesh);
void validateBoundaryModel(const Mesh& mesh);
void initializeBoundaryConditions(Mesh& mesh);
void initializeFlowFields(Mesh& mesh);
