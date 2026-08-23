#pragma once

struct Mesh;

enum class BoundaryType : int {
    MpiGhost = -3,
    VelocityInlet = -2,
    PressureOutlet = -1,
    Interior = 0,
    Wall = 1,
};

inline bool isInterior(int value) {
    return value == static_cast<int>(BoundaryType::Interior);
}

inline bool isPressureOutlet(int value) {
    return value == static_cast<int>(BoundaryType::PressureOutlet);
}

inline bool isVelocityInlet(int value) {
    return value == static_cast<int>(BoundaryType::VelocityInlet);
}

inline bool isMpiGhost(int value) {
    return value == static_cast<int>(BoundaryType::MpiGhost);
}

inline bool isWall(int value) {
    return value >= static_cast<int>(BoundaryType::Wall);
}

inline bool isCoupledCell(int value) {
    return isInterior(value) || isMpiGhost(value);
}

double boundaryU(const Mesh& mesh, int i, int j);
double boundaryV(const Mesh& mesh, int i, int j);
void initializeBoundaryConditions(Mesh& mesh);
void initializeFlowFields(Mesh& mesh);
