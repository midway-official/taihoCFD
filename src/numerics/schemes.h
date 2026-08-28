#pragma once

#include <string_view>

enum class TimeScheme {
    Unset,
    SteadyState,
    BackwardEuler,
};

enum class ConvectionScheme {
    Unset,
    Upwind,
};

enum class GradientScheme {
    Unset,
    Central,
};

enum class LaplacianScheme {
    Unset,
    Orthogonal,
};

enum class InterpolationScheme {
    Unset,
    Linear,
};

struct NumericalSchemes {
    TimeScheme time = TimeScheme::Unset;
    ConvectionScheme velocity_convection = ConvectionScheme::Unset;
    GradientScheme pressure_gradient = GradientScheme::Unset;
    LaplacianScheme velocity_laplacian = LaplacianScheme::Unset;
    InterpolationScheme face_interpolation = InterpolationScheme::Unset;

    static NumericalSchemes steady();
    static NumericalSchemes backwardEuler();
    void validate() const;
};

std::string_view toString(TimeScheme scheme);
std::string_view toString(ConvectionScheme scheme);
std::string_view toString(GradientScheme scheme);
std::string_view toString(LaplacianScheme scheme);
std::string_view toString(InterpolationScheme scheme);
