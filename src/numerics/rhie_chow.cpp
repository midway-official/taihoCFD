#include "numerics/rhie_chow.h"

#include "mesh/boundary.h"
#include "numerics/stencil.h"

void interpolateFaceVelocity(Mesh& mesh, const Equation& momentum) {
    for (int j = 0; j < mesh.nx - 1; ++j) {
        for (int i = 0; i < mesh.ny; ++i) {
            const int left_type = mesh.bctype(i, j);
            const int right_type = mesh.bctype(i, j + 1);

            if (isCoupledCell(left_type) && isCoupledCell(right_type) &&
                (isInterior(left_type) || isInterior(right_type))) {
                const double de = stencil::checkedDistance(
                    mesh.x_c(i, j + 1) - mesh.x_c(i, j), "face de", i, j);
                const double dw = stencil::checkedDistance(
                    mesh.x_c(i, j) - mesh.x_c(i, j - 1), "face dw", i, j);
                const double dee = stencil::checkedDistance(
                    mesh.x_c(i, j + 2) - mesh.x_c(i, j + 1),
                    "face dee", i, j);

                const double p_w =
                    stencil::pressureSample(mesh, i, j - 1, i, j);
                const double p_ee =
                    stencil::pressureSample(mesh, i, j + 2, i, j + 1);
                const double grad_left =
                    (mesh.p(i, j + 1) - p_w) / (de + dw);
                const double grad_right =
                    (p_ee - mesh.p(i, j)) / (dee + de);
                const double face_grad =
                    (mesh.p(i, j + 1) - mesh.p(i, j)) / de;

                const double d_left = mesh.vol(i, j) / momentum.A_p(i, j);
                const double d_right =
                    mesh.vol(i, j + 1) / momentum.A_p(i, j + 1);
                mesh.u_face(i, j) =
                    0.5 * (mesh.u(i, j) + mesh.u(i, j + 1)) +
                    0.5 * (d_left * grad_left + d_right * grad_right) -
                    0.5 * (d_left + d_right) * face_grad;
            } else if (isPressureOutlet(right_type) && isInterior(left_type)) {
                mesh.u_face(i, j) = mesh.u(i, j);
            } else if (isPressureOutlet(left_type) && isInterior(right_type)) {
                mesh.u_face(i, j) = mesh.u(i, j + 1);
            } else if (isVelocityInlet(right_type) && isInterior(left_type)) {
                mesh.u_face(i, j) = boundaryU(mesh, i, j + 1);
            } else if (isVelocityInlet(left_type) && isInterior(right_type)) {
                mesh.u_face(i, j) = boundaryU(mesh, i, j);
            } else if ((isWall(left_type) && isInterior(right_type)) ||
                       (isWall(right_type) && isInterior(left_type))) {
                const int boundary_j = isWall(left_type) ? j : j + 1;
                mesh.u_face(i, j) = boundaryU(mesh, i, boundary_j);
            } else {
                mesh.u_face(i, j) = 0.0;
            }
            stencil::requireFinite(mesh.u_face(i, j), "Rhie-Chow u_face", i, j);
        }
    }

    for (int j = 0; j < mesh.nx; ++j) {
        for (int i = 0; i < mesh.ny - 1; ++i) {
            const int lower_type = mesh.bctype(i, j);
            const int upper_type = mesh.bctype(i + 1, j);

            if (isCoupledCell(lower_type) && isCoupledCell(upper_type) &&
                (isInterior(lower_type) || isInterior(upper_type))) {
                const double ds = stencil::checkedDistance(
                    mesh.y_c(i + 1, j) - mesh.y_c(i, j), "face ds", i, j);
                const double dn = stencil::checkedDistance(
                    mesh.y_c(i, j) - mesh.y_c(i - 1, j), "face dn", i, j);
                const double dss = stencil::checkedDistance(
                    mesh.y_c(i + 2, j) - mesh.y_c(i + 1, j),
                    "face dss", i, j);

                const double p_n =
                    stencil::pressureSample(mesh, i - 1, j, i, j);
                const double p_ss =
                    stencil::pressureSample(mesh, i + 2, j, i + 1, j);
                const double grad_lower =
                    (mesh.p(i + 1, j) - p_n) / (ds + dn);
                const double grad_upper =
                    (p_ss - mesh.p(i, j)) / (dss + ds);
                const double face_grad =
                    (mesh.p(i + 1, j) - mesh.p(i, j)) / ds;

                const double d_lower = mesh.vol(i, j) / momentum.A_p(i, j);
                const double d_upper =
                    mesh.vol(i + 1, j) / momentum.A_p(i + 1, j);
                mesh.v_face(i, j) =
                    0.5 * (mesh.v(i, j) + mesh.v(i + 1, j)) +
                    0.5 * (d_lower * grad_lower + d_upper * grad_upper) -
                    0.5 * (d_lower + d_upper) * face_grad;
            } else if (isPressureOutlet(upper_type) && isInterior(lower_type)) {
                mesh.v_face(i, j) = mesh.v(i, j);
            } else if (isPressureOutlet(lower_type) && isInterior(upper_type)) {
                mesh.v_face(i, j) = mesh.v(i + 1, j);
            } else if (isVelocityInlet(upper_type) && isInterior(lower_type)) {
                mesh.v_face(i, j) = boundaryV(mesh, i + 1, j);
            } else if (isVelocityInlet(lower_type) && isInterior(upper_type)) {
                mesh.v_face(i, j) = boundaryV(mesh, i, j);
            } else if ((isWall(lower_type) && isInterior(upper_type)) ||
                       (isWall(upper_type) && isInterior(lower_type))) {
                const int boundary_i = isWall(lower_type) ? i : i + 1;
                mesh.v_face(i, j) = boundaryV(mesh, boundary_i, j);
            } else {
                mesh.v_face(i, j) = 0.0;
            }
            stencil::requireFinite(mesh.v_face(i, j), "Rhie-Chow v_face", i, j);
        }
    }
}
