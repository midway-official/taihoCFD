#ifndef PARALLEL_H
#define PARALLEL_H

#include "fluid.h"
#include <mpi.h>
#include <omp.h>
// 声明全局变量（其他文件可访问）
extern double total_comm_time; // 通信时间
extern int total_comm_count,totalcount;   // 通信次数
extern double start_time, end_time;


//交互网格数据

void exchangeColumns(MatrixXd& matrix, int rank, int num_procs);
void vectorToMatrix(const VectorXd& x, MatrixXd& phi, const Mesh& mesh);

void matrixToVector(const MatrixXd& phi, VectorXd& x, const Mesh& mesh);

void Parallel_correction(Mesh mesh,Equation equ,MatrixXd &phi1,MatrixXd &phi2);
void Parallel_correction2(Mesh mesh,Equation equ,MatrixXd &phi1,MatrixXd &phi2);



void CG_parallel(Equation& equ, Mesh mesh,
                 VectorXd& b, VectorXd& x,
                 double epsilon, int max_iter,
                 int rank, int num_procs,
                 double& r0,int verbose=0);   

void solveFieldCG(
    Equation& equ,
    Mesh& mesh,
    MatrixXd& field,
    double tol,
    int max_iter,
    int rank,
    int num_procs,
    double& l2_norm,
    int verbose
);
void PCG_parallel(Equation& equ, Mesh mesh,
                 VectorXd& b, VectorXd& x,
                 double epsilon, int max_iter,
                 int rank, int num_procs,
                 double& r0,int verbose=0);   

void solveFieldPCG(
    Equation& equ,
    Mesh& mesh,
    MatrixXd& field,
    double tol,
    int max_iter,
    int rank,
    int num_procs,
    double& l2_norm,
    int verbose
);
#endif // PARALLEL_H