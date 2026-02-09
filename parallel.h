#ifndef PARALLEL_H
#define PARALLEL_H

#include "fluid.h"
#include <mpi.h>
#include <omp.h>
// 声明全局变量（其他文件可访问）
extern double total_comm_time; // 通信时间
extern int total_comm_count,totalcount;   // 通信次数
extern double start_time, end_time;
// 发送矩阵列数据的函数声明
/*
void sendMatrixColumn(const MatrixXd& src_matrix, int src_col, 
                     MatrixXd& dst_matrix, int dst_col, 
                     int target_rank, int tag = 0);

// 接收矩阵列数据的函数声明
void recvMatrixColumn(MatrixXd& dst_matrix, int dst_col,
                     int src_rank, int tag = 0);
*/
double computeHash(const vector<double>& data);
void sendMatrixColumnWithSafety(const MatrixXd& src_matrix, int src_col, 
                                 vector<double>& send_buffer, 
                                 int target_rank, int tag);
void recvMatrixColumnWithSafety(vector<double>& recv_buffer, 
                                 int src_rank, int tag);

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
                 double& r0,int verbose=0);   // 新增：0不打印，1打印


#endif // PARALLEL_H