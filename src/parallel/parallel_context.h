#pragma once

#include <mpi.h>

struct ParallelContext {
    MPI_Comm communicator = MPI_COMM_WORLD;
    int rank = 0;
    int size = 1;

    static ParallelContext world(MPI_Comm communicator = MPI_COMM_WORLD);
    void validate() const;
};
