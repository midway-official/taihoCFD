#include "parallel/parallel_context.h"

#include <stdexcept>

ParallelContext ParallelContext::world(MPI_Comm communicator_value) {
    ParallelContext context;
    context.communicator = communicator_value;
    MPI_Comm_rank(context.communicator, &context.rank);
    MPI_Comm_size(context.communicator, &context.size);
    context.validate();
    return context;
}

void ParallelContext::validate() const {
    if (communicator == MPI_COMM_NULL || size <= 0 ||
        rank < 0 || rank >= size) {
        throw std::invalid_argument("非法 MPI rank/size");
    }
}
