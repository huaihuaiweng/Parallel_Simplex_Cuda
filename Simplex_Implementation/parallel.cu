#include cuda.h
double **tableau;

struct Compare_Max {
    double val = 0;
    int index = -1;
};

int main(int argc, char** argv) {
    int nRow, nCol;
    
}


__global__ void findMaxObjectiveKernel(double *tableau, int nRow, int nCol, Compare_Max *max) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j <= nCol) {
        double val = tableau[nRow * nCol + j];
        if (val < 0.0) {
            val = -val;
            atomicMax(&(max->val), val);
            if (max->val == val) {
                max->index = j;
            }
        }
    }
}

Compare_Max findMaxfromObjectiveFunction() {

}
