#include <cuda.h>
double **tableau;

struct Compare_Max {
    double val = 0;
    int index = -1;
};

int main(int argc, char** argv) {
    int nRow, nCol;
    Compare_Max max = findMaxObjective(tableau, nRow, nCol);
    
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

Compare_Max findMaxObjective(double **tableau, int nRow, int nCol) {
    Compare_Max max;
    max.val = 0.0;
    max.index = -1;

    // Flatten the 2D array
    double *flat_tableau = new double[nRow * nCol];
    for (int i = 0; i < nRow; ++i) {
        for (int j = 0; j < nCol; ++j) {
            flat_tableau[i * nCol + j] = tableau[i][j];
        }
    }

    // Allocate device memory
    double *d_tableau;
    Compare_Max *d_max;
    cudaMalloc((void**)&d_tableau, nRow * nCol * sizeof(double));
    cudaMalloc((void**)&d_max, sizeof(Compare_Max));

    // Copy data to device
    cudaMemcpy(d_tableau, flat_tableau, nRow * nCol * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &max, sizeof(Compare_Max), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(256);
    dim3 gridSize((nCol + blockSize.x - 1) / blockSize.x);
    findMaxObjectiveKernel<<<gridSize, blockSize>>>(d_tableau, nRow, nCol, d_max);

    // Copy result back to host
    cudaMemcpy(&max, d_max, sizeof(Compare_Max), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_tableau);
    cudaFree(d_max);

    // Free host memory
    delete[] flat_tableau;

    // For debugging
    cout << "max.index: " << max.index << endl;

    return max;
}
