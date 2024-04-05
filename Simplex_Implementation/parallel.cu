#include <cuda.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#define NUM_THREADS 256

double* tableau_gpu;
int blks;


struct Compare_Max {
    double val = 0;
    int index = -1;
};

// __device__ static double atomicMax(double* address, double val)
// {
//     int* address_as_i = (int*) address;
//     int old = *address_as_i, assumed;
//     do {
//         assumed = old;
//         old = ::atomicCAS(address_as_i, assumed,
//             __double_as_int(::fmaxf(val, __int_as_double(assumed))));
//     } while (assumed != old);
//     return __int_as_double(old);
// }


// __global__ void findMaxObjectiveKernel(double *tableau, int nRow, int nCol, Compare_Max *max) {
//     int j = threadIdx.x + blockIdx.x * blockDim.x;
//     int arr_size = nRow * nCol;
//     if (j < arr_size) {
//         double val = tableau[j];
//         if (val < 0.0) {
//             val = -val;
//             atomicMax(&(max->val), val);
//             if (max->val == val) {
//                 max->index = j;
//             }
//         }
//     }
// }

int main(int argc, char** argv) {
    int nRow, nCol;
    std::ifstream tableau_file(argv[1]);
    std::cout << "argv[1]: " << argv[1] << std::endl;

    // Getting the row/col lengths
    std::string line;
    std::getline(tableau_file, line);
    std::stringstream ss(line);
    std::string curr;
    ss >> curr;
    nRow = std::stoi(curr);
    ss >> curr;
    nCol = std::stoi(curr);
    ss >> curr;

    // Moving matrix from file to vector
    // std::vector<std::vector<double>> tableau_cpu(nRow, std::vector<double>(nCol, 0.0));
    double* tableau_cpu = (double*) malloc(nRow * nCol * sizeof(double));
    // for (int i = 0; i < nRow; ++i) {
    //     std::getline(tableau_file, line);
    //     std::stringstream ss(line);
    //     for (int j = 0; j < nCol; ++j) {
    //         ss >> curr;
    //         tableau_cpu[i][j] = std::stod(curr);
    //     }
    // }

     for (int i = 0; i < nRow; ++i) {
        std::getline(tableau_file, line);
        std::stringstream ss(line);
        for (int j = 0; j < nCol; ++j) {
            ss >> curr;
            double val = stod(curr);
            tableau_cpu[i * nCol + j] = val;
        }
    }

    // Now, print vector to check that it is loaded into memory.
    // std::cout << "{";
    // for (const auto& row: tableau_cpu) {
    //     std::cout << "{";
    //     for (const double& num: row) {
    //         std::cout << num << " ";
    //     }
    //     std::cout << "}";
    //     std::cout << std::endl;
    // }
    // std::cout << "}";

    cudaMalloc((void**)&tableau_gpu, nRow * nCol * sizeof(double));
    cudaMemcpy(tableau_gpu, tableau_cpu, nRow * nCol * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    // Compare_Max* d_max; // Device pointer for Compare_Max
    // Compare_Max h_max; // Host Compare_Max, to set initial values

    // h_max.val = 0;
    // h_max.index = -1;

    // Allocate memory on the device for Compare_Max
    // cudaMalloc((void**)&d_max, sizeof(Compare_Max));

    // Copy the initialized Compare_Max from host to device
    // cudaMemcpy(d_max, &h_max, sizeof(Compare_Max), cudaMemcpyHostToDevice);

    blks = (nCol * nRow + NUM_THREADS - 1) / NUM_THREADS;
    // findMaxObjectiveKernel<<<blks, NUM_THREADS>>>(tableau_gpu, nRow, nCol, d_max);
    double* min = thrust::min_element(thrust::host, tableau_cpu, tableau_cpu+(nCol * nRow - 1));

    // cudaMemcpy(&h_max, d_max, sizeof(Compare_Max), cudaMemcpyDeviceToHost);
    // std::cout << max->index << std::endl;
    // Now you can access the results on the host
    std::cout << "Min value: " << min << std::endl;
    // std::cout << "Index of max value: " << h_max.index << std::endl;

    
}




// Compare_Max findMaxObjective(double *tableau, int nRow, int nCol) {
//     Compare_Max max;
//     max.val = 0.0;
//     max.index = -1;

//     // Flatten the 2D array
//     double *flat_tableau = new double[nRow * nCol];
//     for (int i = 0; i < nRow; ++i) {
//         for (int j = 0; j < nCol; ++j) {
//             flat_tableau[i * nCol + j] = tableau[i][j];
//         }
//     }

//     // Allocate device memory
//     double *d_tableau;
//     Compare_Max *d_max;
//     cudaMalloc((void**)&d_tableau, nRow * nCol * sizeof(double));
//     cudaMalloc((void**)&d_max, sizeof(Compare_Max));

//     // Copy data to device
//     cudaMemcpy(d_tableau, flat_tableau, nRow * nCol * sizeof(double), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_max, &max, sizeof(Compare_Max), cudaMemcpyHostToDevice);

//     // Launch kernel
//     dim3 blockSize(256);
//     dim3 gridSize((nCol + blockSize.x - 1) / blockSize.x);
//     findMaxObjectiveKernel<<<gridSize, blockSize>>>(d_tableau, nRow, nCol, d_max);

//     // Copy result back to host
//     cudaMemcpy(&max, d_max, sizeof(Compare_Max), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(d_tableau);
//     cudaFree(d_max);

//     // Free host memory
//     delete[] flat_tableau;

//     // For debugging
//     cout << "max.index: " << max.index << endl;

//     return max;
// }
