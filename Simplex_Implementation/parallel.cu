#include <cuda.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <cmath>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#define NUM_THREADS 256

double* tableau_gpu;
double* ratios_gpu;
int blks;
int* count_gpu;
int* count_cpu


__global__ void calcRatio(double* tableau_gpu, double* ratios_gpu, int* count_gpu, int pivot_column_idx, int nCol) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= nRow - 1) {
        return;
    }
    int last_column = nCol - 1;
    double pivot_column_val = tableau_gpu[tid * nCol + pivot_column_idx];
    if (pivot_column_val > 0.0) {
        ratios_gpu[tid] = tableu_gpu[tid * nCol + last_column] / pivot_column_val;
    } else {
        atomicInc(count_gpu, nRows);
    }
}

// Step 4
__global__ void updateAllColumnsInPivotRow(double* tableau_gpu, int pivot_row_idx, int pivot_col_index, int nCol) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= nCol) {
        return;
    }
    double pivot = tableau_gpu[pivot_row_idx * nCol + pivot_column_idx];
    tableau_gpu[pivot_row_idx * nCol + tid] = tableau_gpu[pivot_row_idx * nCol + tid] / pivot;
}

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
    double* tableau_cpu = (double*) malloc(nRow * nCol * sizeof(double));
    for (int i = 0; i < nRow; ++i) {
        std::getline(tableau_file, line);
        std::stringstream ss(line);
        for (int j = 0; j < nCol; ++j) {
            ss >> curr;
            double val = stod(curr);
            tableau_cpu[i * nCol + j] = val;
        }
    }

    // Moving matrix from CPU memory to GPU memory
    cudaMalloc((void**)&tableau_gpu, nRow * nCol * sizeof(double));
    cudaMemcpy(tableau_gpu, tableau_cpu, nRow * nCol * sizeof(double), cudaMemcpyHostToDevice);
    
    // Find the minimum value in the last row after copying the data to device memory (tableau_gpu)
    thrust::device_vector<double> d_tableau(tableau_gpu, tableau_gpu + (nRow * nCol));

    // Pointing to the start of the last row
    auto start = d_tableau.begin() + (nRow - 1) * nCol;
    auto end = d_tableau.begin() + nRow * nCol;

    // Finding the minimum element in the last row
    auto min_iter = thrust::min_element(start, end);

    // Calculating the index of the minimum element in the last row
    int pivot_column_idx = min_iter - start;

    // Dereferencing the iterator to get the minimum value
    double min_value = *min_iter;

    std::cout << "Min value in the last row: " << min_value << std::endl;
    std::cout << "Index of min value in the last row: " << min_index << std::endl;

    blks = (nCol * nRow + NUM_THREADS - 1) / NUM_THREADS;

    // Create a count variable that can be accessed by each CUDA thread.
    cudaMalloc((void**)&count, 1 * sizeof(int));
    // Create an array that stores the ratio of each element of last column of the tableau with the pivot column.
    cudaMalloc((void**)&ratios_gpu, (nRow - 1) * sizeof(double));
    cudaMemset((void*)ratios_gpu, HUGE_VAL, (nRow - 1) * sizeof(double));
    calcRatio<<<blks, NUM_THREADS>>>(tableau_gpu, ratios_gpu, count_gpu, pivot_column_idx, nCol);

    cudaMemcpy(count_cpu, count_gpu, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    if (count_cpu == nRow - 1) {
        std::cout << "There is no solution. Ending program..." << std::endl;
        return -1;
    }
    // If there was a solution, the GPU will continue working.
    thrust::device_vector<double> d_ratios(ratios_gpu, ratios_gpu + (nRow - 1))
    auto ratios_start = d_ratios.begin()
    auto ratio_end = d_ratios.begin() + (nRow - 1);
    auto min_ratio_iter = thrust::min_element(ratios_start, ratio_end);
    int pivot_row_idx = min_ratio_iter - ratio_start;
    double min_ratio_val = *min_ratio_iter;

    // Step 4:
    updateAllColumnsInPivotRow<<<blks, NUM_THREADS>>>(tableau_gpu, pivot_row_idx, pivot_column_idx, nCol);
    cudaDeviceSynchronize();













}


