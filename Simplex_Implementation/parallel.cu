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
unsigned int* count_gpu;
unsigned int* count_cpu;
unsigned int* count2_gpu;
unsigned int* count2_cpu;


__global__ void initDoubleArrayToInf(double *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = HUGE_VAL;
    }
}


__global__ void calcRatio(double* tableau_gpu, double* ratios_gpu, unsigned int* count_gpu, int pivot_col_idx, int nCol, int nRow) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= nRow - 1) {
        return;
    }
    int last_column = nCol - 1;
    double pivot_column_val = tableau_gpu[tid * nCol + pivot_col_idx];
    if (pivot_column_val > 0.0) {
        ratios_gpu[tid] = tableau_gpu[tid * nCol + last_column] / pivot_column_val;
    } else {
        atomicInc(count_gpu, nRow);
    }
}

// Step 4
__global__ void updateAllColumnsInPivotRow(double* tableau_gpu, int pivot_row_idx, int pivot_col_idx, int nCol) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= nCol) {
        return;
    }
    double pivot = tableau_gpu[pivot_row_idx * nCol + pivot_col_idx];
    tableau_gpu[pivot_row_idx * nCol + tid] = tableau_gpu[pivot_row_idx * nCol + tid] / pivot;
}

// Step 5
__global__ void performRowOperations(double* tableau_gpu, int nRow, int nCol, int pivot_row_idx, int pivot_col_idx) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Convert the tid into a row and column component
    int thread_row = tid / nCol;
    int thread_col = tid % nCol;

    // Handles the case when your tid is on the pivot row.
    if (tid >= (nRow - 1) * nCol || thread_row == pivot_row_idx) {
        return;
    }
    double pivot2 = -tableau_gpu[thread_row * nCol + pivot_col_idx];
    tableau_gpu[thread_row * nCol + thread_col] += pivot2 * tableau_gpu[pivot_row_idx * nCol + thread_col];
}

// Step 6
// Potentially you can combine Step 5 and Step 6 for potential speedups.
__global__ void updateObjectiveFunction(double* tableau_gpu, int nRow, int nCol, int pivot_row_idx, int pivot_col_idx, unsigned int* count2_gpu) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= nCol) {
        return;
    }
    double pivot3 = -tableau_gpu[(nRow - 1) * nCol + pivot_col_idx];
    tableau_gpu[(nRow - 1) * nCol + tid] += pivot3 * tableau_gpu[pivot_row_idx * nCol + tid];
    if ((tid < nCol - 1) && tableau_gpu[(nRow - 1) * (nCol) + tid] < 0.0) {
        printf("%d \n", tid);
        atomicInc(count2_gpu, nCol - 1);
    }
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
    int pivot_col_idx = min_iter - start;

    // Dereferencing the iterator to get the minimum value
    double min_value = *min_iter;

    std::cout << "Min value in the last row: " << min_value << std::endl;
    std::cout << "Index of min value in the last row: " << pivot_col_idx << std::endl;

    blks = (nCol * nRow + NUM_THREADS - 1) / NUM_THREADS;
    //
    // Initial cudaMalloc for all steps
    cudaMalloc((void**)&count_gpu, 1 * sizeof(unsigned int));
    cudaMalloc((void**)&ratios_gpu, (nRow - 1) * sizeof(double));
    cudaMalloc((void**)&count2_gpu, 1 * sizeof(unsigned int*));
    count2_cpu = (unsigned int*) malloc(sizeof(unsigned int*));
    count_cpu = (unsigned int*)malloc(sizeof(unsigned int));
    //
    // Start of step 3:
    do {
        // Create a count variable that can be accessed by each CUDA thread.
        cudaMemset(count_gpu, 0, sizeof(unsigned int));
        // Create an array that stores the ratio of each element of last column of the tableau with the pivot column.
        initDoubleArrayToInf<<<blks, NUM_THREADS>>>(ratios_gpu, nRow - 1);
        calcRatio<<<blks, NUM_THREADS>>>(tableau_gpu, ratios_gpu, count_gpu, pivot_col_idx, nCol, nRow);
        std::cout << "Test Step3:#1" << std::endl;
        cudaMemcpy(count_cpu, count_gpu, 1 * sizeof(int), cudaMemcpyDeviceToHost);
        if (*count_cpu == nRow - 1) {
            std::cout << "There is no solution. Ending program..." << std::endl;
            return -1;
        }
        // If there was a solution, the GPU will continue working.
        thrust::device_vector<double> d_ratios(ratios_gpu, ratios_gpu + (nRow - 1));
        auto ratio_start = d_ratios.begin();
        auto ratio_end = d_ratios.begin() + (nRow - 1);
        auto min_ratio_iter = thrust::min_element(ratio_start, ratio_end);
        int pivot_row_idx = min_ratio_iter - ratio_start;
        std::cout << "Test Step3:#2" << std::endl;
        double min_ratio_val = *min_ratio_iter;
        
        std::cout << "Test Step3:#3" << std::endl;
        // Step 4:
        updateAllColumnsInPivotRow<<<blks, NUM_THREADS>>>(tableau_gpu, pivot_row_idx, pivot_col_idx, nCol);
        cudaDeviceSynchronize();

        std::cout << "pivot_col_idx : " << pivot_col_idx << std::endl;
        std::cout << "pivot_row_idx : " << pivot_row_idx << std::endl;
        std::cout << "min_ratio_val : " << min_ratio_val << std::endl;

        std::cout << "Updated matrix" << std::endl;
        cudaMemcpy(tableau_cpu, tableau_gpu, nRow * nCol * sizeof(double), cudaMemcpyDeviceToHost);
        for (int i = 0; i < nRow; ++i){
            std::cout << i << " -th row ";
            for (int j = 0; j < nCol; ++j){
                std::cout << tableau_cpu[i * nCol + j] << " "; 
            }
            std::cout << std::endl;
        }
        performRowOperations<<<blks, NUM_THREADS>>>(tableau_gpu, nRow, nCol, pivot_row_idx, pivot_col_idx);
        cudaDeviceSynchronize();
        std::cout << "Updated matrix after step5" << std::endl;
        cudaMemcpy(tableau_cpu, tableau_gpu, nRow * nCol * sizeof(double), cudaMemcpyDeviceToHost);
        for (int i = 0; i < nRow; ++i){
            std::cout << i << " -th row ";
            for (int j = 0; j < nCol; ++j){
                std::cout << tableau_cpu[i * nCol + j] << " "; 
            }
            std::cout << std::endl;
        }
        cudaMemset(count2_gpu, 0, sizeof(unsigned int));
        
        updateObjectiveFunction<<<blks, NUM_THREADS>>>(tableau_gpu, nRow, nCol, pivot_row_idx, pivot_col_idx, count2_gpu);
        cudaDeviceSynchronize();
        cudaMemcpy(count2_cpu, count2_gpu, 1 * sizeof(int), cudaMemcpyDeviceToHost);
        // Find the minimum value in the last row after copying the data to device memory (tableau_gpu)
        thrust::device_vector<double> d_obj_tableau(tableau_gpu, tableau_gpu + (nRow * nCol));

        // Pointing to the start of the last row
        start = d_obj_tableau.begin() + (nRow - 1) * nCol;
        end = d_obj_tableau.begin() + nRow * nCol;

        // Finding the minimum element in the last row
        auto min_obj_iter = thrust::min_element(start, end);

        // Calculating the index of the minimum element in the last row
        pivot_col_idx = min_obj_iter - start;

        // Dereferencing the iterator to get the minimum value
        min_value = *min_obj_iter;

        
        std::cout << "After step6: Min value in the last row: " << min_value << std::endl;
        std::cout << "After step6: Index of min value in the last row: " << pivot_col_idx << std::endl;
        cudaMemcpy(tableau_cpu, tableau_gpu, nRow * nCol * sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "Updated matrix after step6" << std::endl;
        for (int i = 0; i < nRow; ++i){
            std::cout << i << "-th row ";
            for (int j = 0; j < nCol; ++j){
                std::cout << tableau_cpu[i * nCol + j] << " "; 
            }
            std::cout << std::endl;
        }
        std::cout << "count2: " << *count2_cpu << std::endl;
    } while (*count2_cpu != 0);

    std::cout << "Finished Algorithm:" << std::endl;
}


