#include <cuda.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#define NUM_THREADS 256

double* tableau_gpu;
int blks;

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
    int min_index = min_iter - start;

    // Dereferencing the iterator to get the minimum value
    double min_value = *min_iter;

    std::cout << "Min value in the last row: " << min_value << std::endl;
    std::cout << "Index of min value in the last row: " << min_index << std::endl;

    blks = (nCol * nRow + NUM_THREADS - 1) / NUM_THREADS;
}
