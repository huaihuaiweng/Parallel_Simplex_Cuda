#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>

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

    // // Moving matrix from file to vector
    double* tableau_cpu2 = (double*) malloc(nRow * nCol * sizeof(double));
    // for (int i = 0; i < nRow; ++i) {
    //     std::getline(tableau_file, line);
    //     std::stringstream ss(line);
    //     for (int j = 0; j < nCol; ++j) {
    //         ss >> curr;
            
    //     }
    // }
    
    std::vector<std::vector<double>> tableau_cpu(nRow, std::vector<double>(nCol, 0.0));
    for (int i = 0; i < nRow; ++i) {
        std::getline(tableau_file, line);
        std::stringstream ss(line);
        for (int j = 0; j < nCol; ++j) {
            std::cout << curr  << std::endl;
            ss >> curr;
            double val = stod(curr);
            tableau_cpu2[i * nRow + j] = val;
            tableau_cpu[i][j] = val;
        }
    }

    std::cout << "Correct: " << std::endl;
    for (const auto& row: tableau_cpu) {
        std::cout << "{";
        for (const double& num: row) {
            std::cout << num << " ";
        }
        std::cout << "}" << std::endl;
    }
    std::cout << "}" << std::endl;

    std::cout << "Malloc Version" << std::endl;
    // Now, print vector to check that it is loaded into memory.
    std::cout << "nRow" << nRow << std::endl;
    std::cout << "nCol" << nCol << std::endl;
    std::cout << "{";
    for (int i = 0; i < nRow; ++i) {
        std::cout << "{";
        for (int j = 0; j < nCol; ++j) {
            std::cout << tableau_cpu2[i * nRow + j] << " ";
        }
        std::cout << "}" << std::endl;
    }
    std::cout << "}" << std::endl;



    
    

    
    return 0;
}

// TODO: Move parsing of first row to this method to clean up main function.
std::pair<int,int> read_row_col(std::string line) {

    
}