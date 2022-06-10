#pragma once

#include <vector>
#include <string>

void helper_();

std::vector<double> ECC_(std::string& fileName, int h, int w, int d, bool pad, bool async, bool mt, bool timing, bool verb);

std::vector<double> ECC_folder(std::string& path, int h, int w, int d, bool pad, bool async, bool mt, bool timing, bool verb);

std::vector<double> ECC_vanila(std::string& fileName, int h, int w, bool mt, bool timing, bool verb);

void run_folder(std::string& path, int run_times);

void run_GENERAL(std::string& path, int data_dim, int run_times);