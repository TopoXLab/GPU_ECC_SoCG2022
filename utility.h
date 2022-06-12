#pragma once
#include <chrono>

double get_wall_time();

double get_cpu_time();

int find_and_query_device(
    int argc,
    char** argv,
    bool verbose
);

void print_ECC_results(
    std::vector<float>& ascend_unique_arr,
    int* VCEC_host
);

float* generate_toy_sample(
    const int dim,
    const int index
);

std::vector<std::pair<int, int>> return_chunk_index(
    int h,
    int& chunk_num,
    bool expand_by_one
);

std::vector<std::pair<int, int>> return_equal_size_chunk(
    int h,
    int chunk_num,
    bool expand_by_one
);

void accumulate_VCEC_host_(
    int* VCEC_host_,
    int* VCEC_host_partial_,
    const int binNum
);

float compute_stdev(std::vector<float>& v);

void parse_GRF_name(std::string& fileName,
    std::string& size,
    std::string& dim
);

void parse_VICTRE_name(
    std::string& fileName,
    std::string& d1,
    std::string& d2,
    std::string& d3
);

void dummy2D_thread_(
    float*& data,
    int slice_l, 
    int slice,
    int w
);

void dummy3D_thread_(
    float*& data,
    int slice_l,
    int slice,
    int h,
    int w
);

std::vector<float> accumulate_ascend_unique_arr(std::vector<std::vector<float>>& ascend_unique_arr_local_rec);

std::vector<std::string> fileNames_from_folder(std::string& path);

std::vector<std::string> compose_outfileNames_from_folder(std::string& path, std::string& patho);

void accumulate_VCEC_host_various_binNum_(
	int* VCEC_host_,
	int* VCED_host_partial_,
	const std::vector<float>& ascend_unique_arr,
	const std::vector<float>& ascend_unique_arr_local
);

struct Stopwatch {
    std::chrono::time_point<std::chrono::high_resolution_clock> last;
    Stopwatch() {
        last = std::chrono::high_resolution_clock::now();
    }

    double lap() {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(
            now - last).count();
        last = now;
        return duration;
    }
};

int decide_chunk_num(
    const int h,
    const int w,
    const int d
);