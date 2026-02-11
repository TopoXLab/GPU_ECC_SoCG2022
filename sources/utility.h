#pragma once
#include <chrono>
#include <vector>

double get_wall_time();

std::vector<std::pair<int, int>> return_chunk_index(
    int h,
    int& chunk_num,
    bool expand_by_one
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