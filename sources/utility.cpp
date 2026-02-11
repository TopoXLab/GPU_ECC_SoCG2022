#include <iostream>
#include <assert.h>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <numeric>
#include <functional>
#include <limits>
#include <filesystem>
#include <chrono>
#include <cmath>

#include "utility.h"
#include "kernel.h"
#include "template.h"
#include "helper_cuda.h"

namespace fs = std::filesystem;

std::vector<std::pair<int, int>> return_chunk_index(int h, int& chunk_num, bool expand_by_one) {
	/*
	- 2D input is cut from H; 3D input is cut from D
    - Returned section includes expanded part [[9, 9], 1, 3, 2, [9, 9]]
	*/
	if (expand_by_one) h += 2;
	int start, end;
	chunk_num = (chunk_num > (h - 2)) ? h - 2 : chunk_num;
	if (chunk_num <= 0) {
		std::cout << "Invalid input data shape without padding, please use padding" << std::endl; exit(1); }
	int chunk_size = (h - 2) / chunk_num;

	std::vector<std::pair<int, int>> section;
	for (int i = 0; i < chunk_num; i++) {
		start = i * chunk_size + 1;
		end = (i == chunk_num - 1) ? h - 2 : (i + 1) * chunk_size;
		section.push_back(std::make_pair(start - 1, end + 1));
	}
	return section;
}

void accumulate_VCEC_host_various_binNum_(int* VCEC_host_, int* VCEC_host_partial_,
	const std::vector<float>& ascend_unique_arr, const std::vector<float>& ascend_unique_arr_local
) {
	for (int i = 0; i < ascend_unique_arr_local.size(); i++) {
		auto iterator = std::find(ascend_unique_arr.begin(), ascend_unique_arr.end(), ascend_unique_arr_local[i]);
		VCEC_host_[int(iterator - ascend_unique_arr.begin())] += VCEC_host_partial_[i];
	}
}

void dummy2D_thread_(float*& data, int slice_l, int slice, int w) {
	size_t offset = (slice - slice_l) * (w + 2);
	for (size_t i = 0; i < (w + 2); i++)
		data[i + offset] = std::numeric_limits<float>::max();
}

void dummy3D_thread_(float*& data, int slice_l, int slice, int h, int w) {
	size_t offset = (slice - slice_l) * (h + 2) * (w + 2);
	for (size_t i = 0; i < (h + 2) * (w + 2); i++)
		data[i + offset] = std::numeric_limits<float>::max();
}

std::vector<float> accumulate_ascend_unique_arr(std::vector<std::vector<float>>& ascend_unique_arr_local_rec) {
	std::vector<float> ascend_unique_arr;
	std::unordered_map<float, bool> duplicate_rec;
	for (size_t i = 0; i < ascend_unique_arr_local_rec.size(); i++)
		for (size_t j = 0; j < ascend_unique_arr_local_rec[i].size(); j++) {
			float cur = ascend_unique_arr_local_rec[i][j];
			if (!duplicate_rec[cur]) { duplicate_rec[cur] = true; ascend_unique_arr.push_back(cur); }
		}
	std::sort(ascend_unique_arr.begin(), ascend_unique_arr.end());
	return ascend_unique_arr;
}

double get_wall_time()
{
	using clock = std::chrono::steady_clock; // monotonic, good for intervals
	return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

std::vector<std::string> fileNames_from_folder(std::string& path) {
	if (!fs::exists(path)) { std::cout << "Invalid folder path" << std::endl; exit(1); }

	std::vector<std::string> res;
	for (const auto& entry : fs::directory_iterator(fs::path(path))) {
		if (!entry.is_regular_file()) continue;
		auto ext = entry.path().extension().string();
		if (ext == ".dat" || ext == ".DAT") {
			res.push_back(entry.path().string());
		}
	}

	return res;
}

std::vector<std::string> compose_outfileNames_from_folder(std::string& path, std::string& patho) {
	if (!fs::exists(path)) { std::cout << "Invalid folder path" << std::endl; exit(1); }
	std::vector<std::string> res;
	fs::path out_dir(patho);

	for (const auto& entry : fs::directory_iterator(fs::path(path))) {
		if (!entry.is_regular_file()) continue;
		fs::path in_file = entry.path();
		if (in_file.has_extension() && in_file.extension() == ".txt") continue;
		fs::path n = in_file.stem();
		n += ".txt";
		fs::path o = out_dir / n;
		res.push_back(o.string());
	}
	return res;
}

int decide_chunk_num(const int h, const int w, const int d) {
	/*
		Decide chunk number
		@h: height of the input data
		@w: width of the input data
		@d: depth of the input data
	*/
	if (d > 0) return int(std::sqrt(d));
	else return int(std::sqrt(h));
}