#include <iostream>
#include <assert.h>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <numeric>
#include <functional>
#include <limits>

#include "utility.h"
#include "kernel.h"
#include "template.h"
#include "helper_cuda.h"

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

#define NOMINMAX
#include <Windows.h>

int find_and_query_device(int argc, char** argv, bool verbose) {
	int dev = findCudaDevice(argc, (const char**)argv);

	if (verbose) {
		// Query Cuda device
		cudaDeviceProp deviceProp;
		deviceProp.major = 0;
		deviceProp.minor = 0;

		checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
		printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
			deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
		printf("Constant memory size: %.0f KB\n", static_cast<float>(deviceProp.totalConstMem / 1024.0f));
		printf("Global memory size: %.0f GB\n", static_cast<float>(deviceProp.totalGlobalMem / 1073741824.0f));
		printf("32-bit registers per block: %d\n", deviceProp.regsPerBlock);
		printf("32-bit registers per SM: %d\n", deviceProp.regsPerMultiprocessor);
		printf("Shared memory per block: %.0f KB\n", static_cast<float>(deviceProp.sharedMemPerBlock / 1024.0f));
		printf("Shared memory per SM: %.0f KB\n", static_cast<float>(deviceProp.sharedMemPerMultiprocessor / 1024.0f));
		printf("Maximum Texture Dimension Size (x,y,z): 1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
			deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
			deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
			deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
		printf("Async engine count: %d\n", deviceProp.asyncEngineCount);
	}
	return dev;
}

void print_ECC_results(std::vector<float>& ascend_unique_arr, int* VCEC_host) {
	std::cout << "ECC computation results:" << std::endl;
	long long int accum = 0;
	for (int i = 0; i < ascend_unique_arr.size(); i++) {
		if (VCEC_host[i] == 0) continue;
		std::cout << ascend_unique_arr[i] << " " << accum + VCEC_host[i] << std::endl;
		accum += VCEC_host[i];
	}
}

float* generate_toy_sample(const int dim, const int index) {
	float dum_array_2D[3][9] = { { 1,2,5,6,3,4,8,7,0 }, { 3,3,3,3,3,3,3,3,3 }, { 3,3,3,3,4,3,3,3,3 } };
	float dum_array_3D[2][27] = { { 1,2,5,6,3,4,8,7,0,3,3,3,3,3,3,3,3,3,3,3,3,3,4,3,3,3,3 }, {1.1,2.1,5.1,6.1,3.1,4.1,8.1,7.1,0.1,
		3.1,3.1,3.1,3.1,3.1,3.1,3.1,3.1,3.1,3.1,3.1,3.1,3.1,4.1,3.1,3.1,3.1,3.1} };
	if (dim == 2) { assert(index < 3); return dum_array_2D[index]; }
	else if (dim == 3) { assert(index < 2); return dum_array_3D[index]; }
	else { std::cout << "Invalid index request for toy sample" << std::endl; exit(1); }
}

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

std::vector<std::pair<int, int>> return_equal_size_chunk(int h, int chunk_num, bool expand_by_one) {
	std::vector<std::pair<int, int>> section;
	for (size_t i = 0; i < chunk_num; i++) {
		section.push_back(std::make_pair(0, (expand_by_one) ? h + 1 : h - 1));
	}
	return section;
}

float compute_stdev(std::vector<float>& v) {
	float sum = std::accumulate(v.begin(), v.end(), 0.0);
	float mean = sum / v.size();
	std::vector<float> diff(v.size());
	std::transform(v.begin(), v.end(), diff.begin(),
		std::bind2nd(std::minus<float>(), mean));
	float sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
	float stdev = std::sqrt(sq_sum / v.size());
	return stdev;
}

void accumulate_VCEC_host_(int* VCEC_host_, int* VCEC_host_partial_, const int binNum) {
	for (int i = 0; i < binNum; i++) VCEC_host_[i] += VCEC_host_partial_[i];
}

void accumulate_VCEC_host_various_binNum_(int* VCEC_host_, int* VCEC_host_partial_,
	const std::vector<float>& ascend_unique_arr, const std::vector<float>& ascend_unique_arr_local
) {
	for (int i = 0; i < ascend_unique_arr_local.size(); i++) {
		auto iterator = std::find(ascend_unique_arr.begin(), ascend_unique_arr.end(), ascend_unique_arr_local[i]);
		VCEC_host_[int(iterator - ascend_unique_arr.begin())] += VCEC_host_partial_[i];
	}
}

void parse_GRF_name(std::string& fileName, std::string& size, std::string& dim) {
	std::string tmp("\\");
	std::string name;
	if (fileName.find(tmp) != std::string::npos) {
		size_t found = fileName.find(tmp);
		name = fileName.substr(found + 1, fileName.length());
	}
	else {
		size_t found = fileName.find_last_of('/');
		name = fileName.substr(found + 1, fileName.length());
	}
	dim = name.substr(0, name.find('D'));
	size = name.substr(name.find('_') + 1, name.length());
	size = size.substr(0, size.find('_'));
}

void parse_VICTRE_name(std::string& fileName, std::string& d1, std::string& d2, std::string& d3) {
	std::string tmp("\\");
	std::string name;
	if (fileName.find(tmp) != std::string::npos) {
		size_t found = fileName.find(tmp);
		name = fileName.substr(found + 1, fileName.length());
	}
	else {
		size_t found = fileName.find_last_of('/');
		name = fileName.substr(found + 1, fileName.length());
	}
	std::string p1 = name.substr(name.find('_') + 1, name.length());
	d1 = p1.substr(0, p1.find('_'));
	p1 = p1.substr(p1.find('_') + 1, p1.length());
	d2 = p1.substr(0, p1.find('_'));
	p1 = p1.substr(p1.find('_') + 1, p1.length());
	d3 = p1.substr(0, p1.find('_'));
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

double get_wall_time() {
	LARGE_INTEGER time, freq;
	if (!QueryPerformanceFrequency(&freq)) {
		//  Handle error
		return 0;
	}
	if (!QueryPerformanceCounter(&time)) {
		//  Handle error
		return 0;
	}
	return (double)time.QuadPart / freq.QuadPart;
}

double get_cpu_time() {
	FILETIME a, b, c, d;
	if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0) {
		//  Returns total user time.
		//  Can be tweaked to include kernel times as well.
		return
			(double)(d.dwLowDateTime |
				((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
	}
	else {
		//  Handle error
		return 0;
	}
}

std::vector<std::string> fileNames_from_folder(std::string& path) {
	if (!boost::filesystem::exists(path)) { std::cout << "Invalid folder path" << std::endl; exit(1); }
	std::vector<std::string> res;
	for (auto& entry : boost::make_iterator_range(
		boost::filesystem::directory_iterator(path), {})) {
		std::string t = entry.path().string();
		std::string postfix = t.substr(t.find_last_of('.') + 1);
		if (postfix == std::string("dat")) res.push_back(t);
	}
	return res;
}

std::vector<std::string> compose_outfileNames_from_folder(std::string& path, std::string& patho) {
	if (!boost::filesystem::exists(path)) { std::cout << "Invalid folder path" << std::endl; exit(1); }
	std::vector<std::string> res;
	for (auto& entry : boost::make_iterator_range(
		boost::filesystem::directory_iterator(path), {})) {
		std::string t = entry.path().string();
		std::string n = t.substr(t.find_last_of("/\\") + 1);
		n = n.substr(0, n.find_last_of(".")) + ".txt";
		std::string o = (patho[patho.length() - 1] == '/' || patho[patho.length() - 1] == '\\') ? patho + n : patho + "/" + n;
		res.push_back(o);
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