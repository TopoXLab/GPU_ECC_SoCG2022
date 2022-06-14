#pragma once

#include <unordered_map>
#include "helper_cuda.h"
#include "utility.h"

template<typename input_elem_t>
void init_histogram_2D_(input_elem_t** data2D, const int num, const int size) {
	for (int i = 0; i < num; i++) init_histogram_1D_<input_elem_t>(data2D[i], size);
}

template<typename input_elem_t>
void init_histogram_1D_(input_elem_t* data1D, const int size) {
	for (size_t i = 0; i < size; i++) data1D[i] = (input_elem_t)0;
}

template<typename input_elem_t>
input_elem_t** allocate_device_memory2D_(const int num, const int size) {
	input_elem_t** data2D = new input_elem_t * [num];
	for (int i = 0; i < num; i++) data2D[i] = allocate_device_memory1D_<input_elem_t>(size);
	return data2D;
}

template<typename input_elem_t>
input_elem_t* allocate_device_memory1D_(const int size) {
	input_elem_t* data1D;
	checkCudaErrors(cudaMalloc((void**)&data1D, size * sizeof(input_elem_t)));
	return data1D;
}

template<typename input_elem_t>
input_elem_t** allocate_host_memory2D_(const int num, const int size) {
	input_elem_t** data2D = new input_elem_t * [num];
	for (int i = 0; i < num; i++) data2D[i] = allocate_host_memory1D_<input_elem_t>(size);
	return data2D;
}

template<typename input_elem_t>
input_elem_t* allocate_host_memory1D_(const int size) {
	input_elem_t* data1D = (input_elem_t*)malloc(size * sizeof(input_elem_t));
	return data1D;
}

template<typename input_elem_t>
input_elem_t** allocate_host_memory2D_pinned_(const int num, const int size) {
	input_elem_t** data2D = new input_elem_t * [num];
	for (int i = 0; i < num; i++) data2D[i] = allocate_host_memory1D_pinned_<input_elem_t>(size);
	return data2D;
}

template<typename input_elem_t>
input_elem_t* allocate_host_memory1D_pinned_(const int size) {
	input_elem_t* data1D;
	checkCudaErrors(cudaMallocHost((void**)&data1D, size * sizeof(input_elem_t)));
	return data1D;
}

template<typename input_elem_t>
void free_device_memory2D_(input_elem_t** data2D, const int num) {
	for (int i = 0; i < num; i++) checkCudaErrors(cudaFree(data2D[i]));
	delete data2D;
}

template<typename input_elem_t>
void free_host_memory2D_pinned_(input_elem_t** data2D, const int num) {
	for (int i = 0; i < num; i++) cudaFreeHost(data2D[i]);
	delete data2D;
}

template<typename input_elem_t>
void free_host_memory2D_(input_elem_t** data2D, const int num) {
	for (int i = 0; i < num; i++) free(data2D[i]);
	delete data2D;
}

template<typename input_elemt_t>
void free_vector2D_(std::vector<std::vector<input_elemt_t>>& v) {
	for (int i = 0; i < v.size(); i++) v[i].clear();
	v.clear();
}

template<typename input_elem_t>
input_elem_t* generate_random_data(const int total_size, int range, input_elem_t oscilation) {
	int sign;
	input_elem_t* data = (input_elem_t*)malloc(total_size * sizeof(input_elem_t));
	for (size_t i = 0; i < total_size; i++) { 
		sign = (rand() % 2 == 0) ? 1 : -1;
		data[i] = static_cast<input_elem_t>(rand() % range) + input_elem_t(sign * oscilation);
	}
	return data;
}

template<typename input_elem_t>
input_elem_t* from_random_2D_(const int h, const int w, int range, input_elem_t oscilation) {
	input_elem_t* data = generate_random_data<input_elem_t>(h * w, range, oscilation);
	return data;
}

template<typename input_elem_t>
input_elem_t* from_random_3D_(const int h, const int w, const int d, int range, input_elem_t oscilation) {
	input_elem_t* data = generate_random_data<input_elem_t>(h * w * d, range, oscilation);
	return data;
}

template<typename input_elem_t, typename output_elem_t>
void write_stream_(std::string filename, input_elem_t* data, const int size) {
	std::ofstream wstream(filename.c_str(), std::ios::out | std::ios::binary);
	for (size_t i = 0; i < size; i++) { output_elem_t o = (output_elem_t)data[i]; wstream.write((char*)&o, sizeof(output_elem_t)); }
	wstream.close();
}

template<typename input_elem_t>
float* from_stream_2Dfloat_(std::string filename, int& h, int& w, bool expand_by_one, bool pinned,
	int& binNum_global, int& binNum_max_local, const std::vector<std::pair<int, int>>& section,
	std::vector<float>& ascend_unique_arr, std::vector<std::vector<float>>& ascend_unique_arr_local, float& read_time) {
	/*
	* Converts all data type to float for GPU, because the coded access pattern is designed for float.
	* @filename: filename of the input
	* @h, w: height and width of the input
	* @expand_by_one: pad file by one pixel around it
	*/
	float* data;
	std::unordered_map<float, bool> duplicate_rec;
	ascend_unique_arr_local.resize(section.size());
	binNum_max_local = 0;
	if (pinned)
		(expand_by_one) ? checkCudaErrors(cudaMallocHost((void**)&data, (h + 2) * (w + 2) * sizeof(float)))
		: checkCudaErrors(cudaMallocHost((void**)&data, h * w * sizeof(float)));
	else
		(expand_by_one) ? data = (float*)malloc((h + 2) * (w + 2) * sizeof(float)) : data = (float*)malloc(h * w * sizeof(float));
	std::ifstream stream(filename.c_str(), std::ios::binary);
	if (!stream) std::cout << "<<Error>>: File open failed" << std::endl;
	float max_ = std::numeric_limits<float>::min(), min_ = std::numeric_limits<float>::max();
	using input_value_t = input_elem_t;

	if (expand_by_one) {
		Stopwatch sw_read;
		for (size_t secidx = 0; secidx < section.size(); secidx++) {
			std::unordered_map<float, bool> duplicate_rec_local;
			for (size_t i = section[secidx].first + 1; i <= section[secidx].second - 1; i++)
				for (size_t j = i * (w + 2) + 1; j < (i + 1) * (w + 2) - 1; j++) {
					input_value_t v;
					stream.read(reinterpret_cast<char*>(&v), sizeof(input_value_t));
					data[j] = static_cast<float>(v);
					if (!duplicate_rec[data[j]]) { duplicate_rec[data[j]] = true; ascend_unique_arr.push_back(data[j]); }
					if (!duplicate_rec_local[data[j]]) { duplicate_rec_local[data[j]] = true; ascend_unique_arr_local[secidx].push_back(data[j]); }
					max_ = max(max_, data[j]);
					min_ = min(min_, data[j]);
				}
			std::sort(ascend_unique_arr_local[secidx].begin(), ascend_unique_arr_local[secidx].end());
			binNum_max_local = (ascend_unique_arr_local[secidx].size() > binNum_max_local) ? ascend_unique_arr_local[secidx].size() : binNum_max_local;
		}
		read_time = sw_read.lap() * 1000.0;
		for (int i = 0; i < w + 2; i++) data[i] = max_ + 1;
		for (int i = (h + 1) * (w + 2); i < (h + 2) * (w + 2); i++) data[i] = max_ + 1;
		for (int i = w + 2; i <= h * (w + 2); i += (w + 2)) data[i] = max_ + 1;
		for (int i = 2 * (w + 2) - 1; i <= h * (w + 2) + w + 1; i += (w + 2))  data[i] = max_ + 1;
		h = h + 2;
		w = w + 2;
	}
	else {
		float v_;
		input_value_t v;
		Stopwatch sw_read;
		for (int i = 0; i < w; i++) {
			stream.read(reinterpret_cast<char*>(&v), sizeof(input_value_t));
			data[i] = static_cast<float>(v);
		}
		for (size_t secidx = 0; secidx < section.size(); secidx++) {
			std::unordered_map<float, bool> duplicate_rec_local;
			for (int i = section[secidx].first + 1; i <= section[secidx].second - 1; i++) {
				stream.read(reinterpret_cast<char*>(&v), sizeof(input_value_t));
				data[i * w] = static_cast<float>(v);
				for (int j = 1; j < w - 1; j++) {
					stream.read(reinterpret_cast<char*>(&v), sizeof(input_value_t));
					v_ = static_cast<float>(v);
					data[i * w + j] = v_;
					if (!duplicate_rec[v_]) { duplicate_rec[v_] = true; ascend_unique_arr.push_back(v_); }
					if (!duplicate_rec_local[v_]) { duplicate_rec_local[v_] = true; ascend_unique_arr_local[secidx].push_back(v_); }
					max_ = max(max_, v_);
					min_ = min(min_, v_);
				}
				stream.read(reinterpret_cast<char*>(&v), sizeof(input_value_t));
				data[(i + 1) * w - 1] = static_cast<float>(v);
			}
			std::sort(ascend_unique_arr_local[secidx].begin(), ascend_unique_arr_local[secidx].end());
			binNum_max_local = (ascend_unique_arr_local[secidx].size() > binNum_max_local) ? ascend_unique_arr_local[secidx].size() : binNum_max_local;
		}
		for (int i = 0; i < w; i++) {
			stream.read(reinterpret_cast<char*>(&v), sizeof(input_value_t));
			data[w * (h - 1) + i] = static_cast<float>(v);
		}
		read_time = sw_read.lap() * 1000.0;
	}
	stream.close();
	binNum_global = ascend_unique_arr.size();
	std::sort(ascend_unique_arr.begin(), ascend_unique_arr.end());
	return data;
}

template<typename input_elem_t>
float* from_stream_3Dfloat_(
	std::string filename, int& h, int& w, int& d, bool expand_by_one, bool pinned,
	int& binNum_global, int& binNum_max_local, const std::vector<std::pair<int, int>>& section,
	std::vector<float>& ascend_unique_arr, std::vector<std::vector<float>>& ascend_unique_arr_local, float& read_time) {
	/*
	* Converts all data type to float for GPU, because the coded access pattern is designed for float.
	* @filename: filename of the input
	* @h, w, d: height, width, and depth of the input
	* @expand_by_one: pad file by one pixel around it
	* Note: Huber's input 5x3x2: 5 for depth, 3 for height, 2 for width
	*/
	float* data;
	std::unordered_map<float, bool> duplicate_rec;
	ascend_unique_arr_local.resize(section.size());
	binNum_max_local = 0;
	if (pinned)
		(expand_by_one) ? checkCudaErrors(cudaMallocHost((void**)&data, (h + 2) * (w + 2) * (d + 2) * sizeof(float))) :
		checkCudaErrors(cudaMallocHost((void**)&data, h * w * d * sizeof(float)));
	else
		(expand_by_one) ? data = (float*)malloc((h + 2) * (w + 2) * (d + 2) * sizeof(float)) : data = (float*)malloc(h * w * d * sizeof(float));
	std::ifstream stream(filename.c_str(), std::ios::binary);
	if (!stream) { std::cout << "<<Error>>: File open failed" << std::endl; exit(1); }
	float max_ = std::numeric_limits<float>::min(), min_ = std::numeric_limits<float>::max();
	using input_value_t = input_elem_t;

	if (expand_by_one) {
		int cnt;	
		Stopwatch sw_read;
		for (size_t secidx = 0; secidx < section.size(); secidx++) {
			std::unordered_map<float, bool> duplicate_rec_local;
			for (size_t i = section[secidx].first + 1; i <= section[secidx].second - 1; i++)
				for (size_t j = i * (h + 2) * (w + 2) + w + 2 + 1; j < i * (h + 2) * (w + 2) + (h + 1) * (w + 2) - 1; j++) {
					cnt = 0;
					while (cnt++ < w) {
						input_value_t v;
						stream.read(reinterpret_cast<char*>(&v), sizeof(input_value_t));
						data[j] = static_cast<float>(v);
						if (!duplicate_rec[data[j]]) { duplicate_rec[data[j]] = true; ascend_unique_arr.push_back(data[j]); }
						if (!duplicate_rec_local[data[j]]) { duplicate_rec_local[data[j]] = true; ascend_unique_arr_local[secidx].push_back(data[j]); }
						max_ = std::max(max_, data[j]);
						min_ = std::min(min_, data[j]);
						j++;
					}
					j++;
				}
			std::sort(ascend_unique_arr_local[secidx].begin(), ascend_unique_arr_local[secidx].end());
			binNum_max_local = (ascend_unique_arr_local[secidx].size() > binNum_max_local) ? ascend_unique_arr_local[secidx].size() : binNum_max_local;
		}
		read_time = sw_read.lap() * 1000.0;
		for (int i = 0; i < (w + 2) * (h + 2); i++) data[i] = max_ + 1;
		for (int i = (d + 1) * (w + 2) * (h + 2); i < (d + 2) * (w + 2) * (h + 2); i++) data[i] = max_ + 1;
		for (int i = 1; i <= d; i++) for (int j = (w + 2) * (h + 2) * i; j < (w + 2) * (h + 2) * i + w + 2; j++) data[j] = max_ + 1;
		for (int i = 1; i <= d; i++) for (int j = (w + 2) * (h + 2) * i + (w + 2) * (h + 1); j < (w + 2) * (h + 2) * (i + 1); j++) data[j] = max_ + 1;
		for (int i = 1; i <= d; i++) for (int j = (w + 2) * (h + 2) * i + w + 2; j <= (w + 2) * (h + 2) * i + (w + 2) * h; j += (w + 2)) data[j] = max_ + 1;
		for (int i = 1; i <= d; i++) for (int j = (w + 2) * (h + 2) * i + 2 * (w + 2) - 1; j < (w + 2) * (h + 2) * i + (w + 2) * (h + 1); j += (w + 2)) data[j] = max_ + 1;
		h = h + 2;
		w = w + 2;
		d = d + 2;
	}
	else {
		float v_;
		input_value_t v;
		Stopwatch sw_read;
		for (int i = 0; i < h * w; i++) {
			stream.read(reinterpret_cast<char*>(&v), sizeof(input_value_t));
			data[i] = static_cast<float>(v);
		}
		for (size_t secidx = 0; secidx < section.size(); secidx++) {
			std::unordered_map<float, bool> duplicate_rec_local;
			for (int i = section[secidx].first + 1; i <= section[secidx].second - 1; i++) {
				int j = 0;
				while (j < w) { 
					stream.read(reinterpret_cast<char*>(&v), sizeof(input_value_t));
					data[i * h * w + j++] = static_cast<float>(v);
				}
				for (int k = 1; k < h - 1; k++)
					for (j = 0; j < w; j++) {
						if (j == 0 || j == w - 1) {
							stream.read(reinterpret_cast<char*>(&v), sizeof(input_value_t));
							data[i * h * w + k * w + j] = static_cast<float>(v);
						}
						else {
							stream.read(reinterpret_cast<char*>(&v), sizeof(input_value_t));
							v_ = static_cast<float>(v);
							data[i * h * w + k * w + j] = v_;
							if (!duplicate_rec[v_]) { duplicate_rec[v_] = true; ascend_unique_arr.push_back(v_); }
							if (!duplicate_rec_local[v_]) { duplicate_rec_local[v_] = true; ascend_unique_arr_local[secidx].push_back(v_); }
							max_ = max(max_, v_);
							min_ = min(min_, v_);
						}
					}
				j = 0;
				while (j < w) {
					stream.read(reinterpret_cast<char*>(&v), sizeof(input_value_t));
					data[i * h * w + (h - 1) * w + j++] = static_cast<float>(v);
				}
			}
			std::sort(ascend_unique_arr_local[secidx].begin(), ascend_unique_arr_local[secidx].end());
			binNum_max_local = (ascend_unique_arr_local[secidx].size() > binNum_max_local) ? ascend_unique_arr_local[secidx].size() : binNum_max_local;
		}
		for (int i = 0; i < h * w; i++) {
			stream.read(reinterpret_cast<char*>(&v), sizeof(input_value_t));
			data[h * w * (d - 1) + i] = static_cast<float>(v);
		}
		read_time = sw_read.lap() * 1000.0;
	}
	stream.close();
	binNum_global = ascend_unique_arr.size();
	std::sort(ascend_unique_arr.begin(), ascend_unique_arr.end());
	return data;
}

template<typename input_elem_t>
input_elem_t* local_unique_numbers(input_elem_t* input, const int offset, const int size, const bool pinned, int& binNum) {
	/*
	* @input: input data array
	* @offset: offset in the input array
	* @size: size of the incoming data in elements
	* @pinned: pinned host memory or not
	* @binNum: return value, number of distinct elements in the array
	* return: float array with unique elements sorted in ascending order
	*/
	std::vector<input_elem_t> unique_vals;
	std::unordered_map<input_elem_t, bool> duplicate_rec;
	for (size_t i = offset; i < offset + size; i++)
		if (!duplicate_rec[input[i]]) { duplicate_rec[input[i]] = true; unique_vals.push_back(input[i]); }
	std::sort(unique_vals.begin(), unique_vals.end());

	binNum = unique_vals.size();
	input_elem_t* unique_vals_host_ = (pinned) ? allocate_host_memory1D_pinned_<input_elem_t>(binNum) : allocate_host_memory1D_<input_elem_t>(binNum);
	std::copy(unique_vals.begin(), unique_vals.end(), unique_vals_host_);
	return unique_vals_host_;
}

template<typename input_elem_t>
input_elem_t* from_stream_vanilla_(std::string& fileName, int size) {
	input_elem_t* data = (input_elem_t*)malloc(size * sizeof(input_elem_t));
	std::ifstream stream(fileName.c_str(), std::ios::binary);
	if (!stream) { std::cout << "File open failed" << std::endl; exit(1); }
	for (size_t i = 0; i < size; i++) {
		input_elem_t v;
		stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
		data[i] = static_cast<input_elem_t>(v);
	}
	stream.close();
	return data;
}

template<typename input_elem_t>
input_elem_t** allocate_input_memory(int num, bool async_mode, bool expand_by_one, int& h, int& w, int& d, int chunkH) {
	/*
	* chunkH: if equals -1, program will use d in 3D case and h in 2D case
	*/
	int size;
	if (d > 0) {
		h = (expand_by_one) ? h + 2 : h;
		w = (expand_by_one) ? w + 2 : w;
		d = (expand_by_one) ? d + 2 : d;
		size = (chunkH == -1) ? h * w * d : h * w * chunkH;
	}
	else {
		h = (expand_by_one) ? h + 2 : h;
		w = (expand_by_one) ? w + 2 : w;
		size = (chunkH == -1) ? w * h : w * chunkH;
	}
	input_elem_t** res = (async_mode) ? allocate_host_memory2D_pinned_<input_elem_t>(num, size) : allocate_host_memory2D_<input_elem_t>(num, size);
	return res;
}

template<typename input_elem_t>
std::vector<float> from_stream3D_thread_(std::string& filename, float*& data, int slice_l, int slice_r, int h, int w, bool padded) {
	std::vector<float> ascend_unique_arr_local;
	std::unordered_map<float, bool> duplicate_rec_local;

	std::ifstream stream(filename.c_str(), std::ios::binary);
	if (!stream) { std::cout << "File open failed" << std::endl; exit(1); }
	size_t file_offset = (padded) ? (slice_l - 1) * h * w * sizeof(input_elem_t) : slice_l * h * w * sizeof(input_elem_t);
	stream.seekg(file_offset);

	if (padded) {
		int cnt;
		for (size_t i = slice_l; i <= slice_r; i++) {
			for (size_t j = i * (h + 2) * (w + 2) + w + 2 + 1; j < i * (h + 2) * (w + 2) + (h + 1) * (w + 2) - 1; j++) {
				cnt = 0;
				while (cnt++ < w) {
					input_elem_t v;
					stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
					data[j] = static_cast<float>(v);
					if (!duplicate_rec_local[data[j]]) { duplicate_rec_local[data[j]] = true; ascend_unique_arr_local.push_back(data[j]); }
					j++;
				}
				j++;
			}
		}
	}
	else {
		float v_;
		input_elem_t v;
		for (int i = slice_l; i <= slice_r; i++) {
			int j = 0;
			while (j < w) {
				stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
				data[i * h * w + j++] = static_cast<float>(v);
			}
			for (int k = 1; k < h - 1; k++)
				for (j = 0; j < w; j++) {
					if (j == 0 || j == w - 1) {
						stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
						data[i * h * w + k * w + j] = static_cast<float>(v);
					}
					else {
						stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
						v_ = static_cast<float>(v);
						data[i * h * w + k * w + j] = v_;
						if (!duplicate_rec_local[v_]) { duplicate_rec_local[v_] = true; ascend_unique_arr_local.push_back(v_); }
					}
				}
			j = 0;
			while (j < w) {
				stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
				data[i * h * w + (h - 1) * w + j++] = static_cast<float>(v);
			}
		}
	}
	std::sort(ascend_unique_arr_local.begin(), ascend_unique_arr_local.end());
	stream.close();
	return ascend_unique_arr_local;
}

template<typename input_elem_t>
std::vector<float> from_stream2D_thread_(std::string& filename, float*& data, int slice_l, int slice_r, int w, bool padded) {
	std::vector<float> ascend_unique_arr_local;
	std::unordered_map<float, bool> duplicate_rec_local;

	std::ifstream stream(filename.c_str(), std::ios::binary);
	if (!stream) { std::cout << "File open failed" << std::endl; exit(1); }
	size_t file_offset = (padded) ? (slice_l - 1) * w * sizeof(input_elem_t) : slice_l * w * sizeof(input_elem_t);
	stream.seekg(file_offset);

	if (padded) {
		for (size_t i = slice_l; i <= slice_r; i++) {
			for (size_t j = i * (w + 2) + 1; j < (i + 1) * (w + 2) - 1; j++) {
				input_elem_t v;
				stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
				data[j] = static_cast<float>(v);
				if (!duplicate_rec_local[data[j]]) { duplicate_rec_local[data[j]] = true; ascend_unique_arr_local.push_back(data[j]); }
			}
		}
	}
	else {
		float v_;
		input_elem_t v;
		for (size_t i = slice_l; i <= slice_r; i++) {
			stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
			data[i * w] = static_cast<float>(v);
			for (size_t j = 1; j < w - 1; j++) {
				stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
				v_ = static_cast<float>(v);
				data[i * w + j] = v_;
				if (!duplicate_rec_local[v_]) { duplicate_rec_local[v_] = true; ascend_unique_arr_local.push_back(v_); }
			}
			stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
			data[(i + 1) * w - 1] = static_cast<float>(v);
		}
	}
	std::sort(ascend_unique_arr_local.begin(), ascend_unique_arr_local.end());
	stream.close();
	return ascend_unique_arr_local;
}

template<typename input_elem_t>
float* from_stream_3Dfloat_mt_(
	std::string& filename, int& h, int& w, int& d, bool expand_by_one, bool pinned,
	int& binNum_global, int& binNum_max_local, const std::vector<std::pair<int, int>>& section,
	std::vector<float>& ascend_unique_arr, std::vector<std::vector<float>>& ascend_unique_arr_local, float& read_time) {
	/*
	* Converts all data type to float for GPU, because the coded access pattern is designed for float.
	* @filename: filename of the input
	* @h, w, d: height, width, and depth of the input
	* @expand_by_one: pad file by one pixel around it
	* Note: Huber's input 5x3x2: 5 for depth, 3 for height, 2 for width
	*/
	float* data;
	if (pinned)
		(expand_by_one) ? checkCudaErrors(cudaMallocHost((void**)&data, (h + 2) * (w + 2) * (d + 2) * sizeof(float))) :
		checkCudaErrors(cudaMallocHost((void**)&data, h * w * d * sizeof(float)));
	else
		(expand_by_one) ? data = (float*)malloc((h + 2) * (w + 2) * (d + 2) * sizeof(float)) : data = (float*)malloc(h * w * d * sizeof(float));

	size_t num_threads = std::thread::hardware_concurrency();
	ctpl::thread_pool tp(num_threads);
	ascend_unique_arr_local.resize(section.size());

	Stopwatch sw_read;
	for (size_t secidx = 0; secidx < section.size(); secidx++) {
		tp.push(
			[&, secidx = secidx](int) -> void {
				ascend_unique_arr_local[secidx] = from_stream3D_thread_<input_elem_t>(filename, data,
					section[secidx].first + 1, section[secidx].second - 1, h, w, expand_by_one);
			});
	}
	tp.stop(true);
	read_time = sw_read.lap() * 1000.0;

	binNum_max_local = 0;
	std::unordered_map<float, bool> duplicate_rec;
	float max_ = std::numeric_limits<float>::min(), min_ = std::numeric_limits<float>::max();
	for (size_t i = 0; i < section.size(); i++) {
		max_ = std::max(max_, ascend_unique_arr_local[i][ascend_unique_arr_local[i].size() - 1]);
		min_ = std::min(min_, ascend_unique_arr_local[i][0]);
		binNum_max_local = (ascend_unique_arr_local[i].size() > binNum_max_local) ? ascend_unique_arr_local[i].size() : binNum_max_local;
		for (size_t j = 0; j < ascend_unique_arr_local[i].size(); j++) {
			float cur = ascend_unique_arr_local[i][j];
			if (!duplicate_rec[cur]) { duplicate_rec[cur] = true; ascend_unique_arr.push_back(cur); }
		}
	}
	std::sort(ascend_unique_arr.begin(), ascend_unique_arr.end());
	binNum_global = ascend_unique_arr.size();

	if (expand_by_one) {
		for (size_t i = 0; i < (w + 2) * (h + 2); i++) data[i] = max_ + 1;
		for (size_t i = (d + 1) * (w + 2) * (h + 2); i < (d + 2) * (w + 2) * (h + 2); i++) data[i] = max_ + 1;
		for (size_t i = 1; i <= d; i++) for (size_t j = (w + 2) * (h + 2) * i; j < (w + 2) * (h + 2) * i + w + 2; j++) data[j] = max_ + 1;
		for (size_t i = 1; i <= d; i++) for (size_t j = (w + 2) * (h + 2) * i + (w + 2) * (h + 1); j < (w + 2) * (h + 2) * (i + 1); j++) data[j] = max_ + 1;
		for (size_t i = 1; i <= d; i++) for (size_t j = (w + 2) * (h + 2) * i + w + 2; j <= (w + 2) * (h + 2) * i + (w + 2) * h; j += (w + 2)) data[j] = max_ + 1;
		for (size_t i = 1; i <= d; i++) for (size_t j = (w + 2) * (h + 2) * i + 2 * (w + 2) - 1; j < (w + 2) * (h + 2) * i + (w + 2) * (h + 1); j += (w + 2)) data[j] = max_ + 1;
		h = h + 2;
		w = w + 2;
		d = d + 2;
	}
	else {
		float v_;
		input_elem_t v;
		std::ifstream stream(filename.c_str(), std::ios::binary);
		for (size_t i = 0; i < h * w; i++) {
			stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
			data[i] = static_cast<float>(v);
		}
		stream.seekg((d - 1) * h * w * sizeof(input_elem_t));
		for (size_t i = 0; i < h * w; i++) {
			stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
			data[h * w * (d - 1) + i] = static_cast<float>(v);
		}
		stream.close();
	}
	return data;
}

template<typename input_elem_t>
float* from_stream_2Dfloat_mt_(
	std::string& filename, int& h, int& w, bool expand_by_one, bool pinned,
	int& binNum_global, int& binNum_max_local, const std::vector<std::pair<int, int>>& section,
	std::vector<float>& ascend_unique_arr, std::vector<std::vector<float>>& ascend_unique_arr_local, float& read_time) {
	/*
	* Converts all data type to float for GPU, because the coded access pattern is designed for float.
	* @filename: filename of the input
	* @h, w: height and width of the input
	* @expand_by_one: pad file by one pixel around it
	*/
	float* data;
	if (pinned)
		(expand_by_one) ? checkCudaErrors(cudaMallocHost((void**)&data, (h + 2) * (w + 2) * sizeof(float)))
		: checkCudaErrors(cudaMallocHost((void**)&data, h * w * sizeof(float)));
	else
		(expand_by_one) ? data = (float*)malloc((h + 2) * (w + 2) * sizeof(float)) : data = (float*)malloc(h * w * sizeof(float));

	size_t num_threads = std::thread::hardware_concurrency();
	ctpl::thread_pool tp(num_threads);
	ascend_unique_arr_local.resize(section.size());

	Stopwatch sw_read;
	for (size_t secidx = 0; secidx < section.size(); secidx++) {
		tp.push(
			[&, secidx = secidx](int) -> void {
				ascend_unique_arr_local[secidx] = from_stream2D_thread_<input_elem_t>(filename, data,
					section[secidx].first + 1, section[secidx].second - 1, w, expand_by_one);
			});
	}
	tp.stop(true);
	read_time = sw_read.lap() * 1000.0;

	binNum_max_local = 0;
	std::unordered_map<float, bool> duplicate_rec;
	float max_ = std::numeric_limits<float>::min(), min_ = std::numeric_limits<float>::max();
	for (size_t i = 0; i < section.size(); i++) {
		max_ = std::max(max_, ascend_unique_arr_local[i][ascend_unique_arr_local[i].size() - 1]);
		min_ = std::min(min_, ascend_unique_arr_local[i][0]);
		binNum_max_local = (ascend_unique_arr_local[i].size() > binNum_max_local) ? ascend_unique_arr_local[i].size() : binNum_max_local;
		for (size_t j = 0; j < ascend_unique_arr_local[i].size(); j++) {
			float cur = ascend_unique_arr_local[i][j];
			if (!duplicate_rec[cur]) { duplicate_rec[cur] = true; ascend_unique_arr.push_back(cur); }
		}
	}
	std::sort(ascend_unique_arr.begin(), ascend_unique_arr.end());
	binNum_global = ascend_unique_arr.size();

	if (expand_by_one) {
		for (size_t i = 0; i < w + 2; i++) data[i] = max_ + 1;
		for (size_t i = (h + 1) * (w + 2); i < (h + 2) * (w + 2); i++) data[i] = max_ + 1;
		for (size_t i = w + 2; i <= h * (w + 2); i += (w + 2)) data[i] = max_ + 1;
		for (size_t i = 2 * (w + 2) - 1; i <= h * (w + 2) + w + 1; i += (w + 2))  data[i] = max_ + 1;
		h = h + 2;
		w = w + 2;
	}
	else {
		float v_;
		input_elem_t v;
		std::ifstream stream(filename.c_str(), std::ios::binary);
		for (size_t i = 0; i < w; i++) {
			stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
			data[i] = static_cast<float>(v);
		}
		stream.seekg((h - 1) * w * sizeof(input_elem_t));
		for (size_t i = 0; i < w; i++) {
			stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
			data[w * (h - 1) + i] = static_cast<float>(v);
		}
		stream.close();
	}
	return data;
}

template<typename input_elem_t>
std::vector<float> from_stream3D_thread_slice_(std::string& fileName, float*& data, int slice_base, int slice_end, std::pair<int, int> sec_cur, int h, int w, bool padded) {
	std::ifstream stream(fileName.c_str(), std::ios::binary);
	if (!stream) { std::cout << "File open failed" << std::endl; exit(1); }
	int slice_l = sec_cur.first;
	int slice_r = sec_cur.second;

	int slice_offset;
	float v_f;
	input_elem_t v;
	unsigned long long file_offset = (padded) ? (slice_l - 1) * sizeof(input_elem_t) : slice_l * sizeof(input_elem_t);
	file_offset *= h;
	file_offset *= w;
	stream.seekg(file_offset);

	std::vector<float> ascend_unique_arr_local;
	std::unordered_map<float, bool> duplicate_rec_local;

	if (padded) {
		for (size_t sec_idx = slice_l; sec_idx <= slice_r; sec_idx++) {
			slice_offset = (sec_idx - slice_base) * (h + 2) * (w + 2);
			for (size_t i = 0; i < (h + 2); i++) {
				if (i == 0 || i == h + 1) {
					for (size_t j = 0; j < w + 2; j++)
						data[slice_offset + i * (w + 2) + j] = std::numeric_limits<float>::max();
				}
				else {
					data[slice_offset + i * (w + 2)] = data[slice_offset + i * (w + 2) + w + 1] = std::numeric_limits<float>::max();
					for (size_t j = 1; j < w + 1; j++) {
						stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
						v_f = static_cast<float>(v);
						data[slice_offset + i * (w + 2) + j] = v_f;
						if (!duplicate_rec_local[v_f] && sec_idx != slice_base && sec_idx != slice_end) {
							duplicate_rec_local[v_f] = true; ascend_unique_arr_local.push_back(v_f);
						}
					}
				}
			}
		}
	}
	else {
		for (size_t sec_idx = slice_l; sec_idx <= slice_r; sec_idx++) {
			slice_offset = (sec_idx - slice_base) * h * w;
			for (size_t i = 0; i < w; i++) {
				stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
				data[slice_offset + i] = static_cast<float>(v);
			}
			for (size_t i = 1; i < h - 1; i++) {
				stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
				data[slice_offset + i * w] = static_cast<float>(v);
				for (size_t j = 1; j < w - 1; j++) {
					stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
					v_f = static_cast<float>(v);
					data[slice_offset + i * w + j] = v_f;
					if (!duplicate_rec_local[v_f] && sec_idx != slice_base && sec_idx != slice_end) {
						duplicate_rec_local[v_f] = true; ascend_unique_arr_local.push_back(v_f);
					}
				}
				stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
				data[slice_offset + i * w + w - 1] = static_cast<float>(v);
			}
			for (size_t i = 0; i < w; i++) {
				stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
				data[slice_offset + (h - 1) * w + i] = static_cast<float>(v);
			}
		}
	}
	stream.close();
	return ascend_unique_arr_local;
}

template<typename input_elem_t>
std::vector<float> from_stream2D_thread_row_(std::string& fileName, float*& data, int slice_base, int slice_end, std::pair<int, int> sec_cur, int w, bool padded) {
	std::ifstream stream(fileName.c_str(), std::ios::binary);
	if (!stream) { std::cout << "File open failed" << std::endl; exit(1); }
	int slice_l = sec_cur.first;
	int slice_r = sec_cur.second;

	int slice_offset;
	float v_f;
	input_elem_t v;
	size_t file_offset = (padded) ? (slice_l - 1) * w * sizeof(input_elem_t) : slice_l * w * sizeof(input_elem_t);
	stream.seekg(file_offset);
	std::vector<float> ascend_unique_arr_local;
	std::unordered_map<float, bool> duplicate_rec_local;

	if (padded) {
		for (size_t sec_idx = slice_l; sec_idx <= slice_r; sec_idx++) {
			slice_offset = (sec_idx - slice_base) * (w + 2);
			data[slice_offset] = data[slice_offset + w + 1] = std::numeric_limits<float>::max();
			for (size_t i = 1; i < w + 1; i++) {
				stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
				v_f = static_cast<float>(v);
				data[slice_offset + i] = v_f;
				if (!duplicate_rec_local[v_f] && sec_idx != slice_base && sec_idx != slice_end) 
				{ duplicate_rec_local[v_f] = true; ascend_unique_arr_local.push_back(v_f); }
			}
		}
	}
	else {
		for (size_t sec_idx = slice_l; sec_idx <= slice_r; sec_idx++) {
			slice_offset = (sec_idx - slice_base) * w;
			stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
			data[slice_offset] = static_cast<float>(v);
			for (size_t i = 1; i < w - 1; i++) {
				stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
				v_f = static_cast<float>(v);
				data[slice_offset + i] = v_f;
				if (!duplicate_rec_local[v_f] && sec_idx != slice_base && sec_idx != slice_end)
				{ duplicate_rec_local[v_f] = true; ascend_unique_arr_local.push_back(v_f); }
			}
			stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
			data[slice_offset + w - 1] = static_cast<float>(v);
		}
	}
	std::sort(ascend_unique_arr_local.begin(), ascend_unique_arr_local.end());
	stream.close();
	return ascend_unique_arr_local;
}

template<typename input_elem_t>
std::vector<float> from_stream_3Dfloat_mt_slice_(
	std::string& fileName, int core_num, float*& data, int h, int w, int d, int slice_l, int slice_r, bool expand_by_one, double& time) {
	double w1 = get_wall_time();
	if (expand_by_one) { h -= 2; w -= 2; d -= 2; }
	size_t num_threads = std::thread::hardware_concurrency();
	ctpl::thread_pool tp(num_threads);

	bool slice_l_changed = false, slice_r_changed = false;
	if (expand_by_one && slice_l == 0) { dummy3D_thread_(data, 0, 0, h, w); slice_l++; slice_l_changed = true; }
	if (expand_by_one && slice_r == d + 1) { dummy3D_thread_(data, slice_l, slice_r, h, w); slice_r--; slice_r_changed = true; }
	int start, end;
	std::vector<std::pair<int, int>> sec_;
	core_num = (core_num > slice_r - slice_l + 1) ? slice_r - slice_l + 1 : core_num;
	int sub_chunk_size = (slice_r - slice_l + 1) / core_num;
	for (int i = 0; i < core_num; i++) {
		start = i * sub_chunk_size + slice_l;
		end = (i == core_num - 1) ? slice_r : (i + 1) * sub_chunk_size - 1 + slice_l;
		sec_.push_back(std::make_pair(start, end));
	}
	if (expand_by_one && slice_l_changed) slice_l--;
	if (expand_by_one && slice_r_changed) slice_r++;
	std::vector<std::vector<float>> ascend_unique_arr_local_slice(core_num);

	for (size_t secidx = 0; secidx < core_num; secidx++) {
		tp.push(
			[&, secidx = secidx](int) -> void {
				ascend_unique_arr_local_slice[secidx] = from_stream3D_thread_slice_<input_elem_t>(fileName, data,
					slice_l, slice_r, sec_[secidx], h, w, expand_by_one);
			});
	}
	tp.stop(true);

	std::vector<float> ascend_unique_arr_local;
	std::unordered_map<float, bool> duplicate_rec;
	for (size_t i = 0; i < core_num; i++) {
		for (size_t j = 0; j < ascend_unique_arr_local_slice[i].size(); j++) {
			float cur = ascend_unique_arr_local_slice[i][j];
			if (!duplicate_rec[cur]) { duplicate_rec[cur] = true; ascend_unique_arr_local.push_back(cur); }
		}
	}
	std::sort(ascend_unique_arr_local.begin(), ascend_unique_arr_local.end());
	double w2 = get_wall_time();
	time += (w2 * 1000 - w1 * 1000);
	return ascend_unique_arr_local;
}

template<typename input_elem_t>
std::vector<float> from_stream_2Dfloat_mt_row_(
	std::string& fileName, int core_num, float*& data, int h, int w, int slice_l, int slice_r, bool expand_by_one, double& time) {
	double w1 = get_wall_time();
	if (expand_by_one) { h -= 2; w -= 2; }
	size_t num_threads = std::thread::hardware_concurrency();
	ctpl::thread_pool tp(num_threads);

	bool slice_l_changed = false, slice_r_changed = false;
	if (expand_by_one && slice_l == 0) { dummy2D_thread_(data, 0, 0, w); slice_l_changed = true; }
	if (expand_by_one && slice_r == h + 1) { dummy2D_thread_(data, slice_l, slice_r, w); slice_r_changed = true; }
	if (expand_by_one && slice_l == 0) slice_l++;
	if (expand_by_one && slice_r == h + 1) slice_r--;
	int start, end;
	std::vector<std::pair<int, int>> sec_;
	core_num = (core_num > slice_r - slice_l + 1) ? slice_r - slice_l + 1 : core_num;
	int sub_chunk_size = (slice_r - slice_l + 1) / core_num;
	for (int i = 0; i < core_num; i++) {
		start = i * sub_chunk_size + slice_l;
		end = (i == core_num - 1) ? slice_r : (i + 1) * sub_chunk_size - 1 + slice_l;
		sec_.push_back(std::make_pair(start, end));
	}
	if (expand_by_one && slice_l_changed) slice_l--;
	if (expand_by_one && slice_r_changed) slice_r++;
	std::vector<std::vector<float>> ascend_unique_arr_local_slice(core_num);

	Stopwatch sw_read;
	for (size_t secidx = 0; secidx < core_num; secidx++) {
		tp.push(
			[&, secidx = secidx](int) -> void {
				ascend_unique_arr_local_slice[secidx] =
					from_stream2D_thread_row_<input_elem_t>(fileName, data, slice_l, slice_r, sec_[secidx], w, expand_by_one);
			});
	}
	tp.stop(true);

	std::vector<float> ascend_unique_arr_local;
	std::unordered_map<float, bool> duplicate_rec;
	for (size_t i = 0; i < core_num; i++) {
		for (size_t j = 0; j < ascend_unique_arr_local_slice[i].size(); j++) {
			float cur = ascend_unique_arr_local_slice[i][j];
			if (!duplicate_rec[cur]) { duplicate_rec[cur] = true; ascend_unique_arr_local.push_back(cur); }
		}
	}
	std::sort(ascend_unique_arr_local.begin(), ascend_unique_arr_local.end());
	double w2 = get_wall_time();
	time += (w2 * 1000 - w1 * 1000);
	return ascend_unique_arr_local;
}

template<typename input_elem_t>
void write_txt_ECC(const std::string filename_o, std::vector<input_elem_t>& ascend_unique_arr, int* VCEC_host) {
	std::fstream out(filename_o, std::fstream::out);
	long long int accum = 0;
	for (unsigned int i = 0; i < ascend_unique_arr.size(); i++) {
		if (VCEC_host[i] == 0) continue;
		out << ascend_unique_arr[i] << " " << accum + VCEC_host[i] << std::endl;
		accum += VCEC_host[i];
	}
	out.close();
}