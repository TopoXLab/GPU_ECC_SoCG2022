#pragma once

#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "utility.h"
#include "ctpl_stl.h"

template<typename input_elem_t>
void init_histogram_1D_(input_elem_t* data1D, const int size) {
	for (size_t i = 0; i < size; i++) data1D[i] = (input_elem_t)0;
}

template<typename input_elem_t>
void init_histogram_2D_(input_elem_t** data2D, const int num, const int size) {
	for (int i = 0; i < num; i++) init_histogram_1D_<input_elem_t>(data2D[i], size);
}

template<typename input_elem_t>
input_elem_t* allocate_device_memory1D_(const int size) {
	input_elem_t* data1D;
	checkCudaErrors(cudaMalloc((void**)&data1D, size * sizeof(input_elem_t)));
	return data1D;
}

template<typename input_elem_t>
input_elem_t** allocate_device_memory2D_(const int num, const int size) {
	input_elem_t** data2D = new input_elem_t * [num];
	for (int i = 0; i < num; i++) data2D[i] = allocate_device_memory1D_<input_elem_t>(size);
	return data2D;
}

template<typename input_elem_t>
input_elem_t* allocate_host_memory1D_(const int size) {
	input_elem_t* data1D = (input_elem_t*)malloc(size * sizeof(input_elem_t));
	return data1D;
}

template<typename input_elem_t>
input_elem_t** allocate_host_memory2D_(const int num, const int size) {
	input_elem_t** data2D = new input_elem_t * [num];
	for (int i = 0; i < num; i++) data2D[i] = allocate_host_memory1D_<input_elem_t>(size);
	return data2D;
}

template<typename input_elem_t>
input_elem_t* allocate_host_memory1D_pinned_(const int size) {
	input_elem_t* data1D;
	checkCudaErrors(cudaMallocHost((void**)&data1D, size * sizeof(input_elem_t)));
	return data1D;
}

template<typename input_elem_t>
input_elem_t** allocate_host_memory2D_pinned_(const int num, const int size) {
	input_elem_t** data2D = new input_elem_t * [num];
	for (int i = 0; i < num; i++) data2D[i] = allocate_host_memory1D_pinned_<input_elem_t>(size);
	return data2D;
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
std::vector<float> from_stream3D_thread_slice_(const std::string& fileName, float*& data, int slice_base, int slice_end, std::pair<int, int> sec_cur, int h, int w, bool padded) {
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
std::vector<float> from_stream2D_thread_row_(const std::string& fileName, float*& data, int slice_base, int slice_end, std::pair<int, int> sec_cur, int w, bool padded) {
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
	const std::string& fileName, int core_num, float*& data, int h, int w, int d, int slice_l, int slice_r, bool expand_by_one, double& time) {
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
	const std::string& fileName, int core_num, float*& data, int h, int w, int slice_l, int slice_r, bool expand_by_one, double& time) {
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

template <typename input_elem_t>
std::vector<float> from_array3D_thread_slice_(
	const input_elem_t* arr, 
	float*&				data,
	int					slice_base, 
	int					slice_end,
	std::pair<int, int> sec_cur,
	int					h, 
	int					w, 
	bool				padded
)
{
	int slice_l = sec_cur.first;
	int slice_r = sec_cur.second;

	std::vector<float> ascend_unique_arr_local;
	std::unordered_map<float, bool> duplicate_rec_local;

	if (padded) {
		for (int sec_idx = slice_l; sec_idx <= slice_r; ++sec_idx) {
			size_t dest_slice_offset = static_cast<size_t>(sec_idx - slice_base) * (h + 2) * (w + 2);
			int arr_slice_index = sec_idx - 1;
			for (int i = 0; i < h + 2; ++i) {
				if (i == 0 || i == h + 1) {
					for (int j = 0; j < w + 2; ++j) {
						data[dest_slice_offset + static_cast<size_t>(i) * (w + 2) + j] =
							std::numeric_limits<float>::max();
					}
				}
				else {
					data[dest_slice_offset + static_cast<size_t>(i) * (w + 2)] =
						std::numeric_limits<float>::max();
					data[dest_slice_offset + static_cast<size_t>(i) * (w + 2) + (w + 1)] =
						std::numeric_limits<float>::max();
					size_t arr_row_offset = static_cast<size_t>(arr_slice_index) * h * w + static_cast<size_t>(i - 1) * w;
					for (int j = 1; j < w + 1; ++j) {
						input_elem_t v = arr[arr_row_offset + (j - 1)];
						float v_f = static_cast<float>(v);
						data[dest_slice_offset + static_cast<size_t>(i) * (w + 2) + j] = v_f;
						if (!duplicate_rec_local[v_f] && sec_idx != slice_base && sec_idx != slice_end) {
							duplicate_rec_local[v_f] = true;
							ascend_unique_arr_local.push_back(v_f);
						}
					}
				}
			}
		}
	}
	else {
		for (int sec_idx = slice_l; sec_idx <= slice_r; ++sec_idx) {
			size_t dest_slice_offset = static_cast<size_t>(sec_idx - slice_base) * h * w;
			size_t arr_slice_offset = static_cast<size_t>(sec_idx) * h * w;
			for (int j = 0; j < w; ++j) {
				float v_f = static_cast<float>(arr[arr_slice_offset + j]);
				data[dest_slice_offset + j] = v_f;
			}
			for (int i = 1; i < h - 1; ++i) {
				float v_f = static_cast<float>(arr[arr_slice_offset + static_cast<size_t>(i) * w]);
				data[dest_slice_offset + static_cast<size_t>(i) * w] = v_f;
				for (int j = 1; j < w - 1; ++j) {
					input_elem_t v = arr[arr_slice_offset + static_cast<size_t>(i) * w + j];
					float v_f_inner = static_cast<float>(v);
					data[dest_slice_offset + static_cast<size_t>(i) * w + j] = v_f_inner;
					if (!duplicate_rec_local[v_f_inner] && sec_idx != slice_base && sec_idx != slice_end) {
						duplicate_rec_local[v_f_inner] = true;
						ascend_unique_arr_local.push_back(v_f_inner);
					}
				}
				v_f = static_cast<float>(arr[arr_slice_offset + static_cast<size_t>(i) * w + (w - 1)]);
				data[dest_slice_offset + static_cast<size_t>(i) * w + (w - 1)] = v_f;
			}
			size_t base_offset = dest_slice_offset + static_cast<size_t>(h - 1) * w;
			size_t src_offset = arr_slice_offset + static_cast<size_t>(h - 1) * w;
			for (int j = 0; j < w; ++j) {
				float v_f = static_cast<float>(arr[src_offset + j]);
				data[base_offset + j] = v_f;
			}
		}
	}
	std::sort(ascend_unique_arr_local.begin(), ascend_unique_arr_local.end());
	return ascend_unique_arr_local;
}

template <typename input_elem_t>
std::vector<float> from_array2D_thread_row_(
	const input_elem_t*		arr, 
	float*&					data,
	int						slice_base, 
	int						slice_end,
	std::pair<int, int>		sec_cur,
	int						w, 
	bool					padded
) 
{
	int slice_l = sec_cur.first;
	int slice_r = sec_cur.second;

	std::vector<float> ascend_unique_arr_local;
	std::unordered_map<float, bool> duplicate_rec_local;

	if (padded) {
		for (int sec_idx = slice_l; sec_idx <= slice_r; ++sec_idx) {
			size_t dest_row_offset = static_cast<size_t>(sec_idx - slice_base) * (w + 2);
			data[dest_row_offset] = std::numeric_limits<float>::max();
			data[dest_row_offset + (w + 1)] = std::numeric_limits<float>::max();
			size_t arr_row_offset = static_cast<size_t>(sec_idx - 1) * w;
			for (int i = 1; i < w + 1; ++i) {
				input_elem_t v = arr[arr_row_offset + (i - 1)];
				float v_f = static_cast<float>(v);
				data[dest_row_offset + i] = v_f;
				if (!duplicate_rec_local[v_f] && sec_idx != slice_base && sec_idx != slice_end) {
					duplicate_rec_local[v_f] = true;
					ascend_unique_arr_local.push_back(v_f);
				}
			}
		}
	}
	else {
		for (int sec_idx = slice_l; sec_idx <= slice_r; ++sec_idx) {
			size_t dest_row_offset = static_cast<size_t>(sec_idx - slice_base) * w;
			size_t arr_row_offset = static_cast<size_t>(sec_idx) * w;
			float v_f = static_cast<float>(arr[arr_row_offset]);
			data[dest_row_offset] = v_f;
			for (int i = 1; i < w - 1; ++i) {
				input_elem_t v = arr[arr_row_offset + i];
				float v_f_inner = static_cast<float>(v);
				data[dest_row_offset + i] = v_f_inner;
				if (!duplicate_rec_local[v_f_inner] && sec_idx != slice_base && sec_idx != slice_end) {
					duplicate_rec_local[v_f_inner] = true;
					ascend_unique_arr_local.push_back(v_f_inner);
				}
			}
			v_f = static_cast<float>(arr[arr_row_offset + (w - 1)]);
			data[dest_row_offset + (w - 1)] = v_f;
		}
	}
	std::sort(ascend_unique_arr_local.begin(), ascend_unique_arr_local.end());
	return ascend_unique_arr_local;
}

template<typename input_elem_t>
std::vector<float> from_array_3Dfloat_mt_slice_(
	const input_elem_t* arr, int core_num, float*& data, int h, int w, int d, int slice_l, int slice_r, bool expand_by_one, double& time) {
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
				ascend_unique_arr_local_slice[secidx] = from_array3D_thread_slice_<input_elem_t>(arr, data,
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
std::vector<float> from_array_2Dfloat_mt_row_(
	const input_elem_t* arr, int core_num, float*& data, int h, int w, int slice_l, int slice_r, bool expand_by_one, double& time) {
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
					from_array2D_thread_row_<input_elem_t>(arr, data, slice_l, slice_r, sec_[secidx], w, expand_by_one);
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
std::vector<float> return_results(const std::vector<input_elem_t>& ascend_unique_arr, const int* VCEC_host)
{
	std::vector<float> out;
	out.reserve(ascend_unique_arr.size() * 2);
	std::int64_t accum = 0;

	for (size_t i = 0; i < ascend_unique_arr.size(); ++i) {
		const int v = VCEC_host[i];
		if (v == 0) continue;

		accum += static_cast<std::int64_t>(v);
		out.push_back(static_cast<float>(ascend_unique_arr[i]));
		out.push_back(static_cast<float>(accum));
	}

	return out;
}

template <typename input_elem_t>
void read_3D_file_to_array(const std::string& fileName, float*& arr, int h, int w, int d) {
	std::ifstream stream(fileName.c_str(), std::ios::binary);
	if (!stream) {
		throw std::runtime_error("File open failed: " + fileName);
	}
	size_t total = static_cast<size_t>(d) * h * w;
	arr = new float[total];
	for (size_t idx = 0; idx < total; ++idx) {
		input_elem_t v;
		stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
		arr[idx] = static_cast<float>(v);
	}
	stream.close();
}

template <typename input_elem_t>
void read_2D_file_to_array(const std::string& fileName, float*& arr, int h, int w) {
	std::ifstream stream(fileName.c_str(), std::ios::binary);
	if (!stream) {
		throw std::runtime_error("File open failed: " + fileName);
	}
	size_t total = static_cast<size_t>(h) * w;
	arr = new float[total];
	for (size_t idx = 0; idx < total; ++idx) {
		input_elem_t v;
		stream.read(reinterpret_cast<char*>(&v), sizeof(input_elem_t));
		arr[idx] = static_cast<float>(v);
	}
	stream.close();
}