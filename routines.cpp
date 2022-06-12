#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <numeric>

#include "kernel.h"
#include "helper_cuda.h"

#include "template.h"
#include "utility.h"
#include "ctpl_stl.h"

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

#include "routines.h"

void helper_() {
	std::cout << "Invalid arguments" << std::endl;
	std::cout << "============================================================" << std::endl;
	std::cout << "Usage: ECC_GPU.exe [mode] [input] [output] [height] [width] [depth]" << std::endl;
	std::cout << "--mode:   b for batch mode, s for single mode" << std::endl;
	std::cout << "--input:  directory for batch mode, file for single mode" << std::endl;
	std::cout << "--output: directory for batch mode, file for single mode" << std::endl;
	std::cout << "--height: height of the input file" << std::endl;
	std::cout << "--width:  width of the input file" << std::endl;
	std::cout << "--depth:  depth of the input file, set to 0 in case of 2D file" << std::endl;
	std::cout << "============================================================" << std::endl;
	exit(1);
}

std::vector<double> ECC_(
	std::string& fileName,
	std::string& fileNameo,
	int h, int w, int d,
	bool pad,
	bool async,
	bool mt,
	bool timing,
	bool verb,
	bool output
) {
	int imageH_ = h;
	int imageW_ = w;
	int imageD_ = d;
	bool pad_by_one = pad;
	bool async_mode = async;
	bool manual_timing = timing;
	bool mt_read = mt;
	bool verbose = verb;
	bool data_3D = (d > 0);

	const int pre_binNum = 1024;
	int chunk_num = decide_chunk_num(imageH_, imageW_, imageD_);
	int engine_num = 3;
	engine_num = (engine_num > chunk_num) ? chunk_num : engine_num;

	// ------ Divide into chunks
	std::vector<std::pair<int, int>> section = (data_3D) ? return_chunk_index(imageD_, chunk_num, pad_by_one) : return_chunk_index(imageH_, chunk_num, pad_by_one);

	// ------ Timing
	std::vector<double> timings(2, 0);
	double w1 = get_wall_time();
	// ------- Declare data structures -------
	float** input_host = allocate_input_memory<float>(engine_num, async_mode, pad_by_one, imageH_, imageW_, imageD_, section[chunk_num - 1].second - section[chunk_num - 1].first + 1);
	cudaArray** tex_dataSrc_ = (data_3D) ? allocate_cudaArray_array_(engine_num, imageH_, imageW_, section[chunk_num - 1].second - section[chunk_num - 1].first + 1) :
		allocate_cudaArray_array_(engine_num, section[chunk_num - 1].second - section[chunk_num - 1].first + 1, imageW_);
	cudaTextureObject_t* texSrc_ = create_cudaTextureObject_array_(tex_dataSrc_, engine_num);
	(data_3D) ? setBasicParams_const_(section, imageW_, imageH_) : setBasicParams_const_(section, imageW_);
	// ------- VCEC & bin array -------
	int binNum_local, binNum_global;
	std::vector<std::vector<int>> VCEC_local_rec;
	std::vector<std::vector<float>> ascend_unique_arr_local_rec;
	int** VCEC_host_partial_ = (async_mode) ? allocate_host_memory2D_pinned_<int>(engine_num, pre_binNum) : allocate_host_memory2D_<int>(engine_num, pre_binNum);
	float** ascend_unique_arr_local_host_ = (async_mode) ? allocate_host_memory2D_pinned_<float>(engine_num, pre_binNum) : allocate_host_memory2D_<float>(engine_num, pre_binNum);

	int** VCEC_device_partial_ = allocate_device_memory2D_<int>(engine_num, pre_binNum);
	float** ascend_unique_arr_local_device_ = allocate_device_memory2D_<float>(engine_num, pre_binNum);

	// ----- Allocate and initialize an array of stream handles
	cudaEvent_t* events;
	cudaStream_t* streams;

	// ----- Histogram Initialization -----
	if (async_mode) {
		events = create_cudaEvent_array(engine_num * 3);
		streams = create_cudaStream_array(engine_num);
	}

	// ------ Memory transfer and kernel launch
	for (int i = 0; i < chunk_num; i++) {
		int engidx = i % engine_num;
		int chunkH = section[i].second - section[i].first + 1;
		if (async_mode) {
			checkCudaErrors(cudaEventSynchronize(events[engine_num * 2 + engidx]));
			std::vector<float> ascend_unique_arr_local = (data_3D) ?
				from_stream_3Dfloat_mt_slice_<float>(fileName, 8, input_host[engidx], imageH_, imageW_, imageD_, section[i].first, section[i].second, pad_by_one, timings[0]) :
				from_stream_2Dfloat_mt_row_<float>(fileName, 8, input_host[engidx], imageH_, imageW_, section[i].first, section[i].second, pad_by_one, timings[0]);
			binNum_local = int(ascend_unique_arr_local.size());
			ascend_unique_arr_local_rec.push_back(ascend_unique_arr_local);
			if (binNum_local > pre_binNum) std::cout << "WARNING: increase bin number or chunk number" << std::endl;
			if (data_3D) {
				cudaMemcpy3DParms p = setup_3DCopy_params(tex_dataSrc_[engidx], input_host[engidx], imageH_, imageW_, chunkH, 0);
				checkCudaErrors(cudaMemcpy3DAsync(&p, streams[engidx]));
			}
			else checkCudaErrors(cudaMemcpyToArrayAsync(tex_dataSrc_[engidx], 0, 0, &input_host[engidx][0], chunkH * imageW_ * sizeof(float), cudaMemcpyHostToDevice, streams[engidx]));
			checkCudaErrors(cudaEventRecord(events[engine_num * 2 + engidx], streams[engidx]));
			if (i >= engine_num) checkCudaErrors(cudaEventSynchronize(events[engine_num + engidx]));

			std::copy(ascend_unique_arr_local.begin(), ascend_unique_arr_local.end(), ascend_unique_arr_local_host_[engidx]);
			checkCudaErrors(cudaMemcpyAsync(ascend_unique_arr_local_device_[engidx], ascend_unique_arr_local_host_[engidx], binNum_local * sizeof(float), cudaMemcpyHostToDevice, streams[engidx]));
			checkCudaErrors(cudaEventRecord(events[engine_num + engidx], streams[engidx]));
			if (i >= engine_num) {
				checkCudaErrors(cudaEventSynchronize(events[engidx]));
				std::vector<int> VCEC_local_;
				VCEC_local_.assign(VCEC_host_partial_[engidx], VCEC_host_partial_[engidx] + ascend_unique_arr_local_rec[i - engine_num].size());
				VCEC_local_rec.push_back(VCEC_local_);
			}
			init_VCEC_device(binNum_local, VCEC_device_partial_[engidx], &streams[engidx]);
			if (data_3D) computeECC_3D(imageH_ - 2, imageW_ - 2, chunkH - 2, binNum_local, texSrc_[engidx], VCEC_device_partial_[engidx], ascend_unique_arr_local_device_[engidx], i == (chunk_num - 1), &streams[engidx]);
			else computeECC(chunkH - 2, imageW_ - 2, binNum_local, texSrc_[engidx], VCEC_device_partial_[engidx], ascend_unique_arr_local_device_[engidx], i == (chunk_num - 1), &streams[engidx]);
			checkCudaErrors(cudaMemcpyAsync(VCEC_host_partial_[engidx], VCEC_device_partial_[engidx], binNum_local * sizeof(int), cudaMemcpyDeviceToHost, streams[engidx]));
			checkCudaErrors(cudaEventRecord(events[engidx], streams[engidx]));
		}
		else {
			std::vector<float> ascend_unique_arr_local = (data_3D) ?
				from_stream_3Dfloat_mt_slice_<float>(fileName, 8, input_host[engidx], imageH_, imageW_, imageD_, section[i].first, section[i].second, pad_by_one, timings[0]) :
				from_stream_2Dfloat_mt_row_<float>(fileName, 8, input_host[engidx], imageH_, imageW_, section[i].first, section[i].second, pad_by_one, timings[0]);
			binNum_local = int(ascend_unique_arr_local.size());
			ascend_unique_arr_local_rec.push_back(ascend_unique_arr_local);
			if (binNum_local > pre_binNum) std::cout << "WARNING: increase bin number" << std::endl;
			if (data_3D) { cudaMemcpy3DParms p = setup_3DCopy_params(tex_dataSrc_[engidx], input_host[engidx], imageH_, imageW_, chunkH, 0); checkCudaErrors(cudaMemcpy3D(&p)); }
			else checkCudaErrors(cudaMemcpyToArray(tex_dataSrc_[engidx], 0, 0, &input_host[engidx][0], chunkH * imageW_ * sizeof(float), cudaMemcpyHostToDevice));
			std::copy(ascend_unique_arr_local.begin(), ascend_unique_arr_local.end(), ascend_unique_arr_local_host_[engidx]);
			checkCudaErrors(cudaMemcpy(ascend_unique_arr_local_device_[engidx], ascend_unique_arr_local_host_[engidx], binNum_local * sizeof(float), cudaMemcpyHostToDevice));
			init_histogram_1D_<int>(VCEC_host_partial_[engidx], binNum_local);
			checkCudaErrors(cudaMemcpy(VCEC_device_partial_[engidx], VCEC_host_partial_[engidx], binNum_local * sizeof(int), cudaMemcpyHostToDevice));
			if (data_3D) computeECC_3D(imageH_ - 2, imageW_ - 2, chunkH - 2, binNum_local, texSrc_[engidx], VCEC_device_partial_[engidx], ascend_unique_arr_local_device_[engidx], i == (chunk_num - 1), nullptr);
			else computeECC(chunkH - 2, imageW_ - 2, binNum_local, texSrc_[engidx], VCEC_device_partial_[engidx], ascend_unique_arr_local_device_[engidx], i == (chunk_num - 1), nullptr);
			checkCudaErrors(cudaMemcpy(VCEC_host_partial_[engidx], VCEC_device_partial_[engidx], binNum_local * sizeof(int), cudaMemcpyDeviceToHost));
			std::vector<int> VCEC_local_;
			VCEC_local_.assign(VCEC_host_partial_[engidx], VCEC_host_partial_[engidx] + binNum_local);
			VCEC_local_rec.push_back(VCEC_local_);
		}
	}
	// ------- Process results --------
	std::vector<float> ascend_unique_arr = accumulate_ascend_unique_arr(ascend_unique_arr_local_rec);
	binNum_global = ascend_unique_arr.size();
	int* VCEC_host_ = allocate_host_memory1D_<int>(binNum_global);
	init_histogram_1D_<int>(VCEC_host_, binNum_global);
	if (async_mode) for (int i = chunk_num; i < chunk_num + engine_num; i++) {
		checkCudaErrors(cudaEventSynchronize(events[i % engine_num]));
		std::vector<int> VCEC_local_;
		VCEC_local_.assign(VCEC_host_partial_[i % engine_num], VCEC_host_partial_[i % engine_num] + ascend_unique_arr_local_rec[i - engine_num].size());
		VCEC_local_rec.push_back(VCEC_local_);
	}
	for (size_t i = 0; i < VCEC_local_rec.size(); i++)
		accumulate_VCEC_host_various_binNum_(VCEC_host_, &VCEC_local_rec[i][0], ascend_unique_arr, ascend_unique_arr_local_rec[i]);

	checkCudaErrors(cudaDeviceSynchronize());
	double w2 = get_wall_time();
	timings[1] = (w2 * 1000 - w1 * 1000);
	if (verbose) print_ECC_results(ascend_unique_arr, VCEC_host_);
	if (output) write_txt_ECC<float>(fileNameo, ascend_unique_arr, VCEC_host_);
	if (manual_timing) {
		printf("Number of unique values: %d\n", binNum_global);
		printf("Data read time: %f msecs;\n", timings[0]);
		printf("GPU total time: %f msecs;\n", timings[1]);
	}

	// ------- Free memory --------
	section.clear();
	ascend_unique_arr.clear();
	free_vector2D_<float>(ascend_unique_arr_local_rec);
	(async_mode) ? cudaFreeHost(input_host) : free(input_host);
	(async_mode) ? free_host_memory2D_pinned_<float>(input_host, engine_num) : free_host_memory2D_<float>(input_host, engine_num);
	free(VCEC_host_);
	(async_mode) ? free_host_memory2D_pinned_<int>(VCEC_host_partial_, engine_num) : free_host_memory2D_<int>(VCEC_host_partial_, engine_num);
	(async_mode) ? free_host_memory2D_pinned_<float>(ascend_unique_arr_local_host_, engine_num) : free_host_memory2D_<float>(ascend_unique_arr_local_host_, engine_num);

	free_device_memory2D_<int>(VCEC_device_partial_, engine_num);
	free_cudaArray_array(tex_dataSrc_, engine_num);
	free_device_memory2D_<float>(ascend_unique_arr_local_device_, engine_num);
	if (async_mode) free_cudaEvent_array(events, engine_num * 3);
	if (async_mode) free_cudaStream_array(streams, engine_num);

	checkCudaErrors(cudaDeviceReset());
	return timings;
}

std::vector<double> ECC_folder(
	std::string& path,
	std::string& patho,
	int h, int w, int d,
	bool pad,
	bool async,
	bool mt,
	bool timing,
	bool verb,
	bool output
) {
	int imageH_ = h;
	int imageW_ = w;
	int imageD_ = d;
	bool pad_by_one = pad;
	bool async_mode = async;
	bool manual_timing = timing;
	bool mt_read = mt;
	bool verbose = verb;
	bool data_3D = (d > 0);

	std::vector<std::string> fileNames = fileNames_from_folder(path);
	std::vector<std::string> fileNameso = compose_outfileNames_from_folder(path, patho);

	const int pre_binNum = 1024;
	int chunk_num = decide_chunk_num(imageH_, imageW_, imageD_);
	int engine_num = 3;
	engine_num = (engine_num > chunk_num) ? chunk_num : engine_num;

	// ------ Divide into chunks
	std::vector<std::pair<int, int>> section = (data_3D) ? return_chunk_index(imageD_, chunk_num, pad_by_one) : return_chunk_index(imageH_, chunk_num, pad_by_one);

	// ------ Timing
	std::vector<double> timings(2, 0);
	double w1 = get_wall_time();
	// ------- Declare data structures -------
	float** input_host = allocate_input_memory<float>(engine_num, async_mode, pad_by_one, imageH_, imageW_, imageD_, section[chunk_num - 1].second - section[chunk_num - 1].first + 1);
	cudaArray** tex_dataSrc_ = (data_3D) ? allocate_cudaArray_array_(engine_num, imageH_, imageW_, section[chunk_num - 1].second - section[chunk_num - 1].first + 1) :
		allocate_cudaArray_array_(engine_num, section[chunk_num - 1].second - section[chunk_num - 1].first + 1, imageW_);
	cudaTextureObject_t* texSrc_ = create_cudaTextureObject_array_(tex_dataSrc_, engine_num);
	(data_3D) ? setBasicParams_const_(section, imageW_, imageH_) : setBasicParams_const_(section, imageW_);
	// ------- VCEC & bin array -------
	int binNum_local, binNum_global;
	std::vector<std::vector<int>> VCEC_local_rec;
	std::vector<std::vector<float>> ascend_unique_arr_local_rec;
	int** VCEC_host_partial_ = (async_mode) ? allocate_host_memory2D_pinned_<int>(engine_num, pre_binNum) : allocate_host_memory2D_<int>(engine_num, pre_binNum);
	float** ascend_unique_arr_local_host_ = (async_mode) ? allocate_host_memory2D_pinned_<float>(engine_num, pre_binNum) : allocate_host_memory2D_<float>(engine_num, pre_binNum);

	int** VCEC_device_partial_ = allocate_device_memory2D_<int>(engine_num, pre_binNum);
	float** ascend_unique_arr_local_device_ = allocate_device_memory2D_<float>(engine_num, pre_binNum);

	// ----- Allocate and initialize an array of stream handles
	cudaEvent_t* events;
	cudaStream_t* streams;

	// ----- Histogram Initialization -----
	if (async_mode) {
		events = create_cudaEvent_array(engine_num * 3);
		streams = create_cudaStream_array(engine_num);
	}

	// ------ Loop through each file in the folder ------
	for (int fidx = 0; fidx < fileNames.size(); fidx++) {
		// ------ Memory transfer and kernel launch
		for (int i = 0; i < chunk_num; i++) {
			int engidx = i % engine_num;
			int chunkH = section[i].second - section[i].first + 1;
			if (async_mode) {
				checkCudaErrors(cudaEventSynchronize(events[engine_num * 2 + engidx]));
				std::vector<float> ascend_unique_arr_local = (data_3D) ?
					from_stream_3Dfloat_mt_slice_<float>(fileNames[fidx], 8, input_host[engidx], imageH_, imageW_, imageD_, section[i].first, section[i].second, pad_by_one, timings[0]) :
					from_stream_2Dfloat_mt_row_<float>(fileNames[fidx], 8, input_host[engidx], imageH_, imageW_, section[i].first, section[i].second, pad_by_one, timings[0]);
				binNum_local = int(ascend_unique_arr_local.size());
				ascend_unique_arr_local_rec.push_back(ascend_unique_arr_local);
				if (binNum_local > pre_binNum) std::cout << "WARNING: increase bin number or chunk number" << std::endl;
				if (data_3D) {
					cudaMemcpy3DParms p = setup_3DCopy_params(tex_dataSrc_[engidx], input_host[engidx], imageH_, imageW_, chunkH, 0);
					checkCudaErrors(cudaMemcpy3DAsync(&p, streams[engidx]));
				}
				else checkCudaErrors(cudaMemcpyToArrayAsync(tex_dataSrc_[engidx], 0, 0, &input_host[engidx][0], chunkH * imageW_ * sizeof(float), cudaMemcpyHostToDevice, streams[engidx]));
				checkCudaErrors(cudaEventRecord(events[engine_num * 2 + engidx], streams[engidx]));
				if (i >= engine_num) checkCudaErrors(cudaEventSynchronize(events[engine_num + engidx]));

				std::copy(ascend_unique_arr_local.begin(), ascend_unique_arr_local.end(), ascend_unique_arr_local_host_[engidx]);
				checkCudaErrors(cudaMemcpyAsync(ascend_unique_arr_local_device_[engidx], ascend_unique_arr_local_host_[engidx], binNum_local * sizeof(float), cudaMemcpyHostToDevice, streams[engidx]));
				checkCudaErrors(cudaEventRecord(events[engine_num + engidx], streams[engidx]));
				if (i >= engine_num) {
					checkCudaErrors(cudaEventSynchronize(events[engidx]));
					std::vector<int> VCEC_local_;
					VCEC_local_.assign(VCEC_host_partial_[engidx], VCEC_host_partial_[engidx] + ascend_unique_arr_local_rec[i - engine_num].size());
					VCEC_local_rec.push_back(VCEC_local_);
				}
				init_VCEC_device(binNum_local, VCEC_device_partial_[engidx], &streams[engidx]);
				if (data_3D) computeECC_3D(imageH_ - 2, imageW_ - 2, chunkH - 2, binNum_local, texSrc_[engidx], VCEC_device_partial_[engidx], ascend_unique_arr_local_device_[engidx], i == (chunk_num - 1), &streams[engidx]);
				else computeECC(chunkH - 2, imageW_ - 2, binNum_local, texSrc_[engidx], VCEC_device_partial_[engidx], ascend_unique_arr_local_device_[engidx], i == (chunk_num - 1), &streams[engidx]);
				checkCudaErrors(cudaMemcpyAsync(VCEC_host_partial_[engidx], VCEC_device_partial_[engidx], binNum_local * sizeof(int), cudaMemcpyDeviceToHost, streams[engidx]));
				checkCudaErrors(cudaEventRecord(events[engidx], streams[engidx]));
			}
			else {
				std::vector<float> ascend_unique_arr_local = (data_3D) ?
					from_stream_3Dfloat_mt_slice_<float>(fileNames[fidx], 8, input_host[engidx], imageH_, imageW_, imageD_, section[i].first, section[i].second, pad_by_one, timings[0]) :
					from_stream_2Dfloat_mt_row_<float>(fileNames[fidx], 8, input_host[engidx], imageH_, imageW_, section[i].first, section[i].second, pad_by_one, timings[0]);
				binNum_local = int(ascend_unique_arr_local.size());
				ascend_unique_arr_local_rec.push_back(ascend_unique_arr_local);
				if (binNum_local > pre_binNum) std::cout << "WARNING: increase bin number" << std::endl;
				if (data_3D) { cudaMemcpy3DParms p = setup_3DCopy_params(tex_dataSrc_[engidx], input_host[engidx], imageH_, imageW_, chunkH, 0); checkCudaErrors(cudaMemcpy3D(&p)); }
				else checkCudaErrors(cudaMemcpyToArray(tex_dataSrc_[engidx], 0, 0, &input_host[engidx][0], chunkH * imageW_ * sizeof(float), cudaMemcpyHostToDevice));
				std::copy(ascend_unique_arr_local.begin(), ascend_unique_arr_local.end(), ascend_unique_arr_local_host_[engidx]);
				checkCudaErrors(cudaMemcpy(ascend_unique_arr_local_device_[engidx], ascend_unique_arr_local_host_[engidx], binNum_local * sizeof(float), cudaMemcpyHostToDevice));
				init_histogram_1D_<int>(VCEC_host_partial_[engidx], binNum_local);
				checkCudaErrors(cudaMemcpy(VCEC_device_partial_[engidx], VCEC_host_partial_[engidx], binNum_local * sizeof(int), cudaMemcpyHostToDevice));
				if (data_3D) computeECC_3D(imageH_ - 2, imageW_ - 2, chunkH - 2, binNum_local, texSrc_[engidx], VCEC_device_partial_[engidx], ascend_unique_arr_local_device_[engidx], i == (chunk_num - 1), nullptr);
				else computeECC(chunkH - 2, imageW_ - 2, binNum_local, texSrc_[engidx], VCEC_device_partial_[engidx], ascend_unique_arr_local_device_[engidx], i == (chunk_num - 1), nullptr);
				checkCudaErrors(cudaMemcpy(VCEC_host_partial_[engidx], VCEC_device_partial_[engidx], binNum_local * sizeof(int), cudaMemcpyDeviceToHost));
				std::vector<int> VCEC_local_;
				VCEC_local_.assign(VCEC_host_partial_[engidx], VCEC_host_partial_[engidx] + binNum_local);
				VCEC_local_rec.push_back(VCEC_local_);
			}
		}
		// ------- Process results --------
		std::vector<float> ascend_unique_arr = accumulate_ascend_unique_arr(ascend_unique_arr_local_rec);
		binNum_global = ascend_unique_arr.size();
		int* VCEC_host_ = allocate_host_memory1D_<int>(binNum_global);
		init_histogram_1D_<int>(VCEC_host_, binNum_global);
		if (async_mode) for (int i = chunk_num; i < chunk_num + engine_num; i++) {
			checkCudaErrors(cudaEventSynchronize(events[i % engine_num]));
			std::vector<int> VCEC_local_;
			VCEC_local_.assign(VCEC_host_partial_[i % engine_num], VCEC_host_partial_[i % engine_num] + ascend_unique_arr_local_rec[i - engine_num].size());
			VCEC_local_rec.push_back(VCEC_local_);
		}
		for (size_t i = 0; i < VCEC_local_rec.size(); i++)
			accumulate_VCEC_host_various_binNum_(VCEC_host_, &VCEC_local_rec[i][0], ascend_unique_arr, ascend_unique_arr_local_rec[i]);
		if (output) write_txt_ECC<float>(fileNameso[fidx], ascend_unique_arr, VCEC_host_);
		ascend_unique_arr_local_rec.clear();
		VCEC_local_rec.clear();
		ascend_unique_arr.clear();
		free(VCEC_host_);
	}

	checkCudaErrors(cudaDeviceSynchronize());
	double w2 = get_wall_time();
	timings[1] = (w2 * 1000 - w1 * 1000);
	if (manual_timing) {
		printf("Number of unique values: %d\n", binNum_global);
		printf("Data read time: %f msecs;\n", timings[0]);
		printf("GPU total time: %f msecs;\n", timings[1]);
	}

	// ------- Free memory --------
	section.clear();
	
	free_vector2D_<float>(ascend_unique_arr_local_rec);
	(async_mode) ? cudaFreeHost(input_host) : free(input_host);
	(async_mode) ? free_host_memory2D_pinned_<float>(input_host, engine_num) : free_host_memory2D_<float>(input_host, engine_num);
	
	(async_mode) ? free_host_memory2D_pinned_<int>(VCEC_host_partial_, engine_num) : free_host_memory2D_<int>(VCEC_host_partial_, engine_num);
	(async_mode) ? free_host_memory2D_pinned_<float>(ascend_unique_arr_local_host_, engine_num) : free_host_memory2D_<float>(ascend_unique_arr_local_host_, engine_num);

	free_device_memory2D_<int>(VCEC_device_partial_, engine_num);
	free_cudaArray_array(tex_dataSrc_, engine_num);
	free_device_memory2D_<float>(ascend_unique_arr_local_device_, engine_num);
	if (async_mode) free_cudaEvent_array(events, engine_num * 3);
	if (async_mode) free_cudaStream_array(streams, engine_num);

	checkCudaErrors(cudaDeviceReset());
	return timings;
}

//std::vector<double> ECC_folder(
//	std::string& path,
//	std::string& patho,
//	int h, int w, int d,
//	bool pad,
//	bool async,
//	bool mt,
//	bool timing,
//	bool verb,
//	bool output
//) {
//	int imageH_ = h;
//	int imageW_ = w;
//	int imageD_ = d;
//	bool pad_by_one = pad;
//	bool async_mode = async;
//	bool manual_timing = timing;
//	bool mt_read = mt;
//	bool verbose = verb;
//	bool data_3D = (d > 0);
//
//	std::vector<std::string> fileNames = fileNames_from_folder(path);
//	const int pre_binNum = 1024;
//	int chunk_num = fileNames.size();
//	int engine_num = 3;
//	engine_num = (engine_num > chunk_num) ? chunk_num : engine_num;
//
//	// ------ Divide into chunks
//	std::vector<std::pair<int, int>> section = (data_3D) ? return_equal_size_chunk(imageD_, chunk_num, pad_by_one) : return_equal_size_chunk(imageH_, chunk_num, pad_by_one);
//
//	// ------ Timing
//	std::vector<double> timings(2, 0);
//	double w1 = get_wall_time();
//	// ------- Declare data structures -------
//	float** input_host = allocate_input_memory<float>(engine_num, async_mode, pad_by_one, imageH_, imageW_, imageD_, -1);
//	cudaArray** tex_dataSrc_ = (data_3D) ? allocate_cudaArray_array_(engine_num, imageH_, imageW_, imageD_) : allocate_cudaArray_array_(engine_num, imageH_, imageW_);
//	cudaTextureObject_t* texSrc_ = create_cudaTextureObject_array_(tex_dataSrc_, engine_num);
//	(data_3D) ? setBasicParams_const_(section, imageW_, imageH_) : setBasicParams_const_(section, imageW_);
//	// ------- VCEC & bin array -------
//	int binNum_local;
//	std::vector<std::vector<int>> VCEC_local_rec;
//	std::vector<std::vector<float>> ascend_unique_arr_local_rec;
//	int** VCEC_host_partial_ = (async_mode) ? allocate_host_memory2D_pinned_<int>(engine_num, pre_binNum) : allocate_host_memory2D_<int>(engine_num, pre_binNum);
//	float** ascend_unique_arr_local_host_ = (async_mode) ? allocate_host_memory2D_pinned_<float>(engine_num, pre_binNum) : allocate_host_memory2D_<float>(engine_num, pre_binNum);
//
//	int** VCEC_device_partial_ = allocate_device_memory2D_<int>(engine_num, pre_binNum);
//	float** ascend_unique_arr_local_device_ = allocate_device_memory2D_<float>(engine_num, pre_binNum);
//
//	// ----- Allocate and initialize an array of stream handles
//	cudaEvent_t* events;
//	cudaStream_t* streams;
//
//	// ----- Histogram Initialization -----
//	if (async_mode) {
//		events = create_cudaEvent_array(engine_num * 3);
//		streams = create_cudaStream_array(engine_num);
//	}
//
//	// ------ Memory transfer and kernel launch
//	for (int i = 0; i < chunk_num; i++) {
//		int engidx = i % engine_num;
//		int chunkH = (data_3D) ? imageD_ : imageH_;
//		std::string fileName = fileNames[i];
//
//		if (async_mode) {
//			checkCudaErrors(cudaEventSynchronize(events[engine_num * 2 + engidx]));
//			std::vector<float> ascend_unique_arr_local = (data_3D) ?
//				from_stream_3Dfloat_mt_slice_<float>(fileName, 8, input_host[engidx], imageH_, imageW_, imageD_, 0, imageD_ - 1, pad_by_one, timings[0]) :
//				from_stream_2Dfloat_mt_row_<float>(fileName, 8, input_host[engidx], imageH_, imageW_, 0, imageH_ - 1, pad_by_one, timings[0]);
//			binNum_local = int(ascend_unique_arr_local.size());
//			ascend_unique_arr_local_rec.push_back(ascend_unique_arr_local);
//			if (binNum_local > pre_binNum) std::cout << "WARNING: increase bin number" << std::endl;
//			if (data_3D) {
//				cudaMemcpy3DParms p = setup_3DCopy_params(tex_dataSrc_[engidx], input_host[engidx], imageH_, imageW_, chunkH, 0);
//				checkCudaErrors(cudaMemcpy3DAsync(&p, streams[engidx]));
//			}
//			else checkCudaErrors(cudaMemcpyToArrayAsync(tex_dataSrc_[engidx], 0, 0, &input_host[engidx][0], chunkH * imageW_ * sizeof(float), cudaMemcpyHostToDevice, streams[engidx]));
//			checkCudaErrors(cudaEventRecord(events[engine_num * 2 + engidx], streams[engidx]));
//			if (i >= engine_num) checkCudaErrors(cudaEventSynchronize(events[engine_num + engidx]));
//
//			std::copy(ascend_unique_arr_local.begin(), ascend_unique_arr_local.end(), ascend_unique_arr_local_host_[engidx]);
//			checkCudaErrors(cudaMemcpyAsync(ascend_unique_arr_local_device_[engidx], ascend_unique_arr_local_host_[engidx], binNum_local * sizeof(float), cudaMemcpyHostToDevice, streams[engidx]));
//			checkCudaErrors(cudaEventRecord(events[engine_num + engidx], streams[engidx]));
//			if (i >= engine_num) {
//				checkCudaErrors(cudaEventSynchronize(events[engidx]));
//				std::vector<int> VCEC_local_;
//				VCEC_local_.assign(VCEC_host_partial_[engidx], VCEC_host_partial_[engidx] + ascend_unique_arr_local_rec[i - engine_num].size());
//				VCEC_local_rec.push_back(VCEC_local_);
//			}
//			init_VCEC_device(binNum_local, VCEC_device_partial_[engidx], &streams[engidx]);
//			if (data_3D) computeECC_3D(imageH_ - 2, imageW_ - 2, chunkH - 2, binNum_local, texSrc_[engidx], VCEC_device_partial_[engidx], ascend_unique_arr_local_device_[engidx], i == (chunk_num - 1), &streams[engidx]);
//			else computeECC(chunkH - 2, imageW_ - 2, binNum_local, texSrc_[engidx], VCEC_device_partial_[engidx], ascend_unique_arr_local_device_[engidx], i == (chunk_num - 1), &streams[engidx]);
//			checkCudaErrors(cudaMemcpyAsync(VCEC_host_partial_[engidx], VCEC_device_partial_[engidx], binNum_local * sizeof(int), cudaMemcpyDeviceToHost, streams[engidx]));
//			checkCudaErrors(cudaEventRecord(events[engidx], streams[engidx]));
//		}
//		else {
//			std::vector<float> ascend_unique_arr_local = (data_3D) ?
//				from_stream_3Dfloat_mt_slice_<float>(fileName, 8, input_host[engidx], imageH_, imageW_, imageD_, 0, imageD_ - 1, pad_by_one, timings[0]) :
//				from_stream_2Dfloat_mt_row_<float>(fileName, 8, input_host[engidx], imageH_, imageW_, 0, imageH_ - 1, pad_by_one, timings[0]);
//			binNum_local = int(ascend_unique_arr_local.size());
//			ascend_unique_arr_local_rec.push_back(ascend_unique_arr_local);
//			if (binNum_local > pre_binNum) std::cout << "WARNING: increase bin number" << std::endl;
//			if (data_3D) { cudaMemcpy3DParms p = setup_3DCopy_params(tex_dataSrc_[engidx], input_host[engidx], imageH_, imageW_, chunkH, 0); checkCudaErrors(cudaMemcpy3D(&p)); }
//			else checkCudaErrors(cudaMemcpyToArray(tex_dataSrc_[engidx], 0, 0, &input_host[engidx][0], chunkH * imageW_ * sizeof(float), cudaMemcpyHostToDevice));
//			std::copy(ascend_unique_arr_local.begin(), ascend_unique_arr_local.end(), ascend_unique_arr_local_host_[engidx]);
//			checkCudaErrors(cudaMemcpy(ascend_unique_arr_local_device_[engidx], ascend_unique_arr_local_host_[engidx], binNum_local * sizeof(float), cudaMemcpyHostToDevice));
//			init_histogram_1D_<int>(VCEC_host_partial_[engidx], binNum_local);
//			checkCudaErrors(cudaMemcpy(VCEC_device_partial_[engidx], VCEC_host_partial_[engidx], binNum_local * sizeof(int), cudaMemcpyHostToDevice));
//			if (data_3D) computeECC_3D(imageH_ - 2, imageW_ - 2, chunkH - 2, binNum_local, texSrc_[engidx], VCEC_device_partial_[engidx], ascend_unique_arr_local_device_[engidx], i == (chunk_num - 1), nullptr);
//			else computeECC(chunkH - 2, imageW_ - 2, binNum_local, texSrc_[engidx], VCEC_device_partial_[engidx], ascend_unique_arr_local_device_[engidx], i == (chunk_num - 1), nullptr);
//			checkCudaErrors(cudaMemcpy(VCEC_host_partial_[engidx], VCEC_device_partial_[engidx], binNum_local * sizeof(int), cudaMemcpyDeviceToHost));
//			std::vector<int> VCEC_local_;
//			VCEC_local_.assign(VCEC_host_partial_[engidx], VCEC_host_partial_[engidx] + binNum_local);
//			VCEC_local_rec.push_back(VCEC_local_);
//		}
//	}
//	// ------- Process results --------
//	if (async_mode) for (int i = chunk_num; i < chunk_num + engine_num; i++) {
//		checkCudaErrors(cudaEventSynchronize(events[i % engine_num]));
//		std::vector<int> VCEC_local_;
//		VCEC_local_.assign(VCEC_host_partial_[i % engine_num], VCEC_host_partial_[i % engine_num] + ascend_unique_arr_local_rec[i - engine_num].size());
//		VCEC_local_rec.push_back(VCEC_local_);
//	}
//
//	checkCudaErrors(cudaDeviceSynchronize());
//	double w2 = get_wall_time();
//	timings[1] = (w2 * 1000 - w1 * 1000);
//	if (output) {
//		std::vector<std::string> fileNameso = compose_outfileNames_from_folder(path, patho);
//		for (unsigned int i = 0; i < fileNameso.size(); i++) write_txt_ECC<float>(fileNameso[i], ascend_unique_arr_local_rec[i], &VCEC_local_rec[i][0]);
//	}
//	if (manual_timing) {
//		printf("Data read time: %f msecs;\n", timings[0]);
//		printf("GPU total time: %f msecs;\n", timings[1]);
//	}
//
//	// ------- Free memory --------
//	section.clear();
//	free_vector2D_<float>(ascend_unique_arr_local_rec);
//	(async_mode) ? cudaFreeHost(input_host) : free(input_host);
//	(async_mode) ? free_host_memory2D_pinned_<float>(input_host, engine_num) : free_host_memory2D_<float>(input_host, engine_num);
//	(async_mode) ? free_host_memory2D_pinned_<int>(VCEC_host_partial_, engine_num) : free_host_memory2D_<int>(VCEC_host_partial_, engine_num);
//	(async_mode) ? free_host_memory2D_pinned_<float>(ascend_unique_arr_local_host_, engine_num) : free_host_memory2D_<float>(ascend_unique_arr_local_host_, engine_num);
//
//	free_device_memory2D_<int>(VCEC_device_partial_, engine_num);
//	free_cudaArray_array(tex_dataSrc_, engine_num);
//	free_device_memory2D_<float>(ascend_unique_arr_local_device_, engine_num);
//	if (async_mode) free_cudaEvent_array(events, engine_num * 3);
//	if (async_mode) free_cudaStream_array(streams, engine_num);
//
//	checkCudaErrors(cudaDeviceReset());
//	return timings;
//}

std::vector<double> ECC_vanila(std::string& fileName, int h, int w, bool mt, bool timing, bool verb) {
	int imageH_          = h;
	int imageW_          = w;
	int imageD_          = 0;
	bool manual_timing   = timing;
	bool mt_read         = mt;
	bool verbose         = verb;

	const int blur_times = 1;
	const int pre_binNum = 700;
	int chunk_num        = 1;
	int engine_num       = 1;
	engine_num = (engine_num > chunk_num) ? chunk_num : engine_num;

	// ------ Divide into chunks
	std::vector<std::pair<int, int>> section = return_equal_size_chunk(imageH_, chunk_num, false);
	// ------ Timing
	std::vector<double> timings(5, 0);
	double w1 = get_wall_time();
	// ------- Declare data structures -------
	float** input_host = allocate_input_memory<float>(engine_num, false, false, imageH_, imageW_, imageD_, -1);
	cudaArray** tex_dataSrc_ = allocate_cudaArray_array_(engine_num, imageH_, imageW_);
	cudaTextureObject_t* texSrc_ = create_cudaTextureObject_array_(tex_dataSrc_, engine_num);
	setBasicParams_const_(section, imageW_);
	// ------- VCEC & bin array -------
	std::vector<std::vector<int>> VCEC_rec;
	int* VCEC_host_ = allocate_host_memory1D_<int>(pre_binNum);
	int* VCEC_device_ = allocate_device_memory1D_<int>(pre_binNum);
	float* ascend_unique_arr_host_ = allocate_host_memory1D_<float>(pre_binNum);
	float* ascend_unique_arr_device_ = allocate_device_memory1D_<float>(pre_binNum);
	for (size_t i = 0; i < pre_binNum; i++) ascend_unique_arr_host_[i] = i;

	// ----- Read input -----
	std::vector<float> ascend_unique_arr_local = from_stream_2Dfloat_mt_row_<float>(fileName, 8, input_host[0], imageH_, imageW_, 0, imageH_ - 1, false, timings[0]);
	assert(ascend_unique_arr_local.size() <= pre_binNum);

	// ----- Gaussian Blur -----
	double kernel_t1 = get_wall_time();
	double g1 = get_wall_time();
	float* tmp_;
	//float* blurred_h = allocate_host_memory1D_<float>(imageH_ * imageW_);
	float* source_d  = allocate_device_memory1D_<float>(imageH_ * imageW_);
	float* blurred_d = allocate_device_memory1D_<float>(imageH_ * imageW_);
	int ksize = generate_kernel(1.5);
	checkCudaErrors(cudaMemcpy(source_d, input_host[0], imageH_ * imageW_ * sizeof(float), cudaMemcpyHostToDevice));
	double g2 = get_wall_time();
	timings[2] += (g2 - g1) * 1000;

	// ---- Kernel Launch -----
	double mem_t1 = get_wall_time();
	checkCudaErrors(cudaMemcpyToArray(tex_dataSrc_[0], 0, 0, source_d, imageH_ * imageW_ * sizeof(float), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(ascend_unique_arr_device_, ascend_unique_arr_host_, pre_binNum * sizeof(float), cudaMemcpyHostToDevice));
	double mem_t2 = get_wall_time();
	timings[3] += (mem_t2 - mem_t1) * 1000;

	for (int i = 0; i < blur_times; i++) {
		init_VCEC_device(pre_binNum, VCEC_device_, nullptr);
		computeECC(imageH_ - 2, imageW_ - 2, pre_binNum, texSrc_[0], VCEC_device_, ascend_unique_arr_device_, true, nullptr);
		checkCudaErrors(cudaDeviceSynchronize());
		mem_t1 = get_wall_time();
		checkCudaErrors(cudaMemcpy(VCEC_host_, VCEC_device_, pre_binNum * sizeof(int), cudaMemcpyDeviceToHost));
		std::vector<int> VCEC_local_;
		VCEC_local_.assign(VCEC_host_, VCEC_host_ + pre_binNum);
		VCEC_rec.push_back(VCEC_local_);
		mem_t2 = get_wall_time();
		timings[3] += (mem_t2 - mem_t1) * 1000;

		g1 = get_wall_time();
		gaussianVerticalCuda(source_d, blurred_d, imageW_, imageH_, ksize);
		if (i != blur_times - 1) {
			//gaussianVerticalCuda(source_d, blurred_d, imageW_, imageH_, ksize);
			checkCudaErrors(cudaMemcpyToArray(tex_dataSrc_[0], 0, 0, blurred_d, imageH_ * imageW_ * sizeof(float), cudaMemcpyDeviceToDevice));
		}
		
		//checkCudaErrors(cudaMemcpy(blurred_h, blurred_d, imageW_ * imageH_ * sizeof(float), cudaMemcpyDeviceToHost));
		//cv::Mat img_uint8, image_uint8_large;
		//cv::Mat img(imageH_, imageW_, CV_32FC1, blurred_h);
		//img.convertTo(img_uint8, CV_8UC1);
		//resize(img_uint8, image_uint8_large, cv::Size(2 * img_uint8.cols, 2 * img_uint8.rows), cv::INTER_CUBIC);
		//cv::imwrite("E:/Data2/SoCG2022/Repeated_Gaussian/Blur_" + std::to_string(i + 1) + ".png", image_uint8_large);
		//cv::imshow("Window", img_uint8);
		//cv::waitKey();

		checkCudaErrors(cudaDeviceSynchronize());
		tmp_ = blurred_d; blurred_d = source_d; source_d = tmp_;
		g2 = get_wall_time();
		timings[2] += (g2 - g1) * 1000;
	}
	double kernel_t2 = get_wall_time();
	timings[4] += (kernel_t2 - kernel_t1) * 1000;

	// ----- Post-processing -----
	checkCudaErrors(cudaDeviceSynchronize());
	double w2 = get_wall_time();
	timings[1] = (w2 * 1000 - w1 * 1000);
	if (manual_timing) {
		printf("GPU total time:   %.3f msecs;\n", timings[1]);
		printf("ECC memory transfer time:   %.3f msecs;\n", timings[3] / blur_times);
		printf("ECC kernel execution time:   %.3f msecs;\n", (timings[4] - timings[2] - timings[3]) / blur_times);
		printf("Gaussian blur total time:   %.3f msecs;\n", timings[2] / blur_times);
		printf("Data read time:   %.3f msecs;\n", timings[0]);
	}

	// ------- Free memory --------
	section.clear();
	//free(blurred_h);
	free(VCEC_host_);
	free(ascend_unique_arr_host_);
	checkCudaErrors(cudaFree(source_d));
	checkCudaErrors(cudaFree(blurred_d));
	checkCudaErrors(cudaFree(VCEC_device_));
	checkCudaErrors(cudaFree(ascend_unique_arr_device_));
	free_host_memory2D_<float>(input_host, engine_num);
	free_cudaArray_array(tex_dataSrc_, engine_num);

	checkCudaErrors(cudaDeviceReset());
	return timings;
}

void run_folder(std::string& path, int run_times) {
	if (!boost::filesystem::exists(path)) { std::cout << "Invalid folder path" << std::endl; exit(1); }
	std::vector<std::string> res = fileNames_from_folder(path);
	std::string rec_txt = path + "/gpu_rec.txt";
	std::fstream out(rec_txt, std::fstream::out | std::fstream::trunc);

	time_t now = time(0);
	char* dt = ctime(&now);
	out << dt;
	out << "=====================================" << std::endl;

	std::unordered_map<int, std::vector<float>> instance_total_time_rec_2D;
	std::unordered_map<int, std::vector<float>> instance_execution_time_rec_2D;
	std::unordered_map<int, std::vector<float>> instance_load_time_rec_2D;
	std::unordered_map<int, std::vector<float>> instance_total_stdev_rec_2D;

	std::unordered_map<int, std::vector<float>> instance_total_time_rec_3D;
	std::unordered_map<int, std::vector<float>> instance_execution_time_rec_3D;
	std::unordered_map<int, std::vector<float>> instance_load_time_rec_3D;
	std::unordered_map<int, std::vector<float>> instance_total_stdev_rec_3D;

	bool pad_by_one = true;
	bool async_mode = true;
	bool manual_timing = false;
	bool mt_read = true;
	bool verbose = false;

	for (int i = 0; i < res.size(); i++) {
		std::cout << "Processing: " << res[i] << std::endl;
		std::string dim, size;
		parse_GRF_name(res[i], size, dim);

		int imageH_ = std::stoi(size);
		int imageW_ = std::stoi(size);
		int imageD_ = std::stoi(size);

		std::vector<float> total_time, execution_time, load_time;
		for (int j = 0; j < run_times; j++) {
			std::vector<double> time;
			if (dim == std::string("2"))
				time = ECC_(res[i], std::string(""), imageH_, imageW_, -1, pad_by_one, async_mode, mt_read, manual_timing, verbose, false);
			else
				time = ECC_(res[i], std::string(""), imageH_, imageW_, imageD_, pad_by_one, async_mode, mt_read, manual_timing, verbose ,false);
			total_time.push_back(time[1]);
			execution_time.push_back(time[1] - time[0]);
			load_time.push_back(time[0]);
		}
		if (dim == std::string("2")) {
			instance_total_time_rec_2D[std::stoi(size)].push_back(float(std::accumulate(total_time.begin(), total_time.end(), 0.0)) / total_time.size() * 1.0);
			instance_execution_time_rec_2D[std::stoi(size)].push_back(float(std::accumulate(execution_time.begin(), execution_time.end(), 0.0)) / execution_time.size() * 1.0);
			instance_load_time_rec_2D[std::stoi(size)].push_back(float(std::accumulate(load_time.begin(), load_time.end(), 0.0)) / load_time.size() * 1.0);
			instance_total_stdev_rec_2D[std::stoi(size)].push_back(compute_stdev(total_time));
		}
		else {
			instance_total_time_rec_3D[std::stoi(size)].push_back(float(std::accumulate(total_time.begin(), total_time.end(), 0.0)) / total_time.size() * 1.0);
			instance_execution_time_rec_3D[std::stoi(size)].push_back(float(std::accumulate(execution_time.begin(), execution_time.end(), 0.0)) / execution_time.size() * 1.0);
			instance_load_time_rec_3D[std::stoi(size)].push_back(float(std::accumulate(load_time.begin(), load_time.end(), 0.0)) / load_time.size() * 1.0);
			instance_total_stdev_rec_3D[std::stoi(size)].push_back(compute_stdev(total_time));
		}
	}

	for (auto s : instance_total_time_rec_2D) {
		int num = s.second.size();
		int size = s.first;
		float avg_total = std::accumulate(s.second.begin(), s.second.end(), 0.0) / num;
		float avg_exec = std::accumulate(instance_execution_time_rec_2D[size].begin(), instance_execution_time_rec_2D[size].end(), 0.0) / num;
		float avg_load = std::accumulate(instance_load_time_rec_2D[size].begin(), instance_load_time_rec_2D[size].end(), 0.0) / num;
		float outer_stdev_total = compute_stdev(s.second);
		float outer_stdev_exec = compute_stdev(instance_execution_time_rec_2D[size]);
		float outer_stdev_load = compute_stdev(instance_load_time_rec_2D[size]);
		float inner_stdev_total = std::accumulate(instance_total_stdev_rec_2D[size].begin(), instance_total_stdev_rec_2D[size].end(), 0.0) / num;

		out << "2D Size (" << size << " " << size << ") instance #: " << num <<
			" || avg_total: " << avg_total << " ms" <<
			" || avg_execution: " << avg_exec << " ms" <<
			" || avg_load: " << avg_load << " ms" <<
			" || outer_stdev_total: " << outer_stdev_total << " ms" <<
			" || outer_stdev_execution: " << outer_stdev_exec << " ms" <<
			" || outer_stdev_load: " << outer_stdev_load << " ms" <<
			" || inner_stdev_load: " << inner_stdev_total << " ms" << std::endl;
	}

	for (auto s : instance_total_time_rec_3D) {
		int num = s.second.size();
		int size = s.first;
		float avg_total = std::accumulate(s.second.begin(), s.second.end(), 0.0) / num;
		float avg_exec = std::accumulate(instance_execution_time_rec_3D[size].begin(), instance_execution_time_rec_3D[size].end(), 0.0) / num;
		float avg_load = std::accumulate(instance_load_time_rec_3D[size].begin(), instance_load_time_rec_3D[size].end(), 0.0) / num;
		float outer_stdev_total = compute_stdev(s.second);
		float outer_stdev_exec = compute_stdev(instance_execution_time_rec_3D[size]);
		float outer_stdev_load = compute_stdev(instance_load_time_rec_3D[size]);
		float inner_stdev_total = std::accumulate(instance_total_stdev_rec_3D[size].begin(), instance_total_stdev_rec_3D[size].end(), 0.0) / num;

		out << "3D Size (" << size << " " << size << ") instance #: " << num <<
			" || avg_total: " << avg_total << " ms" <<
			" || avg_execution: " << avg_exec << " ms" <<
			" || avg_load: " << avg_load << " ms" <<
			" || outer_stdev_total: " << outer_stdev_total << " ms" <<
			" || outer_stdev_execution: " << outer_stdev_exec << " ms" <<
			" || outer_stdev_load: " << outer_stdev_load << " ms" <<
			" || inner_stdev_load: " << inner_stdev_total << " ms" << std::endl;
	}

	out.close();
}

void run_GENERAL(std::string& path, int data_dim, int run_times) {
	if (!boost::filesystem::exists(path)) { std::cout << "Invalid folder path" << std::endl; exit(1); }
	std::vector<std::string> res = fileNames_from_folder(path);
	std::string rec_txt = path + "/gpu_rec.txt";
	std::fstream out(rec_txt, std::fstream::out | std::fstream::trunc);

	time_t now = time(0);
	char* dt = ctime(&now);
	out << dt;
	out << "=====================================" << std::endl;

	bool pad_by_one = true;
	bool async_mode = true;
	bool manual_timing = false;
	bool mt_read = true;
	bool verbose = false;

	for (int i = 0; i < res.size(); i++) {
		std::cout << "Processing: " << res[i] << std::endl;
		std::string d1, d2, d3;
		parse_VICTRE_name(res[i], d1, d2, d3);
		int imageH_ = std::stoi(d1);
		int imageW_ = std::stoi(d2);
		int imageD_ = std::stoi(d3);

		std::vector<float> total_time, execution_time, load_time;
		for (int j = 0; j < run_times; j++) {
			std::vector<double> time;
			time = (data_dim == 2) ? ECC_(res[i], std::string(""), imageH_, imageW_, -1, pad_by_one, async_mode, mt_read, manual_timing, verbose, false) :
				ECC_(res[i], std::string(""), imageH_, imageW_, imageD_, pad_by_one, async_mode, mt_read, manual_timing, verbose, false);
			total_time.push_back(time[0] + time[1]);
			execution_time.push_back(time[1]);
			load_time.push_back(time[0]);
		}

		out << "File: " << res[i] << std::endl <<
			" || avg_total: " << float(std::accumulate(total_time.begin(), total_time.end(), 0.0) / total_time.size() * 1.0) << " ms" <<
			" || avg_execution: " << float(std::accumulate(execution_time.begin(), execution_time.end(), 0.0) / execution_time.size() * 1.0) << " ms" <<
			" || avg_load: " << float(std::accumulate(load_time.begin(), load_time.end(), 0.0) / load_time.size() * 1.0) << " ms" << std::endl <<
			" || stdev_total: " << compute_stdev(total_time) << " ms" <<
			" || stdev_execution: " << compute_stdev(execution_time) << " ms" <<
			" || stdev_load: " << compute_stdev(load_time) << " ms" << std::endl;
	}
	out.close();
}