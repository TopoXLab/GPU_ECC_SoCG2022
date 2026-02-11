#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <numeric>
#include <iostream>
#include <filesystem>
#include <cassert>

#include "kernel.h"
#include "helper_cuda.h"

#include "template.h"
#include "utility.h"
#include "ctpl_stl.h"
#include "routines.h"

namespace fs = std::filesystem;

ECC::ECC(int h, int w, int d) : imageH_(h), imageW_(w), imageD_(d), pre_binNum(1024) {
	pad_by_one		= true;
	mt_read			= true;
	verbose			= false;
	data_3D			= (imageD_ > 0);

	chunk_num	= decide_chunk_num(imageH_, imageW_, imageD_);
	engine_num	= 3;
	engine_num	= (engine_num > chunk_num) ? chunk_num : engine_num;

	// ------ Divide into chunks
	section = (data_3D) ? return_chunk_index(imageD_, chunk_num, pad_by_one) : return_chunk_index(imageH_, chunk_num, pad_by_one);

	// ------- Declare data structures -------
	input_host						= allocate_input_memory<float>(engine_num, true, pad_by_one, imageH_, imageW_, imageD_, section[chunk_num - 1].second - section[chunk_num - 1].first + 1);
	tex_dataSrc_					= (data_3D) ? allocate_cudaArray_array_(engine_num, imageH_, imageW_, section[chunk_num - 1].second - section[chunk_num - 1].first + 1) :
		allocate_cudaArray_array_(engine_num, section[chunk_num - 1].second - section[chunk_num - 1].first + 1, imageW_);
	(data_3D) ? setBasicParams_const_(section, imageW_, imageH_) : setBasicParams_const_(section, imageW_);

	// ------- VCEC & bin array -------
	VCEC_host_partial_				= allocate_host_memory2D_pinned_<int>(engine_num, pre_binNum);
	ascend_unique_arr_local_host_	= allocate_host_memory2D_pinned_<float>(engine_num, pre_binNum);
	VCEC_device_partial_			= allocate_device_memory2D_<int>(engine_num, pre_binNum);
	ascend_unique_arr_local_device_ = allocate_device_memory2D_<float>(engine_num, pre_binNum);

	// ----- Allocate and initialize an array of stream handles
	events	= create_cudaEvent_array(engine_num * 3);
	streams	= create_cudaStream_array(engine_num);
}

ECC::~ECC() {
	section.clear();

	free_host_memory2D_pinned_	<float>(input_host, engine_num);
	free_host_memory2D_pinned_	<int>(VCEC_host_partial_, engine_num);
	free_host_memory2D_pinned_	<float>(ascend_unique_arr_local_host_, engine_num);
	free_device_memory2D_		<int>(VCEC_device_partial_, engine_num);
	free_device_memory2D_		<float>(ascend_unique_arr_local_device_, engine_num);

	free_cudaArray_array(tex_dataSrc_, engine_num);
	free_cudaEvent_array(events, engine_num * 3);
	free_cudaStream_array(streams, engine_num);
}

std::vector<float> ECC::run_frmFile(const std::string& filename) {

	int binNum_local, binNum_global;
	std::vector<double> timings(2, 0);
	std::vector<std::vector<int>>	VCEC_local_rec;
	std::vector<std::vector<float>> ascend_unique_arr_local_rec;
	cudaTextureObject_t* texSrc_ = create_cudaTextureObject_array_(tex_dataSrc_, engine_num);

	// ------ Memory transfer and kernel launch
	for (int i = 0; i < chunk_num; i++) {
		int engidx = i % engine_num;
		int chunkH = section[i].second - section[i].first + 1;

		checkCudaErrors(cudaEventSynchronize(events[engine_num * 2 + engidx]));
		std::vector<float> ascend_unique_arr_local = (data_3D) ?
			from_stream_3Dfloat_mt_slice_<float>(filename, 8, input_host[engidx], imageH_, imageW_, imageD_, section[i].first, section[i].second, pad_by_one, timings[0]) :
			from_stream_2Dfloat_mt_row_<float>(filename, 8, input_host[engidx], imageH_, imageW_, section[i].first, section[i].second, pad_by_one, timings[0]);
		binNum_local = int(ascend_unique_arr_local.size());
		ascend_unique_arr_local_rec.push_back(ascend_unique_arr_local);
		if (binNum_local > pre_binNum) std::cout << "WARNING: increase bin number or chunk number" << std::endl;
		if (data_3D) {
			cudaMemcpy3DParms p = setup_3DCopy_params(tex_dataSrc_[engidx], input_host[engidx], imageH_, imageW_, chunkH, 0);
			checkCudaErrors(cudaMemcpy3DAsync(&p, streams[engidx]));
		}
		else {
			const size_t spitch = static_cast<size_t>(imageW_) * sizeof(float);
			const size_t width = static_cast<size_t>(imageW_) * sizeof(float);
			const size_t height = static_cast<size_t>(chunkH);
			checkCudaErrors(cudaMemcpy2DToArrayAsync(tex_dataSrc_[engidx], 0, 0, input_host[engidx], spitch, width, height, cudaMemcpyHostToDevice, streams[engidx]));
		}
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
	// ------- Process results --------
	std::vector<float> ascend_unique_arr = accumulate_ascend_unique_arr(ascend_unique_arr_local_rec);
	binNum_global = ascend_unique_arr.size();
	int* VCEC_host_ = allocate_host_memory1D_<int>(binNum_global);
	init_histogram_1D_<int>(VCEC_host_, binNum_global);
	for (int i = chunk_num; i < chunk_num + engine_num; i++) {
		checkCudaErrors(cudaEventSynchronize(events[i % engine_num]));
		std::vector<int> VCEC_local_;
		VCEC_local_.assign(VCEC_host_partial_[i % engine_num], VCEC_host_partial_[i % engine_num] + ascend_unique_arr_local_rec[i - engine_num].size());
		VCEC_local_rec.push_back(VCEC_local_);
	}
	for (size_t i = 0; i < VCEC_local_rec.size(); i++)
		accumulate_VCEC_host_various_binNum_(VCEC_host_, &VCEC_local_rec[i][0], ascend_unique_arr, ascend_unique_arr_local_rec[i]);

	std::vector<float> res = return_results(ascend_unique_arr, VCEC_host_);

	checkCudaErrors(cudaDeviceSynchronize());
	ascend_unique_arr_local_rec.clear();
	VCEC_local_rec.clear();
	ascend_unique_arr.clear();
	timings.clear();
	free(VCEC_host_);

	return res;
}

std::vector<float> ECC::run_frmArr(float* src) {

	int binNum_local, binNum_global;
	std::vector<double> timings(2, 0);
	std::vector<std::vector<int>>	VCEC_local_rec;
	std::vector<std::vector<float>> ascend_unique_arr_local_rec;
	cudaTextureObject_t* texSrc_ = create_cudaTextureObject_array_(tex_dataSrc_, engine_num);

	// ------ Memory transfer and kernel launch
	for (int i = 0; i < chunk_num; i++) {
		int engidx = i % engine_num;
		int chunkH = section[i].second - section[i].first + 1;

		checkCudaErrors(cudaEventSynchronize(events[engine_num * 2 + engidx]));
		std::vector<float> ascend_unique_arr_local = (data_3D) ?
			from_array_3Dfloat_mt_slice_<float>(src, 8, input_host[engidx], imageH_, imageW_, imageD_, section[i].first, section[i].second, pad_by_one, timings[0]) :
			from_array_2Dfloat_mt_row_<float>(src, 8, input_host[engidx], imageH_, imageW_, section[i].first, section[i].second, pad_by_one, timings[0]);
		binNum_local = int(ascend_unique_arr_local.size());
		ascend_unique_arr_local_rec.push_back(ascend_unique_arr_local);
		if (binNum_local > pre_binNum) std::cout << "WARNING: increase bin number or chunk number" << std::endl;
		if (data_3D) {
			cudaMemcpy3DParms p = setup_3DCopy_params(tex_dataSrc_[engidx], input_host[engidx], imageH_, imageW_, chunkH, 0);
			checkCudaErrors(cudaMemcpy3DAsync(&p, streams[engidx]));
		}
		else {
			const size_t spitch = static_cast<size_t>(imageW_) * sizeof(float);
			const size_t width = static_cast<size_t>(imageW_) * sizeof(float);
			const size_t height = static_cast<size_t>(chunkH);
			checkCudaErrors(cudaMemcpy2DToArrayAsync(tex_dataSrc_[engidx], 0, 0, input_host[engidx], spitch, width, height, cudaMemcpyHostToDevice, streams[engidx]));
		}
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
	// ------- Process results --------
	std::vector<float> ascend_unique_arr = accumulate_ascend_unique_arr(ascend_unique_arr_local_rec);
	binNum_global = ascend_unique_arr.size();
	int* VCEC_host_ = allocate_host_memory1D_<int>(binNum_global);
	init_histogram_1D_<int>(VCEC_host_, binNum_global);
	for (int i = chunk_num; i < chunk_num + engine_num; i++) {
		checkCudaErrors(cudaEventSynchronize(events[i % engine_num]));
		std::vector<int> VCEC_local_;
		VCEC_local_.assign(VCEC_host_partial_[i % engine_num], VCEC_host_partial_[i % engine_num] + ascend_unique_arr_local_rec[i - engine_num].size());
		VCEC_local_rec.push_back(VCEC_local_);
	}
	for (size_t i = 0; i < VCEC_local_rec.size(); i++)
		accumulate_VCEC_host_various_binNum_(VCEC_host_, &VCEC_local_rec[i][0], ascend_unique_arr, ascend_unique_arr_local_rec[i]);

	std::vector<float> res = return_results(ascend_unique_arr, VCEC_host_);

	checkCudaErrors(cudaDeviceSynchronize());
	ascend_unique_arr_local_rec.clear();
	VCEC_local_rec.clear();
	ascend_unique_arr.clear();
	timings.clear();
	free(VCEC_host_);

	return res;
}