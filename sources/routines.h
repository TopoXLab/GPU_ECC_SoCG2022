#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <cuda_runtime.h>

class ECC {
public:
	ECC(int, int, int);

	~ECC();

	std::vector<float> run_frmFile(const std::string&);
	std::vector<float> run_frmArr(float*);
	
private:
	int									imageH_;
	int									imageW_;
	int									imageD_;
	int									engine_num;
	int									chunk_num;
	const int							pre_binNum;
	bool								pad_by_one;
	bool								mt_read;
	bool								verbose;
	bool								data_3D;

	int**								VCEC_host_partial_;
	int**								VCEC_device_partial_;
	float**								input_host;
	float**								ascend_unique_arr_local_host_;
	float**								ascend_unique_arr_local_device_;
	std::vector<std::pair<int, int>>	section;

	cudaArray**							tex_dataSrc_;
	cudaEvent_t*						events;
	cudaStream_t*						streams;
};