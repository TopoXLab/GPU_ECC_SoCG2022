/*
* This codes implements a GPU version ECC computation without futher optimization.
* Author: Fan Wang
* Date: 09/08/2021
*/

#include <iostream>

#include "routines.h"

using namespace std;

int main(int argc, char **argv)
{
	if (argc != 7) helper_();

	bool pad_by_one		 = true;
	bool async_mode		 = true;
	bool mt_read	     = true;
	bool manual_timing   = true;
	bool verbose         = false;
	bool output			 = true;
	int imageH_			 = std::atoi(argv[4]);
	int imageW_			 = std::atoi(argv[5]);
	int imageD_			 = std::atoi(argv[6]);
	std::vector<std::string> args;
	args.assign(argv + 1, argv + 4);
	if (args[0] == "b1")
		std::vector<double> time = ECC_folder_sequential(args[1], args[2], imageH_, imageW_, imageD_, pad_by_one, async_mode, mt_read, manual_timing, verbose, output);
	else if (args[0] == "b2")
		std::vector<double> time = ECC_folder_multiple(args[1], args[2], imageH_, imageW_, imageD_, pad_by_one, async_mode, mt_read, manual_timing, verbose, output);
	else if (args[0] == "s")
		std::vector<double> time = ECC_(args[1], args[2], imageH_, imageW_, imageD_, pad_by_one, async_mode, mt_read, manual_timing, verbose, output);
	else helper_();


	//// Note: some data is not in float format, like planck data. Need to change the input data type to uint8_t!
	//// ------ Read/Generate input data
	//std::string fileName0 = "E:/WorkBench/ECC_CPU/example_data/test_float_1000x1000x100.raw";
	//std::string fileName4 = "E:/Data2/SoCG2022/GaussRandomField/3D/3D_512_0.dat";
	//std::string fileName1 = "E:/Data2/SoCG2022/UniformRandom/3D/3D_2048_0.dat";
	//std::string fileName2 = "E:/Data2/SoCG2022/VICTRE/dense_287_359_202_250um_01.dat";
	//std::string fileName3 = "E:/Data2/SoCG2022/CMB/planck13_1500_750.dat";
	//std::string fileName5 = "E:/Data2/SoCG2022/Binary/3D/3D_512_0.dat";
	//std::string fileName6 = "E:/Data2/SoCG2022/GaussRandomField/2D/2D_128_0.dat";
	//std::string fileName7 = "E:/Data2/SoCG2022/Repeated_Gaussian/2D_1024_11.dat";

	//std::string path1 = "E:/WorkBench/ECC_v1.0/ECC_v1.0/TopoXLab_Github/GPU_ECC_SoCG2022/GaussRandomField/2D";
	//std::string outpath = "E:/WorkBench/ECC_v1.0/ECC_v1.0/TopoXLab_Github/GPU_ECC_SoCG2022/GaussRandomField";

	//int imageH_        = 128;
	//int imageW_        = 128;
	//int imageD_        = 0;
	//bool pad_by_one    = true;
	//bool async_mode    = true;
	//bool manual_timing = true;
	//bool data_3D       = false;
	//bool mt_read       = true;
	//bool verbose       = false;
	//bool output        = true;

	//// ------- Read in data -------
	////float* toy2D = generate_toy_sample(2, 0);
	////float* toy3D = generate_toy_sample(3, 0);
	////float* input_host = from_random_2D_<float>(imageH_, imageW_, 256, 0.2);
	////float* input_host = from_random_3D_<float>(imageH_, imageW_, imageD_, 256, 0.2);
	////write_stream_<float, float>(fileName, input_host, imageH_* imageW_* imageD_);

	////std::vector<double> time = ECC_(path1, outpath, imageH_, imageW_, imageD_, pad_by_one, async_mode, mt_read, manual_timing, verbose, output);
	////std::vector<double> time = ECC_folder_sequential(path1, outpath, imageH_, imageW_, imageD_, pad_by_one, async_mode, mt_read, manual_timing, verbose, output);
	//std::vector<double> time = ECC_folder_multiple(path1, outpath, imageH_, imageW_, imageD_, pad_by_one, async_mode, mt_read, manual_timing, verbose, output);
	////std::vector<double> time = ECC_vanila(fileName7, imageH_, imageW_, mt_read, manual_timing, verbose);
	////run_folder(fileName1, 10);
	////run_GENERAL(fileName3, 2, 10);

	return 0;
}