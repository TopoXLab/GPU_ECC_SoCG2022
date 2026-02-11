/*
* This codes implements a GPU version ECC computation without futher optimization.
* Author: Fan Wang
* Date: 09/08/2021
*/

#include <iostream>
#include <filesystem>

#include "template.h"
#include "routines.h"

using namespace std;
namespace fs = std::filesystem;

void write_results(const std::vector<float>& res, const std::string& filename_o)
{
	if (res.size() % 2 != 0) {
		throw std::runtime_error("ECC vector length must be even (x,y pairs).");
	}
	std::ofstream out(filename_o);
	if (!out.is_open()) {
		throw std::runtime_error("Failed to open output file: " + filename_o);
	}

	for (size_t i = 0; i < res.size(); i += 2) {
		out << res[i] << " " << res[i + 1] << "\n";
	}
}

int parse_int_checked(const char* s, const char* what) {
	errno = 0;
	char* end = nullptr;
	long v = std::strtol(s, &end, 10);

	if (errno != 0 || end == s || *end != '\0' || v > INT_MAX || v < INT_MIN) {
		throw std::runtime_error(std::string("Invalid integer for ") + what + ": " + s);
	}
	return static_cast<int>(v);
}

void validate_mode_and_paths(const std::string& mode,
	const std::string& in,
	const std::string& out)
{
	if (mode == "b") {
		if (!fs::exists(in) || !fs::is_directory(in)) {
			throw std::runtime_error("Batch mode expects input to be an existing directory: " + in);
		}
		// Ensure output directory exists (create it)
		if (!fs::exists(out)) fs::create_directories(out);
		if (!fs::is_directory(out)) {
			throw std::runtime_error("Batch mode expects output to be a directory: " + out);
		}
	}
	else if (mode == "s") {
		if (!fs::exists(in) || !fs::is_regular_file(in)) {
			throw std::runtime_error("Single mode expects input to be an existing file: " + in);
		}
		// Ensure output file's parent folder exists
		fs::path op(out);
		if (op.has_parent_path() && !fs::exists(op.parent_path())) {
			fs::create_directories(op.parent_path());
		}
	}
	else {
		throw std::runtime_error("Mode must be 'b' (batch) or 's' (single). Got: " + mode);
	}
}

[[noreturn]] void helper_(const char* prog) {
	std::cerr
		<< "Invalid arguments\n"
		<< "============================================================\n"
		<< "Usage: " << prog << " [mode] [input] [output] [height] [width] [depth]\n"
		<< "--mode:   b for batch mode, s for single mode\n"
		<< "--input:  directory for batch mode, file for single mode\n"
		<< "--output: directory for batch mode, file for single mode\n"
		<< "--height: height of the input file (> 0)\n"
		<< "--width:  width of the input file (> 0)\n"
		<< "--depth:  depth of the input file (>= 0). Set to 0 for 2D.\n"
		<< "============================================================\n";
	std::exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
    try {
        if (argc != 7) helper_(argv[0]);

        // Parse fixed arguments
        std::vector<std::string> args(argv + 1, argv + 4); // mode, input, output
        const std::string& mode = args[0];
        const std::string& input = args[1];
        const std::string& output = args[2];

        int imageH_ = parse_int_checked(argv[4], "height");
        int imageW_ = parse_int_checked(argv[5], "width");
        int imageD_ = parse_int_checked(argv[6], "depth");

        // Corner cases
        if (imageH_ <= 0 || imageW_ <= 0) {
            throw std::runtime_error("height and width must be > 0");
        }
        if (imageD_ < 0) {
            throw std::runtime_error("depth must be >= 0");
        }

        validate_mode_and_paths(mode, input, output);

        // Construct ECC
        ECC ecc(imageH_, imageW_, imageD_);

        if (mode == "b") {
            std::vector<std::string> fileNames = fileNames_from_folder(const_cast<std::string&>(input));
            std::vector<std::string> fileNameso = compose_outfileNames_from_folder(const_cast<std::string&>(input),
                const_cast<std::string&>(output));

            if (fileNames.empty()) {
                throw std::runtime_error("No input files found in folder: " + input);
            }
            if (fileNames.size() != fileNameso.size()) {
                throw std::runtime_error("Internal error: input/output filename counts differ.");
            }

            for (size_t i = 0; i < fileNames.size(); i++) {
                std::vector<float> res = ecc.run_frmFile(fileNames[i]);
                write_results(res, fileNameso[i]);
            }
        }
        else if (mode == "s") {
            std::vector<float> res = ecc.run_frmFile(input);
            write_results(res, output);
        }
        else {
            helper_(argv[0]);
        }

        // One sync + check at the end
        cudaError_t st = cudaDeviceSynchronize();
        if (st != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA error after synchronize: ") +
                cudaGetErrorString(st));
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return 1;
    }
}