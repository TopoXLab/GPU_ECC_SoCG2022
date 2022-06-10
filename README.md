# GPU ECC
Authors: Fan Wang, Hubert Wagner, Chao Chen <br/>
Maintainer: Fan Wang
## Introduction ##
GPU ECC is an optimized GPU implementation of the Euler Characteristic Curve computation for 2D and 3D grayscale images.

## Installation ##
GPU ECC is tested under Windows only at the moment. We plan for a Linux version in the near future. <br/>
Dependencies <br/>
* boost
Tools <br/>
* CMake 3.14 or above: [link](https://cmake.org/download/)
* Microsoft Visual Studio 2019: [link](https://visualstudio.microsoft.com/vs/older-downloads/)

## How to compile ##
Skip this part if you are familiar with using CMake for compilation. <br/>
<details>
  <summary>1. Setup source and destination folder</summary>
  <p>In CMake-GUI, the folder where you downloaded the source files will be the "source" folder. Create a folder named "build" as the desination folder where the compiled binaries will be saved.</p>
</details>
<details>
  <summary>2. Specify compiler</summary>
  <p>Choose Visual Studio 16 2019 as the compiler. Other compilers are not tested.</p>
</details>
<details>
  <summary>3. Missing dependencies</summary>
  <p>Make sure to check box "Grouped" and "Advanced" in CMake-GUI. If one or more of the dependencies are not installed at the default locations and cannot be found by CMake, you need to tell CMake where to find those dependencies.</p>
    1. dsf
    2. sdf
  
</details>

## Run from command line ##
To run GPU ECC from command line, go into the folder where the executable is located and type: <br/>
`GPU_ECC.exe [mode] [input_name] [output_name] [height] [width] [depth]` <br/>
Arguments:
<pre>
--mode:         GPU ECC can compute for a single file or a batch of files. Use 's' for single mode or 'b' for batch mode.
--input_name:   Path to a single file in single mode or a directory containing files in batch mode.
--output_name:  Path to a single flie in single mode or a directory in batch mode. In case of batch mode, the output file 
                will have the same name as input file.
--height:       Height of the input file. In case of batch mode, same height is assumed for every file under the directory.
--width:        Width of the input file. In case of batch mode, same width is assumed for every file under the directory.
--depth:        Depth of the input file. In case of 2D file, set depth to 0.
</pre>
An example command: <br/>
`GPU_ECC.exe b C:/input_directory C:/output_directory 256 256 0` <br/>
