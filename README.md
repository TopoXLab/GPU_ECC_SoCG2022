# GPU ECC
Authors: Fan Wang, Hubert Wagner, Chao Chen <br/>
Maintainer: Fan Wang
## Introduction ##
GPU ECC is an optimized GPU implementation of the Euler Characteristic Curve computation for 2D and 3D grayscale images.

## Installation ##
GPU ECC is tested under Windows only at the moment. We plan for a Linux version in the near future.

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
  <p>Make sure to check box "Grouped" and "Advanced" in CMake-GUI. If one or more of the dependencies are not installed at the default locations and cannot be found by CMake, you need to tell CMake where to find those dependencies.
    
    1. OpenCV: expand "Ungrouped Entries" and set "OpenCV_DIR" as the directory where you installed/compiled your openCV
    binaries. An example would be "D:/opencv/build/x64/vc14/lib". Click "Configure" in CMake-GUI.
    2. Boost: expand "Boost" and set both "Boost_DIR" and "Boost_INCLUDE_DIR" as the root folder of boost (e.g. D:/boost_1_77_0).
    Set "Boost_FILESYSTEM_LIBRARY_DEBUG" and "Boost_FILESYSTEM_LIBRARY_RELEASE" as the folder where you built your own boost
    binaries (e.g. D:/boost_1_77_0/lib64-msvc-14.2). Press "Configure" button. In some versions of CMake, another Boost entry
    will appear, expand it and make sure to set "Boost_LIBRARY_DIR_DEBUG" and "Boost_LIBRARY_DIR_DEBUG" with the same folder
    you used earlier for "Boost_FILESYSTEM_LIBRARY_DEBUG" and "Boost_FILESYSTEM_LIBRARY_RELEASE". Once all the errors go away,
    press "Generate".
  </p>
</details>
<details>
  <summary>4. Build GPU ECC with Visual Studio</summary>
  <p>Locate file "GPU_ECC.sln" under the "build" folder and open it with MSVC. Swith to "Release" mode and build the solution.</p>
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
