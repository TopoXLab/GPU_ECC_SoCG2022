# GPU ECC
Authors: Fan Wang, Hubert Wagner, Chao Chen <br/>
Maintainer: Fan Wang
## Introduction ##
GPU ECC is an optimized GPU implementation of the Euler Characteristic Curve computation for 2D and 3D grayscale images.
## Installation ##

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
