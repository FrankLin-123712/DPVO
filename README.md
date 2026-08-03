# Deep Patch Visual Odometry/SLAM
This repository contains the source code for our papers:

[Deep Patch Visual Odometry](https://arxiv.org/pdf/2208.04726.pdf)<br/>
Zachary Teed<sup>\*</sup>, Lahav Lipson<sup>\*</sup>, Jia Deng <sub></sub><br/>
[Deep Patch Visual SLAM](http://arxiv.org/pdf/2408.01654)<br/>
Lahav Lipson, Zachary Teed, Jia Deng<br/>
<a target="_blank" href="https://colab.research.google.com/drive/1VSFGNB7YCveqKF7XNz4RlV9EnfQA3fhQ?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a><a target="_blank" href="https://github.com/princeton-vl/DPVO_Docker">
  <img src="https://img.shields.io/badge/Docker-grey?logo=Docker" alt="Open In Colab"/>
</a>

[<img src="https://i.imgur.com/6ZQPbR1.png?1" width="600">](https://www.youtube.com/watch?v=e5wanf71YFs)

```
@article{teed2023deep,
   title={Deep Patch Visual Odometry},
   author={Teed, Zachary and Lipson, Lahav and Deng, Jia},
   journal={Advances in Neural Information Processing Systems},
   year={2023}
 }
```
```
@inproceedings{lipson2024deep,
    author={Lipson, Lahav and Teed, Zachary and Deng, Jia},
    title={{Deep Patch Visual SLAM}},
    booktitle={European Conference on Computer Vision},
    year={2024}
}
```
## Setup and Installation
The code was tested on Ubuntu 20/22 and Cuda 11/12.</br>

Clone the repo
```
git clone https://github.com/princeton-vl/DPVO.git --recursive
cd DPVO
```
Create and activate the dpvo anaconda environment
```
conda env create -f environment.yml
conda activate dpvo
```

Next install the DPVO package
```bash
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty

# install DPVO
# DPVO's setup.py imports torch while building CUDA extensions. With recent pip
# versions, plain `pip install .` may fail because pip builds in an isolated env
# that cannot see the PyTorch package installed in the active conda env.
# Use --no-build-isolation so setup.py can use that active PyTorch install.
#
# CUDA_HOME and TORCH_CUDA_ARCH_LIST are machine-dependent:
#   1. CUDA_HOME must point to a CUDA toolkit that provides nvcc.
#   2. The nvcc CUDA major version should match torch.version.cuda.
#      For example, PyTorch cu121 should use a CUDA 12.x toolkit.
#   3. TORCH_CUDA_ARCH_LIST should match the target GPU compute capability.
#      8.6+PTX is suitable for many Ampere GPUs, but may not be optimal for
#      Turing, Volta, Ada, Hopper, or other GPU generations.
#
# Check the active PyTorch CUDA version with:
#   python -c "import torch; print(torch.__version__, torch.version.cuda)"
# Check the selected CUDA compiler with:
#   $CUDA_HOME/bin/nvcc --version
#
# Example for this machine:
CUDA_HOME=/usr/local/cuda-12.3 TORCH_CUDA_ARCH_LIST="8.6+PTX" pip install --no-build-isolation .

# download models and data (~2GB)
./download_models_and_data.sh
```


### Recommended - Install the Pangolin Viewer
Note: You will need to have CUDA 11 and CuDNN installed on your system.

1. Step 1: Install Pangolin (need the custom version included with the repo)
```
./Pangolin/scripts/install_prerequisites.sh recommended
mkdir Pangolin/build && cd Pangolin/build
cmake ..
make -j8
sudo make install
cd ../..
```

2. Step 2: Install the viewer
```bash
pip install ./DPViewer
```

For installation issues, our [Docker Image](https://github.com/princeton-vl/DPVO_Docker) supports the visualizer.

### Classical Backend (optional)

We provide a classical backend for closing very large loops, which requires extra installation.

Step 1. Install the OpenCV C++ API. On Ubuntu, you can use
```bash
sudo apt-get install -y libopencv-dev
```
Step 2. Install DBoW2
```bash
cd DBoW2
mkdir -p build && cd build
cmake .. # tested with cmake 3.22.1 and gcc/cc 11.4.0 on Ubuntu
make # tested with GNU Make 4.3
sudo make install
cd ../..
```

Step 3. Install the image retrieval
```bash
pip install ./DPRetrieval
```

## Demos
DPVO can be run on any video or image directory with a single command. Note you will need to have installed DPViewer to visualize the reconstructions in real-time. You can also save the completed reconstructions and view them in COLMAP. The pretrained models can be downloaded from google drive [models.zip](https://drive.google.com/file/d/1dRqftpImtHbbIPNBIseCv9EvrlHEnjhX/view?usp=sharing) if you have not already run the download script. 


```bash
python demo.py \
    --imagedir=<path to image directory or video> \
    --calib=<path to calibration file> \
    --viz # enable visualization
    --plot # save trajectory plot
    --save_ply # save point cloud as a .ply file
    --save_trajectory # save the predicted trajectory as .txt in TUM format
    --save_colmap # save point cloud + trajectory in the standard COLMAP text format
```

### iPhone
```bash
python demo.py --imagedir=movies/IMG_0492.MOV --calib=calib/iphone.txt --stride=5 --plot --viz
```

### TartanAir
Download a sequence from [TartanAir](https://theairlab.org/tartanair-dataset/) (several samples are availabe from download directly from the webpage)
```bash
python demo.py --imagedir=<path to image_left> --calib=calib/tartan.txt --stride=1 --plot --viz
```

### EuRoC
Download a sequence from [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) (download ASL format)
```bash
python demo.py --imagedir=<path to mav0/cam0/data/> --calib=calib/euroc.txt --stride=2 --plot --viz
```

## SLAM Backends
To run DPVO with a SLAM backend (i.e., DPV-SLAM), add
```bash
--opts LOOP_CLOSURE True
```
to any `evaluate_X.py` script or to `demo.py`

If installed, the classical backend can also be enabled using 
```
--opts CLASSIC_LOOP_CLOSURE True
```

## Evaluation
We provide evaluation scripts for TartanAir, EuRoC, TUM-RGBD and ICL-NUIM. Up to date result logs on these datasets can be found in the `logs` directory.

### TartanAir:
Results on the validation split and test set can be obtained with the command:
```
python evaluate_tartan.py --trials=5 --split=validation --plot --save_trajectory
```

### EuRoC:
```
python evaluate_euroc.py --trials=5 --plot --save_trajectory
```

### TUM-RGBD:
```
python evaluate_tum.py --trials=5 --plot --save_trajectory
```

### ICL-NUIM:
```
python evaluate_icl_nuim.py --trials=5 --plot --save_trajectory
```

### KITTI:
```
python evaluate_kitti.py --trials=5 --plot --save_trajectory
```

## Tools

The scripts under `tools/` cover ONNX export, workload estimation, runtime parity data generation, and video frame extraction. Run the commands below from the repository root after activating the `dpvo` environment. Use `python tools/<path-to-script>.py --help` for the complete option list.

| Script | Purpose |
| --- | --- |
| `export2onnx/export2onnx.sh` | Export the feature encoder and update block with the standard defaults. |
| `export2onnx/export_models.py` | Configurable ONNX exporter used by `export2onnx/export2onnx.sh`. |
| `export2onnx/verify_onnxmodel.py` | Print ONNX metadata and run the ONNX structural checker. |
| `analyze_dpvo_workload.py` | Statically estimate module-level MACs, memory traffic, and operational intensity. |
| `gen_testdata/gen_testdata.sh` | Generate the predefined end-to-end and component parity datasets. |
| `gen_testdata/generate_dpvo_python_testdata.py` | Generate a resized input sequence and golden outputs from the Python DPVO tracker. |
| `gen_testdata/generate_dpvo_runner_parity_testdata.py` | Generate patchify, correlation, update, and bundle-adjustment parity cases. |
| `turn_mov2png.sh` | Convert every `.mov`/`.MOV` file in a directory to PNG frames. |
| `gen_testdata/dpvo_runner_parity_common.py` | Internal support module for the runner parity generator; it is not a standalone command. |

### ONNX export and validation

The ONNX commands require the Python `onnx` package in addition to the installed DPVO environment.

Export both supported models:

```bash
./tools/export2onnx/export2onnx.sh
```

`export2onnx/export2onnx.sh` uses `$HOME/miniconda3/envs/dpvo/bin/python` when available and otherwise uses `python` from `PATH`. Its defaults can be overridden with environment variables:

```bash
PYTHON_BIN=/path/to/python \
WEIGHTS=./dpvo.pth \
OUT_DIR=./exported_models \
HEIGHT=480 WIDTH=640 EDGES=256 OPSET=13 \
./tools/export2onnx/export2onnx.sh
```

For finer control, invoke the exporter directly:

```bash
python tools/export2onnx/export_models.py \
    --weights ./dpvo.pth \
    --out ./exported_models \
    --height 480 \
    --width 640 \
    --edges 256 \
    --opset 13
```

Add `--skip-feature` or `--skip-update` to export only one model. The command writes:

- `exported_models/feature_extractor.onnx`
- `exported_models/update_block.onnx`

The feature extractor accepts `images` and returns `fmap` and `imap`. It contains only the image encoders; Python `patchify` logic and custom DPVO runtime operations are not included.

The update block accepts `net`, `ctx`, `corr`, `ii`, `jj`, `kk`, `ix`, and `jx`, and returns `net_out`, `delta`, and `weight`. The runtime must compute the correlation/context tensors and graph indices, including the explicit previous/next neighbor links `ix` and `jx`, before inference. This explicit-neighbor interface is the supported replacement for exporting `fastba.neighbors(...)`.

Inspect either model and run `onnx.checker`:

```bash
python tools/export2onnx/verify_onnxmodel.py exported_models/update_block.onnx
```

The default model, when the positional path is omitted, is `exported_models/update_block.onnx`.

### Workload and roofline estimation

`analyze_dpvo_workload.py` is a dependency-free static estimator, not a runtime profiler. It models the feature encoder, update block, correlation lookup, BA Jacobian construction, and Schur complement using `config/default.yaml` unless another config is supplied.

Print the default Markdown table:

```bash
python tools/analyze_dpvo_workload.py
```

Estimate a particular graph and classify each module using a hardware roofline:

```bash
python tools/analyze_dpvo_workload.py \
    --height 480 --width 640 \
    --edges 8192 --unique-patches 960 \
    --peak-macs 1T --bandwidth 100GB/s
```

Results can be written as Markdown, CSV, or JSON:

```bash
python tools/analyze_dpvo_workload.py \
    --edge-mode new-frame \
    --format csv \
    --output workload.csv
```

Without `--edges`, the edge count is estimated from `PATCHES_PER_FRAME`, `PATCH_LIFETIME`, and `REMOVAL_WINDOW`. Use `--edge-mode steady` for the active steady-state graph or `--edge-mode new-frame` for newly appended edges. Roofline arguments must be supplied in pairs: either `--peak-macs` with `--bandwidth`, or `--peak-macs-per-cycle` with `--bandwidth-per-cycle`.

### Test and parity data generation

These generators expect an installed DPVO package and `dpvo.pth`. The end-to-end and runner parity generators also execute DPVO CUDA/custom operations; use a CUDA-enabled DPVO environment. Generated tensors are raw contiguous `.bin` files described by a `manifest.txt` file.

Input frames are discovered in filename order and may be PNG or JPEG. Calibration files must contain at least `fx fy cx cy`; additional values are treated as distortion coefficients by the end-to-end generator.

Generate all predefined datasets:

```bash
./tools/gen_testdata/gen_testdata.sh 0
```

The mode selects which dataset to generate:

| Mode | Output under `testdata/` | Description |
| --- | --- | --- |
| `0` | All outputs below | Generate every predefined dataset. This is also the default when no mode is given. |
| `1` | `dpvo_runner_parity_small/` | Four small component parity cases. |
| `2` | `dpvo_python_medium_fast/` | 32-frame end-to-end case, 256-pixel maximum long edge, 16 patches per frame. |
| `3` | `dpvo_python_medium/` | 32-frame end-to-end case, 512-pixel maximum long edge, 64 patches per frame. |

The wrapper defaults to `dpvo.pth`, `subset_0493/`, `calib/iphone.txt`, and `testdata/`. Override them when using another sequence:

```bash
PYTHON_BIN=/path/to/python \
WEIGHTS=/path/to/dpvo.pth \
IMAGES=/path/to/frames \
CALIB=/path/to/calib.txt \
TESTDATA_ROOT=/path/to/testdata \
./tools/gen_testdata/gen_testdata.sh 1
```

To customize an end-to-end case directly:

```bash
python tools/gen_testdata/generate_dpvo_python_testdata.py \
    --weights ./dpvo.pth \
    --images ./subset_0493 \
    --calib ./calib/iphone.txt \
    --output-root ./testdata/dpvo_python_small \
    --frame-start 1 --frame-count 12 --frame-step 1 \
    --max-long-edge 256 \
    --patches-per-frame 32 \
    --seed 7 --dump-state
```

At least eight frames are required. `--width` and `--height` may be supplied together instead of `--max-long-edge`; output dimensions are aligned down to multiples of 16 and cannot exceed the source dimensions. The output contains preprocessed frames, adjusted calibration, patch-center and bootstrap-depth data, metadata, golden tensors, and a TUM-format golden trajectory. `--dump-state` adds the final patch-graph state and update trace. `--dump-update-parity-cases` additionally writes one update-block case per tracker update. Existing `images/`, `golden/`, `centers/`, and `bootstrap_depths/` directories under the selected output root are replaced.

Generate the four component-level cases directly:

```bash
python tools/gen_testdata/generate_dpvo_runner_parity_testdata.py \
    --weights ./dpvo.pth \
    --images ./subset_0493 \
    --calib ./calib/iphone.txt \
    --output-root ./testdata/dpvo_runner_parity_small \
    --width 256 --height 144 \
    --frame-start 1 --frame-count 4 \
    --patches-per-frame 8
```

This creates `patchify_small/`, `correlation_small/`, `update_small/`, and `bundle_adjustment_small/`. Its defaults match the wrapper defaults for `dpvo.pth`, `subset_0493/`, `calib/iphone.txt`, and `testdata/dpvo_runner_parity_small/`. Bundle-adjustment golden generation requires CUDA.

### Convert MOV videos to PNG sequences

`turn_mov2png.sh` requires `ffmpeg`. By default it converts every `.mov`/`.MOV` under `movies/` at 30 FPS and writes each video to `sequences/<video-name>/`:

```bash
./tools/turn_mov2png.sh
```

Override the input, output, or frame rate with environment variables:

```bash
INPUT_DIR=/path/to/movies \
OUTPUT_DIR=/path/to/sequences \
FPS=15 \
./tools/turn_mov2png.sh
```

## Training
Make sure you have run `./download_models_and_data.sh`. Your directory structure should look as follows

```Shell
├── datasets
    ├── TartanAir.pickle
    ├── TartanAir
        ├── abandonedfactory
        ├── abandonedfactory_night
        ├── ...
        ├── westerndesert
    ...
```

To train (log files will be written to `runs/<your name>`). Model will be run on the validation split every 10k iterations
```
python train.py --steps=240000 --lr=0.00008 --name=<your name>
```

## Change Log
* **Aug 2022**: Initial release
* **Sep 2022**: Add link to docker
* **Mar 2023**: Google Colab, TUM + ICL-NUIM evaluation code, flags for saving output
* **July 2024**: Add DPV-SLAM. Update output-saving utilities.


## Acknowledgements
* Our Viewer is adapted from DSO.
