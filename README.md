# GPU Heat Diffusion Simulation
This project simulates 2D heat distribution over time using a discrete Laplace operator. It uses a custom OpenCL kernel to leverage the massive parallelism of gpus.
## Solutions

### 1. Advanced Memory Management: Tiling & Async Copies
To minimize expensive global memory access, the OpenCL kernel uses a Tiled memory strategy:
* **Local Memory (SRAM):** Data is loaded into `__local` memory tiles. This allows worker threads within a work-group to share data for neighboring cells (the stencil) without hitting global VRAM.
* **Asynchronous 2D Copies:** I utilized the `cl_khr_extended_async_copies` extension. Using `async_work_group_copy_2D2D` allows the hardware to perform DMA (Direct Memory Access) transfers in the background, overlapping data movement with computation where possible.

### 2. Stencil Computation & Boundary Handling
The simulation uses a 5-point stencil to update each "cell" based on its neighbors and a diffusion constant.
* **Boundary Conditions** The implementation handles boundaries by "skipping" edges where the constant temperature is fixed at zero.
* **Double Buffering:** The host code implements a "ping-pong" buffer strategy, swapping input and output pointers after each iteration to avoid redundant memory copies between the GPU and CPU.

### 3. Compilation Pipeline
The project includes a dedicated "compiler" utility that:
* Reads the OpenCL `.cl` source.
* Compiles it for the specific GPU architecture detected at runtime.
* Serializes the resulting program binary to disk. 
* The main simulation can then load this binary directly for near-instant startup times in subsequent runs.
* All this is done automatically when calling make. 

---

## Performance Insights
By using Local Memory Tiling, the kernel achieves much higher memory bandwidth utilization than a "naive" implementation. The transition from $O(n)$ global memory reads to localized shared access significantly reduces the "memory wall" bottleneck typical in stencil-based simulations.



## Usage
The program expects an `init` file containing grid dimensions and starting hot-spots.

```bash
# Compile the kernel and the host program
make

# Run the simulation
# -n: Number of time steps
# -d: Diffusion constant
./diffusion -n100 -d0.02
