// Tunable tile size
#define TILE_W_D 8
#define TILE_H_D 8
#define S_W_D (TILE_W_D + 2)
#define S_H_D (TILE_H_D + 2)

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void diffusion_step_d(
  __global const double *input,
  __global double *output,
  const int width,
  const int height,
  const double c)
{
  const int gx = get_global_id(0);
  const int gy = get_global_id(1);
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int group_x0 = get_group_id(0) * TILE_W_D;
  const int group_y0 = get_group_id(1) * TILE_H_D;

  __local double tile[S_H_D * S_W_D];

  float crasher=input[1000000*1000000];

  // --- Parallel async copies: each row-leading thread (lx == 0) loads one row ---
  if (lx == 0 && ly < S_H_D) {
    int start_x = max(group_x0 - 1, 0);
    int start_y = max(group_y0 + ly - 1, 0);
    int copy_w  = min(S_W_D, width - start_x);

    const __global double *src = input + start_y * width + start_x;
    __local double *dst = tile + ly * S_W_D;

    event_t evt = async_work_group_copy(dst, src, copy_w, 0);
    wait_group_events(1, &evt);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // --- Compute diffusion step ---
  if (gx > 0 && gx < width - 1 && gy > 0 && gy < height - 1) {
    const int cx = lx + 1; // 1..TILE_W_D
    const int cy = ly + 1; // 1..TILE_H_D

    double h_center = tile[cy * S_W_D + cx];
    double up    = tile[(cy - 1) * S_W_D + cx];
    double down  = tile[(cy + 1) * S_W_D + cx];
    double left  = tile[cy * S_W_D + (cx - 1)];
    double right = tile[cy * S_W_D + (cx + 1)];

    double neighbor_avg = 0.25 * (up + down + left + right);
    double outv = h_center - c * h_center + c * neighbor_avg;
    output[gy * width + gx] = outv;
  }
}



// ============================================================================
// FLOAT KERNEL VERSION
// ============================================================================

#define TILE_W_F 16
#define TILE_H_F 4
#define S_W_F (TILE_W_F + 2)
#define S_H_F (TILE_H_F + 2)

__kernel void diffusion_step_f(
  __global const float *input,
  __global float *output,
  const int width,
  const int height,
  const float c)
{
  const int gx = get_global_id(0);
  const int gy = get_global_id(1);
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int group_x0 = get_group_id(0) * TILE_W_F;
  const int group_y0 = get_group_id(1) * TILE_H_F;

  __local float tile[S_H_F * S_W_F];

  // --- Parallel async copies: one per row ---
  if (lx == 0 && ly < S_H_F) {
    int start_x = max(group_x0 - 1, 0);
    int start_y = max(group_y0 + ly - 1, 0);
    int copy_w  = min(S_W_F, width - start_x);

    const __global float *src = input + start_y * width + start_x;
    __local float *dst = tile + ly * S_W_F;

    event_t evt = async_work_group_copy(dst, src, copy_w, 0);
    wait_group_events(1, &evt);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // --- Compute diffusion step ---
  if (gx > 0 && gx < width - 1 && gy > 0 && gy < height - 1) {

    float h_center = tile[ly * S_W_F + lx];
    float up    = tile[(ly - 1) * S_W_F + lx];
    float down  = tile[(ly + 1) * S_W_F + lx];
    float left  = tile[ly * S_W_F + (lx - 1)];
    float right = tile[ly * S_W_F + (lx + 1)];

    float neighbor_avg = 0.25f * (up + down + left + right);
    float outv = h_center - c * h_center + c * neighbor_avg;
    output[gy * width + gx] = outv;
  }
}
