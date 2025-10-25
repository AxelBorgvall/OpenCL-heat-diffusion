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
  __local event_t events[S_H_F];

  // Boundary skips for Y
  int upskip = (group_y0 == 0) ? 1 : 0;
  int downskip = (group_y0 + TILE_H_F >= height) ? group_y0 + TILE_H_F + 1 - height : 0;
  int firsty = group_y0 - 1 + upskip;
  int num_rows = S_H_F - upskip - downskip;

  // Boundary skips for X
  int leftskip = (group_x0 == 0) ? 1 : 0;
  int rightskip = (group_x0 + TILE_W_F >= width) ? group_x0 + TILE_W_F + 1 - width : 0;
  int start_x = group_x0 - 1 + leftskip;
  int copy_w = S_W_F - leftskip - rightskip;

  // Collective asynchronous loading: all threads issue the same copy calls for each row
  int num_events = 0;
  for (int r = 0; r < num_rows; ++r) {
    int start_y = firsty + r;
    const __global float *src = input + start_y * width + start_x;
    __local float *dst = tile + (r + upskip) * S_W_F + leftskip;

    events[r] = async_work_group_copy(dst, src, copy_w, 0);
    num_events++;
  }

  // Wait collectively for all copies to complete (no barrier needed here, as params are uniform)
  wait_group_events(num_events, events);

  // Compute diffusion step
  if (gx > 0 && gx < width - 1 && gy > 0 && gy < height - 1) {
    float h_center = tile[(ly + 1) * S_W_F + (lx + 1)];
    float up    = tile[(ly)     * S_W_F + (lx + 1)];
    float down  = tile[(ly + 2) * S_W_F + (lx + 1)];
    float left  = tile[(ly + 1) * S_W_F + (lx)];
    float right = tile[(ly + 1) * S_W_F + (lx + 2)];

    float neighbor_avg = 0.25f * (up + down + left + right);
    float outv = h_center - c * h_center + c * neighbor_avg;
    output[gy * width + gx] = outv;
  }
}
