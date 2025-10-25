// Tunable tile size:
// assume mem fed is padded, no 0 BC
#define TILE_W_D 8 // will need to load 10 length rows
#define TILE_H_D 8 // will need to load 4 of them
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void diffusion_step_d(
  __global double *input,  // flattened input image (row-major)
  __global double *output,       // flattened output image (row-major)
  const int width,
  const int height,
  const double c)
{
  //global coords
  const int gx = get_global_id(0); // x in [0, global_size_x)
  const int gy = get_global_id(1); // y in [0, global_size_y)

  //local coords
  const int lx = get_local_id(0);  // in [0, get_local_size(0))
  const int ly = get_local_id(1);
  
  //group coords (upperleft corner)
  const int group_x0 = get_group_id(0) * TILE_W_D;
  const int group_y0 = get_group_id(1) * TILE_H_D;

  //allocate local memory for the tile
  S_W=TILE_W_D+2;
  S_H=TILE_H_D+2;
  __local double tile[S_H*S_W];

  const int local_size_x = get_local_size(0);
  const int local_size_y = get_local_size(1);

  int esize=sizeof(double);
  int leftadd=(gx!=0)?1:0;
  int rightadd=(gx>=width-1)?width-1-gx:1;

  int topadd=(gy!=0)?1:0;
  int botadd=(gy>=height-1)?height-1-gy:1;

  
  /*
  we want to load height+topadd+botadd rows of length 
  width + leftadd + rightadd
  starting at idx [-leftadd,-topadd]
  */
  int workeridx=lx+ly*S_W;

  if (workeridx<TILE_H+botadd+topadd){
    memcpy(&tile+(S_W)*workeridx,input+(group_y0-topadd)*width+(group_x0-leftadd),(TILE_W+leftadd+rightadd)*esize);
  }

  //wait for whole tile to be loaded
  barrier(CLK_LOCAL_MEM_FENCE);

  //ensure coords are within actual window not including bdy
  if (gx>0&&gx < width-1 &&gy>0&& gy < height-1) {

    const int cx = lx + 1; // 1..TILE_W
    const int cy = ly + 1; // 1..TILE_H
    
    //access relevant elements
    double h_center = tile[cy * S_W + cx];

    double up    = tile[(cy - 1) * S_W + cx];
    double down  = tile[(cy + 1) * S_W + cx];
    double left  = tile[cy * S_W + (cx - 1)];
    double right = tile[cy * S_W + (cx + 1)];

    //neighbor average
    double neighbor_avg = 0.25 * (up + down + left + right);

    //compute and set output
    double outv = h_center-c*h_center + c * neighbor_avg ;
    output[gy * width + gx] = outv;
  }
}






// Tunable tile size:
// assume mem fed is padded, no 0 BC

#define TILE_W_F 16 // will need to load 10 length rows
#define TILE_H_F 4 // will need to load 4 of them

//#define TILE_W_F 8
//#define TILE_H_F 8

__kernel void diffusion_step_f(
  __global float *input,  // flattened input image (row-major)
  __global float *output,       // flattened output image (row-major)
  const int width,
  const int height,
  const float c)
{
  //global coords
  const int gx = get_global_id(0); // x in [0, global_size_x)
  const int gy = get_global_id(1); // y in [0, global_size_y)

  //local coords
  const int lx = get_local_id(0);  // in [0, get_local_size(0))
  const int ly = get_local_id(1);
  
  //group coords (upperleft corner)
  const int group_x0 = get_group_id(0) * TILE_W_D;
  const int group_y0 = get_group_id(1) * TILE_H_D;

  //allocate local memory for the tile
  S_W=TILE_W_D+2;
  S_H=TILE_H_D+2;
  __local float tile[S_H*S_W];

  const int local_size_x = get_local_size(0);
  const int local_size_y = get_local_size(1);

  int esize=sizeof(float);
  int leftadd=(gx!=0)?1:0;
  int rightadd=(gx>=width-1)?width-1-gx:1;

  int topadd=(gy!=0)?1:0;
  int botadd=(gy>=height-1)?height-1-gy:1;

  
  /*
  we want to load height+topadd+botadd rows of length 
  width + leftadd + rightadd
  starting at idx [-leftadd,-topadd]
  */
  int workeridx=lx+ly*S_W;

  if (workeridx<TILE_H+botadd+topadd){
    memcpy(&tile+(S_W)*workeridx,input+(group_y0-topadd)*width+(group_x0-leftadd),(TILE_W+leftadd+rightadd)*esize);
  }

  //wait for whole tile to be loaded
  barrier(CLK_LOCAL_MEM_FENCE);

  //ensure coords are within actual window not including bdy
  if (gx>0&&gx < width-1 &&gy>0&& gy < height-1) {

    const int cx = lx + 1; // 1..TILE_W
    const int cy = ly + 1; // 1..TILE_H
    
    //access relevant elements
    float h_center = tile[cy * S_W + cx];

    float up    = tile[(cy - 1) * S_W + cx];
    float down  = tile[(cy + 1) * S_W + cx];
    float left  = tile[cy * S_W + (cx - 1)];
    float right = tile[cy * S_W + (cx + 1)];

    //neighbor average
    float neighbor_avg = 0.25 * (up + down + left + right);

    //compute and set output
    float outv = h_center-c*h_center + c * neighbor_avg ;
    output[gy * width + gx] = outv;
  }
}
















