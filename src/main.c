#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

#include "../include/config.h"
#include "../include/util.h"

static void check_cl(cl_int err, const char *msg) {
  if (err != CL_SUCCESS) {
    fprintf(stderr, "OpenCL error %d: %s\n", err, msg);
    exit(1);
  }
}

int main(int argc, char *argv[]) {
  // argument parsing
  uint32_t n = 1;        // number of iterations (unused here, we perform one step)
  ARGTYPE d = 0.01;

  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], "-n", 2) == 0) {
      if (argv[i][2] != '\0') n = atoi(argv[i] + 2);
      else if (++i < argc) n = atoi(argv[i]);
    } else if (strncmp(argv[i], "-d", 2) == 0) {
      if (argv[i][2] != '\0') d = atof(argv[i] + 2);
      else if (++i < argc) d = atof(argv[i]);
    }
  }

  FILE *init = fopen("init", "r");
  if (!init) { perror("open init"); return 1; }
  Matrix input = readfile(init);
  fclose(init);
  if (!input.data) { fprintf(stderr,"Failed to read input\n"); return 1; }

  // prepare output matrix
  Matrix output;
  output.width  = input.width;
  output.height = input.height;
  output.data = calloc((size_t)output.width * output.height, sizeof(ARGTYPE));
  if (!output.data) { perror("calloc output"); freemat(&input); return 1; }

  // OpenCL setup
  cl_int err;
  cl_platform_id platform = NULL;
  cl_device_id device = NULL;
  err = clGetPlatformIDs(1, &platform, NULL);
  check_cl(err, "clGetPlatformIDs");


  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  if (err != CL_SUCCESS) {
    // fallback to CPU if GPU not present
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    check_cl(err, "clGetDeviceIDs CPU fallback");
  }

  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  check_cl(err, "clCreateContext");

  // create command queue (legacy)
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
  check_cl(err, "clCreateCommandQueue");

  // try to open precompiled binary first
  FILE* fbin = fopen("bin/diffstep.bin", "rb");
  cl_program program = NULL;
  if (fbin) {
    fseek(fbin, 0, SEEK_END);
    size_t bin_size = ftell(fbin);
    rewind(fbin);
    unsigned char* bin = (unsigned char*)malloc(bin_size);
    if (!bin) { fclose(fbin); fprintf(stderr,"malloc bin\n"); goto compile_source; }
    if (fread(bin, 1, bin_size, fbin) != bin_size) { free(bin); fclose(fbin); goto compile_source; }
    fclose(fbin);

    const unsigned char* bins[] = { bin };
    program = clCreateProgramWithBinary(context, 1, &device, &bin_size, bins, NULL, &err);
    free(bin);
    check_cl(err, "clCreateProgramWithBinary");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
      // fallthrough to compile source if binary build fails
      clReleaseProgram(program);
      program = NULL;
      fprintf(stderr, "Binary build failed, will try compiling source\n");
      goto compile_source;
    }
  } else {
    compile_source:
    // compile from diffstep.cl (fallback)
    FILE* fsrc = fopen("src/diffstep.cl", "r");
    if (!fsrc) { fprintf(stderr,"No binary and cannot open diffstep.cl\n"); goto cleanup; }
    fseek(fsrc, 0, SEEK_END);
    size_t src_size = ftell(fsrc);
    rewind(fsrc);
    char* src = malloc(src_size + 1);
    fread(src,1,src_size,fsrc);
    src[src_size] = '\0';
    fclose(fsrc);
    program = clCreateProgramWithSource(context, 1, (const char**)&src, &src_size, &err);
    free(src);
    check_cl(err, "clCreateProgramWithSource");
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
      size_t log_size = 0;
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
      char *log = malloc(log_size + 1);
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
      log[log_size] = '\0';
      fprintf(stderr, "Build log:\n%s\n", log);
      free(log);
      goto cleanup;
    }
  }

  // kernel name must match file: diffusion_step
  #ifdef USE_FLOAT
    cl_kernel kernel = clCreateKernel(program, "diffusion_step_f", &err);
  #else
    cl_kernel kernel = clCreateKernel(program, "diffusion_step_d", &err);
  #endif
 
  check_cl(err, "clCreateKernel");

  // create buffers
  size_t nbytes = (size_t)input.width * input.height * sizeof(ARGTYPE);


  cl_mem g_input  = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbytes, input.data, &err);
  check_cl(err, "clCreateBuffer g_input");
  cl_mem g_output = clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &err);
  check_cl(err, "clCreateBuffer g_output");

  // set kernel args (kernel signature: (__global const ARGTYPE*, __global float*, int width, int height, float c))
  int kw = (int)input.width;
  int kh = (int)input.height;
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &g_input);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &g_output);
  err |= clSetKernelArg(kernel, 2, sizeof(int), &kw);
  err |= clSetKernelArg(kernel, 3, sizeof(int), &kh);
  err |= clSetKernelArg(kernel, 4, sizeof(ARGTYPE), &d);
  check_cl(err, "clSetKernelArg");

  // compute global/local sizes (make them size_t)
  size_t local[2] = { TILE_W, TILE_H };
  size_t global[2];
  global[0] = ((input.width  + TILE_W - 1) / TILE_W) * TILE_W;
  global[1] = ((input.height + TILE_H - 1) / TILE_H) * TILE_H;

  for (uint32_t step = 0; step < n; ++step) {
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
    check_cl(err, "clEnqueueNDRangeKernel");

    cl_mem tmp = g_input;
    g_input = g_output;
    g_output = tmp;

    // update kernel args
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &g_input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &g_output);
    check_cl(err, "clSetKernelArg swap");
  }

  // g_input now contains the final result
  err = clEnqueueReadBuffer(queue, g_input, CL_TRUE, 0, nbytes, output.data, 0, NULL, NULL);
  check_cl(err, "clEnqueueReadBuffer final output");


  ARGTYPE avg=0;
  ARGTYPE denom_inv=(double)1/(input.width*input.height);
  uint64_t N=(uint64_t)kw*kh;
  for (uint64_t i=0; i<N; ++i) {
    avg+=output.data[i];
  }
  avg*=denom_inv;
  printf("average: %f\n",avg);
  ARGTYPE avg_abs_diff=0;
  for (uint64_t i=0; i<N; ++i) {
    avg_abs_diff+=fabs(output.data[i]-avg);
  }
  avg_abs_diff*=denom_inv;
  printf("average absolute difference: %f\n",avg_abs_diff);


  // cleanup
  cleanup:
  if (g_input) clReleaseMemObject(g_input);
  if (g_output) clReleaseMemObject(g_output);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);

  freemat(&input);
  freemat(&output);

  return 0;
}
