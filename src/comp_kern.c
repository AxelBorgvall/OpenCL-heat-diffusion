#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

int main(){
  FILE* f=fopen("src/diffstep.cl", "r");
  if (!f){perror("Failed ot open kernel source");return 1;}

  fseek(f, 0, SEEK_END);
  size_t src_size=ftell(f);
  rewind(f);

  char* src=(char*)malloc(src_size+1);

  fread(src,1,src_size,f);
  src[src_size]='\0';
  fclose(f);

  cl_platform_id platform;
  cl_device_id device;
  cl_int err;
  clGetPlatformIDs(1, &platform, NULL);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  cl_program program = clCreateProgramWithSource(context, 1, (const char**)&src, &src_size, &err);
  free(src);

  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char* log = (char*)malloc(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    printf("Build log:\n%s\n", log);
    free(log);
    return 1;
  }
  // Get binary
  size_t bin_size;
  clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_size, NULL);
  unsigned char* bin = (unsigned char*)malloc(bin_size);
  unsigned char* bins[] = { bin };
  clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(bins), bins, NULL);

  FILE* bout = fopen("bin/diffstep.bin", "wb");
  fwrite(bin, 1, bin_size, bout);
  fclose(bout);

  printf("Kernel compiled successfully to diffstep.bin\n");

  clReleaseProgram(program);
  clReleaseContext(context);
  free(bin);
  return 0;

}



