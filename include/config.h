#ifndef CONFIG_H
#define CONFIG_H

#include <stdint.h>

#define USE_FLOAT

#ifdef USE_FLOAT
  typedef float ARGTYPE;
  #define TILE_W 16
  #define TILE_H 4
#else
  typedef double ARGTYPE;
  #define TILE_W 8
  #define TILE_H 8
#endif


typedef struct{
  ARGTYPE* data;
  uint32_t width;
  uint32_t height;
} Matrix;

#endif // !CONFIG_H
