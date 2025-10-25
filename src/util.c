#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "../include/config.h"
#include "../include/util.h"


Matrix readfile(FILE *f) {
  Matrix mat;
  mat.data = NULL;
  mat.width = mat.height = 0;
  char line[256];

  if (!fgets(line, sizeof(line), f)) {
    fprintf(stderr, "Error reading dimensions\n");
    return mat;
  }
  if (sscanf(line, "%u %u", &mat.width, &mat.height) != 2) {
    fprintf(stderr, "Invalid dimensions format\n");
    return mat;
  }

  mat.height+=2;
  mat.width+=2;

  mat.data = calloc((size_t)mat.height * mat.width, sizeof(ARGTYPE));
  if (!mat.data) { perror("calloc"); return mat; }

  while (fgets(line, sizeof(line), f)) {
    uint32_t x, y;
    double val;
    char *token = strtok(line, " \t\n");
    if (!token) continue;
    x = (uint32_t)strtoul(token, NULL, 10);

    token = strtok(NULL, " \t\n");
    if (!token) continue;
    y = (uint32_t)strtoul(token, NULL, 10);

    token = strtok(NULL, " \t\n");
    if (!token) continue;
    val = strtod(token, NULL);

    if ( x > mat.width || y > mat.height) {
      fprintf(stderr, "Invalid coordinates: %u %u\n", x, y);
      continue;
    }

    // convert 1-based to 0-based
    uint32_t xi = x+1;
    uint32_t yi = y+1;
    mat.data[yi * mat.width + xi] = (ARGTYPE)val;
  }
  return mat;
}

void freemat(Matrix *mat) {
  if (!mat) return;
  if (mat->data) free(mat->data);
  mat->data = NULL;
  mat->width = mat->height = 0;
}
