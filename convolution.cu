#include <iostream>
#include <string>
#include "dnn.cuh"

#ifndef Sy
  #define Sy 1
  #define Sx 1
#endif

#ifndef Tnn
  #define Tnn 32
  #define Tn  16
  #define Ti  16
  #define Ty  8
  #define Tx  8
#endif

#define NYPAD (Ny + Ky)
#define NXPAD (Nx + Kx)
#define NYSCL (Ny / Sy)
#define NXSCL (Nx / Sx)

#define SYNAPSE_SIZE (1L * Ky * Kx * Nn * Ni)

__global__ void generate_synapse (VTYPE (&synapse)[Ky][Kx][Nn][Ni]) {
  int k_row = blockIdx.y * blockDim.y + threadIdx.y;
  int k_col = blockIdx.x * blockDim.x + threadIdx.x;
  printf("blockIdx.x: %d, blockDim.x: %d, threadIdx.x: %d blockIdx.y: %d, blockDim.y: %d, threadIdx.y: %d row: %d, col: %d\n", blockIdx.x, blockDim.x, threadIdx.x, blockIdx.y, blockDim.y, threadIdx.y, k_row, k_col);
  for (int nn = 0; nn < Nn; ++nn) {
    for (int ni = 0; ni < Ni; ++ni) {
      synapse[k_row][k_col][nn][ni] = 1;
    }
  }
}

__device__ VTYPE d_transfer(VTYPE i) {
  return (i > 0) ? i : i / 4;
}

void fill_convolution_shared_simple (VTYPE (&synapse)[Ky][Kx][Nn][Ni], VTYPE (&neuron_i)[NYPAD][NXPAD][Ni]) {
  for (int yy = 0; yy < Ky; ++yy) {
    for (int xx = 0; xx < Kx; ++xx) {
      for (int nn = 0; nn < Nn; ++nn) {
        for (int ni = 0; ni < Ni; ++ni) {
          synapse[yy][xx][nn][ni] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 0.5f;
        }
      }
    }
  }
  for (int yy = 0; yy < NYPAD; ++yy) {
    for (int xx = 0; xx < NXPAD; ++xx) {
      for (int ni = 0; ni < Ni; ++ni) {
        neuron_i[yy][xx][ni] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 0.5f;
      }
    }
  }
}

__global__ void d_convolution_layer (VTYPE (&synapse)[Ky][Kx][Nn][Ni], VTYPE (&neuron_i)[NYPAD][NXPAD][Ni], VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
  VTYPE sum[Nn] = {0};

  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  // int yout = 0;
  // for (int y = 0; y < Ny; y += Sy)  {
  //   int xout = 0;
  //   for (int x = 0; x < Nx; x += Sx) {
      for (int nn = 0; nn < Nn; nn += Tn) {
        for (int n = nn; n < nn + Tn; ++n) {
          sum[n] = 0;
        }

        for (int ky = 0; ky < Ky; ++ky) {
          for (int kx = 0; kx < Kx; ++kx) {
            for (int n = nn; n < nn + Tn; ++n) {
              for (int i = 0; i < Ni; ++i) {
                VTYPE sv = synapse[ky][kx][n][i];
                VTYPE nv = neuron_i[ky + y][kx + x][i];
                sum[n] += sv * nv;
              }
            }
          }
        }

        for (int n = nn; n < nn + Tn; ++n) {
          neuron_n[y][x][n] = d_transfer(sum[n]);
        }
      }
  //     xout++;
  //   }
  //   yout++;
  // }
}

void convolution_layer (VTYPE (&synapse)[Ky][Kx][Nn][Ni], VTYPE (&neuron_i)[NYPAD][NXPAD][Ni], VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
  VTYPE sum[Nn] = {0};

  int yout = 0;
  for (int y = 0; y < Ny; y += Sy)  {
    int xout = 0;
    for (int x = 0; x < Nx; x += Sx) {
      for (int nn = 0; nn < Nn; nn += Tn) {
        for (int n = nn; n < nn + Tn; ++n) {
          sum[n] = 0;
        }

        for (int ky = 0; ky < Ky; ++ky) {
          for (int kx = 0; kx < Kx; ++kx) {
            for (int n = nn; n < nn + Tn; ++n) {
              for (int i = 0; i < Ni; ++i) {
                VTYPE sv = synapse[ky][kx][n][i];
                VTYPE nv = neuron_i[ky + y][kx + x][i];
                sum[n] += sv * nv;
              }
            }
          }
        }

        for (int n = nn; n < nn + Tn; ++n) {
          neuron_n[yout][xout][n] = transfer(sum[n]);
        }
      }
      xout++;
    }
    yout++;
  }
}

void convolution_layer_blocked(VTYPE (&synapse)[Ky][Kx][Nn][Ni], VTYPE (&neuron_i)[NYPAD][NXPAD][Ni], VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
  VTYPE sum[Nn]={0};

  for (int yy = 0; yy < Ny; yy += Ty) {
    for (int xx = 0; xx < Nx; xx += Tx) {
      for (int nnn = 0; nnn < Nn; nnn += Tnn) {
        int yout = yy/Sy;
        for (int y = yy; y < yy + Ty; y += Sy) { // tiling for y;
          int xout = xx/Sx;

          for (int x = xx; x < xx + Tx; x += Sx) { // tiling for x;

            for (int nn = nnn; nn < nnn + Tnn; nn += Tn) {
              for (int n = nn; n < nn + Tn; n++) {
                sum[n] = 0;
              }

              for (int ky = 0; ky < Ky; ky++) {  // sliding window;
                for (int kx = 0; kx < Kx; kx++) {

                  int ii = 0;
                  VTYPE sum_sc;

                  for (; ii < Ni -Ti+1; ii += Ti) {
                    for (int n = nn; n < nn + Tn; n++) {
                      sum_sc=0;
                      for (int i = ii; i < ii + Ti; i++) {
                        VTYPE sv = synapse[ky][kx][n][i];
                        VTYPE nv = neuron_i[ky + y][kx + x][i];
                        sum_sc+=sv*nv;
                      }
                      sum[n]+=sum_sc;
                    }
                  }
                }
              }
              //transfer
              for (int n = nn; n < nn + Tn; n++) {
                neuron_n[yout][xout][n] = transfer(sum[n]);
              }
            }
            xout++; 
          }
          yout++;
        }
      }
    }
  }
}

int main (const int argc, const char** argv) {

  VTYPE (*synapse)[Ky][Kx][Nn][Ni];
  VTYPE (*neuron_i)[NYPAD][NXPAD][Ni];
  VTYPE (*neuron_n)[NYSCL][NXSCL][Nn];
  VTYPE (*neuron_n2)[NYSCL][NXSCL][Nn];

  VTYPE (*d_synapse)[Ky][Kx][Nn][Ni];
  VTYPE (*d_neuron_i)[NYPAD][NXPAD][Ni];
  VTYPE (*d_neuron_n)[NYSCL][NXSCL][Nn];
  VTYPE (*d_neuron_n2)[NYSCL][NXSCL][Nn];

  std::cout << "allocating memory\n";
  
  // synapse = (VTYPE (*)[Ky][Kx][Nn][Ni]) aligned_malloc(64, SYNAPSE_SIZE * sizeof(VTYPE));
  // neuron_i = (VTYPE (*)[NYPAD][NXPAD][Ni]) aligned_malloc(64, NYPAD * NXPAD * Ni * sizeof(VTYPE));
  // neuron_n = (VTYPE (*)[NYSCL][NXSCL][Nn]) aligned_malloc(64, NYSCL * NXSCL * Nn * sizeof(VTYPE));
  // neuron_n2 = (VTYPE (*)[NYSCL][NXSCL][Nn]) aligned_malloc(64, NYSCL * NXSCL * Nn * sizeof(VTYPE));

  // d_synapse = (VTYPE (*)[Ky][Kx][Nn][Ni]) aligned_cuda_malloc(64, SYNAPSE_SIZE * sizeof(VTYPE));
  // d_neuron_i = (VTYPE (*)[NYPAD][NXPAD][Ni]) aligned_cuda_malloc(64, NYPAD * NXPAD * Ni * sizeof(VTYPE));
  // d_neuron_n = (VTYPE (*)[NYSCL][NXSCL][Nn]) aligned_cuda_malloc(64, NYSCL * NXSCL * Nn * sizeof(VTYPE));


  synapse = (VTYPE (*)[Ky][Kx][Nn][Ni]) malloc(SYNAPSE_SIZE * sizeof(VTYPE));
  neuron_i = (VTYPE (*)[NYPAD][NXPAD][Ni]) malloc(NYPAD * NXPAD * Ni * sizeof(VTYPE));
  neuron_n = (VTYPE (*)[NYSCL][NXSCL][Nn]) malloc(NYSCL * NXSCL * Nn * sizeof(VTYPE));
  neuron_n2 = (VTYPE (*)[NYSCL][NXSCL][Nn]) malloc(NYSCL * NXSCL * Nn * sizeof(VTYPE));

  cudaMalloc(&d_synapse, SYNAPSE_SIZE * sizeof(VTYPE));
  cudaMalloc(&d_neuron_i, NYPAD * NXPAD * Ni * sizeof(VTYPE));
  cudaMalloc(&d_neuron_n, NYSCL * NXSCL * Nn * sizeof(VTYPE));
  // cudaMalloc(&d_synapse, SYNAPSE_SIZE * sizeof(VTYPE));

  std::cout << "filling matrices\n";
  fill_convolution_shared_simple(*synapse, *neuron_i);

  // int threads = 16;
  // dim3 num_threads(threads, threads);
  // dim3 num_blocks((Ky + threads - 1) / threads, (Kx + threads - 1) / threads);
  // std::cout << threads << " " << (Ky + threads - 1) / threads << " " << (Kx + threads - 1) / threads << std::endl;
  // generate_synapse<<<num_blocks, num_threads>>>(*d_synapse);
  // cudaDeviceSynchronize(); 

  cudaMemcpy(d_synapse, synapse, SYNAPSE_SIZE * sizeof(VTYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(d_neuron_i, neuron_i, NYPAD * NXPAD * Ni * sizeof(VTYPE), cudaMemcpyHostToDevice);

  // cudaMemcpy(synapse, d_synapse, SYNAPSE_SIZE * sizeof(VTYPE), cudaMemcpyDeviceToHost);
  // std::cout << *synapse[0][0][0][0] << std::endl;
  // std::cout << *synapse[0][0][0][1] << std::endl;
  // std::cout << *synapse[0][0][0][2] << std::endl;
  // std::cout << *synapse[0][0][0][3] << std::endl;
  // std::cout << *synapse[0][0][0][1] << std::endl;
  // std::cout << *synapse[0][0][1][0] << std::endl;
  // std::cout << *synapse[0][1][0][0] << std::endl;
  // std::cout << *synapse[1][0][0][0] << std::endl;
  // std::cout << *d_synapse[0][0][0][0] << std::endl;

  int threads = 16;
  dim3 num_threads(threads, threads);
  dim3 num_blocks((Ny + threads - 1) / threads, (Nx + threads - 1) / threads);

  begin_roi();
  d_convolution_layer<<<num_blocks, num_threads>>>(*d_synapse, *d_neuron_i, *d_neuron_n);
  cudaDeviceSynchronize();
  end_roi();

  cudaMemcpy(neuron_n, d_neuron_n, NYSCL * NXSCL * Nn * sizeof(VTYPE), cudaMemcpyDeviceToHost);

  std::cout << *synapse[0][0][0][0] << std::endl;

  begin_roi();
  convolution_layer_blocked(*synapse,*neuron_i,*neuron_n2);
  end_roi();
  compare((VTYPE*)*neuron_n,(VTYPE*)*neuron_n2,NYSCL*NXSCL*Nn);

  cudaFree(d_synapse);
  cudaFree(d_neuron_i);
  cudaFree(d_neuron_n);
}
