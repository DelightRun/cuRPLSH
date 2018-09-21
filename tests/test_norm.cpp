#include <initializer_list>
#include <iostream>

#include "dataset.h"
#include "device_resources.h"
#include "device_tensor.h"
#include "internal/constants.h"
#include "internal/norm.h"
#include "timer.h"

typedef std::chrono::high_resolution_clock::time_point time_point;
time_point now() { return std::chrono::high_resolution_clock::now(); }

double getmilliseconds(time_point start, time_point end) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() /
         1000.0;
}

int main(int argc, const char** argv) {
  int device = getCudaDevice(argc, argv);

  curplsh::DeviceScope scope(device);

  curplsh::DatasetSIFT sift("/home/changxu/Datasets/sift/");

  int dim = sift.getDimension();
  int num = sift.getNumBase();

  curplsh::DeviceTensor<float, 2> vectors(const_cast<float*>(sift.getBase()),
                                          {num, dim}, sift.getMemorySpace());
  curplsh::DeviceTensor<float, 1> norms({num}, sift.getMemorySpace());

  for (int i = 0; i < 100; ++i) {
    curplsh::DeviceTimer timer;
    curplsh::computeL2Norm(vectors, norms, true);
  }

  float* hostNorms = (float*)malloc(num * sizeof(float));
  norms.toHost(hostNorms);
  for (int i = 0; i < 100; i++) {
    std::cout << "Norm #" << i << " = " << hostNorms[i] << std::endl;
  }
  for (int i = norms.getSize(0) - 100; i < norms.getSize(0); i++) {
    std::cout << "Norm #" << i << " = " << hostNorms[i] << std::endl;
  }

  return 0;
}
