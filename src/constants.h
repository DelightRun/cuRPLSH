#pragma once

#include <limits>

namespace curplsh {

typedef unsigned int uint;
typedef unsigned short ushort;

constexpr int kWarpSize = 32;
constexpr int kMaxThreadsPerBlock = 1024;

constexpr float kFloatMin = std::numeric_limits<float>::lowest();
constexpr float kFloatMax = std::numeric_limits<float>::max();

constexpr int kIntMin = std::numeric_limits<int>::lowest();
constexpr int kIntMax = std::numeric_limits<int>::max();

constexpr uint kUintMin = std::numeric_limits<uint>::lowest();
constexpr uint kUintMax = std::numeric_limits<uint>::max();

}
