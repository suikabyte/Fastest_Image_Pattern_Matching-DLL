#pragma once

#include <cstdint>

// SIMD 卷积函数声明
int32_t IM_Conv_SIMD(uint8_t* pCharKernel, uint8_t* pCharConv, int iLength);
