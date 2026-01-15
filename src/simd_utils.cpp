#include "simd_utils.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    // x86/x64 平台使用 SSE2
    #include <emmintrin.h>

    inline int32_t hsum_epi32_sse2(__m128i x) {
        __m128i hi64 = _mm_unpackhi_epi64(x, x);
        __m128i sum64 = _mm_add_epi32(hi64, x);
        __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
        __m128i sum32 = _mm_add_epi32(sum64, hi32);
        return _mm_cvtsi128_si32(sum32);
    }

    int32_t IM_Conv_SIMD(uint8_t* pCharKernel, uint8_t* pCharConv, int iLength) {
        const int iBlockSize = 16, Block = iLength / iBlockSize;
        __m128i SumV = _mm_setzero_si128();
        __m128i Zero = _mm_setzero_si128();

        for (int Y = 0; Y < Block * iBlockSize; Y += iBlockSize) {
            __m128i SrcK = _mm_loadu_si128((__m128i*)(pCharKernel + Y));
            __m128i SrcC = _mm_loadu_si128((__m128i*)(pCharConv + Y));
            __m128i SrcK_L = _mm_unpacklo_epi8(SrcK, Zero);
            __m128i SrcK_H = _mm_unpackhi_epi8(SrcK, Zero);
            __m128i SrcC_L = _mm_unpacklo_epi8(SrcC, Zero);
            __m128i SrcC_H = _mm_unpackhi_epi8(SrcC, Zero);
            __m128i SumT = _mm_add_epi32(_mm_madd_epi16(SrcK_L, SrcC_L), 
                                         _mm_madd_epi16(SrcK_H, SrcC_H));
            SumV = _mm_add_epi32(SumV, SumT);
        }

        int32_t Sum = hsum_epi32_sse2(SumV);

        for (int Y = Block * iBlockSize; Y < iLength; Y++) {
            Sum += pCharKernel[Y] * pCharConv[Y];
        }

        return Sum;
    }

#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
    // ARM 平台使用 NEON
    #include <arm_neon.h>

    inline int32_t vaddvq_s32(int32x4_t v) {
        int32x2_t tmp = vpadd_s32(vget_low_s32(v), vget_high_s32(v));
        return vget_lane_s32(vpadd_s32(tmp, tmp), 0);
    }

    int32_t IM_Conv_SIMD(uint8_t* pCharKernel, uint8_t* pCharConv, int iLength) {
        const int iBlockSize = 16, Block = iLength / iBlockSize;
        int32x4_t SumV = vdupq_n_s32(0);
        uint8x16_t Zero = vdupq_n_u8(0);

        for (int Y = 0; Y < Block * iBlockSize; Y += iBlockSize) {
            uint8x16_t SrcK = vld1q_u8(pCharKernel + Y);
            uint8x16_t SrcC = vld1q_u8(pCharConv + Y);
            int16x8_t SrcK_L = vmovl_u8(vget_low_u8(SrcK));
            int16x8_t SrcK_H = vmovl_u8(vget_high_u8(SrcK));
            int16x8_t SrcC_L = vmovl_u8(vget_low_u8(SrcC));
            int16x8_t SrcC_H = vmovl_u8(vget_high_u8(SrcC));
            int32x4_t SumT = vaddq_s32(vmull_s16(vget_low_s16(SrcK_L), vget_low_s16(SrcC_L)), 
                                      vmull_s16(vget_low_s16(SrcK_H), vget_low_s16(SrcC_H)));
            SumV = vaddq_s32(SumV, SumT);
        }

        int32_t Sum = vaddvq_s32(SumV);

        for (int Y = Block * iBlockSize; Y < iLength; Y++) {
            Sum += pCharKernel[Y] * pCharConv[Y];
        }

        return Sum;
    }

#else
    // 通用实现（无SIMD优化）
    int32_t IM_Conv_SIMD(uint8_t* pCharKernel, uint8_t* pCharConv, int iLength) {
        int32_t Sum = 0;
        for (int Y = 0; Y < iLength; Y++) {
            Sum += pCharKernel[Y] * pCharConv[Y];
        }
        return Sum;
    }

#endif
