#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
    #ifdef TEMPLATEMATCHER_EXPORTS
        #define TM_API __declspec(dllexport)
    #else
        #define TM_API __declspec(dllimport)
    #endif
#else
    #define TM_API __attribute__((visibility("default")))
#endif

// 不透明句柄
typedef void* TM_MatcherHandle;
typedef void* TM_ImageHandle;

// 匹配结果结构
typedef struct {
    double centerX;
    double centerY;
    double angle;
    double score;
    double ptLT_x, ptLT_y;
    double ptRT_x, ptRT_y;
    double ptRB_x, ptRB_y;
    double ptLB_x, ptLB_y;
} TM_MatchResult;

// 匹配配置结构
typedef struct {
    int maxPos;
    double maxOverlap;
    double score;
    double toleranceAngle;
    int minReduceArea;
    int useSIMD;
    int subPixelEstimation;
    int bitwiseNot;
    int stopLayer1;
    int toleranceRange;
    double tolerance1;
    double tolerance2;
    double tolerance3;
    double tolerance4;
} TM_MatchConfig;

// 序列化类型
typedef enum {
  TM_FMT_BINARY          = 0,
  TM_FMT_BINARY_PORTABLE = 1,
  TM_FMT_JSON            = 2,
  TM_FMT_XML             = 3
} TM_Format;

// 创建匹配器
TM_API TM_MatcherHandle TM_CreateMatcher();

// 销毁匹配器
TM_API void TM_DestroyMatcher(TM_MatcherHandle handle);

// 学习模板（从文件路径）
TM_API int TM_LearnPatternFromFile(TM_MatcherHandle handle, const char* filepath);

// 学习模板（从内存数据，BGR格式，width x height）
TM_API int TM_LearnPatternFromData(TM_MatcherHandle handle, const unsigned char* data, 
                                   int width, int height, int channels);

// 写入模板到文件
TM_API int TM_WritePatternToFile(TM_MatcherHandle handle, const char* filepath, TM_Format format);

// 从文件读取模板
TM_API int TM_ReadPatternFromFile(TM_MatcherHandle handle, const char* filepath, TM_Format format);

// 写入模板到内存
TM_API int TM_WritePatternToData(TM_MatcherHandle handle, unsigned char* data,
                                 int dataSize, int* writeSize, TM_Format format);

// 从内存读取模板
TM_API int TM_ReadPatternFromData(TM_MatcherHandle handle, const unsigned char* data,
                                  int size, TM_Format format);

// 执行匹配（从文件路径）
TM_API int TM_MatchFromFile(TM_MatcherHandle handle, const char* sourceFile, 
                            const TM_MatchConfig* config, TM_MatchResult* results, 
                            int maxResults, double* executionTimeMs);

// 执行匹配（从内存数据）
TM_API int TM_MatchFromData(TM_MatcherHandle handle, const unsigned char* data, 
                            int width, int height, int channels,
                            const TM_MatchConfig* config, TM_MatchResult* results, 
                            int maxResults, double* executionTimeMs);

// 在图像上绘制匹配结果（从文件路径）
TM_API int TM_DrawMatchesToFile(const char* inputFile, const char* outputFile,
                                const TM_MatchResult* results, int numResults);

// 在图像上绘制匹配结果（从内存数据）
TM_API int TM_DrawMatchesToData(const unsigned char* inputData, int width, int height, 
                                int channels, unsigned char* outputData,
                                const TM_MatchResult* results, int numResults);

// 获取默认配置
TM_API void TM_GetDefaultConfig(TM_MatchConfig* config);

#ifdef __cplusplus
}
#endif
