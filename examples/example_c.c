#include "TemplateMatcherC.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <template_image> <source_image> [output_image]\n", argv[0]);
        return 1;
    }

    // 创建匹配器
    TM_MatcherHandle matcher = TM_CreateMatcher();
    if (!matcher) {
        fprintf(stderr, "Failed to create matcher\n");
        return 1;
    }

    // 学习模板
    printf("Learning pattern from: %s\n", argv[1]);
    if (!TM_LearnPatternFromFile(matcher, argv[1])) {
        fprintf(stderr, "Failed to learn pattern\n");
        TM_DestroyMatcher(matcher);
        return 1;
    }
    printf("Pattern learned successfully\n");

    // 配置匹配参数
    TM_MatchConfig config;
    TM_GetDefaultConfig(&config);
    config.maxPos = 10;
    config.score = 0.6;
    config.toleranceAngle = 30.0;
    config.useSIMD = 1;
    config.subPixelEstimation = 1;

    // 执行匹配
    printf("Matching in: %s\n", argv[2]);
    TM_MatchResult results[100];
    double execTime;
    int count = TM_MatchFromFile(matcher, argv[2], &config, results, 100, &execTime);

    // 处理结果
    if (count > 0) {
        printf("Found %d matches in %.2f ms\n", count, execTime);

        for (int i = 0; i < count; i++) {
            printf("Match %d:\n", i);
            printf("  Score: %.3f\n", results[i].score);
            printf("  Angle: %.2f degrees\n", results[i].angle);
            printf("  Center: (%.2f, %.2f)\n", results[i].centerX, results[i].centerY);
        }

        // 可视化
        const char* outputFile = (argc >= 4) ? argv[3] : "result.bmp";
        if (TM_DrawMatchesToFile(argv[2], outputFile, results, count)) {
            printf("Result saved to: %s\n", outputFile);
        } else {
            fprintf(stderr, "Failed to save result image\n");
        }
    } else {
        printf("No matches found\n");
    }

    // 清理
    TM_DestroyMatcher(matcher);
    return 0;
}
