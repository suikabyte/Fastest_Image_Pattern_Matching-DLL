#include "TemplateMatcher.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace TemplateMatching;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <template_image> <source_image> [output_image]" << std::endl;
        return 1;
    }

    // 创建匹配器
    TemplateMatcher matcher;

    // 学习模板
    std::cout << "Learning pattern from: " << argv[1] << std::endl;
    cv::Mat templateImg = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (templateImg.empty()) {
        std::cerr << "Failed to load template image: " << argv[1] << std::endl;
        return 1;
    }

    if (!matcher.LearnPattern(templateImg)) {
        std::cerr << "Failed to learn pattern" << std::endl;
        return 1;
    }
    std::cout << "Pattern learned successfully" << std::endl;

    // 配置匹配参数
    MatchConfig config;
    config.iMaxPos = 10;
    config.dScore = 0.6;
    config.dToleranceAngle = 30.0;
    config.bUseSIMD = true;
    config.bSubPixelEstimation = true;

    // 执行匹配
    std::cout << "Matching in: " << argv[2] << std::endl;
    cv::Mat sourceImg = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    if (sourceImg.empty()) {
        std::cerr << "Failed to load source image: " << argv[2] << std::endl;
        return 1;
    }

    MatchResult result = matcher.Match(sourceImg, config);

    // 处理结果
    if (result.success) {
        std::cout << "Found " << result.matches.size() << " matches in " 
                  << result.executionTimeMs << " ms" << std::endl;

        for (size_t i = 0; i < result.matches.size(); i++) {
            const auto& match = result.matches[i];
            std::cout << "Match " << i << ":" << std::endl;
            std::cout << "  Score: " << match.dMatchScore << std::endl;
            std::cout << "  Angle: " << match.dMatchedAngle << " degrees" << std::endl;
            std::cout << "  Center: (" << match.ptCenter.x << ", " 
                      << match.ptCenter.y << ")" << std::endl;
        }

        // 可视化
        cv::Mat visImg = sourceImg.clone();
        cv::cvtColor(visImg, visImg, cv::COLOR_GRAY2BGR);
        Visualization::DrawMatchResult(visImg, result.matches);

        const char* outputFile = (argc >= 4) ? argv[3] : "result.bmp";
        cv::imwrite(outputFile, visImg);
        std::cout << "Result saved to: " << outputFile << std::endl;
    } else {
        std::cout << "No matches found" << std::endl;
    }

    return 0;
}
