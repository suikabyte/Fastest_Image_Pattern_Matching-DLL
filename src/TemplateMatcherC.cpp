#include "TemplateMatcherC.h"
#include "TemplateMatcher.h"
#include <opencv2/opencv.hpp>
#include <cstring>

using namespace TemplateMatching;

extern "C" {

TM_MatcherHandle TM_CreateMatcher() {
    return new TemplateMatcher();
}

void TM_DestroyMatcher(TM_MatcherHandle handle) {
    if (handle) {
        delete static_cast<TemplateMatcher*>(handle);
    }
}

int TM_LearnPatternFromFile(TM_MatcherHandle handle, const char* filepath) {
    if (!handle || !filepath)
        return 0;
    
    TemplateMatcher* matcher = static_cast<TemplateMatcher*>(handle);
    cv::Mat img = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
    if (img.empty())
        return 0;
    
    return matcher->LearnPattern(img) ? 1 : 0;
}

int TM_LearnPatternFromData(TM_MatcherHandle handle, const unsigned char* data, 
                           int width, int height, int channels) {
    if (!handle || !data || width <= 0 || height <= 0)
        return 0;
    
    TemplateMatcher* matcher = static_cast<TemplateMatcher*>(handle);
    cv::Mat img;
    
    if (channels == 1) {
        img = cv::Mat(height, width, CV_8UC1, (void*)data);
    } else if (channels == 3) {
        img = cv::Mat(height, width, CV_8UC3, (void*)data);
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    } else {
        return 0;
    }
    
    return matcher->LearnPattern(img) ? 1 : 0;
}

int TM_MatchFromFile(TM_MatcherHandle handle, const char* sourceFile, 
                    const TM_MatchConfig* config, TM_MatchResult* results, 
                    int maxResults, double* executionTimeMs) {
    if (!handle || !sourceFile || !config || !results || maxResults <= 0)
        return 0;
    
    TemplateMatcher* matcher = static_cast<TemplateMatcher*>(handle);
    cv::Mat sourceImg = cv::imread(sourceFile, cv::IMREAD_GRAYSCALE);
    if (sourceImg.empty())
        return 0;
    
    MatchConfig matchConfig;
    matchConfig.iMaxPos = config->maxPos;
    matchConfig.dMaxOverlap = config->maxOverlap;
    matchConfig.dScore = config->score;
    matchConfig.dToleranceAngle = config->toleranceAngle;
    matchConfig.iMinReduceArea = config->minReduceArea;
    matchConfig.bUseSIMD = config->useSIMD != 0;
    matchConfig.bSubPixelEstimation = config->subPixelEstimation != 0;
    matchConfig.bBitwiseNot = config->bitwiseNot != 0;
    matchConfig.bStopLayer1 = config->stopLayer1 != 0;
    matchConfig.bToleranceRange = config->toleranceRange != 0;
    matchConfig.dTolerance1 = config->tolerance1;
    matchConfig.dTolerance2 = config->tolerance2;
    matchConfig.dTolerance3 = config->tolerance3;
    matchConfig.dTolerance4 = config->tolerance4;
    
    MatchResult result = matcher->Match(sourceImg, matchConfig);
    
    if (executionTimeMs)
        *executionTimeMs = result.executionTimeMs;
    
    if (!result.success)
        return 0;
    
    int count = std::min((int)result.matches.size(), maxResults);
    for (int i = 0; i < count; i++) {
        const auto& match = result.matches[i];
        results[i].centerX = match.ptCenter.x;
        results[i].centerY = match.ptCenter.y;
        results[i].angle = match.dMatchedAngle;
        results[i].score = match.dMatchScore;
        results[i].ptLT_x = match.ptLT.x;
        results[i].ptLT_y = match.ptLT.y;
        results[i].ptRT_x = match.ptRT.x;
        results[i].ptRT_y = match.ptRT.y;
        results[i].ptRB_x = match.ptRB.x;
        results[i].ptRB_y = match.ptRB.y;
        results[i].ptLB_x = match.ptLB.x;
        results[i].ptLB_y = match.ptLB.y;
    }
    
    return count;
}

int TM_MatchFromData(TM_MatcherHandle handle, const unsigned char* data, 
                    int width, int height, int channels,
                    const TM_MatchConfig* config, TM_MatchResult* results, 
                    int maxResults, double* executionTimeMs) {
    if (!handle || !data || !config || !results || maxResults <= 0 || 
        width <= 0 || height <= 0)
        return 0;
    
    TemplateMatcher* matcher = static_cast<TemplateMatcher*>(handle);
    cv::Mat sourceImg;
    
    if (channels == 1) {
        sourceImg = cv::Mat(height, width, CV_8UC1, (void*)data).clone();
    } else if (channels == 3) {
        sourceImg = cv::Mat(height, width, CV_8UC3, (void*)data);
        cv::cvtColor(sourceImg, sourceImg, cv::COLOR_BGR2GRAY);
    } else {
        return 0;
    }
    
    MatchConfig matchConfig;
    matchConfig.iMaxPos = config->maxPos;
    matchConfig.dMaxOverlap = config->maxOverlap;
    matchConfig.dScore = config->score;
    matchConfig.dToleranceAngle = config->toleranceAngle;
    matchConfig.iMinReduceArea = config->minReduceArea;
    matchConfig.bUseSIMD = config->useSIMD != 0;
    matchConfig.bSubPixelEstimation = config->subPixelEstimation != 0;
    matchConfig.bBitwiseNot = config->bitwiseNot != 0;
    matchConfig.bStopLayer1 = config->stopLayer1 != 0;
    matchConfig.bToleranceRange = config->toleranceRange != 0;
    matchConfig.dTolerance1 = config->tolerance1;
    matchConfig.dTolerance2 = config->tolerance2;
    matchConfig.dTolerance3 = config->tolerance3;
    matchConfig.dTolerance4 = config->tolerance4;
    
    MatchResult result = matcher->Match(sourceImg, matchConfig);
    
    if (executionTimeMs)
        *executionTimeMs = result.executionTimeMs;
    
    if (!result.success)
        return 0;
    
    int count = std::min((int)result.matches.size(), maxResults);
    for (int i = 0; i < count; i++) {
        const auto& match = result.matches[i];
        results[i].centerX = match.ptCenter.x;
        results[i].centerY = match.ptCenter.y;
        results[i].angle = match.dMatchedAngle;
        results[i].score = match.dMatchScore;
        results[i].ptLT_x = match.ptLT.x;
        results[i].ptLT_y = match.ptLT.y;
        results[i].ptRT_x = match.ptRT.x;
        results[i].ptRT_y = match.ptRT.y;
        results[i].ptRB_x = match.ptRB.x;
        results[i].ptRB_y = match.ptRB.y;
        results[i].ptLB_x = match.ptLB.x;
        results[i].ptLB_y = match.ptLB.y;
    }
    
    return count;
}

int TM_DrawMatchesToFile(const char* inputFile, const char* outputFile,
                        const TM_MatchResult* results, int numResults) {
    if (!inputFile || !outputFile || !results || numResults <= 0)
        return 0;
    
    cv::Mat img = cv::imread(inputFile);
    if (img.empty())
        return 0;
    
    std::vector<SingleTargetMatch> matches;
    for (int i = 0; i < numResults; i++) {
        SingleTargetMatch match;
        match.ptCenter = cv::Point2d(results[i].centerX, results[i].centerY);
        match.dMatchedAngle = results[i].angle;
        match.dMatchScore = results[i].score;
        match.ptLT = cv::Point2d(results[i].ptLT_x, results[i].ptLT_y);
        match.ptRT = cv::Point2d(results[i].ptRT_x, results[i].ptRT_y);
        match.ptRB = cv::Point2d(results[i].ptRB_x, results[i].ptRB_y);
        match.ptLB = cv::Point2d(results[i].ptLB_x, results[i].ptLB_y);
        matches.push_back(match);
    }
    
    Visualization::DrawMatchResult(img, matches);
    
    return cv::imwrite(outputFile, img) ? 1 : 0;
}

int TM_DrawMatchesToData(const unsigned char* inputData, int width, int height, 
                        int channels, unsigned char* outputData,
                        const TM_MatchResult* results, int numResults) {
    if (!inputData || !outputData || !results || numResults <= 0 || 
        width <= 0 || height <= 0)
        return 0;
    
    cv::Mat img;
    if (channels == 1) {
        img = cv::Mat(height, width, CV_8UC1, (void*)inputData).clone();
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    } else if (channels == 3) {
        img = cv::Mat(height, width, CV_8UC3, (void*)inputData).clone();
    } else {
        return 0;
    }
    
    std::vector<SingleTargetMatch> matches;
    for (int i = 0; i < numResults; i++) {
        SingleTargetMatch match;
        match.ptCenter = cv::Point2d(results[i].centerX, results[i].centerY);
        match.dMatchedAngle = results[i].angle;
        match.dMatchScore = results[i].score;
        match.ptLT = cv::Point2d(results[i].ptLT_x, results[i].ptLT_y);
        match.ptRT = cv::Point2d(results[i].ptRT_x, results[i].ptRT_y);
        match.ptRB = cv::Point2d(results[i].ptRB_x, results[i].ptRB_y);
        match.ptLB = cv::Point2d(results[i].ptLB_x, results[i].ptLB_y);
        matches.push_back(match);
    }
    
    Visualization::DrawMatchResult(img, matches);
    
    if (channels == 1) {
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        std::memcpy(outputData, gray.data, width * height);
    } else {
        std::memcpy(outputData, img.data, width * height * 3);
    }
    
    return 1;
}

void TM_GetDefaultConfig(TM_MatchConfig* config) {
    if (!config)
        return;
    
    config->maxPos = 70;
    config->maxOverlap = 0.0;
    config->score = 0.5;
    config->toleranceAngle = 0.0;
    config->minReduceArea = 256;
    config->useSIMD = 1;
    config->subPixelEstimation = 0;
    config->bitwiseNot = 0;
    config->stopLayer1 = 0;
    config->toleranceRange = 0;
    config->tolerance1 = 40.0;
    config->tolerance2 = 60.0;
    config->tolerance3 = -110.0;
    config->tolerance4 = -100.0;
}

} // extern "C"
