#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

namespace TemplateMatching {

// 常量定义
constexpr double VISION_TOLERANCE = 0.0000001;
constexpr double D2R = CV_PI / 180.0;
constexpr double R2D = 180.0 / CV_PI;
constexpr int MATCH_CANDIDATE_NUM = 5;

// 模板数据结构
struct TemplData {
    std::vector<cv::Mat> vecPyramid;
    std::vector<cv::Scalar> vecTemplMean;
    std::vector<double> vecTemplNorm;
    std::vector<double> vecInvArea;
    std::vector<bool> vecResultEqual1;
    bool bIsPatternLearned;
    int iBorderColor;

    TemplData() : bIsPatternLearned(false), iBorderColor(0) {}
    
    void clear() {
        std::vector<cv::Mat>().swap(vecPyramid);
        std::vector<double>().swap(vecTemplNorm);
        std::vector<double>().swap(vecInvArea);
        std::vector<cv::Scalar>().swap(vecTemplMean);
        std::vector<bool>().swap(vecResultEqual1);
    }
    
    void resize(int iSize) {
        vecTemplMean.resize(iSize);
        vecTemplNorm.resize(iSize, 0);
        vecInvArea.resize(iSize, 1);
        vecResultEqual1.resize(iSize, false);
    }
};

// 匹配参数结构
struct MatchParameter {
    cv::Point2d pt;
    double dMatchScore;
    double dMatchAngle;
    cv::Rect rectRoi;
    double dAngleStart;
    double dAngleEnd;
    cv::RotatedRect rectR;
    cv::Rect rectBounding;
    bool bDelete;

    double vecResult[3][3];  // for subpixel
    int iMaxScoreIndex;      // for subpixel
    bool bPosOnBorder;
    cv::Point2d ptSubPixel;
    double dNewAngle;

    MatchParameter() 
        : dMatchScore(0), dMatchAngle(0), bDelete(false), 
          dNewAngle(0.0), bPosOnBorder(false), iMaxScoreIndex(0) {
        memset(vecResult, 0, sizeof(vecResult));
    }

    MatchParameter(cv::Point2f ptMinMax, double dScore, double dAngle)
        : pt(ptMinMax), dMatchScore(dScore), dMatchAngle(dAngle),
          bDelete(false), dNewAngle(0.0), bPosOnBorder(false), iMaxScoreIndex(0) {
        memset(vecResult, 0, sizeof(vecResult));
    }
};

// 单个目标匹配结果
struct SingleTargetMatch {
    cv::Point2d ptLT, ptRT, ptRB, ptLB, ptCenter;
    double dMatchedAngle;
    double dMatchScore;
};

// 块最大值结构（用于优化）
struct BlockMax {
    struct Block {
        cv::Rect rect;
        double dMax;
        cv::Point ptMaxLoc;
        Block() {}
        Block(cv::Rect rect_, double dMax_, cv::Point ptMaxLoc_)
            : rect(rect_), dMax(dMax_), ptMaxLoc(ptMaxLoc_) {}
    };

    std::vector<Block> vecBlock;
    cv::Mat matSrc;

    BlockMax() {}
    BlockMax(const cv::Mat& matSrc_, cv::Size sizeTemplate);
    void UpdateMax(cv::Rect rectIgnore);
    void GetMaxValueLoc(double& dMax, cv::Point& ptMaxLoc);
};

// 匹配参数配置
struct MatchConfig {
    int iMaxPos = 70;
    double dMaxOverlap = 0.0;
    double dScore = 0.5;
    double dToleranceAngle = 0.0;
    int iMinReduceArea = 256;
    bool bUseSIMD = true;
    bool bSubPixelEstimation = false;
    bool bBitwiseNot = false;
    bool bStopLayer1 = false;  // FastMode: 设置为true时粗匹配，牺牲精度提升速度
    bool bToleranceRange = false;
    double dTolerance1 = 40.0;
    double dTolerance2 = 60.0;
    double dTolerance3 = -110.0;
    double dTolerance4 = -100.0;
};

// 匹配结果
struct MatchResult {
    std::vector<SingleTargetMatch> matches;
    double executionTimeMs = 0.0;
    bool success = false;
};

// 模板匹配器主类
class TemplateMatcher {
public:
    TemplateMatcher();
    ~TemplateMatcher();

    // 学习模板（从图像）
    bool LearnPattern(const cv::Mat& templateImage);

    // 执行匹配
    MatchResult Match(const cv::Mat& sourceImage, const MatchConfig& config = MatchConfig());

    // 获取模板数据（用于保存/加载）
    const TemplData& GetTemplateData() const { return m_TemplData; }

    // 设置模板数据（用于加载）
    void SetTemplateData(const TemplData& data) { m_TemplData = data; }

    // 检查是否已学习模板
    bool IsPatternLearned() const { return m_TemplData.bIsPatternLearned; }

private:
    TemplData m_TemplData;

    // 核心算法函数
    int GetTopLayer(const cv::Mat* matTempl, int iMinDstLength);
    void MatchTemplate(cv::Mat& matSrc, TemplData* pTemplData, cv::Mat& matResult, 
                       int iLayer, bool bUseSIMD);
    void GetRotatedROI(cv::Mat& matSrc, cv::Size size, cv::Point2f ptLT, 
                       double dAngle, cv::Mat& matROI);
    void CCOEFF_Denominator(cv::Mat& matSrc, TemplData* pTemplData, 
                            cv::Mat& matResult, int iLayer);
    cv::Size GetBestRotationSize(cv::Size sizeSrc, cv::Size sizeDst, double dRAngle);
    cv::Point2f ptRotatePt2f(cv::Point2f ptInput, cv::Point2f ptOrg, double dAngle);
    void FilterWithScore(std::vector<MatchParameter>* vec, double dScore);
    void FilterWithRotatedRect(std::vector<MatchParameter>* vec, int iMethod, 
                               double dMaxOverLap);
    cv::Point GetNextMaxLoc(cv::Mat& matResult, cv::Point ptMaxLoc, 
                            cv::Size sizeTemplate, double& dMaxValue, double dMaxOverlap);
    cv::Point GetNextMaxLoc(cv::Mat& matResult, cv::Point ptMaxLoc, 
                            cv::Size sizeTemplate, double& dMaxValue, 
                            double dMaxOverlap, BlockMax& blockMax);
    void SortPtWithCenter(std::vector<cv::Point2f>& vecSort);
    bool SubPixEstimation(std::vector<MatchParameter>* vec, double* dNewX, 
                          double* dNewY, double* dNewAngle, double dAngleStep, 
                          int iMaxScoreIndex);
};

// 可视化函数
namespace Visualization {
    // 在图像上绘制匹配结果
    void DrawMatchResult(cv::Mat& image, const std::vector<SingleTargetMatch>& matches,
                        const cv::Scalar& color = cv::Scalar(0, 255, 0), 
                        int thickness = 1, bool drawLabels = true);

    // 绘制虚线
    void DrawDashLine(cv::Mat& matDraw, cv::Point ptStart, cv::Point ptEnd,
                     const cv::Scalar& color1 = cv::Scalar(0, 0, 255),
                     const cv::Scalar& color2 = cv::Scalar(255, 255, 255));

    // 绘制十字标记
    void DrawMarkCross(cv::Mat& matDraw, int iX, int iY, int iLength,
                      const cv::Scalar& color = cv::Scalar(0, 255, 0),
                      int iThickness = 1);
}

} // namespace TemplateMatching
