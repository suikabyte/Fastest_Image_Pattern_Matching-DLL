#include "TemplateMatcher.h"
#include "simd_utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <limits>

namespace TemplateMatching {

// 比较函数
static bool compareScoreBig2Small(const MatchParameter& lhs, const MatchParameter& rhs) {
    return lhs.dMatchScore > rhs.dMatchScore;
}

static bool comparePtWithAngle(const std::pair<cv::Point2f, double>& lhs, 
                                const std::pair<cv::Point2f, double>& rhs) {
    return lhs.second < rhs.second;
}

// BlockMax 实现
BlockMax::BlockMax(const cv::Mat& matSrc_, cv::Size sizeTemplate) {
    matSrc = matSrc_;
    int iBlockW = sizeTemplate.width * 2;
    int iBlockH = sizeTemplate.height * 2;

    int iCol = matSrc.cols / iBlockW;
    bool bHResidue = matSrc.cols % iBlockW != 0;

    int iRow = matSrc.rows / iBlockH;
    bool bVResidue = matSrc.rows % iBlockH != 0;

    if (iCol == 0 || iRow == 0) {
        vecBlock.clear();
        return;
    }

    vecBlock.resize(iCol * iRow);
    int iCount = 0;
    for (int y = 0; y < iRow; y++) {
        for (int x = 0; x < iCol; x++) {
            cv::Rect rectBlock(x * iBlockW, y * iBlockH, iBlockW, iBlockH);
            vecBlock[iCount].rect = rectBlock;
            double dMax;
            cv::Point ptMaxLoc;
            cv::minMaxLoc(matSrc(rectBlock), 0, &dMax, 0, &ptMaxLoc);
            vecBlock[iCount].dMax = dMax;
            vecBlock[iCount].ptMaxLoc = ptMaxLoc + rectBlock.tl();
            iCount++;
        }
    }

    if (bHResidue && bVResidue) {
        cv::Rect rectRight(iCol * iBlockW, 0, matSrc.cols - iCol * iBlockW, matSrc.rows);
        Block blockRight;
        blockRight.rect = rectRight;
        double dMax;
        cv::Point ptMaxLoc;
        cv::minMaxLoc(matSrc(rectRight), 0, &dMax, 0, &ptMaxLoc);
        blockRight.dMax = dMax;
        blockRight.ptMaxLoc = ptMaxLoc + rectRight.tl();
        vecBlock.push_back(blockRight);

        cv::Rect rectBottom(0, iRow * iBlockH, iCol * iBlockW, matSrc.rows - iRow * iBlockH);
        Block blockBottom;
        blockBottom.rect = rectBottom;
        cv::minMaxLoc(matSrc(rectBottom), 0, &blockBottom.dMax, 0, &blockBottom.ptMaxLoc);
        blockBottom.ptMaxLoc += rectBottom.tl();
        vecBlock.push_back(blockBottom);
    } else if (bHResidue) {
        cv::Rect rectRight(iCol * iBlockW, 0, matSrc.cols - iCol * iBlockW, matSrc.rows);
        Block blockRight;
        blockRight.rect = rectRight;
        double dMax;
        cv::Point ptMaxLoc;
        cv::minMaxLoc(matSrc(rectRight), 0, &dMax, 0, &ptMaxLoc);
        blockRight.dMax = dMax;
        blockRight.ptMaxLoc = ptMaxLoc + rectRight.tl();
        vecBlock.push_back(blockRight);
    } else if (bVResidue) {
        cv::Rect rectBottom(0, iRow * iBlockH, matSrc.cols, matSrc.rows - iRow * iBlockH);
        Block blockBottom;
        blockBottom.rect = rectBottom;
        cv::minMaxLoc(matSrc(rectBottom), 0, &blockBottom.dMax, 0, &blockBottom.ptMaxLoc);
        blockBottom.ptMaxLoc += rectBottom.tl();
        vecBlock.push_back(blockBottom);
    }
}

void BlockMax::UpdateMax(cv::Rect rectIgnore) {
    if (vecBlock.size() == 0)
        return;
    int iSize = vecBlock.size();
    for (int i = 0; i < iSize; i++) {
        cv::Rect rectIntersec = rectIgnore & vecBlock[i].rect;
        if (rectIntersec.width == 0 && rectIntersec.height == 0)
            continue;
        double dMax;
        cv::Point ptMaxLoc;
        cv::minMaxLoc(matSrc(vecBlock[i].rect), 0, &dMax, 0, &ptMaxLoc);
        vecBlock[i].dMax = dMax;
        vecBlock[i].ptMaxLoc = ptMaxLoc + vecBlock[i].rect.tl();
    }
}

void BlockMax::GetMaxValueLoc(double& dMax, cv::Point& ptMaxLoc) {
    int iSize = vecBlock.size();
    if (iSize == 0) {
        cv::minMaxLoc(matSrc, 0, &dMax, 0, &ptMaxLoc);
        return;
    }
    int iIndex = 0;
    dMax = vecBlock[0].dMax;
    for (int i = 1; i < iSize; i++) {
        if (vecBlock[i].dMax >= dMax) {
            iIndex = i;
            dMax = vecBlock[i].dMax;
        }
    }
    ptMaxLoc = vecBlock[iIndex].ptMaxLoc;
}

// TemplateMatcher 实现
TemplateMatcher::TemplateMatcher() {
}

TemplateMatcher::~TemplateMatcher() {
}

bool TemplateMatcher::LearnPattern(const cv::Mat& templateImage) {
    if (templateImage.empty())
        return false;

    m_TemplData.clear();

    int iTopLayer = GetTopLayer(&templateImage, (int)sqrt((double)256)); // 默认最小区域256
    cv::buildPyramid(templateImage, m_TemplData.vecPyramid, iTopLayer);
    
    m_TemplData.iBorderColor = cv::mean(templateImage).val[0] < 128 ? 255 : 0;
    int iSize = m_TemplData.vecPyramid.size();
    m_TemplData.resize(iSize);

    for (int i = 0; i < iSize; i++) {
        double invArea = 1.0 / ((double)m_TemplData.vecPyramid[i].rows * 
                               m_TemplData.vecPyramid[i].cols);
        cv::Scalar templMean, templSdv;
        double templNorm = 0;

        cv::meanStdDev(m_TemplData.vecPyramid[i], templMean, templSdv);
        templNorm = templSdv[0] * templSdv[0] + templSdv[1] * templSdv[1] + 
                   templSdv[2] * templSdv[2] + templSdv[3] * templSdv[3];

        if (templNorm < DBL_EPSILON) {
            m_TemplData.vecResultEqual1[i] = true;
        }

        double templSum2 = templNorm + templMean[0] * templMean[0] + 
                          templMean[1] * templMean[1] + 
                          templMean[2] * templMean[2] + 
                          templMean[3] * templMean[3];
        templSum2 /= invArea;
        templNorm = std::sqrt(templNorm);
        templNorm /= std::sqrt(invArea);

        m_TemplData.vecInvArea[i] = invArea;
        m_TemplData.vecTemplMean[i] = templMean;
        m_TemplData.vecTemplNorm[i] = templNorm;
    }
    m_TemplData.bIsPatternLearned = true;
    return true;
}

int TemplateMatcher::GetTopLayer(const cv::Mat* matTempl, int iMinDstLength) {
    int iTopLayer = 0;
    int iMinReduceArea = iMinDstLength * iMinDstLength;
    int iArea = matTempl->cols * matTempl->rows;
    while (iArea > iMinReduceArea) {
        iArea /= 4;
        iTopLayer++;
    }
    return iTopLayer;
}

void TemplateMatcher::MatchTemplate(cv::Mat& matSrc, TemplData* pTemplData, 
                                    cv::Mat& matResult, int iLayer, bool bUseSIMD) {
    if (bUseSIMD) {
        matResult.create(matSrc.rows - pTemplData->vecPyramid[iLayer].rows + 1,
                        matSrc.cols - pTemplData->vecPyramid[iLayer].cols + 1, CV_32FC1);
        matResult.setTo(0);
        cv::Mat& matTemplate = pTemplData->vecPyramid[iLayer];

        int t_r_end = matTemplate.rows;
        for (int r = 0; r < matResult.rows; r++) {
            float* r_matResult = matResult.ptr<float>(r);
            uchar* r_source = matSrc.ptr<uchar>(r);
            uchar* r_template, *r_sub_source;
            for (int c = 0; c < matResult.cols; ++c, ++r_matResult, ++r_source) {
                r_template = matTemplate.ptr<uchar>();
                r_sub_source = r_source;
                for (int t_r = 0; t_r < t_r_end; ++t_r, 
                     r_sub_source += matSrc.cols, r_template += matTemplate.cols) {
                        *r_matResult = *r_matResult + IM_Conv_SIMD(r_template, r_sub_source, 
                                                                matTemplate.cols);
                }
            }
        }
    } else {
        cv::matchTemplate(matSrc, pTemplData->vecPyramid[iLayer], matResult, 
                         cv::TM_CCOEFF);
    }
    CCOEFF_Denominator(matSrc, pTemplData, matResult, iLayer);
}

void TemplateMatcher::CCOEFF_Denominator(cv::Mat& matSrc, TemplData* pTemplData, 
                                         cv::Mat& matResult, int iLayer) {
    if (pTemplData->vecResultEqual1[iLayer]) {
        matResult = cv::Scalar::all(1);
        return;
    }

    cv::Mat sum, sqsum;
    cv::integral(matSrc, sum, sqsum, CV_64F);

    double* q0 = (double*)sqsum.data;
    double* q1 = q0 + pTemplData->vecPyramid[iLayer].cols;
    double* q2 = (double*)(sqsum.data + pTemplData->vecPyramid[iLayer].rows * sqsum.step);
    double* q3 = q2 + pTemplData->vecPyramid[iLayer].cols;

    double* p0 = (double*)sum.data;
    double* p1 = p0 + pTemplData->vecPyramid[iLayer].cols;
    double* p2 = (double*)(sum.data + pTemplData->vecPyramid[iLayer].rows * sum.step);
    double* p3 = p2 + pTemplData->vecPyramid[iLayer].cols;

    int sumstep = sum.data ? (int)(sum.step / sizeof(double)) : 0;
    int sqstep = sqsum.data ? (int)(sqsum.step / sizeof(double)) : 0;

    double dTemplMean0 = pTemplData->vecTemplMean[iLayer][0];
    double dTemplNorm = pTemplData->vecTemplNorm[iLayer];
    double dInvArea = pTemplData->vecInvArea[iLayer];

    for (int i = 0; i < matResult.rows; i++) {
        float* rrow = matResult.ptr<float>(i);
        int idx = i * sumstep;
        int idx2 = i * sqstep;

        for (int j = 0; j < matResult.cols; j += 1, idx += 1, idx2 += 1) {
            double num = rrow[j], t;
            double wndMean2 = 0, wndSum2 = 0;

            t = p0[idx] - p1[idx] - p2[idx] + p3[idx];
            wndMean2 += t * t;
            num -= t * dTemplMean0;
            wndMean2 *= dInvArea;

            t = q0[idx2] - q1[idx2] - q2[idx2] + q3[idx2];
            wndSum2 += t;

            double diff2 = (wndSum2 - wndMean2 > 0) ? (wndSum2 - wndMean2) : 0.0;
            if (diff2 <= std::min(0.5, 10.0 * FLT_EPSILON * wndSum2))
                t = 0;
            else
                t = std::sqrt(diff2) * dTemplNorm;

            if (fabs(num) < t)
                num /= t;
            else if (fabs(num) < t * 1.125)
                num = num > 0 ? 1 : -1;
            else
                num = 0;

            rrow[j] = (float)num;
        }
    }
}

void TemplateMatcher::GetRotatedROI(cv::Mat& matSrc, cv::Size size, cv::Point2f ptLT, 
                                    double dAngle, cv::Mat& matROI) {
    double dAngle_radian = dAngle * D2R;
    cv::Point2f ptC((matSrc.cols - 1) / 2.0f, (matSrc.rows - 1) / 2.0f);
    cv::Point2f ptLT_rotate = ptRotatePt2f(ptLT, ptC, dAngle_radian);
    cv::Size sizePadding(size.width + 6, size.height + 6);

    cv::Mat rMat = cv::getRotationMatrix2D(ptC, dAngle, 1);
    rMat.at<double>(0, 2) -= ptLT_rotate.x - 3;
    rMat.at<double>(1, 2) -= ptLT_rotate.y - 3;
    cv::warpAffine(matSrc, matROI, rMat, sizePadding, cv::INTER_LINEAR, 
                   cv::BORDER_CONSTANT, cv::Scalar(m_TemplData.iBorderColor));
}

cv::Size TemplateMatcher::GetBestRotationSize(cv::Size sizeSrc, cv::Size sizeDst, 
                                               double dRAngle) {
    double dRAngle_radian = dRAngle * D2R;
    cv::Point ptLT(0, 0), ptLB(0, sizeSrc.height - 1), 
              ptRB(sizeSrc.width - 1, sizeSrc.height - 1), 
              ptRT(sizeSrc.width - 1, 0);
    cv::Point2f ptCenter((sizeSrc.width - 1) / 2.0f, (sizeSrc.height - 1) / 2.0f);
    cv::Point2f ptLT_R = ptRotatePt2f(cv::Point2f(ptLT), ptCenter, dRAngle_radian);
    cv::Point2f ptLB_R = ptRotatePt2f(cv::Point2f(ptLB), ptCenter, dRAngle_radian);
    cv::Point2f ptRB_R = ptRotatePt2f(cv::Point2f(ptRB), ptCenter, dRAngle_radian);
    cv::Point2f ptRT_R = ptRotatePt2f(cv::Point2f(ptRT), ptCenter, dRAngle_radian);

    float fTopY = std::max(std::max(ptLT_R.y, ptLB_R.y), std::max(ptRB_R.y, ptRT_R.y));
    float fBottomY = std::min(std::min(ptLT_R.y, ptLB_R.y), std::min(ptRB_R.y, ptRT_R.y));
    float fRightX = std::max(std::max(ptLT_R.x, ptLB_R.x), std::max(ptRB_R.x, ptRT_R.x));
    float fLeftX = std::min(std::min(ptLT_R.x, ptLB_R.x), std::min(ptRB_R.x, ptRT_R.x));

    double dAngle = dRAngle;
    if (dAngle > 360)
        dAngle -= 360;
    else if (dAngle < 0)
        dAngle += 360;

    if (fabs(fabs(dAngle) - 90) < VISION_TOLERANCE || 
        fabs(fabs(dAngle) - 270) < VISION_TOLERANCE) {
        return cv::Size(sizeSrc.height, sizeSrc.width);
    } else if (fabs(dAngle) < VISION_TOLERANCE || 
               fabs(fabs(dAngle) - 180) < VISION_TOLERANCE) {
        return sizeSrc;
    }

    if (dAngle > 0 && dAngle < 90) {
        // 保持原角度
    } else if (dAngle > 90 && dAngle < 180) {
        dAngle -= 90;
    } else if (dAngle > 180 && dAngle < 270) {
        dAngle -= 180;
    } else if (dAngle > 270 && dAngle < 360) {
        dAngle -= 270;
    }

    float fH1 = sizeDst.width * sin(dAngle * D2R) * cos(dAngle * D2R);
    float fH2 = sizeDst.height * sin(dAngle * D2R) * cos(dAngle * D2R);

    int iHalfHeight = (int)ceil(fTopY - ptCenter.y - fH1);
    int iHalfWidth = (int)ceil(fRightX - ptCenter.x - fH2);

    cv::Size sizeRet(iHalfWidth * 2, iHalfHeight * 2);

    bool bWrongSize = (sizeDst.width < sizeRet.width && sizeDst.height > sizeRet.height) ||
                      (sizeDst.width > sizeRet.width && sizeDst.height < sizeRet.height) ||
                      sizeDst.area() > sizeRet.area();
    if (bWrongSize)
        sizeRet = cv::Size((int)(fRightX - fLeftX + 0.5), (int)(fTopY - fBottomY + 0.5));

    return sizeRet;
}

cv::Point2f TemplateMatcher::ptRotatePt2f(cv::Point2f ptInput, cv::Point2f ptOrg, 
                                          double dAngle) {
    double dWidth = ptOrg.x * 2;
    double dHeight = ptOrg.y * 2;
    double dY1 = dHeight - ptInput.y, dY2 = dHeight - ptOrg.y;

    double dX = (ptInput.x - ptOrg.x) * cos(dAngle) - (dY1 - ptOrg.y) * sin(dAngle) + ptOrg.x;
    double dY = (ptInput.x - ptOrg.x) * sin(dAngle) + (dY1 - ptOrg.y) * cos(dAngle) + dY2;

    dY = -dY + dHeight;
    return cv::Point2f((float)dX, (float)dY);
}

void TemplateMatcher::FilterWithScore(std::vector<MatchParameter>* vec, double dScore) {
    std::sort(vec->begin(), vec->end(), compareScoreBig2Small);
    int iSize = vec->size();
    int iIndexDelete = iSize + 1;
    for (int i = 0; i < iSize; i++) {
        if ((*vec)[i].dMatchScore < dScore) {
            iIndexDelete = i;
            break;
        }
    }
    if (iIndexDelete != iSize + 1) {
        vec->erase(vec->begin() + iIndexDelete, vec->end());
    }
}

void TemplateMatcher::FilterWithRotatedRect(std::vector<MatchParameter>* vec, int iMethod, 
                                            double dMaxOverLap) {
    int iMatchSize = vec->size();
    cv::RotatedRect rect1, rect2;
    for (int i = 0; i < iMatchSize - 1; i++) {
        if (vec->at(i).bDelete)
            continue;
        for (int j = i + 1; j < iMatchSize; j++) {
            if (vec->at(j).bDelete)
                continue;
            rect1 = vec->at(i).rectR;
            rect2 = vec->at(j).rectR;
            std::vector<cv::Point2f> vecInterSec;
            int iInterSecType = cv::rotatedRectangleIntersection(rect1, rect2, vecInterSec);
            if (iInterSecType == cv::INTERSECT_NONE)
                continue;
            else if (iInterSecType == cv::INTERSECT_FULL) {
                int iDeleteIndex;
                if (iMethod == cv::TM_SQDIFF)
                    iDeleteIndex = (vec->at(i).dMatchScore <= vec->at(j).dMatchScore) ? j : i;
                else
                    iDeleteIndex = (vec->at(i).dMatchScore >= vec->at(j).dMatchScore) ? j : i;
                vec->at(iDeleteIndex).bDelete = true;
            } else {
                if (vecInterSec.size() < 3)
                    continue;
                else {
                    int iDeleteIndex;
                    SortPtWithCenter(vecInterSec);
                    double dArea = cv::contourArea(vecInterSec);
                    double dRatio = dArea / rect1.size.area();
                    if (dRatio > dMaxOverLap) {
                        if (iMethod == cv::TM_SQDIFF)
                            iDeleteIndex = (vec->at(i).dMatchScore <= vec->at(j).dMatchScore) ? j : i;
                        else
                            iDeleteIndex = (vec->at(i).dMatchScore >= vec->at(j).dMatchScore) ? j : i;
                        vec->at(iDeleteIndex).bDelete = true;
                    }
                }
            }
        }
    }
    vec->erase(std::remove_if(vec->begin(), vec->end(), 
                              [](const MatchParameter& p) { return p.bDelete; }),
               vec->end());
}

cv::Point TemplateMatcher::GetNextMaxLoc(cv::Mat& matResult, cv::Point ptMaxLoc, 
                                         cv::Size sizeTemplate, double& dMaxValue, 
                                         double dMaxOverlap) {
    int iStartX = ptMaxLoc.x - sizeTemplate.width * (1 - dMaxOverlap);
    int iStartY = ptMaxLoc.y - sizeTemplate.height * (1 - dMaxOverlap);
    cv::rectangle(matResult, 
                  cv::Rect(iStartX, iStartY, 
                          2 * sizeTemplate.width * (1 - dMaxOverlap), 
                          2 * sizeTemplate.height * (1 - dMaxOverlap)), 
                  cv::Scalar(-1), cv::FILLED);
    cv::Point ptNewMaxLoc;
    cv::minMaxLoc(matResult, 0, &dMaxValue, 0, &ptNewMaxLoc);
    return ptNewMaxLoc;
}

cv::Point TemplateMatcher::GetNextMaxLoc(cv::Mat& matResult, cv::Point ptMaxLoc, 
                                         cv::Size sizeTemplate, double& dMaxValue, 
                                         double dMaxOverlap, BlockMax& blockMax) {
    int iStartX = (int)(ptMaxLoc.x - sizeTemplate.width * (1 - dMaxOverlap));
    int iStartY = (int)(ptMaxLoc.y - sizeTemplate.height * (1 - dMaxOverlap));
    cv::Rect rectIgnore(iStartX, iStartY, 
                        (int)(2 * sizeTemplate.width * (1 - dMaxOverlap)),
                        (int)(2 * sizeTemplate.height * (1 - dMaxOverlap)));
    cv::rectangle(matResult, rectIgnore, cv::Scalar(-1), cv::FILLED);
    blockMax.UpdateMax(rectIgnore);
    cv::Point ptReturn;
    blockMax.GetMaxValueLoc(dMaxValue, ptReturn);
    return ptReturn;
}

void TemplateMatcher::SortPtWithCenter(std::vector<cv::Point2f>& vecSort) {
    int iSize = vecSort.size();
    cv::Point2f ptCenter;
    for (int i = 0; i < iSize; i++)
        ptCenter += vecSort[i];
    ptCenter /= iSize;

    std::vector<std::pair<cv::Point2f, double>> vecPtAngle(iSize);
    for (int i = 0; i < iSize; i++) {
        vecPtAngle[i].first = vecSort[i];
        cv::Point2f vec1(vecSort[i].x - ptCenter.x, vecSort[i].y - ptCenter.y);
        float fNormVec1 = vec1.x * vec1.x + vec1.y * vec1.y;
        float fDot = vec1.x;

        if (vec1.y < 0) {
            vecPtAngle[i].second = acos(fDot / sqrt(fNormVec1)) * R2D;
        } else if (vec1.y > 0) {
            vecPtAngle[i].second = 360 - acos(fDot / sqrt(fNormVec1)) * R2D;
        } else {
            if (vec1.x > 0)
                vecPtAngle[i].second = 0;
            else
                vecPtAngle[i].second = 180;
        }
    }
    std::sort(vecPtAngle.begin(), vecPtAngle.end(), comparePtWithAngle);
    for (int i = 0; i < iSize; i++)
        vecSort[i] = vecPtAngle[i].first;
}

bool TemplateMatcher::SubPixEstimation(std::vector<MatchParameter>* vec, double* dNewX, 
                                       double* dNewY, double* dNewAngle, double dAngleStep, 
                                       int iMaxScoreIndex) {
    cv::Mat matA(27, 10, CV_64F);
    cv::Mat matZ(10, 1, CV_64F);
    cv::Mat matS(27, 1, CV_64F);

    double dX_maxScore = (*vec)[iMaxScoreIndex].pt.x;
    double dY_maxScore = (*vec)[iMaxScoreIndex].pt.y;
    double dTheata_maxScore = (*vec)[iMaxScoreIndex].dMatchAngle;
    int iRow = 0;

    for (int theta = 0; theta <= 2; theta++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                double dX = dX_maxScore + x;
                double dY = dY_maxScore + y;
                double dT = (dTheata_maxScore + (theta - 1) * dAngleStep) * D2R;
                matA.at<double>(iRow, 0) = dX * dX;
                matA.at<double>(iRow, 1) = dY * dY;
                matA.at<double>(iRow, 2) = dT * dT;
                matA.at<double>(iRow, 3) = dX * dY;
                matA.at<double>(iRow, 4) = dX * dT;
                matA.at<double>(iRow, 5) = dY * dT;
                matA.at<double>(iRow, 6) = dX;
                matA.at<double>(iRow, 7) = dY;
                matA.at<double>(iRow, 8) = dT;
                matA.at<double>(iRow, 9) = 1.0;
                matS.at<double>(iRow, 0) = (*vec)[iMaxScoreIndex + (theta - 1)].vecResult[x + 1][y + 1];
                iRow++;
            }
        }
    }

    matZ = (matA.t() * matA).inv() * matA.t() * matS;
    cv::Mat matZ_t;
    cv::transpose(matZ, matZ_t);
    double* dZ = matZ_t.ptr<double>(0);
    cv::Mat matK1 = (cv::Mat_<double>(3, 3) << 
        (2 * dZ[0]), dZ[3], dZ[4], 
        dZ[3], (2 * dZ[1]), dZ[5], 
        dZ[4], dZ[5], (2 * dZ[2]));
    cv::Mat matK2 = (cv::Mat_<double>(3, 1) << -dZ[6], -dZ[7], -dZ[8]);
    cv::Mat matDelta = matK1.inv() * matK2;

    *dNewX = matDelta.at<double>(0, 0);
    *dNewY = matDelta.at<double>(1, 0);
    *dNewAngle = matDelta.at<double>(2, 0) * R2D;
    return true;
}

// Match 函数实现（核心匹配算法）
MatchResult TemplateMatcher::Match(const cv::Mat& sourceImage, const MatchConfig& config) {
    MatchResult result;
    result.success = false;

    if (sourceImage.empty() || !m_TemplData.bIsPatternLearned)
        return result;

    if (sourceImage.size().area() < m_TemplData.vecPyramid[0].size().area())
        return result;

    double d1 = clock();

    int iTopLayer = GetTopLayer(&m_TemplData.vecPyramid[0], 
                                (int)sqrt((double)config.iMinReduceArea));
    std::vector<cv::Mat> vecMatSrcPyr;
    if (config.bBitwiseNot) {
        cv::Mat matNewSrc = 255 - sourceImage;
        cv::buildPyramid(matNewSrc, vecMatSrcPyr, iTopLayer);
    } else {
        cv::buildPyramid(sourceImage, vecMatSrcPyr, iTopLayer);
    }

    TemplData* pTemplData = &m_TemplData;

    // 第一阶段：顶层匹配
    double dAngleStep = atan(2.0 / std::max(pTemplData->vecPyramid[iTopLayer].cols, 
                                            pTemplData->vecPyramid[iTopLayer].rows)) * R2D;

    std::vector<double> vecAngles;
    if (config.bToleranceRange) {
        for (double dAngle = config.dTolerance1; dAngle < config.dTolerance2 + dAngleStep; 
             dAngle += dAngleStep)
            vecAngles.push_back(dAngle);
        for (double dAngle = config.dTolerance3; dAngle < config.dTolerance4 + dAngleStep; 
             dAngle += dAngleStep)
            vecAngles.push_back(dAngle);
    } else {
        if (config.dToleranceAngle < VISION_TOLERANCE)
            vecAngles.push_back(0.0);
        else {
            for (double dAngle = 0; dAngle < config.dToleranceAngle + dAngleStep; 
                 dAngle += dAngleStep)
                vecAngles.push_back(dAngle);
            for (double dAngle = -dAngleStep; dAngle > -config.dToleranceAngle - dAngleStep; 
                 dAngle -= dAngleStep)
                vecAngles.push_back(dAngle);
        }
    }

    int iTopSrcW = vecMatSrcPyr[iTopLayer].cols, 
        iTopSrcH = vecMatSrcPyr[iTopLayer].rows;
    cv::Point2f ptCenter((iTopSrcW - 1) / 2.0f, (iTopSrcH - 1) / 2.0f);

    int iSize = vecAngles.size();
    std::vector<MatchParameter> vecMatchParameter;

    // 计算每层最低分数
    std::vector<double> vecLayerScore(iTopLayer + 1, config.dScore);
    for (int iLayer = 1; iLayer <= iTopLayer; iLayer++)
        vecLayerScore[iLayer] = vecLayerScore[iLayer - 1] * 0.9;

    cv::Size sizePat = pTemplData->vecPyramid[iTopLayer].size();
    bool bCalMaxByBlock = (vecMatSrcPyr[iTopLayer].size().area() / sizePat.area() > 500) && 
                          config.iMaxPos > 10;

    for (int i = 0; i < iSize; i++) {
        cv::Mat matRotatedSrc, matR = cv::getRotationMatrix2D(ptCenter, vecAngles[i], 1);
        cv::Mat matResult;
        cv::Point ptMaxLoc;
        double dValue, dMaxVal;

        cv::Size sizeBest = GetBestRotationSize(vecMatSrcPyr[iTopLayer].size(), 
                                               pTemplData->vecPyramid[iTopLayer].size(), 
                                               vecAngles[i]);

        float fTranslationX = (sizeBest.width - 1) / 2.0f - ptCenter.x;
        float fTranslationY = (sizeBest.height - 1) / 2.0f - ptCenter.y;
        matR.at<double>(0, 2) += fTranslationX;
        matR.at<double>(1, 2) += fTranslationY;
        cv::warpAffine(vecMatSrcPyr[iTopLayer], matRotatedSrc, matR, sizeBest, 
                      cv::INTER_LINEAR, cv::BORDER_CONSTANT, 
                      cv::Scalar(pTemplData->iBorderColor));

        MatchTemplate(matRotatedSrc, pTemplData, matResult, iTopLayer, config.bUseSIMD);

        if (bCalMaxByBlock) {
            BlockMax blockMax(matResult, pTemplData->vecPyramid[iTopLayer].size());
            blockMax.GetMaxValueLoc(dMaxVal, ptMaxLoc);
            if (dMaxVal < vecLayerScore[iTopLayer])
                continue;
            vecMatchParameter.push_back(MatchParameter(
                cv::Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), 
                dMaxVal, vecAngles[i]));
            for (int j = 0; j < config.iMaxPos + MATCH_CANDIDATE_NUM - 1; j++) {
                ptMaxLoc = GetNextMaxLoc(matResult, ptMaxLoc, 
                                        pTemplData->vecPyramid[iTopLayer].size(), 
                                        dValue, config.dMaxOverlap, blockMax);
                if (dValue < vecLayerScore[iTopLayer])
                    break;
                vecMatchParameter.push_back(MatchParameter(
                    cv::Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), 
                    dValue, vecAngles[i]));
            }
        } else {
            cv::minMaxLoc(matResult, 0, &dMaxVal, 0, &ptMaxLoc);
            if (dMaxVal < vecLayerScore[iTopLayer])
                continue;
            vecMatchParameter.push_back(MatchParameter(
                cv::Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), 
                dMaxVal, vecAngles[i]));
            for (int j = 0; j < config.iMaxPos + MATCH_CANDIDATE_NUM - 1; j++) {
                ptMaxLoc = GetNextMaxLoc(matResult, ptMaxLoc, 
                                        pTemplData->vecPyramid[iTopLayer].size(), 
                                        dValue, config.dMaxOverlap);
                if (dValue < vecLayerScore[iTopLayer])
                    break;
                vecMatchParameter.push_back(MatchParameter(
                    cv::Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), 
                    dValue, vecAngles[i]));
            }
        }
    }
    std::sort(vecMatchParameter.begin(), vecMatchParameter.end(), compareScoreBig2Small);

    int iMatchSize = vecMatchParameter.size();
    int iDstW = pTemplData->vecPyramid[iTopLayer].cols, 
        iDstH = pTemplData->vecPyramid[iTopLayer].rows;

    // 第二阶段：逐层细化
    bool bSubPixelEstimation = config.bSubPixelEstimation;
    int iStopLayer = config.bStopLayer1 ? 1 : 0;
    std::vector<MatchParameter> vecAllResult;

    for (int i = 0; i < (int)vecMatchParameter.size(); i++) {
        double dRAngle = -vecMatchParameter[i].dMatchAngle * D2R;
        cv::Point2f ptInput(vecMatchParameter[i].pt.x, vecMatchParameter[i].pt.y);
        cv::Point2f ptLT = ptRotatePt2f(ptInput, ptCenter, dRAngle);

        double dAngleStep = atan(2.0 / std::max(iDstW, iDstH)) * R2D;
        vecMatchParameter[i].dAngleStart = vecMatchParameter[i].dMatchAngle - dAngleStep;
        vecMatchParameter[i].dAngleEnd = vecMatchParameter[i].dMatchAngle + dAngleStep;

        if (iTopLayer <= iStopLayer) {
            double scale = (iTopLayer == 0) ? 1.0 : 2.0;
            vecMatchParameter[i].pt = cv::Point2d(ptLT.x * scale, ptLT.y * scale);
            vecAllResult.push_back(vecMatchParameter[i]);
        } else {
            for (int iLayer = iTopLayer - 1; iLayer >= iStopLayer; iLayer--) {
                dAngleStep = atan(2.0 / std::max(pTemplData->vecPyramid[iLayer].cols, 
                                                pTemplData->vecPyramid[iLayer].rows)) * R2D;
                std::vector<double> vecAngles;
                double dMatchedAngle = vecMatchParameter[i].dMatchAngle;
                if (config.bToleranceRange) {
                    for (int k = -1; k <= 1; k++)
                        vecAngles.push_back(dMatchedAngle + dAngleStep * k);
                } else {
                    if (config.dToleranceAngle < VISION_TOLERANCE)
                        vecAngles.push_back(0.0);
                    else
                        for (int k = -1; k <= 1; k++)
                            vecAngles.push_back(dMatchedAngle + dAngleStep * k);
                }
                cv::Point2f ptSrcCenter((vecMatSrcPyr[iLayer].cols - 1) / 2.0f, 
                                       (vecMatSrcPyr[iLayer].rows - 1) / 2.0f);
                iSize = vecAngles.size();
                std::vector<MatchParameter> vecNewMatchParameter(iSize);
                int iMaxScoreIndex = 0;
                double dBigValue = -1;
                for (int j = 0; j < iSize; j++) {
                    cv::Mat matResult, matRotatedSrc;
                    double dMaxValue = 0;
                    cv::Point ptMaxLoc;
                    cv::Point2f ptLT_scaled(ptLT.x * 2, ptLT.y * 2);
                    GetRotatedROI(vecMatSrcPyr[iLayer], 
                                 pTemplData->vecPyramid[iLayer].size(), 
                                 ptLT_scaled, vecAngles[j], matRotatedSrc);

                    MatchTemplate(matRotatedSrc, pTemplData, matResult, iLayer, 
                                 config.bUseSIMD);
                    cv::minMaxLoc(matResult, 0, &dMaxValue, 0, &ptMaxLoc);
                    vecNewMatchParameter[j] = MatchParameter(ptMaxLoc, dMaxValue, 
                                                            vecAngles[j]);

                    if (vecNewMatchParameter[j].dMatchScore > dBigValue) {
                        iMaxScoreIndex = j;
                        dBigValue = vecNewMatchParameter[j].dMatchScore;
                    }

                    if (ptMaxLoc.x == 0 || ptMaxLoc.y == 0 || 
                        ptMaxLoc.x == matResult.cols - 1 || 
                        ptMaxLoc.y == matResult.rows - 1)
                        vecNewMatchParameter[j].bPosOnBorder = true;
                    if (!vecNewMatchParameter[j].bPosOnBorder) {
                        for (int y = -1; y <= 1; y++)
                            for (int x = -1; x <= 1; x++)
                                vecNewMatchParameter[j].vecResult[x + 1][y + 1] = 
                                    matResult.at<float>(ptMaxLoc + cv::Point(x, y));
                    }
                }
                if (vecNewMatchParameter[iMaxScoreIndex].dMatchScore < vecLayerScore[iLayer])
                    break;

                if (bSubPixelEstimation && iLayer == 0 && 
                    (!vecNewMatchParameter[iMaxScoreIndex].bPosOnBorder) && 
                    iMaxScoreIndex != 0 && iMaxScoreIndex != 2) {
                    double dNewX = 0, dNewY = 0, dNewAngle = 0;
                    SubPixEstimation(&vecNewMatchParameter, &dNewX, &dNewY, &dNewAngle, 
                                   dAngleStep, iMaxScoreIndex);
                    vecNewMatchParameter[iMaxScoreIndex].pt = cv::Point2d(dNewX, dNewY);
                    vecNewMatchParameter[iMaxScoreIndex].dMatchAngle = dNewAngle;
                }

                double dNewMatchAngle = vecNewMatchParameter[iMaxScoreIndex].dMatchAngle;
                cv::Point2f ptLT_scaled(ptLT.x * 2, ptLT.y * 2);
                cv::Point2f ptPaddingLT = ptRotatePt2f(ptLT_scaled, ptSrcCenter, 
                                                      dNewMatchAngle * D2R) - 
                                        cv::Point2f(3, 3);
                cv::Point2f pt(vecNewMatchParameter[iMaxScoreIndex].pt.x + ptPaddingLT.x, 
                              vecNewMatchParameter[iMaxScoreIndex].pt.y + ptPaddingLT.y);
                pt = ptRotatePt2f(pt, ptSrcCenter, -dNewMatchAngle * D2R);

                if (iLayer == iStopLayer) {
                    double scale = (iStopLayer == 0) ? 1.0 : 2.0;
                    vecNewMatchParameter[iMaxScoreIndex].pt = cv::Point2d(pt.x * scale, pt.y * scale);
                    vecAllResult.push_back(vecNewMatchParameter[iMaxScoreIndex]);
                } else {
                    vecMatchParameter[i].dMatchAngle = dNewMatchAngle;
                    vecMatchParameter[i].dAngleStart = vecMatchParameter[i].dMatchAngle - 
                                                       dAngleStep / 2;
                    vecMatchParameter[i].dAngleEnd = vecMatchParameter[i].dMatchAngle + 
                                                     dAngleStep / 2;
                    ptLT = pt;
                }
            }
        }
    }

    FilterWithScore(&vecAllResult, config.dScore);

    // 最后过滤重叠
    iDstW = pTemplData->vecPyramid[iStopLayer].cols * (iStopLayer == 0 ? 1 : 2);
    iDstH = pTemplData->vecPyramid[iStopLayer].rows * (iStopLayer == 0 ? 1 : 2);

    for (int i = 0; i < (int)vecAllResult.size(); i++) {
        cv::Point2f ptLT, ptRT, ptRB, ptLB;
        double dRAngle = -vecAllResult[i].dMatchAngle * D2R;
        ptLT = cv::Point2f((float)vecAllResult[i].pt.x, (float)vecAllResult[i].pt.y);
        ptRT = cv::Point2f(ptLT.x + iDstW * (float)cos(dRAngle), 
                          ptLT.y - iDstW * (float)sin(dRAngle));
        ptLB = cv::Point2f(ptLT.x + iDstH * (float)sin(dRAngle), 
                          ptLT.y + iDstH * (float)cos(dRAngle));
        ptRB = cv::Point2f(ptRT.x + iDstH * (float)sin(dRAngle), 
                          ptRT.y + iDstH * (float)cos(dRAngle));
        vecAllResult[i].rectR = cv::RotatedRect(ptLT, ptRT, ptRB);
    }
    FilterWithRotatedRect(&vecAllResult, cv::TM_CCOEFF_NORMED, config.dMaxOverlap);

    std::sort(vecAllResult.begin(), vecAllResult.end(), compareScoreBig2Small);

    result.executionTimeMs = (clock() - d1) * 1000.0 / CLOCKS_PER_SEC;

    if (vecAllResult.size() == 0) {
        result.success = false;
        return result;
    }

    int iW = pTemplData->vecPyramid[0].cols, iH = pTemplData->vecPyramid[0].rows;

    for (int i = 0; i < (int)vecAllResult.size() && i < config.iMaxPos; i++) {
        SingleTargetMatch sstm;
        double dRAngle = -vecAllResult[i].dMatchAngle * D2R;

        sstm.ptLT = vecAllResult[i].pt;
        sstm.ptRT = cv::Point2d(sstm.ptLT.x + iW * cos(dRAngle), 
                               sstm.ptLT.y - iW * sin(dRAngle));
        sstm.ptLB = cv::Point2d(sstm.ptLT.x + iH * sin(dRAngle), 
                               sstm.ptLT.y + iH * cos(dRAngle));
        sstm.ptRB = cv::Point2d(sstm.ptRT.x + iH * sin(dRAngle), 
                               sstm.ptRT.y + iH * cos(dRAngle));
        sstm.ptCenter = cv::Point2d((sstm.ptLT.x + sstm.ptRT.x + sstm.ptRB.x + sstm.ptLB.x) / 4, 
                                    (sstm.ptLT.y + sstm.ptRT.y + sstm.ptRB.y + sstm.ptLB.y) / 4);
        sstm.dMatchedAngle = -vecAllResult[i].dMatchAngle;
        sstm.dMatchScore = vecAllResult[i].dMatchScore;

        if (sstm.dMatchedAngle < -180)
            sstm.dMatchedAngle += 360;
        if (sstm.dMatchedAngle > 180)
            sstm.dMatchedAngle -= 360;
        result.matches.push_back(sstm);
    }

    result.success = true;
    return result;
}

// 可视化函数实现
namespace Visualization {

void DrawMatchResult(cv::Mat& image, const std::vector<SingleTargetMatch>& matches,
                    const cv::Scalar& color, int thickness, bool drawLabels) {
    if (image.channels() == 1)
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);

    for (size_t i = 0; i < matches.size(); i++) {
        const auto& match = matches[i];
        std::vector<cv::Point> pts = {
            cv::Point((int)match.ptLT.x, (int)match.ptLT.y),
            cv::Point((int)match.ptRT.x, (int)match.ptRT.y),
            cv::Point((int)match.ptRB.x, (int)match.ptRB.y),
            cv::Point((int)match.ptLB.x, (int)match.ptLB.y)
        };

        // 绘制虚线框
        DrawDashLine(image, pts[0], pts[1], color);
        DrawDashLine(image, pts[1], pts[2], color);
        DrawDashLine(image, pts[2], pts[3], color);
        DrawDashLine(image, pts[3], pts[0], color);

        // 绘制角落标记
        cv::Point ptDis1, ptDis2;
        if (pts[2].x - pts[0].x > pts[2].y - pts[0].y) {
            ptDis1 = (pts[3] - pts[0]) / 3;
            ptDis2 = (pts[1] - pts[0]) / 3;
        } else {
            ptDis1 = (pts[3] - pts[0]) / 3;
            ptDis2 = (pts[1] - pts[0]) / 3;
        }
        cv::line(image, pts[0], pts[0] + ptDis1 / 2, color, 1, cv::LINE_AA);
        cv::line(image, pts[0], pts[0] + ptDis2 / 2, color, 1, cv::LINE_AA);

        // 绘制中心十字
        DrawMarkCross(image, (int)match.ptCenter.x, (int)match.ptCenter.y, 5, color, 1);

        // 绘制标签
        if (drawLabels) {
            cv::Point textPos = (pts[0] + pts[1]) / 2;
            std::string str = std::to_string(i);
            cv::putText(image, str, textPos, cv::FONT_HERSHEY_PLAIN, 1, color);
        }
    }
}

void DrawDashLine(cv::Mat& matDraw, cv::Point ptStart, cv::Point ptEnd,
                 const cv::Scalar& color1, const cv::Scalar& color2) {
    cv::LineIterator itLine(matDraw, ptStart, ptEnd, 8, 0);
    int iCount = itLine.count;
    for (int i = 0; i < iCount; i += 1, ++itLine) {
        if (i % 3 == 0) {
            (*itLine)[0] = (uchar)color2.val[0];
            (*itLine)[1] = (uchar)color2.val[1];
            (*itLine)[2] = (uchar)color2.val[2];
        } else {
            (*itLine)[0] = (uchar)color1.val[0];
            (*itLine)[1] = (uchar)color1.val[1];
            (*itLine)[2] = (uchar)color1.val[2];
        }
    }
}

void DrawMarkCross(cv::Mat& matDraw, int iX, int iY, int iLength,
                  const cv::Scalar& color, int iThickness) {
    if (matDraw.empty())
        return;
    cv::Point ptC(iX, iY);
    cv::line(matDraw, ptC - cv::Point(iLength, 0), ptC + cv::Point(iLength, 0), 
            color, iThickness);
    cv::line(matDraw, ptC - cv::Point(0, iLength), ptC + cv::Point(0, iLength), 
            color, iThickness);
}

} // namespace Visualization

} // namespace TemplateMatching
