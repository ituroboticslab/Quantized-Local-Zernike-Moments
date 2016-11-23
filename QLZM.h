#ifndef QLZM_H
#define QLZM_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


typedef std::complex<double> ComplexValue;
typedef std::vector<ComplexValue> ComplexVector;
typedef std::vector<ComplexVector> ComplexMatrix;


class QLZM
{
public:
    QLZM(int patchSize = 7, int gridSize = 5, int momentOrder = 2);
    virtual ~QLZM() { }

    int getPatchSize() const { return _patchSize; }
    int getGridSize() const { return _gridSize; }
    int getMomentOrder() const { return _momentOrder; }
    std::vector<cv::Mat> getFilters() const { return _filters; }

    cv::Mat extractPatterns(cv::Mat image);
    cv::Mat computeDescriptor(cv::Mat patterns);

private:
    int _patchSize;
    int _gridSize;
    int _momentOrder;
    std::vector<cv::Mat> _filters;

    double factorial(int x);
    std::vector<int> getMomentMask(int order);
    std::vector<cv::Mat> computeFilters(int size, int order, ComplexMatrix matrix);
    ComplexMatrix computeComplexMatrix(int size, int order);
    ComplexVector computeComplexVector(int size, int n, int m);
    ComplexValue computeComplexValue(int size, int n, int m, int x, int y);
    cv::Mat getGaussianKernel(cv::Size size, double sigma) const;
    cv::Mat getRegionHistogram(const cv::Mat &region, int binCount, const cv::Mat &gaussianKernel);
};


#endif //QLZM_H
