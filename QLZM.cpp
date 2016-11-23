#include "QLZM.h"


QLZM::QLZM(int patchSize, int gridSize, int momentOrder)
        : _patchSize(patchSize), _gridSize(gridSize), _momentOrder(momentOrder)
{
    CV_Assert(patchSize > 2);
    CV_Assert(gridSize > 0);
    CV_Assert(momentOrder > 0);

    ComplexMatrix matrix = computeComplexMatrix(patchSize, momentOrder);
    _filters = computeFilters(patchSize, momentOrder, matrix);
}

cv::Mat QLZM::extractPatterns(cv::Mat image)
{
    CV_Assert(!image.empty());
    CV_Assert(image.channels() == 1);
    CV_Assert(image.rows > 0 && image.cols > 0);

    cv::Mat input;
    image.convertTo(input, CV_64F);

    int rows = (input.rows - _patchSize) / _patchSize + 1;
    int cols = (input.cols - _patchSize) / _patchSize + 1;

    cv::Mat patterns = cv::Mat::zeros(rows, cols, CV_8U);

    for (int i = 0; i < patterns.rows; i++)
    {
        for (int j = 0; j < patterns.cols; j++)
        {
            cv::Mat src = input(cv::Rect(j * _patchSize, i * _patchSize, _patchSize, _patchSize));

            uchar val = 0;
            for (int k = 0; k < _filters.size(); k++)
            {
                double sum = 0;
                for (int ii = 0; ii < _patchSize; ii++)
                {
                    for (int jj = 0; jj < _patchSize; jj++)
                    {
                        sum += (src.at<double>(ii, jj) * _filters[k].at<double>(ii, jj));
                    }
                }

                val |= (uchar) (sum > 0) << k;
            }

            patterns.at<uchar>(i, j) = val;
        }
    }

    return patterns;
}

cv::Mat QLZM::computeDescriptor(cv::Mat patterns)
{
    CV_Assert(!patterns.empty() && patterns.rows > _patchSize && patterns.cols > _patchSize);

    cv::Size regionSize = cv::Size(patterns.cols / _gridSize, patterns.rows / _gridSize);
    cv::Mat gaussianKernel = getGaussianKernel(regionSize, 8);

    int binCount = (int) std::pow(2, _filters.size());
    int descriptorSize = binCount * (_gridSize * _gridSize + (_gridSize - 1) * (_gridSize - 1));
    cv::Mat descriptor(1, descriptorSize, CV_64F);

    // Compute the histograms for the complete grid
    for (int i = 0; i < _gridSize; i++)
    {
        for (int j = 0; j < _gridSize; j++)
        {
            cv::Rect roi = cv::Rect(i * regionSize.width, j * regionSize.height, regionSize.width, regionSize.height);
            cv::Mat region = patterns(roi);

            cv::Mat regionHistogram = getRegionHistogram(region, binCount, gaussianKernel);

            cv::Rect targetLocation = cv::Rect(binCount * (j * _gridSize + i), 0, binCount, 1);
            cv::Mat targetRegion = descriptor(targetLocation);
            regionHistogram.copyTo(targetRegion);
        }
    }

    // Compute the histograms for the slided grid
    int numRegions = _gridSize * _gridSize;
    for (uint i = 0; i < _gridSize - 1; i++)
    {
        for (uint j = 0; j < _gridSize - 1; j++)
        {
            cv::Range rowRange = cv::Range(j * regionSize.height + regionSize.height / 2,
                                           (j + 1) * regionSize.height + regionSize.height / 2);
            cv::Range colRange = cv::Range(i * regionSize.width + regionSize.width / 2,
                                           (i + 1) * regionSize.width + regionSize.width / 2);
            cv::Mat region = patterns(rowRange, colRange).clone();

            cv::Mat regionHistogram = getRegionHistogram(region, binCount, gaussianKernel);

            cv::Rect targetLocation = cv::Rect(binCount * (numRegions + j * (_gridSize - 1) + i), 0, binCount, 1);
            cv::Mat targetRegion = descriptor(targetLocation);
            regionHistogram.copyTo(targetRegion);
        }
    }

    return descriptor;
}

cv::Mat QLZM::getGaussianKernel(cv::Size size, double sigma) const
{
    CV_Assert(sigma > 0);

    int kernelSize = (size.height % 2 != 0) ? size.height : size.height + 1;

    cv::Mat kernel = cv::getGaussianKernel(kernelSize, sigma);
    kernel = kernel * kernel.t();

    return kernel;
}

cv::Mat QLZM::getRegionHistogram(const cv::Mat &region, int binCount, const cv::Mat &gaussianKernel)
{
    cv::Mat histogram = cv::Mat::zeros(1, binCount, CV_64F);

    for (int i = 0; i < region.rows; i++)
    {
        for (int j = 0; j < region.cols; j++)
        {
            int bin = region.at<uchar>(i, j);
            histogram.at<double>(0, bin) += gaussianKernel.at<double>(i, j);
        }
    }

    histogram = histogram / (cv::norm(histogram, cv::NORM_L2) + std::numeric_limits<double>::epsilon());

    return histogram;
}

double QLZM::factorial(int x)
{
    double result = 1.0;
    while (x > 0)
    {
        result = result * x;
        x--;
    }
    return result;
}

std::vector<int> QLZM::getMomentMask(int order)
{
    std::vector<int> mask;

    for (int n = 0; n <= order; n++)
    {
        for (int m = 0; m <= n; m++)
        {
            if ((n - m) % 2 != 0)
            {
                continue;
            }

            if (m != 0)
            {
                mask.push_back(1);
            }
            else
            {
                mask.push_back(0);
            }
        }
    }

    return mask;
}

std::vector<cv::Mat> QLZM::computeFilters(int size, int order, ComplexMatrix matrix)
{
    std::vector<cv::Mat> filters;

    std::vector<int> momentMask = getMomentMask(order);

    for (int i = 0; i < matrix.size(); i++)
    {
        if (!momentMask[i])
        {
            continue;
        }

        cv::Mat realValFilter(size, size, CV_64F);
        cv::Mat imagValFilter(size, size, CV_64F);

        for (int j = 0; j < matrix[i].size(); j++)
        {
            int row = j / size;
            int col = j % size;

            double reel = std::real(std::conj(matrix[i][j]));
            double imag = std::imag(std::conj(matrix[i][j]));

            realValFilter.at<double>(row, col) = reel;
            imagValFilter.at<double>(row, col) = imag;
        }

        filters.push_back(realValFilter);
        filters.push_back(imagValFilter);
    }

    return filters;
}

ComplexMatrix QLZM::computeComplexMatrix(int size, int order)
{
    ComplexMatrix matrix;

    for (int n = 0; n <= order; n++)
    {
        for (int m = 0; m <= n; m++)
        {
            // m must be satisfy n - |m| is even
            if ((n - m) % 2 != 0)
            {
                continue;
            }

            ComplexVector vec = computeComplexVector(size, n, m);
            matrix.push_back(vec);
        }
    }

    return matrix;
}

ComplexVector QLZM::computeComplexVector(int size, int n, int m)
{
    ComplexVector vector;

    for (int y = 0; y < size; y++)
    {
        for (int x = 0; x < size; x++)
        {
            ComplexValue value = computeComplexValue(size, n, m, x, y);
            vector.push_back(value);
        }
    }

    return vector;
}

ComplexValue QLZM::computeComplexValue(int size, int n, int m, int x, int y)
{
    ComplexValue value;

    double D = (double) size * std::sqrt(2.0);
    double xn = (double) (2 * x + 1 - size) / D;
    double yn = (double) (2 * y + 1 - size) / D;

    for (int s = 0; s <= (n - m) / 2; s++)
    {
        // theta must be between the range of (0,2PI)
        double theta = std::atan2(yn, xn);
        if (theta < 0)
        {
            theta = 2 * M_PI + theta;
        }

        value += (pow(-1.0, (double) s)) * (factorial(n - s)) /
                 (factorial(s) * (factorial((n - 2 * s + m) / 2)) *
                  (factorial((n - 2 * s - m) / 2))) *
                 (pow(sqrt(xn * xn + yn * yn), (n - 2. * s))) *
                 4.0 / (D * D) * std::polar(1., m * theta);
    }

    return value;
}
