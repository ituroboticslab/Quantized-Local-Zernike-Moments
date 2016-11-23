#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include "QLZM.h"

int main()
{
    cv::Mat image = cv::imread("image.png", CV_LOAD_IMAGE_GRAYSCALE);

    int patchSize = 7;
    int gridSize = 5;
    int momentOrder = 2;

    cv::Ptr<QLZM> qlzm = new QLZM(patchSize, gridSize, momentOrder);

    cv::Mat patterns = qlzm->extractPatterns(image);
    cv::Mat descriptor = qlzm->computeDescriptor(patterns);

    std::cout << descriptor << std::endl;

    return 0;
}
