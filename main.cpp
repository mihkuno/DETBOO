#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    cv::Mat image = cv::imread("C:/Users/Mihkuno/Desktop/catto2.png");
    if (image.empty())
    {
        std::cout << "Error: Could not open image file." << std::endl;
        return 1;
    }
    cv::imshow("Image", image);
    cv::waitKey(0); 
    return 0;
}