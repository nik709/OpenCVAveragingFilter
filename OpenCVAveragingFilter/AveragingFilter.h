#pragma once
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <vector>

#define MIN_TRESH_HOLD 60
#define MAX_TRESH_HOLD 100

class AveragingFilter {
private:
	cv::Mat image;
	cv::Mat resultImage;
	void sobel(const cv::Mat gaus, cv::Mat &angles, cv::Mat &gradients);
	void nonMaximumSuppression(const cv::Mat gaus, cv::Mat &angles, cv::Mat &gradients, cv::Mat &supressed);
	std::vector<uchar> getPixelNeighbours(cv::Mat image, int x, int y);
	void doubleTreshHold(const cv::Mat supressed, int minTreshHold, float maxTreshHold, cv::Mat &result);
	int checkNeighbour(cv::Mat img, cv::Point c);
	void tracingEdges(cv::Mat &src);
	void reverseImage(cv::Mat image);
public:
	AveragingFilter(cv::Mat image);
	AveragingFilter inGrayScale();
	AveragingFilter methodCanny();
	AveragingFilter distanceMap();
	void averagingFilter(cv::Mat src);

	cv::Mat getImage();
};

