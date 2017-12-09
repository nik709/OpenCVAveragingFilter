#include "AveragingFilter.h"

AveragingFilter::AveragingFilter(cv::Mat image) {
	image.copyTo(this->image);
}

cv::Mat AveragingFilter::getImage() {
	return image;
}

AveragingFilter AveragingFilter::inGrayScale() {
	if (image.channels() == 3) {
		cv::cvtColor(image, image, CV_BGR2GRAY);
	}
	return *this;
}

AveragingFilter AveragingFilter::methodCanny() {
	cv::Mat gaus, gradients, supressed, cannyImage;
	cv::Mat angles(image.rows, image.cols, CV_8UC1);

	cv::GaussianBlur(image, gaus, cv::Size(3, 3), 1.4f, 1.4f);
	sobel(gaus, angles, gradients);
	nonMaximumSuppression(gaus, angles, gradients, supressed);
	cv::imshow("Suppressed", supressed);

	doubleTreshHold(supressed, MIN_TRESH_HOLD, MAX_TRESH_HOLD, cannyImage);
	tracingEdges(cannyImage);
	cv::imshow("Canny image", cannyImage);

	image = cannyImage;

	return *this;
}

AveragingFilter AveragingFilter::distanceMap() {

	reverseImage(image);
	cv::imshow("reversed image", image);
	//cv::Mat tmp;
	cv::distanceTransform(image, image, CV_DIST_L2, 3);
	//cv::imshow("distance map", tmp);
	return *this;
}

void AveragingFilter::sobel(const cv::Mat gaus, cv::Mat &angles, cv::Mat &gradients) {
	gaus.copyTo(gradients);
	int rowNum = gaus.rows;
	int colNum = gaus.cols;

	for (int i = 1; i < rowNum - 1; i++) {
		for (int j = 1; j < colNum - 1; j++) {
			std::vector<uchar> neighbors = getPixelNeighbours(gaus, i, j);

			int gx = (neighbors.at(1) + 2 * neighbors.at(4) + neighbors.at(7)) - (neighbors.at(0) + 2 * neighbors.at(3) + neighbors.at(5));
			int gy = (neighbors.at(0) + 2 * neighbors.at(1) + neighbors.at(2)) - (neighbors.at(5) + 2 * neighbors.at(6) + neighbors.at(7));
			float g = sqrt((float)(gx*gx) + (float)(gy*gy));
			float dir = (atan2((float)gy, gx) / CV_PI) * 180.0f;

			if (((dir < 22.5) && (dir >= -22.5)) || (dir >= 157.5) || (dir < -157.5))
				dir = 0;
			if (((dir >= 22.5) && (dir < 67.5)) || ((dir < -112.5) && (dir >= -157.5)))
				dir = 45;
			if (((dir >= 67.5) && (dir < 112.5)) || ((dir < -67.5) && (dir >= -112.5)))
				dir = 90;
			if (((dir >= 112.5) && (dir < 157.5)) || ((dir < -22.5) && (dir >= -67.5)))
				dir = 135;

			gradients.at<uchar>(i, j) = g;
			angles.at<uchar>(j, i) = dir;
		}
	}
	cv::imshow("angles", angles);
	cv::imshow("gradients", gradients);
}

std::vector<uchar> AveragingFilter::getPixelNeighbours(cv::Mat image, int x, int y) {
	std::vector<uchar> neighbors;

	for (int i = x - 1; i <= x + 1; i++) {
		for (int j = y - 1; j <= y + 1; j++) {
			if (i != x || j != y) {
				neighbors.push_back(image.at<uchar>(i, j));
			}
		}
	}
	return neighbors;
}

void AveragingFilter::nonMaximumSuppression(const cv::Mat gaus, cv::Mat &angles, cv::Mat &gradients, cv::Mat &supressed) {
	int rows = gaus.rows;
	int cols = gaus.cols;
	cv::Mat dst = cv::Mat(gradients.rows, gradients.cols, CV_8UC1);
	for (int y = 1; y < rows - 1; y++)
		for (int x = 1; x < cols - 1; x++) {
			cv::Point c = cv::Point(x, y);

			std::vector<uchar> gradientsNeighbors = getPixelNeighbours(gradients, y, x);

			if (((angles.at<uchar>(c) == 0) && (gradients.at<uchar>(c) >= gradientsNeighbors.at(3)) && (gradients.at<uchar>(y, x) >= gradientsNeighbors.at(4))) ||
				((angles.at<uchar>(c) == 45) && (gradients.at<uchar>(c) >= gradientsNeighbors.at(2)) && (gradients.at<uchar>(y, x) >= gradientsNeighbors.at(5))) ||
				((angles.at<uchar>(c) == 90) && (gradients.at<uchar>(c) >= gradientsNeighbors.at(1)) && (gradients.at<uchar>(y, x) >= gradientsNeighbors.at(5))) ||
				((angles.at<uchar>(c) == 135) && (gradients.at<uchar>(c) >= gradientsNeighbors.at(0)) && (gradients.at<uchar>(y, x) >= gradientsNeighbors.at(4))))
				dst.at<uchar>(c) = gradients.at<uchar>(c);
			else
				dst.at<uchar>(c) = 0;
		}
	dst.copyTo(supressed);
}

void AveragingFilter::doubleTreshHold(const cv::Mat supressed, int minTreshHold, float maxTreshHold, cv::Mat &result) {
	cv::Mat dst = cv::Mat(supressed.rows, supressed.cols, CV_8UC1);
	int rowNum = supressed.rows;
	int colNum = supressed.cols;

	for (int y = 1; y < rowNum - 1; y++)
		for (int x = 1; x < colNum - 1; x++)
		{
			uchar pix = supressed.at<uchar>(y, x);
			if (pix <= minTreshHold)
				pix = 0;
			else if (pix > minTreshHold && pix < maxTreshHold)
				pix = 127;
			else if (pix >= maxTreshHold)
				pix = 255;

			dst.at<uchar>(y, x) = pix;
		}
	result = dst;
}

int AveragingFilter::checkNeighbour(cv::Mat img, cv::Point c)
{
	if (c.inside(cv::Rect(0, 0, img.cols, img.rows)))
	{
		if (img.at<uchar>(c) != 127)
			return 0;

		img.at<uchar>(c) = 255;

		cv::Point c2 = cv::Point(c.x, c.y - 1);
		cv::Point c3 = cv::Point(c.x + 1, c.y - 1);
		cv::Point c4 = cv::Point(c.x + 1, c.y);
		cv::Point c5 = cv::Point(c.x + 1, c.y + 1);
		cv::Point c6 = cv::Point(c.x, c.y + 1);
		cv::Point c7 = cv::Point(c.x - 1, c.y + 1);
		cv::Point c8 = cv::Point(c.x - 1, c.y);
		cv::Point c9 = cv::Point(c.x - 1, c.y - 1);

		checkNeighbour(img, c2);
		checkNeighbour(img, c3);
		checkNeighbour(img, c4);
		checkNeighbour(img, c5);
		checkNeighbour(img, c6);
		checkNeighbour(img, c7);
		checkNeighbour(img, c8);
		checkNeighbour(img, c9);
	}
	return 0;
}

void AveragingFilter::tracingEdges(cv::Mat &src) {
	int rows = src.rows;
	int cols = src.cols;

	for (int y = 1; y < rows - 1; y++)
		for (int x = 1; x < cols - 1; x++)
		{
			cv::Point c = cv::Point(x, y);
			if (src.at<uchar>(c) == 255)
			{
				cv::Point c2 = cv::Point(x, y - 1);
				cv::Point c3 = cv::Point(x + 1, y - 1);
				cv::Point c4 = cv::Point(x + 1, y);
				cv::Point c5 = cv::Point(x + 1, y + 1);
				cv::Point c6 = cv::Point(x, y + 1);
				cv::Point c7 = cv::Point(x - 1, y + 1);
				cv::Point c8 = cv::Point(x - 1, y);
				cv::Point c9 = cv::Point(x - 1, y - 1);

				checkNeighbour(src, c2);
				checkNeighbour(src, c3);
				checkNeighbour(src, c4);
				checkNeighbour(src, c5);
				checkNeighbour(src, c6);
				checkNeighbour(src, c7);
				checkNeighbour(src, c8);
				checkNeighbour(src, c9);
			}
		}

	for (int y = 1; y < rows - 1; y++)
		for (int x = 1; x < cols - 1; x++)
			if (src.at<uchar>(y, x) == 127)
				src.at<uchar>(y, x) = 0;
}

void AveragingFilter::reverseImage(cv::Mat image) {
	int rowNum = image.rows;
	int colNum = image.cols;

	for (int y = 0; y < rowNum - 1; y++)
		for (int x = 0; x < colNum - 1; x++) {
			image.at<uchar>(y, x) = 255 - image.at<uchar>(y, x);
		}
}

void AveragingFilter::averagingFilter(cv::Mat src) {

	cv::Mat integralImage;
	cv::integral(src, integralImage);
	cv::Mat resultImage = cv::Mat(src);

    int rowNum = src.rows;
    int colNum = src.cols;
    int size = 0;

	for (int i = 1; i < rowNum + 1; i++) {
		for (int j = 1; j < colNum + 1; j++) {
			size = (int)(image.at<float>(i - 1, j - 1));
			size += (1 - size % 2);

			while ((i - size / 2 < 0) || (j - size / 2 < 0) || (i + size / 2 + 1 > rowNum) || (j + size / 2 + 1 > colNum)) {
				size -= 2;
			}

			if (size > 0) {
				cv::Vec3i A = integralImage.at<cv::Vec3i>(i - size / 2, j - size / 2);
				cv::Vec3i B = integralImage.at<cv::Vec3i>(i - size / 2, j + size / 2 + 1);
				cv::Vec3i C = integralImage.at<cv::Vec3i>(i + size / 2 + 1, j - size / 2);
				cv::Vec3i D = integralImage.at<cv::Vec3i>(i + size / 2 + 1, j + size / 2 + 1);
				resultImage.at<cv::Vec3b>(i - 1, j - 1) = (A + D - B - C) / (size*size);
			}
		}
	}
	cv::imshow("Result", resultImage);
}