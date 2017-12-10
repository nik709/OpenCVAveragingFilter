#include "ImageLoader.h"
#include "AveragingFilter.h"

void showImage(std::string windowName, cv::Mat image) {
	cv::imshow(windowName, image);
}

void main() {
	std::string path = "testImages/2.png";
	ImageLoader imageLoader(path);
	imageLoader.load();
	showImage("Source image", imageLoader.getImage());

	AveragingFilter filter(imageLoader.getImage());
	filter.inGrayScale()
		.methodCanny()
		.distanceMap()
		.averagingFilter(imageLoader.getImage());

	cv::waitKey(0);
}
