#pragma once
#include <opencv\cv.h>
#include <opencv\highgui.h>

class ImageLoader {
private:
	cv::Mat image;
	std::string path;

public:
	ImageLoader(std::string path);
	void load();
	cv::Mat getImage();

};
