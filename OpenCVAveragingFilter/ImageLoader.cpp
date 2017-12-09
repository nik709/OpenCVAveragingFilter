#include "ImageLoader.h"

#define endLine std::endl

ImageLoader::ImageLoader(std::string path) {
	this->path = path;
}

void ImageLoader::load() {
	if (path.empty()) {
		std::cout << "Path is empty" << endLine;
		return;
	}

	image = cv::imread(path);
}

cv::Mat ImageLoader::getImage() {
	return image;
}