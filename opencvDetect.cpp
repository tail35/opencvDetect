// opencvDetect.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "common.h"


void DetectAndDrawFace() {
 std::string xmlPath = "haarcascade_frontalface_alt.xml";
 cv::CascadeClassifier ccf;
 ccf.load(xmlPath);
 cv::VideoCapture cap(0);

 while (waitKey(30) != 27) {
  std::chrono::time_point<std::chrono::steady_clock>  start = std::chrono::steady_clock::now();
  cv::Mat img;
  cap >> img;

  cv::Mat src(img);
  cv::resize(src, img, cv::Size(img.cols * 0.4, img.rows * 0.4));  

  std::vector<Rect> faces;
  cv::Mat gray;
  cvtColor(img, gray, CV_BGR2GRAY);
  equalizeHist(gray, gray);
  ccf.detectMultiScale(gray, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30)); //检测人脸
  for (std::vector<Rect>::const_iterator iter = faces.begin(); iter != faces.end(); iter++)
  {
   rectangle(img, *iter, Scalar(0, 0, 255), 10, 8); //画出脸部矩形
  }

  std::chrono::time_point<std::chrono::steady_clock>  end = std::chrono::steady_clock::now();
  auto dur = end - start;
  std::cout << "dur:" << dur.count() / 1000 / 1000 << std::endl;
  imshow("faces", img);
 }
}

int main()
{
  DetectAndDrawFace();
  return 0;
}
