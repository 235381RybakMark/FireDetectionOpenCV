#include <opencv2\opencv.hpp>
#include <iostream>
#define CVUI_IMPLEMENTATION
#include "cvui.h"


using namespace std;
using namespace cv;



class FireDetector {

public:
	Ptr<BackgroundSubtractor> mog2;
	Mat kernel;

	vector<vector<Point>> contours;
	vector<Vec4i> hireachy;

	int errorSize;

	FireDetector()
	{
		mog2 = createBackgroundSubtractorMOG2();
		kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
		errorSize = 10;

	}

	~FireDetector()
	{
	}

	Mat checkRGB(Mat &original)
	{
		Mat fgmask;
		//InputArray origina, Otput fgmask, 
		mog2->apply(original, fgmask, 0.001);//uczułość na ruch 0.001
		Mat result;
		result.create(original.size(), CV_8UC1);
		int Rt = 115;//próg dla koloru czerwonego
		int St = 45;//próg dla nasyconości
		Mat RGB[3];
		split(original, RGB);
		for (int i = 0; i < original.rows; i++)//sprawdzamy wszystkie pixele original
		{
			for (int j = 0; j < original.cols; j++)
			{
				float R = RGB[2].at<uchar>(i, j);
				float G = RGB[1].at<uchar>(i, j);
				float B = RGB[0].at<uchar>(i, j);
				
				
				double Sat = 0;
				int minValue = min(R, min(G, B));
				int maxValue = max(R, max(G, B)) / 255;
				if (maxValue == 0) {
					Sat = 0;
				}
				else {
					Sat = (maxValue - minValue) / maxValue;
				}
				double S = (1 - 3.0 * minValue / (R + G + B));//nasyconość koloru pixela (ważne, żeby wykryć ogień)
				if (fgmask.at<uchar>(i, j) > 0 && R > Rt && R >= G && G >= B && S > 0.20 && S > ((255 - R) * St / Rt))//dobry warunek, żeby rozpoznać ogień
				{
					result.at<uchar>(i, j) = 255;//Jeśli jest ogień
				}
				else
				{
					result.at<uchar>(i, j) = 0;//Jeśli nie ma ogniu 
				}
			}
		}
		dilate(result, result, kernel, Point(-1, -1));
		return result;
	}


	void drawContours(Mat &orginal, Mat input)
	{
		findContours(input, contours, hireachy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
		for (int i = 0; i < contours.size(); i++) {
			Rect selection = boundingRect(contours[i]);
			if (selection.width < errorSize || selection.height < errorSize) {
				std::cout << "Nie znalieziony ogien\n";
				continue;
			}
				
			rectangle(orginal, selection, Scalar(0, 255, 0), 2, 8, 0);
			std::cout << "Znalieziony ogien: " << selection << "\n";
		}
	}
};


int main(void)
{
	VideoCapture cap;

	int videoOption = 2;
	
	//tu podaj ścieżkę do video plkiu
	cap.open("fire.avi");

	Mat frame, computerVision;
	Mat options(100, 340, CV_8UC3, Scalar(0, 0, 0));
	int segments = 1;

	namedWindow("Err", WINDOW_NORMAL);
	namedWindow("Fire", WINDOW_AUTOSIZE);


	FireDetector f1;

	int tr1 = 25;
	createTrackbar("error size:", "Err", &f1.errorSize, 200, NULL);

	while (1)
	{
		cap >> frame;

		
		computerVision = f1.checkRGB(frame);

		f1.drawContours(frame, computerVision);

		cv::imshow("What computer see", computerVision);
		cv::imshow("Fire", frame);
		cv::imshow("Err", options);

		if (cv::waitKey(30) == 27) {
			cap.release();
			destroyAllWindows();
			return 0;
		}
	}
	cap.release();
	cv::waitKey(0);

}

