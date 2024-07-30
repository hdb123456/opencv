#include <opencv2/opencv.hpp>
#include <linux/videodev2.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <cmath>
#include <main.h>


using namespace cv;
using namespace std;


const char* dilation_name = "dilation";
const char* Erosion_name = "Erosion";
const char* morph_name = "morph";
const char* redpoint = "redpoint";

int ksize = 3;
double sigma = 1.0;

int morph_operator = 0;//开运算
int morph_elem = 2;//圆形
int morph_size = 3;//核大小

int const max_operator = 4;
//使用中
int dilation_elem = 0;
int dilation_size = 3;
int erosion_elem = 0;
int erosion_size = 3;

int fd = -1;
struct v4l2_capability cap;
struct v4l2_format fmt;
struct v4l2_requestbuffers reqbuf;
struct v4l2_buffer buf;


const int max_elem = 2;
const int max_kernel_size = 21;

Mat src, dst;
Mat dilation_dst, erosion_dst;
vector<vector<Point>> squares;
// HSV阈值
Mat hsv_img;
int LOW_H = 0;
int LOW_S = 0;
int LOW_V = 30;
int UP_H = 27;
int UP_S = 213;
int UP_V = 255;
const int MAX_COLOR = 180;
const int MAX_SV = 255;
//霍夫圆变换参数
// 全局变量
Mat binaryWhite, binaryBlack;
vector<Vec3f> circlesWhite, circlesBlack;
int dp = 2;
int minDist = 62;
int param1 = 65;
int param2 = 36;
int minRadius = 23;
int maxRadius = 33;

void Dilation(int, void*);
void Erosion(int, void*);
void Morphology_Operations(int, void*);
double angle(Point pt1, Point pt2, Point pt0);
void findSquares(const Mat& image, vector<vector<Point>>& squares);
void create_windows(void);
int video_start(void);
void cleanup(int fd, void** buffers, int buffer_count);
void adjust(int, void*);
void handle( Mat src);
void detectAndDrawCircles(int, void*);


int main(int argc, char *argv[]) 
{
	return video_start();
}

// 形态学膨胀操作的回调函数
void Dilation(int, void*) {
    int dilation_type = dilation_elem == 0 ? MORPH_RECT :
                        dilation_elem == 1 ? MORPH_CROSS : MORPH_ELLIPSE;
    Mat element = getStructuringElement(dilation_type,
                                        Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                        Point(dilation_size, dilation_size));
    dilate(binaryBlack, binaryBlack, element);
    imshow(dilation_name, binaryBlack);
}

// 形态学腐蚀操作的回调函数
void Erosion(int, void*) {
    int erosion_type = erosion_elem == 0 ? MORPH_RECT :
                       erosion_elem == 1 ? MORPH_CROSS : MORPH_ELLIPSE;
    Mat element = getStructuringElement(0,
                                        Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                        Point(erosion_size, erosion_size));
    erode(binaryBlack, binaryBlack, element);
    imshow(Erosion_name, binaryBlack);
}

// 计算三个点形成的角度
double angle(Point pt1, Point pt2, Point pt0) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2));
}

// 找到图像中的矩形轮廓
void findSquares(const Mat& image, vector<vector<Point>>& squares) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(image, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    
    vector<Point> approx;
    for (size_t i = 0; i < contours.size(); i++) {
        // 只处理内部轮廓
        if (hierarchy[i][2] == -1) continue;
        
        approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);
        if (approx.size() == 4 && fabs(contourArea(approx)) > 1000 ) {
            double maxCosine = 0;
            for (int j = 2; j < approx.size(); j++) {
                double cosine = fabs(angle(approx[j % approx.size()], approx[j - 2], approx[j - 1]));
                maxCosine = MAX(maxCosine, cosine);
            }
            if (maxCosine < 0.5) {
                // 计算矩形的重心
                Moments m = moments(approx, true);
                Point center(m.m10/m.m00, m.m01/m.m00);
                // 如果中心在图像的中间区域
                if (center.x > image.cols / 4 && center.x < 3 * image.cols / 4 &&
                    center.y > image.rows / 4 && center.y < 3 * image.rows / 4) {
                    squares.push_back(approx);
                }
            }
        }
    }
     // 仅保留最里面的轮廓
    if (squares.size() == 2) {
        squares.erase(squares.begin());
    } 
}


// 形态学操作的回调函数
void Morphology_Operations(int, void*) {
    // 选择形态学操作类型
    int operation = morph_operator + 2;
 
    Mat element = getStructuringElement(morph_elem, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
 
    morphologyEx(binaryBlack, binaryBlack, operation, element);
    imshow(morph_name, binaryBlack);
}

void create_windows(void)
{
    // 创建窗口
    //形态学操作
    // 膨胀
    namedWindow(dilation_name, WINDOW_AUTOSIZE);
    createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", dilation_name, &dilation_elem, max_elem, Dilation);
    createTrackbar("Kernel size:\n 2n +1", dilation_name, &dilation_size, max_kernel_size, Dilation);
    //腐蚀
    namedWindow(Erosion_name, WINDOW_AUTOSIZE);
    createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", Erosion_name, &erosion_elem, max_elem, Erosion);
    createTrackbar("Kernel size:\n 2n +1", Erosion_name, &erosion_size, max_kernel_size, Erosion);
    //更多形态学操作
    namedWindow(morph_name, WINDOW_AUTOSIZE); // Create window
    createTrackbar("Operator:\n 0: Opening - 1: Closing \n 2: Gradient - 3: Top Hat \n 4: Black Hat", morph_name, &morph_operator, max_operator, Morphology_Operations);
    createTrackbar("Element:\n 0: Rect - 1: Cross - 2: Ellipse", morph_name, &morph_elem, max_elem, Morphology_Operations);
    createTrackbar("Kernel size:\n 2n +1", morph_name, &morph_size, max_kernel_size, Morphology_Operations);
    // 颜色阈值
    namedWindow(redpoint, WINDOW_AUTOSIZE);
    createTrackbar("LOW_H", redpoint, &LOW_H, MAX_COLOR, adjust);
    createTrackbar("LOW_S", redpoint, &LOW_S, MAX_SV, adjust);
    createTrackbar("LOW_V", redpoint, &LOW_V, MAX_SV, adjust);
    createTrackbar("UP_H", redpoint, &UP_H, MAX_COLOR, adjust);
    createTrackbar("UP_S", redpoint, &UP_S, MAX_SV, adjust);
    createTrackbar("UP_V", redpoint, &UP_V, MAX_SV, adjust);
    namedWindow("Detected Circles", WINDOW_AUTOSIZE);//霍夫圆变换
    //霍夫圆变换
    createTrackbar("dp", "Detected Circles", &dp, 10, detectAndDrawCircles);
    createTrackbar("minDist", "Detected Circles", &minDist, 100, detectAndDrawCircles);
    createTrackbar("param1", "Detected Circles", &param1, 100, detectAndDrawCircles);
    createTrackbar("param2", "Detected Circles", &param2, 100, detectAndDrawCircles);
    createTrackbar("minRadius", "Detected Circles", &minRadius, 100, detectAndDrawCircles);
    createTrackbar("maxRadius", "Detected Circles", &maxRadius, 100, detectAndDrawCircles);
       	
}

void adjust(int, void*)
{
	Scalar lower_black(0, 0, 144);
	Scalar upper_black(180, 255, 255);
	inRange(hsv_img, lower_black, upper_black, binaryBlack);
	imshow("binaryBlack",binaryBlack);

    Scalar lower_white(0, 0, 150);
	Scalar upper_white(180, 51, 255);
	inRange(hsv_img, lower_white, upper_white, binaryWhite);
	imshow("binaryWhite",binaryWhite);
}

int video_start(void)
{
    // 打开视频设备
    if ((fd = open("/dev/video4", O_RDWR)) == -1) {
        perror("open video device failed");
        return 1;
    }

    // 查询视频设备功能
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
        perror("query video device capability failed");
        close(fd);
        return 1;
    }

    // 设置视频格式
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = 640;
    fmt.fmt.pix.height = 480;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
        perror("set video format failed");
        close(fd);
        return 1;
    }
    
     // 设置帧率
    struct v4l2_streamparm parm;
    memset(&parm, 0, sizeof(parm));
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = 60; // 设置帧率为60 fps

    if (ioctl(fd, VIDIOC_S_PARM, &parm) == -1) {
        perror("set frame rate failed");
        close(fd);
        return 1;
    }

    // 请求视频缓冲区
    memset(&reqbuf, 0, sizeof(reqbuf));
    reqbuf.count = 4;
    reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    reqbuf.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd, VIDIOC_REQBUFS, &reqbuf) == -1) {
        perror("request video buffer failed");
        close(fd);
        return 1;
    }

    // 映射缓冲区到用户空间
    void* buffers[4];
    for (int i = 0; i < reqbuf.count; ++i) {
        memset(&buf, 0, sizeof(buf));
        buf.index = i;
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd, VIDIOC_QUERYBUF, &buf) == -1) {
            perror("query video buffer failed");
            close(fd);
            return 1;
        }

        buffers[i] = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        if (buffers[i] == MAP_FAILED) {
            perror("mmap video buffer failed");
            close(fd);
            return 1;
        }
    }

    // 将缓冲区入队
    for (int i = 0; i < reqbuf.count; ++i) {
        memset(&buf, 0, sizeof(buf));
        buf.index = i;
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) {
            perror("enqueue video buffer failed");
            close(fd);
            return 1;
        }
    }

    // 启动视频捕捉
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) == -1) {
        perror("start video capture failed");
        close(fd);
        return 1;
    }
    // 设置自动白平衡
    struct v4l2_control ctrl;
    ctrl.id = V4L2_CID_AUTO_WHITE_BALANCE;
    ctrl.value = 0; // 0 表示关闭自动白平衡
    if (ioctl(fd, VIDIOC_S_CTRL, &ctrl) == -1) {
        perror("set auto white balance failed");
        close(fd);
        return 1;
    }
    
    create_windows();//创造窗口
    
    while (true) 
	{
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(fd, &fds);
        timeval tv = {2, 0};
        int r = select(fd + 1, &fds, NULL, NULL, &tv);

        if (r == -1) {
            perror("select failed");
            break;
        }
        if (r == 0) {
            printf("select timeout\n");
            break;
        }

        // 取出缓冲区
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
            perror("dequeue video buffer failed");
            continue;
        }

        unsigned char* frame = (unsigned char*)buffers[buf.index];
        Mat yuyv_img(Size(640, 480), CV_8UC2, frame);
        // 色彩转换
        cvtColor(yuyv_img, src, COLOR_YUV2BGR_YUYV);
        imshow("src", src);
        
        // 色彩校正（根据需要调整），提高对比度
        Mat lookupTable(1, 256, CV_8U);
        uchar* p = lookupTable.ptr();
        for (int i = 0; i < 256; ++i)
            p[i] = (uchar)(pow((double)i / 255.0, 1.2) * 255.0); // Gamma correction
        LUT(src, lookupTable, src);
        //图像处理函数
        handle(src);    
        // 将缓冲区重新入队
        if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) {
            perror("enqueue video buffer failed");
            break;
        }

        if (waitKey(30) == ' ') break;
	}
    cleanup(fd, buffers, reqbuf.count);
    return 0;
}

//关闭相机输入通道
void cleanup(int fd, void** buffers, int buffer_count) {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMOFF, &type) == -1) {
                        perror("stop video capture failed");
        }
    for (int i = 0; i < buffer_count; ++i) {
        munmap(buffers[i], buf.length);
    }
    close(fd);
}

// 绘制圆形的函数
void drawCircles(Mat& src, const vector<Vec3f>& circles, const Scalar& color) {
    for (const auto& circle : circles) {
        Point center(cvRound(circle[0]), cvRound(circle[1]));
        int radius = cvRound(circle[2]);
        cv::circle(src, center, radius, color, 2);
        cv::circle(src, center, 2, color, 3);

        // 显示圆心坐标
        stringstream ss;
        ss << "(" << center.x << ", " << center.y << ")";
        putText(src, ss.str(), center, FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    }
}

void detectAndDrawCircles(int, void*) {
    vector<Vec3f> circlesWhiteTemp, circlesBlackTemp;
    
    // 检测白色棋子的圆形
    HoughCircles(
        binaryWhite,
        circlesWhiteTemp,
        HOUGH_GRADIENT,
        dp,
        minDist,
        param1,
        param2,
        minRadius,
        maxRadius
    );
    circlesWhite = circlesWhiteTemp;

    // 检测黑色棋子的圆形
    HoughCircles(
        binaryBlack,
        circlesBlackTemp,
        HOUGH_GRADIENT,
        dp,
        minDist,
        param1,
        param2,
        minRadius,
        maxRadius
    );
    circlesBlack = circlesBlackTemp;

    // 清空原图像的圆圈
    Mat displayImage = src.clone();
    drawCircles(displayImage, circlesWhite, Scalar(0, 255, 0));  // 绿色圆圈表示白色棋子
    drawCircles(displayImage, circlesBlack, Scalar(0, 0, 255));  // 红色圆圈表示黑色棋子

    // 显示结果
    imshow("Detected Circles", displayImage);
}

void handle( Mat src)
{
    
    // 边缘检测
    Mat gray2;
    cvtColor(src, gray2, COLOR_BGR2GRAY);
    Canny(gray2, dst, 50, 100, 3);
    
    // 形态学去噪
   // Morphology_Operations(0, 0);
    Mat element1 = getStructuringElement(0, Size(3, 3), Point(1, 1));
    morphologyEx(dst, dst, 3, element1);
    //膨胀
    int dilation_type =  MORPH_RECT ;//MORPH_RECT: MORPH_CROSS : MORPH_ELLIPSE;
    Mat element2 = getStructuringElement(dilation_type, Size(2 * 3 + 1, 2 * 3 + 1), Point(3, 3));
    dilate(dst, dst, element2);
    imshow("Dilation of squares", dst);
    //腐蚀
    int erosion_type =  MORPH_RECT ;//MORPH_RECT: MORPH_CROSS : MORPH_ELLIPSE;
    Mat element = getStructuringElement(erosion_type, Size(2 * 3 + 1, 2 * 3 + 1), Point(3, 3));
    erode(dst, dst, element);
    imshow("Erosion of squares", dst);

    // 绘制矩形轮廓   
    vector<vector<Point>> squares;
    findSquares(dst, squares);
        
    for (const auto& square : squares) 
    {
        polylines(src, square, true, Scalar(0, 255, 0), 1, LINE_AA); // 使用抗锯齿绘制

        for (size_t i = 0; i < square.size(); ++i) {
            // 画角点
            circle(src, square[i], 2, Scalar(0, 0, 255), -1); // 红色圆点
            // 添加角点坐标
            putText(src, "(" + to_string(square[i].x) + ", " + to_string(square[i].y) + ")", square[i], FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);    
        }

        
    }
    imshow("hello", src);
     // 仅保留检测到的第一个方形区域
    if (squares.size() > 0) {
        vector<Point> square = squares[0];

        // 创建一个掩码
        Mat mask = Mat::zeros(src.size(), src.type());
        fillConvexPoly(mask, square, Scalar(255, 255, 255));
        bitwise_and(src, mask, src);

        // 绘制检测到的方形区域
        polylines(src, square, true, Scalar(0, 255, 0), 3, LINE_AA);
    }
    imshow("Squares", src);
			
            
    //黑白色棋子二值化
    cvtColor(src, hsv_img, COLOR_BGR2HSV);
    // 黑白棋子掩码
    adjust(0, 0);
    // 形态学操作去噪
    bitwise_not(binaryBlack, binaryBlack);

    Erosion(0, 0);//0,3
    Dilation(0, 0);//0,3

    imshow("binaryBlack", binaryBlack);
    imshow("binaryWhite", binaryWhite);
    // 检测圆形
    detectAndDrawCircles(0, 0);
    
}