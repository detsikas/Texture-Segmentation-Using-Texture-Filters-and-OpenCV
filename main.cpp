//
//  main.cpp
//  TextureSegmentation
//

#include <iostream>
#include <sstream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <getopt.h>
#include <map>

using namespace std;
using namespace cv;

unsigned int const HISTOGRAM_DIMENSION = 9;
unsigned int const HISTOGRAM_EDGE_OFFSET = (HISTOGRAM_DIMENSION-1)/2;
float const TEXTURE_EDGE_THRESHOLD = 204;//0.8;
unsigned int const AREA_THRESHOLD = 2000; //originally 2000

typedef vector<Point> Contour;
typedef vector<Contour> Contours;
typedef vector<Vec4i> Hierarchy;
typedef map<unsigned int, int> HDM; //Histogram differences map

static void displayImage(string const & windowName, Mat image, bool withExit=false)
{
    namedWindow(windowName, WINDOW_AUTOSIZE );
    imshow(windowName, image);
    waitKey(0);
    
    if (withExit)
        exit(0);
}

template<typename T>
static void getIteratorAtPoint(Mat const & image, Point const & p, MatConstIterator_<T> & it)
{
    Size iSize = image.size();
    it = (image.begin<T>()+p.x*iSize.width+p.y);
}

template<typename T>
static MatConstIterator_<T> getIteratorAtPoint(Mat const & image, Point const & p)
{
    Size iSize = image.size();
    return (image.begin<T>()+p.x*iSize.width+p.y);
}

template<typename T>
static MatIterator_<T> getIteratorAtPoint(Mat & image, Point const & p)
{
    Size iSize = image.size();
    return (image.begin<T>()+p.x*iSize.width+p.y);    
}

//Point p represents the top point of the column to be removed
static void removeColumn(Mat const & image, Mat & histogram, Point const & p, HDM & m)
{
    MatConstIterator_<uchar> it;
    MatConstIterator_<uchar> begin = getIteratorAtPoint<uchar>(image, p);
    MatConstIterator_<uchar> end = begin+HISTOGRAM_DIMENSION*image.cols;
    HDM::iterator itHDM;
    
    for (it = begin; it!=end; it+=image.cols)
    {
        itHDM = m.find((uchar)*it);
        if (itHDM!=m.end())
        {
            itHDM->second--;
        }
        else
        {
            m.insert(pair<unsigned int, int>(*it,-1));
        }
    }
}

//Point p represents the top point of the column to be added
static void addColumn(Mat const & image, Mat & histogram, Point const & p, HDM & m)
{
    MatConstIterator_<uchar> it;
    MatConstIterator_<uchar> begin = getIteratorAtPoint<uchar>(image, p);
    MatConstIterator_<uchar> end = begin+HISTOGRAM_DIMENSION*image.cols;
    HDM::iterator itHDM;
    
    for (it = begin; it!=end; it+=image.cols)
    {      
        itHDM = m.find((uchar)*it);
        if (itHDM!=m.end())
        {
            itHDM->second++;
        }
        else
        {
            m.insert(pair<unsigned int, int>(*it,1));
        }
    }
}

//Point p represents the leftmost point of the row to be removed
static void removeRow(Mat const & image, Mat & histogram, Point const & p, HDM & m)
{
    MatConstIterator_<uchar> it;
    MatConstIterator_<uchar> begin = getIteratorAtPoint<uchar>(image, p);
    MatConstIterator_<uchar> end = begin+HISTOGRAM_DIMENSION;
    HDM::iterator itHDM;

    for (it = begin; it!=end; it++)
    {
        itHDM = m.find((uchar)*it);
        if (itHDM!=m.end())
        {
            itHDM->second--;
        }
        else
        {
            m.insert(pair<unsigned int, int>(*it,-1));
        }
    }
}

//Point p represents the rightmost point of the row to be added
static void addRow(Mat const & image, Mat & histogram, Point const & p, HDM & m)
{
    MatConstIterator_<uchar> it;
    MatConstIterator_<uchar> begin = getIteratorAtPoint<uchar>(image, p);
    MatConstIterator_<uchar> end = begin+HISTOGRAM_DIMENSION;
    HDM::iterator itHDM;

    for (it = begin; it!=end; it++)
    {
        itHDM = m.find((uchar)*it);
        if (itHDM!=m.end())
        {
            itHDM->second++;
        }
        else
        {
            m.insert(pair<unsigned int, int>(*it,1));
        }
    }
}

static double calculateEntropy(Mat const & histogram)
{
    double entropyVal = 0;
    MatConstIterator_<float> it;

    for (it=histogram.begin<float>(); it!=histogram.end<float>(); it++)
    {
        int value = *it;
        if (*it<0)
        {
            cout<<*it<<endl;
        }
        
        if ((*it)>0)
            entropyVal -= (value*log2(value));
    }
    
    entropyVal/=255;
    return entropyVal;
}

static double calculateEntropyAndUpdateHistogram(HDM & m, double previousPixelEntropyValue, Mat & histogram)
{
    double pixelEntropyValue = previousPixelEntropyValue;
    MatIterator_<float> hbegin = histogram.begin<float>();
    for (HDM::const_iterator it=m.begin(); it!=m.end(); it++)
    {
        if (it->second!=0)
        {
            //Remove previous entropy value (watch for the negative signs in the sum
            float histVal = *(hbegin+(int)it->first);
            if (histVal>0)
                pixelEntropyValue+=(histVal*log2(histVal)/255);
            
            //Update histogram
            histVal+=it->second;
            *(hbegin+(int)it->first) = histVal;
            
            //Add new entropy value
            if (histVal>0)
                pixelEntropyValue-=(histVal*log2(histVal)/255);
        }
    }
    m.clear();
    return pixelEntropyValue;
}

void calculateInitialHistogram(Mat const & image, Mat & histogram)
{
    Size iSize = image.size();
    /// Establish the number of bins
    int histSize = 256;
    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 };
    /// Set the channels
    int channels = 0;
    const float* histRange = { range };
    bool uniform = true; bool accumulate = false;
    Mat initialHistogramMask = Mat::ones(HISTOGRAM_DIMENSION, HISTOGRAM_DIMENSION, CV_8UC1);
    copyMakeBorder(initialHistogramMask, initialHistogramMask, 0, iSize.height-HISTOGRAM_DIMENSION, 0, iSize.width-HISTOGRAM_DIMENSION, BORDER_CONSTANT, 0);
    calcHist(&image, 1, &channels, initialHistogramMask, histogram, 1, &histSize, &histRange, uniform, accumulate);
}


static void entropyFilt(Mat const & image, Mat & entropy)
{
    double pixelEntropyValue;

    Point const OFFSET_POINT(HISTOGRAM_EDGE_OFFSET, HISTOGRAM_EDGE_OFFSET);
    Size iSize = image.size();
    bool stop = false;
    MatIterator_<float> ite = entropy.begin<float>();

    Mat histogram;
    
    calculateInitialHistogram(image, histogram);
    
    int direction = 1;
    Point currentPoint(HISTOGRAM_EDGE_OFFSET,HISTOGRAM_EDGE_OFFSET);
    
    pixelEntropyValue = calculateEntropy(histogram);
    *ite = calculateEntropy(histogram);

    Point previousPoint = currentPoint;
    currentPoint.y++;
    HDM m;
    
    while(!stop)
    {
        //Get current pointers
        ite = getIteratorAtPoint<float>(entropy, currentPoint-OFFSET_POINT);
        
        //Calculate (update) histogram
        if (previousPoint.y!=currentPoint.y)
        {
            //Remove previous column
            removeColumn(image, histogram, Point(previousPoint.x-HISTOGRAM_EDGE_OFFSET, previousPoint.y-direction*HISTOGRAM_EDGE_OFFSET), m);
            
            //Add current column
            addColumn(image, histogram, Point(currentPoint.x-HISTOGRAM_EDGE_OFFSET, currentPoint.y+direction*HISTOGRAM_EDGE_OFFSET), m);
        }
        
        if (previousPoint.x!=currentPoint.x)
        {
            //Remove previous row
            removeRow(image, histogram, Point(previousPoint.x-HISTOGRAM_EDGE_OFFSET, previousPoint.y-HISTOGRAM_EDGE_OFFSET), m);
            
            //Add current row
            addRow(image, histogram, Point(currentPoint.x+HISTOGRAM_EDGE_OFFSET, currentPoint.y-HISTOGRAM_EDGE_OFFSET), m);
        }

        pixelEntropyValue = calculateEntropyAndUpdateHistogram(m, pixelEntropyValue, histogram);
        *ite = pixelEntropyValue;
        
        //Advance pointers
        previousPoint = currentPoint;
        currentPoint.y+=direction;
        if (currentPoint.y==(iSize.width-HISTOGRAM_EDGE_OFFSET))
        {
            currentPoint = Point(currentPoint.x+1, currentPoint.y-1);
            direction = -1;
        }
        else if (currentPoint.y==(HISTOGRAM_EDGE_OFFSET-1))
        {
            currentPoint = Point(currentPoint.x+1, HISTOGRAM_EDGE_OFFSET);
            direction = 1;
        }
        
        if (currentPoint.x==(iSize.height-HISTOGRAM_EDGE_OFFSET))
            stop = true;
    }
}

static void calculateSymetricPoint(Point const & inPoint, Point & outPoint, Size const & psize)
{
    if (inPoint.y<HISTOGRAM_EDGE_OFFSET)
        outPoint.y = 2*HISTOGRAM_EDGE_OFFSET-inPoint.y-1;
    else if (inPoint.y>(psize.width-HISTOGRAM_EDGE_OFFSET-1))
        outPoint.y = psize.width-2*HISTOGRAM_EDGE_OFFSET+psize.width-inPoint.y-1;
    else
        outPoint.y = inPoint.y;
    
    if (inPoint.x<HISTOGRAM_EDGE_OFFSET)
        outPoint.x = 2*HISTOGRAM_EDGE_OFFSET-inPoint.x-1;
    else if (inPoint.x>(psize.height-HISTOGRAM_EDGE_OFFSET-1))
        outPoint.x = psize.height-2*HISTOGRAM_EDGE_OFFSET+psize.height-inPoint.x-1;
    else
        outPoint.x = inPoint.x;
}

static void symmetricPadding(Mat const & image, Mat & paddedImage)
{
    //Padd with zeros
    copyMakeBorder(image, paddedImage, HISTOGRAM_EDGE_OFFSET, HISTOGRAM_EDGE_OFFSET, HISTOGRAM_EDGE_OFFSET, HISTOGRAM_EDGE_OFFSET, 0);
    MatIterator_<uchar> its;
    MatIterator_<uchar> it;
    
    Size psize = paddedImage.size();
    for (unsigned int j=0; j<psize.width; j++)
    {
        for (unsigned int i=0; i<HISTOGRAM_EDGE_OFFSET; i++)
        {
            Point sp;
            Point p(i,j);
            calculateSymetricPoint(p, sp, psize);
            its = paddedImage.begin<uchar>()+sp.x*paddedImage.cols+sp.y;
            it = paddedImage.begin<uchar>()+p.x*paddedImage.cols+p.y;
            *it = *its;
        }

        for (unsigned int i=psize.height-HISTOGRAM_EDGE_OFFSET; i<psize.height; i++)
        {
            Point sp;
            Point p(i,j);
            calculateSymetricPoint(p, sp, psize);
            its = paddedImage.begin<uchar>()+sp.x*paddedImage.cols+sp.y;
            it = paddedImage.begin<uchar>()+p.x*paddedImage.cols+p.y;
            *it = *its;
        }
    }
    
    
    for (unsigned int i=HISTOGRAM_EDGE_OFFSET; i<psize.height-HISTOGRAM_EDGE_OFFSET; i++)
    {
        for (unsigned int j=0; j<HISTOGRAM_EDGE_OFFSET; j++)
        {
            Point sp;
            Point p(i,j);
            calculateSymetricPoint(p, sp, psize);
            its = paddedImage.begin<uchar>()+sp.x*paddedImage.cols+sp.y;
            it = paddedImage.begin<uchar>()+p.x*paddedImage.cols+p.y;
            *it = *its;
        }
        
        for (unsigned int j=psize.width-HISTOGRAM_EDGE_OFFSET; j<psize.width; j++)
        {
            Point sp;
            Point p(i,j);
            calculateSymetricPoint(p, sp, psize);
            its = paddedImage.begin<uchar>()+sp.x*paddedImage.cols+sp.y;
            it = paddedImage.begin<uchar>()+p.x*paddedImage.cols+p.y;
            *it = *its;
        }
    }
}

static void bwareaopen(Mat const & input, Mat & output)
{
    // Find all contours
    Contours contours;
    findContours(input.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    
    for (int i = 0; i < contours.size(); i++)
    {
        // Calculate contour area
        double area = contourArea(contours[i]);
        
        // Remove small objects by drawing the contour with black color
        //if (area > 0 && area <= AREA_THRESHOLD)
        if (area>AREA_THRESHOLD)
            drawContours(output, contours, i, 255, -1);
    }
}

static void imclose(Mat & image)
{
    morphologyEx(image, image, MORPH_CLOSE, cv::getStructuringElement(MORPH_RECT, Size(9,9)));
}

static void imfill(Mat & image)
{
    // Floodfill from point (0, 0)
    Mat im_floodfill = image.clone();
    floodFill(im_floodfill, cv::Point(0,0), Scalar(255));
    
    // Invert floodfilled image
    Mat im_floodfill_inv;
    bitwise_not(im_floodfill, im_floodfill_inv);
    
    // Combine the two images to get the foreground.
    image = (image | im_floodfill_inv);
}

static void maskImage(Mat const & image, Mat const & mask, Mat & output)
{
    output = image & mask;
}

static void convertion(Mat const & input, Mat & output)
{
    double minVal, maxVal;
    minMaxLoc(input, &minVal, &maxVal);
    input.convertTo(output, CV_8UC1, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
}

int main(int argc, const char * argv[])
{
    // Load the image in grayscale mode
    Mat input = imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
    displayImage("Image", input);
    
    //Padd image
    Mat paddedImage;
    symmetricPadding(input, paddedImage);
    displayImage("Padded image", paddedImage);
    
    //Calculate entropy matrix
    Mat entropyMat = Mat::zeros(input.rows, input.cols, CV_32FC1);
    entropyFilt(paddedImage, entropyMat);
    
    //Rescale entropy matrix
    normalize(entropyMat, entropyMat, 0, 1, NORM_MINMAX, -1, Mat());
    displayImage("Normalized entropy", entropyMat);
    
    //Convert to [0, 255]]
    Mat convertedEntropyMat;
    convertion(entropyMat, convertedEntropyMat);

    //Threshold textures
    threshold(convertedEntropyMat, convertedEntropyMat, TEXTURE_EDGE_THRESHOLD, 255/*1*/, CV_THRESH_BINARY);
    displayImage("Threshold Normalized entropy", convertedEntropyMat);
    
    //Filter areas
    Mat filteredEntropyMat = Mat::zeros(input.rows, input.cols, CV_8UC1);
    bwareaopen(convertedEntropyMat, filteredEntropyMat);
    displayImage("Filtered entropy", filteredEntropyMat);
    
    //Smooth edges and close open holes
    imclose(filteredEntropyMat);
    displayImage("Closed image", filteredEntropyMat);
    
    //Fill holes
    imfill(filteredEntropyMat);
    displayImage("Filled image", filteredEntropyMat);
    
    //Mask image
    Mat output;
    maskImage(input, filteredEntropyMat, output);
    displayImage("Output", output);
    
    return 0;
}
