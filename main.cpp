#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

int main()
{
    std::cout << "\n\nDetective BOO is suiting up...\n\n";

    const std::string YOLO_CONFIG_PATH = "assets/yolov3.cfg";
    const std::string YOLO_WEIGHT_PATH = "assets/yolov3.weights";
    const std::string IMAGE_PATH       = "assets/meme.png";
    const std::string LABEL_PATH       = "assets/coco.names";

    // get the corresponding labels of detection class ids
    std::string LABELS[80];
    std::ifstream read(LABEL_PATH);

    short i = 0;
    std::string l;
    while (getline(read, l)) { 
        LABELS[i] = l;
        i++; 
    }

    read.close();


    // setup darknet neural network to use the YOLOv3 weight and config
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(YOLO_CONFIG_PATH, YOLO_WEIGHT_PATH);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);


    // setup the image and convert it into a blob format for the neural network to i/o
    cv::Mat image = cv::imread(IMAGE_PATH);

    if (image.empty())
    {
        std::cout << "Error: Could not open image file." << std::endl;
        return 1;
    }

    const cv::Mat INPUT_BLOB = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(320, 320), cv::Scalar(), true, false);

    // input the blob to darknet
    net.setInput(INPUT_BLOB); 


    // unconnected layers are the three output layers that have no subsequent layers
    std::vector<cv::String> OUTPUT_LAYERS = net.getUnconnectedOutLayersNames();
    
    // do a forward pass to our blob input into the neural network
    std::vector<cv::Mat> OUTPUT_BLOBS;
    net.forward(OUTPUT_BLOBS, OUTPUT_LAYERS);

    const float CONF_THRESHOLD = 0.5f;
    const float NMS_THRESHOLD = 0.3f;

    std::vector<cv::Rect> coordinates;
    std::vector<float>    confidence;
    std::vector<int>      classification;

    // three output layers with varied image scale
    for (cv::Mat &blob : OUTPUT_BLOBS) {
        
        // iterate through each layer detection info
        for (int rx = 0; rx < blob.rows; rx++) {

            /*  
                
                Note: these are percentage

                0: x coordinate 
                1: y coordinate
                2: width 
                3: height 
                4: object probability
                ...
                : class probability

            */

            // skip the bounding box coordinate data from the first five elements
            // find the class id index and confidence score
            int   idx   = 0;
            float conf  = 0;

            for (int cx = 5; cx < blob.cols; cx++) {
                float col_val = blob.row(rx).at<float>(cx);
                
                if (col_val > conf) {
                    conf = col_val;
                    idx  = cx-5; 
                }
            }

            if (conf >= CONF_THRESHOLD) {
                // extract the bounding box detection data from the first five elements
                const float bx = blob.row(rx).at<float>(0);
                const float by = blob.row(rx).at<float>(1);
                const float bw = blob.row(rx).at<float>(2);
                const float bh = blob.row(rx).at<float>(3);

                const int w = static_cast<int>(bw * image.cols);
                const int h = static_cast<int>(bh * image.cols);
                const int x = static_cast<int>((bx * image.cols) - w / 2);
                const int y = static_cast<int>((by * image.cols) - h / 2); 

                confidence.push_back(conf);
                coordinates.push_back(cv::Rect(x, y, w, h));
                classification.push_back(idx);
            }   
        }
    }

    std::cout << "\n\nDetective BOO is done investigating...\n\n";

    // non-maximum supressions removes duplicate boxes on the same object
    std::vector<int> indices;
    cv::dnn::NMSBoxes(coordinates, confidence, CONF_THRESHOLD, NMS_THRESHOLD, indices);


    // draw a square around the detection coordinates
    for (int &i : indices) {
        const int w = coordinates[i].width;
        const int h = coordinates[i].height;
        
        const int tx = coordinates[i].x;
        const int ty = coordinates[i].y;
        const int bx = tx + w;
        const int by = tx + h;

        const std::string label = LABELS[classification[i]];
        
        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << confidence[i];
        const std::string conf = stream.str();

        cv::putText(image, label, cv::Point(tx, ty-25-10),   cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(250, 0, 250), 1);
        cv::putText(image, conf,  cv::Point(tx+w-35, ty-25-10), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(250, 0, 250), 1);
        
        cv::rectangle(image, cv::Point(tx, ty-25), cv::Point(bx, by+50), cv::Scalar(250, 0, 250), 2);
    }
    
    cv::imshow("Image", image); 
    cv::waitKey(0); 

    return 0;
}