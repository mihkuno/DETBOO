#include <iostream>
#include <string>
#include <fstream>  
#include <opencv2/opencv.hpp>

int main()
{


    std::cout << "\n\nDetective BOO is suiting up...\n\n";

    const std::string YOLO_CONFIG_PATH = "assets/yolov3.cfg";
    const std::string YOLO_WEIGHT_PATH = "assets/yolov3.weights";
    const std::string LABEL_PATH       = "assets/coco.names";
    const std::string INPUT_PATH       = "assets/input.mp4";
    const std::string OUTPUT_PATH      = "assets/output.avi";


    // get the corresponding labels of detection class ids
    std::string LABELS[80];
    std::ifstream read(LABEL_PATH);

    short i = 0;
    std::string l;

    while (std::getline(read, l)) { 
        LABELS[i] = l;
        i++; 
    }

    read.close();


    // Open input video file
    cv::VideoCapture cap(INPUT_PATH);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open input video file" << std::endl;
        return -1;
    }
    

    // Get input video properties
    int frame_width  = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    double fps       = cap.get(cv::CAP_PROP_FPS);
    int codec        = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

    // Create output video writer
    cv::VideoWriter writer(OUTPUT_PATH, codec, fps, cv::Size(frame_width, frame_height));

    // Check if output video writer is opened successfully
    if (!writer.isOpened()) {
        std::cerr << "Failed to open output video file" << std::endl;
        return -1;
    }


    // setup darknet neural network to use the YOLOv3 weight and config
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(YOLO_CONFIG_PATH, YOLO_WEIGHT_PATH);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // unconnected layers are the three output layers that have no subsequent layers
    std::vector<cv::String> OUTPUT_LAYERS = net.getUnconnectedOutLayersNames();


    const float CONF_THRESHOLD = 0.5f;
    const float NMS_THRESHOLD = 0.3f;


    // Loop through each frame of the input video and write to the output video file
    int frame_count = 0;
    cv::Mat frame;

    while (cap.read(frame)) {

        // convert frame to blob
        const cv::Mat INPUT_BLOB = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(320, 320), cv::Scalar(), true, false);

        // input the blob to darknet
        net.setInput(INPUT_BLOB); 

        // do a forward pass to our blob input into the neural network
        std::vector<cv::Mat> OUTPUT_BLOBS;
        net.forward(OUTPUT_BLOBS, OUTPUT_LAYERS);


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

                    const int w = static_cast<int>(bw * frame.cols);
                    const int h = static_cast<int>(bh * frame.cols);
                    const int x = static_cast<int>((bx * frame.cols) - w / 2);
                    const int y = static_cast<int>((by * frame.cols) - h / 2); 

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

            cv::putText(frame, label, cv::Point(tx, ty-25-10),   cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(250, 0, 250), 1);
            cv::putText(frame, conf,  cv::Point(tx+w-35, ty-25-10), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(250, 0, 250), 1);
            
            cv::rectangle(frame, cv::Point(tx, ty-25), cv::Point(bx, by+50), cv::Scalar(250, 0, 250), 2);
        }


        // Write frame to output video file
        writer.write(frame);

        // Increment frame count
        frame_count++;

        // Calculate progress as a percentage
        int progress = (int)(((double)frame_count / total_frames) * 100);

        // Print progress to console
        std::cout << "Progress: " << progress << "%" << std::endl;

        cv::imshow("Render", frame);
        cv::waitKey(1);
    }

    // Release input video file and output video writer
    cap.release();
    writer.release();

    return 0;
}
