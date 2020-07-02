#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#include <vector>
#include <sys/ioctl.h>
#include <dirent.h>
#include "Net.hpp"
#include "util.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"

/******************************************************************************
* function : show usage
******************************************************************************/

void getAllFiles(const char *path, std::vector<std::string> &files);

void Mat2bgr(cv::Mat im, unsigned char *data);

void Usage(char *pchPrgName)
{
    printf("Usage : %s <index> \n", pchPrgName);
    printf("index:\n");
    printf("\t 1) Yolov3(Read File).\n");
    printf("\t 2) Yolov2(Read File).\n");
}


void yolov2DetectDemo(const char *model_path, const char *image_path)
{
    nnie::Net yolov2;

    yolov2.load_model(model_path);

    printf("yolov2 start\n");

    struct timeval tv1;
    struct timeval tv2;
    long t1, t2, time;
    unsigned char *data = (unsigned char *) malloc(sizeof(unsigned char) * 416 * 416 * 3);
    cv::Mat img = cv::imread(image_path);

    cv::resize(img, img, cv::Size(416, 416));

    int step = img.step;
    int h = img.rows;
    int w = img.cols;
    int c = img.channels();

    printf("h=%d\n", h);
    printf("w=%d\n", w);
    printf("c=%d\n", c);

    unsigned char *data1 = (unsigned char *) img.data;
    int count = 0;
    for (int k = 0; k < c; k++)
    {
        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                data[count++] = data1[i * step + j * c + k];
            }
        }
    }

    gettimeofday(&tv1, NULL);

    yolov2.run(data);

    gettimeofday(&tv2, NULL);
    t1 = tv2.tv_sec - tv1.tv_sec;
    t2 = tv2.tv_usec - tv1.tv_usec;
    time = (long) (t1 * 1000 + t2 / 1000);
    printf("yolov2 NNIE inference time : %dms\n", time);

    gettimeofday(&tv1, NULL);

    nnie::Mat output0 = yolov2.getOutputTensor(0);

    printf("%d\n", output0.width);
    printf("%d\n", output0.height);
    printf("%d\n", output0.channel);

    /*yolov2的参数*/
    int img_width = 416;
    int img_height = 416;
    int num_classes = 6;
    int kBoxPerCell = 2;

    int feature_index0 = 0;
    int feature_index1 = 1;
    int feature_index2 = 2;

    float conf_threshold = 0.2;
    float nms_threshold = 0.2;
    int is_nms = 1;

    std::vector<int> ids;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;

    const std::vector<std::vector<cv::Size2f>> anchors = {
            {{3.638, 5.409}, {3.281, 4.764}}};

    parseYolov2Feature(img_width,
                       img_height,
                       num_classes,
                       kBoxPerCell,
                       feature_index0,
                       conf_threshold,
                       anchors[0],
                       output0,
                       ids,
                       boxes,
                       confidences);

    std::vector<int> indices;

    char *cls_names[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                         "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                         "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U",
                         "V", "W", "X", "Y", "Z"};

    std::vector<nnie::objInfo> detection_results;

    if (is_nms)
    {
        cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
    } else
    {
        for (int i = 0; i < boxes.size(); ++i)
        {
            indices.push_back(i);
        }
    }

    printf("\n%d\n", indices.size());
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];

        // remap box in src input size.
        auto remap_box = RemapBoxOnSrc(cv::Rect2d(box), img_width, img_height);
        nnie::objInfo object_detection;
        object_detection.box = remap_box;
        object_detection.cls_id = ids[idx];
        object_detection.confidence = confidences[idx];
        detection_results.push_back(std::move(object_detection));

        float xmin = object_detection.box.x;
        float ymin = object_detection.box.y;
        float xmax = object_detection.box.x + object_detection.box.width;
        float ymax = object_detection.box.y + object_detection.box.height;
        float confidence = object_detection.confidence;
        int cls_id = object_detection.cls_id;
        char *cls_name = cls_names[cls_id];
        printf("%d %s %.3f %.3f %.3f %.3f %.3f\n", cls_id, cls_name, confidence, xmin, ymin, xmax, ymax);
    }

    gettimeofday(&tv2, NULL);
    t1 = tv2.tv_sec - tv1.tv_sec;
    t2 = tv2.tv_usec - tv1.tv_usec;
    time = (long) (t1 * 1000 + t2 / 1000);
    printf("yolov2 postProcess time: %dms\n", time);
    yolov2.finish();
    free(data);
    printf("yolov2 finish\n");
}

void yolov3DetectDemo(const char *model_path, const char *image_path)
{
    nnie::Net yolov3_mnas;

    yolov3_mnas.load_model(model_path);

    printf("yolov3 start\n");

    struct timeval tv1;
    struct timeval tv2;
    long t1, t2, time;

    gettimeofday(&tv1, NULL);
    yolov3_mnas.run(image_path);

    gettimeofday(&tv2, NULL);
    t1 = tv2.tv_sec - tv1.tv_sec;
    t2 = tv2.tv_usec - tv1.tv_usec;
    time = (long) (t1 * 1000 + t2 / 1000);
    printf("yolov3 NNIE inference time : %dms\n", time);

    gettimeofday(&tv1, NULL);

    nnie::Mat output0 = yolov3_mnas.getOutputTensor(0);
    nnie::Mat output1 = yolov3_mnas.getOutputTensor(1);
    nnie::Mat output2 = yolov3_mnas.getOutputTensor(2);

    /*yolov3的参数*/
    int img_width = 416;
    int img_height = 416;
    int num_classes = 80;
    int kBoxPerCell = 3;

    int feature_index0 = 0;
    int feature_index1 = 1;
    int feature_index2 = 2;

    float conf_threshold = 0.2;
    float nms_threshold = 0.5;
    int is_nms = 1;

    std::vector<int> ids;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;

    const std::vector<std::vector<cv::Size2f>> anchors = {
            {{116, 90}, {156, 198}, {373, 326}},
            {{30, 61}, {62, 45}, {59, 119}},
            {{10, 13}, {16, 30}, {33, 23}}};

    parseYolov3Feature(img_width,
                       img_height,
                       num_classes,
                       kBoxPerCell,
                       feature_index0,
                       conf_threshold,
                       anchors[0],
                       output0,
                       ids,
                       boxes,
                       confidences);

    parseYolov3Feature(img_width,
                       img_height,
                       num_classes,
                       kBoxPerCell,
                       feature_index1,
                       conf_threshold,
                       anchors[1],
                       output1,
                       ids,
                       boxes,
                       confidences);

    parseYolov3Feature(img_width,
                       img_height,
                       num_classes,
                       kBoxPerCell,
                       feature_index2,
                       conf_threshold,
                       anchors[2],
                       output2,
                       ids,
                       boxes,
                       confidences);

    std::vector<int> indices;
    /*print result, this sample has 81 classes:
      class 0:background      class 1:person       class 2:bicycle         class 3:car            class 4:motorbike      class 5:aeroplane
      class 6:bus             class 7:train        class 8:truck           class 9:boat           class 10:traffic light
      class 11:fire hydrant   class 12:stop sign   class 13:parking meter  class 14:bench         class 15:bird
      class 16:cat            class 17:dog         class 18:horse          class 19:sheep         class 20:cow
      class 21:elephant       class 22:bear        class 23:zebra          class 24:giraffe       class 25:backpack
      class 26:umbrella       class 27:handbag     class 28:tie            class 29:suitcase      class 30:frisbee
      class 31:skis           class 32:snowboard   class 33:sports ball    class 34:kite          class 35:baseball bat
      class 36:baseball glove class 37:skateboard  class 38:surfboard      class 39:tennis racket class 40bottle
      class 41:wine glass     class 42:cup         class 43:fork           class 44:knife         class 45:spoon
      class 46:bowl           class 47:banana      class 48:apple          class 49:sandwich      class 50orange
      class 51:broccoli       class 52:carrot      class 53:hot dog        class 54:pizza         class 55:donut
      class 56:cake           class 57:chair       class 58:sofa           class 59:pottedplant   class 60bed
      class 61:diningtable    class 62:toilet      class 63:vmonitor       class 64:laptop        class 65:mouse
      class 66:remote         class 67:keyboard    class 68:cell phone     class 69:microwave     class 70:oven
      class 71:toaster        class 72:sink        class 73:refrigerator   class 74:book          class 75:clock
      class 76:vase           class 77:scissors    class 78:teddy bear     class 79:hair drier    class 80:toothbrush*/
    char *cls_names[] = {"background", "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
                         "boat",

                         "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                         "horse", "sheep",

                         "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
                         "suitcase",

                         "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                         "skateboard", "surfboard", "tennis racket",

                         "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "donut", "apple", "sandwich",

                         "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
                         "pottedplant",

                         "bed", "diningtable", "toilet", "vmonitor", "laptop", "mouse", "remote", "keyboard",
                         "cell phone", "microwave",

                         "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                         "hair drier", "toothbrush"};
    std::vector<nnie::objInfo> detection_results;

    if (is_nms)
    {
        cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
    } else
    {
        for (int i = 0; i < boxes.size(); ++i)
        {
            indices.push_back(i);
        }
    }

    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];

        // remap box in src input size.
        auto remap_box = RemapBoxOnSrc(cv::Rect2d(box), img_width, img_height);
        nnie::objInfo object_detection;
        object_detection.box = remap_box;
        object_detection.cls_id = ids[idx] + 1;
        object_detection.confidence = confidences[idx];
        detection_results.push_back(std::move(object_detection));

        float xmin = object_detection.box.x;
        float ymin = object_detection.box.y;
        float xmax = object_detection.box.x + object_detection.box.width;
        float ymax = object_detection.box.y + object_detection.box.height;
        float confidence = object_detection.confidence;
        int cls_id = object_detection.cls_id;
        char *cls_name = cls_names[cls_id];
        printf("%d %s %.3f %.3f %.3f %.3f %.3f\n", cls_id, cls_name, confidence, xmin, ymin, xmax, ymax);
    }

    gettimeofday(&tv2, NULL);
    t1 = tv2.tv_sec - tv1.tv_sec;
    t2 = tv2.tv_usec - tv1.tv_usec;
    time = (long) (t1 * 1000 + t2 / 1000);
    printf("yolov3 postProcess : %dms\n", time);

    printf("yolov3 finish\n");
}

/******************************************************************************
* function 
******************************************************************************/
int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        Usage(argv[0]);
        return -1;
    }
    switch (*argv[1])
    {
        case '1':
        {
            const char *image_path = "./data/nnie_image/detection/dog_bike_car.bgr";
            const char *model_path = "./data/nnie_model/detection/inst_yolov3_cycle.wk";
            yolov3DetectDemo(model_path, image_path);
        }
            break;

        case '2':
        {

            const char *image_path = "./data/nnie_image/detection/street_cars.png";
            cv::Mat im = cv::imread(image_path, cv::IMREAD_COLOR);
            const char *model_path = "./data/nnie_model/detection/inst_yolov2_cycle.wk";
            yolov2DetectDemo(model_path, image_path);

        }
            break;
        default:
        {
            printf("--------------------\n");
//            Usage(argv[0]);
        }
            break;
    }

    return 1;
}


void getAllFiles(const char *path, std::vector<std::string> &files)
{
    DIR *dir;
    struct dirent *ptr;
    if ((dir = opendir(path)) == NULL)
    {
        perror("Open dri error...");
        exit(1);
    }
    while ((ptr = readdir(dir)) != NULL)
    {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)
            continue;
        else if (ptr->d_type == 8) //file
        {
            files.push_back(ptr->d_name);
        }
    }
    sort(files.begin(), files.end());
    closedir(dir);
}

void Mat2bgr(cv::Mat im, unsigned char *data)
{
    int h = im.rows;
    int w = im.cols;
    int c = im.channels();
    int step = im.step;
    unsigned char *data1 = (unsigned char *) im.data;
    int count = 0;
    for (int k = 0; k < c; k++)
    {
        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                data[count++] = data1[i * step + j * c + k];
            }
        }
    }
}