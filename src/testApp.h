#pragma once

#include "ofMain.h"

#include "RBM.h"

class testApp: public ofBaseApp {

    vector<ofImage *> images;

    // MNIST dataset info
    ifstream data_file;
    ifstream labels_file;

    float *training_data;
    float *training_labels;

    int magic_number;
    int number_of_images;
    int n_rows;
    int n_cols;
    int n_images_read;

    RBM *rbm;

    bool readBatch(int n);

public:
    void setup();
    void update();
    void draw();

    void keyPressed(int key);
    void keyReleased(int key);
    void mouseMoved(int x, int y);
    void mouseDragged(int x, int y, int button);
    void mousePressed(int x, int y, int button);
    void mouseReleased(int x, int y, int button);
    void windowResized(int w, int h);
    void dragEvent(ofDragInfo dragInfo);
    void gotMessage(ofMessage msg);
};
