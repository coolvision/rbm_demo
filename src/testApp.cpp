#include <iostream>
#include "ofMain.h"

#include "testApp.h"

bool continuous_update = false;
bool update_step = false;
bool next_img = false;

//--------------------------------------------------------------
void testApp::draw() {

    if (update_step || continuous_update) {
        update_step = false;

        rbm->updateMiniBatch();
        //rbm->update();

        rbm->makeImages();
    }

    float off_x = 350;
    ofDrawBitmapStringHighlight("sample: " + ofToString(rbm->sample_i), off_x,
            25);
    ofDrawBitmapStringHighlight("batch: " + ofToString(rbm->batch_i), off_x, 40);
    ofDrawBitmapStringHighlight("epoch: " + ofToString(rbm->epoch_i), off_x, 55);
    ofDrawBitmapStringHighlight("n: " + ofToString(rbm->n_training_samples),
            off_x, 70);

    // draw dataset images
    int image_i = 0;
    for (int i = 0; i < images.size(); i++) {
        images[i]->draw(ofGetWindowWidth() - 280 - 10 + 28 * (image_i % 10),
                520 + 28 * (image_i / 10));
        ofDrawBitmapStringHighlight(ofToString(training_labels[i]),
                ofGetWindowWidth() - 280 - 10 + 28 * (image_i % 10),
                520 + 28 * (image_i / 10));
        image_i++;
    }

    int img_size = 80;
    for (int i = 0; i < rbm->images_n && i < rbm->batch_size; i++) {
        rbm->v_data_images[i]->update();
        rbm->h_data_images[i]->update();
        rbm->v_prob_images[i]->update();
        rbm->v_images[i]->update();

        rbm->v_data_images[i]->draw(10, 10 + img_size * i, img_size, img_size);
        rbm->h_data_images[i]->draw(10 + img_size, 10 + img_size * i, img_size, img_size);
        rbm->v_prob_images[i]->draw(10 + img_size * 2, 10 + img_size * i, img_size, img_size);
        rbm->v_images[i]->draw(10 + img_size * 3, 10 + img_size * i, img_size, img_size);
    }

    //rbm->v_data_image->draw(10, 10, img_size, img_size);

//    rbm->h_data_prob_image->draw(10, 10 + (img_size + 10) * 1, img_size,
//            img_size);
//    rbm->h_data_image->draw(20 + img_size, 10 + (img_size + 10) * 1, img_size,
//            img_size);
//
//    rbm->v_prob_image->draw(10, 10 + (img_size + 10) * 2, img_size, img_size);
//    rbm->v_image->draw(20 + img_size, 10 + (img_size + 10) * 2, img_size,
//            img_size);
//
//    rbm->h_prob_image->draw(10, 10 + (img_size + 10) * 3, img_size, img_size);
//    rbm->h_image->draw(20 + img_size, 10 + (img_size + 10) * 3, img_size,
//            img_size);



    int fiter_size = 45;
    int side = rbm->h_image_side;
    for (int i = 0; i < rbm->filters.size(); i++) {
        rbm->filters[i]->draw(300 + img_size * 2 + fiter_size * (i / side),
                10 + fiter_size * (i % side), fiter_size, fiter_size);
    }

    fiter_size = 80;
    for (int i = 0; i < rbm->filters.size(); i++) {
        ofDrawBitmapStringHighlight(ofToString(rbm->W[i]),
                610 + img_size * 2 + fiter_size * (i / side),
                700 + 50 * (i % side));
        ofDrawBitmapStringHighlight(ofToString(rbm->mean_activity[i]),
                610 + img_size * 2 + fiter_size * (i / side),
                700 + 50 * (i % side) + 15, ofColor::red);
    }

    img_size = 150;
    rbm->v_bias->draw(1200, 10, img_size, img_size);
    rbm->h_bias->draw(1200, 10 + img_size, img_size, img_size);
}

//--------------------------------------------------------------
int reverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
}

//--------------------------------------------------------------
void testApp::setup() {

    n_images_read = 0;

    ofSetFrameRate(60);

    ofSeedRandom();

    string path = ofToDataPath("train-images-idx3-ubyte", true);
    data_file.open(path.c_str(), ios::binary);
    path = ofToDataPath("train-labels-idx1-ubyte", true);
    labels_file.open(path.c_str(), ios::binary);

    int n_vis_images = 10;

    if (!data_file.is_open() || !labels_file.is_open()) {
        exit();
    }

    data_file.read((char*) &magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    data_file.read((char*) &number_of_images, sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);
    data_file.read((char*) &n_rows, sizeof(n_rows));
    n_rows = reverseInt(n_rows);
    data_file.read((char*) &n_cols, sizeof(n_cols));
    n_cols = reverseInt(n_cols);


    training_data = new float[number_of_images * n_cols * n_rows];
    float *dr = training_data;

    int images_n = 10000;

    // read all of the data
    for (int i = 0; i < images_n; i++) {

        // add an image
        unsigned char *px;
        if (i < n_vis_images) {
            images.push_back(new ofImage());
            n_images_read++;
            images.back()->allocate(28, 28, OF_IMAGE_GRAYSCALE);
            px = images.back()->getPixels();
        }

        if (i % 1000 == 0) {
            cout << "images: " << i << endl;
        }

        // fill with bytes
        for (int r = 0; r < n_rows; r++) {
            for (int c = 0; c < n_cols; c++) {
                uint8_t tmp;
                data_file.read((char *) &tmp, 1);

                *dr = (float) tmp;
                dr++;

                if (i < n_vis_images) {
                    *px = tmp;
                    px++;
                }
            }
        }

        if (i < n_vis_images) {
            images.back()->update();
        }
    }

    labels_file.read((char*) &magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    labels_file.read((char*) &number_of_images, sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);

    training_labels = new float[number_of_images];
    float *lr = training_labels;

    for (int i = 0; i < number_of_images; i++) {
        uint8_t tmp;
        labels_file.read((char *) &tmp, 1);
        *lr = (float) tmp;
        lr++;
    }

    rbm = new RBM();
    rbm->init(28, 15, training_data, training_labels, images_n, 10);
}

bool testApp::readBatch(int n) {

    if (!data_file.is_open()) {
        return false;
    }

    return true;
}

//--------------------------------------------------------------
void testApp::update() {

}

//--------------------------------------------------------------
void testApp::keyPressed(int key) {

    if (key == 'u') {
        update_step = true;
    }

    if (key == 'n') {
        next_img = true;
    }

    if (key == 'c') {
        continuous_update = !continuous_update;
    }
}

//--------------------------------------------------------------
void testApp::keyReleased(int key) {

}

//--------------------------------------------------------------
void testApp::mouseMoved(int x, int y) {

}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button) {

}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button) {

}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button) {

}

//--------------------------------------------------------------
void testApp::windowResized(int w, int h) {

}

//--------------------------------------------------------------
void testApp::gotMessage(ofMessage msg) {

}

//--------------------------------------------------------------
void testApp::dragEvent(ofDragInfo dragInfo) {

}
