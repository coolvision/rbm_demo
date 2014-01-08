#include "testApp.h"
#include <iostream>
#include "ofMain.h"

//--------------------------------------------------------------
void testApp::draw() {

    // draw dataset images
    int image_i = 0;
    for (deque<ofImage *>::iterator i = images.begin(); i < images.end(); i++) {
        if (image_i > 100) {
            break;
        }
        (*i)->draw(ofGetWindowWidth() - 280 - 10 +
                        28 * (image_i % 10), 10 + 28 * (image_i / 10));
        image_i++;
    }

    // get the next image
//    if (images.empty()) {
//        readBatch(100);
//        return;
//    }


    // remove this image from the storage
//    img->clear();
//    delete img;

    rbm->update();
    rbm->makeImages();

    int img_size = rbm->image_side * 3;

    rbm->v_data_image->draw(10, 10, img_size, img_size);

    rbm->h_prob_image->draw(10, 10 + (img_size + 10) * 1, img_size , img_size);
    rbm->h_image->draw(20 + img_size, 10 + (img_size + 10) * 1, img_size, img_size);

    rbm->v_n_prob_image->draw(10, 10 + (img_size + 10) * 2, img_size, img_size);
    rbm->v_n_image->draw(20 + img_size, 10 + (img_size + 10) * 2, img_size, img_size);

    rbm->h_n_prob_image->draw(10, 10 + (img_size + 10) * 3, img_size, img_size);
    rbm->h_n_image->draw(20 + img_size, 10 + (img_size + 10) * 3, img_size, img_size);

    int fiter_size = 28;
    for (int i = 0; i < rbm->filters.size(); i++) {
        rbm->filters[i]->draw(30 + img_size * 2 + fiter_size * (i / 10),
                                 10 + fiter_size * (i % 10));
    }
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

    ofSetFrameRate(60);

    rbm = new RBM(28, 100);
    rbm->randomInit();

    ofSeedRandom();

    // make an images array

    string path = ofToDataPath("train-images-idx3-ubyte", true);
    data_file.open(path.c_str(), ios::binary);

    if (data_file.is_open()) {
        data_file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        data_file.read((char*) &number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        data_file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        data_file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);
    }

    readBatch(100);

    images.pop_back();
    images.pop_back();
    ofImage *img = images.back();

    // put the image into the visible nodes
    unsigned char *px = img->getPixels();
    for (int i = 0; i < rbm->n_visible; i++) {
        rbm->v_data[i] = (float) (px[i] > 128);
    }
}

bool testApp::readBatch(int n) {

    if (!data_file.is_open()) {
        return false;
    }

    for (int i = 0; i < n; i++) {

        // add an image
        images.push_back(new ofImage());
        images.back()->allocate(28, 28, OF_IMAGE_GRAYSCALE);
        unsigned char *px = images.back()->getPixels();

        if (i % 100 == 0) {
            cout << "images: " << i << endl;
        }

        // fill with bytes
        for (int r = 0; r < n_rows; r++) {
            for (int c = 0; c < n_cols; c++) {
                if (data_file.eof()) {
                    return false;
                }
                data_file.read((char*) px, 1);
                px++;
            }
        }

        images.back()->update();
    }

    return true;
}

//--------------------------------------------------------------
void testApp::update() {

}

//--------------------------------------------------------------
void testApp::keyPressed(int key) {

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
