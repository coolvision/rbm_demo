#include "testApp.h"
#include <iostream>
#include "ofMain.h"

//--------------------------------------------------------------
void testApp::draw() {

    // draw dataset images
    for (int i = 0; i < images.size(); i++) {
        images[i]->draw(ofGetWindowWidth() - 280 - 10 +
                        28 * (i % 10), 10+ 28 * (i / 10));
    }

    rbm->update();
    rbm->makeImages();

    int img_size = rbm->image_side * 4;

    rbm->v_image->draw(10, 10, img_size, img_size);
    rbm->v_prob_image->draw(10, img_size + 20, img_size, img_size);

    rbm->h_image->draw(img_size + 20, 10, img_size, img_size);
    rbm->h_prob_image->draw(img_size + 20, img_size + 20, img_size, img_size);

    int fiter_size = 28;
    for (int i = 0; i < rbm->filters.size(); i++) {
        rbm->filters[i]->draw(10 + fiter_size * (i / 10),
                              img_size * 2 + 30 + fiter_size * (i % 10));
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

    rbm = new RBM(28, 100);
    rbm->randomInit();

    ofSeedRandom();

    // make an images array

    string path = ofToDataPath("train-images-idx3-ubyte", true);
    ifstream file(path.c_str(), ios::binary);

    if (file.is_open()) {

        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*) &number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);

        ofImage img;

        for (int i = 0; i < 200; i++) {

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
                    //unsigned char tmp;
                    file.read((char*) px, 1);
                    px++;
                }
            }

            images.back()->update();
        }
    }

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
