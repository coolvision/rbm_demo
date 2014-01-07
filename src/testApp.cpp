#include "testApp.h"
#include <iostream>
#include "ofMain.h"

float sigmoid(float x) { return (1.0 / (1.0 + exp(-x))); }

RBM::RBM(int image_side, int n_hidden) {

    this->n_visible = image_side * image_side;
    this->n_hidden = n_hidden;

    // allocate memory for units and weights
    v = new float[n_visible];
    h = new float[n_hidden];

    v_prob = new float[n_visible];
    h_prob = new float[n_hidden];

    b = new float[n_visible];
    c = new float[n_hidden];

    W = new float[n_visible * n_hidden];

    filters.resize(n_hidden);
    for (int i = 0; i < n_hidden; i++) {
        filters[i] = new ofImage();
        filters.back()->allocate(image_side, image_side, OF_IMAGE_GRAYSCALE);
    }

    v_image = new ofImage();
    v_image->allocate(image_side, image_side, OF_IMAGE_GRAYSCALE);

    h_image = new ofImage();
    h_image->allocate(image_side, image_side, OF_IMAGE_GRAYSCALE);

    v_prob_image = new ofImage();
    v_prob_image->allocate(image_side, image_side, OF_IMAGE_GRAYSCALE);

    h_prob_image = new ofImage();
    h_prob_image->allocate(image_side, image_side, OF_IMAGE_GRAYSCALE);
}

RBM::~RBM() {

    delete[] v;
    delete[] h;

    delete[] b;
    delete[] c;

    delete[] W;
}

void RBM::randomInit() {

    // set biases to 0
    // set units activations randomly
    memset(b, 0, n_visible * sizeof(float));
    memset(c, 0, n_hidden * sizeof(float));

    for (int i = 0; i < n_visible * n_hidden; i++) {
        W[i] = ofRandom(1.0f);
    }

    for (int i = 0; i < n_visible; i++) {
        v_prob[i] = ofRandom(1.0f);
        v[i] = ofRandom(1.0f) > v_prob[i] ? 1.0f : 0.0f;
    }
    for (int i = 0; i < n_hidden; i++) {
        h_prob[i] = ofRandom(1.0f);
        h[i] = ofRandom(1.0f) > h_prob[i] ? 1.0f : 0.0f;
    }
}

void RBM::makeImages() {

    // put units values and weights into images

    // filters from weights
    for (int i = 0; i < n_hidden; i++) {

        for (int k = 0; k < image_side; k++) {
            for (int l = 0; l < image_side; l++) {
                // pixel index
                int p_i = k * image_side + l;
                // weight index
                int w_i = i * n_visible + p_i;
                filters[i]->setColor(k, l, ofColor(W[w_i] * 250.f));
            }
        }
    }

    // activation probabilities and sampled activations
    for (int k = 0; k < image_side; k++) {
        for (int l = 0; l < image_side; l++) {
            // pixel index
            int p_i = k * image_side + l;
            v_prob_image->setColor(k, l, ofColor(v_prob[p_i] * 250.f));
            h_prob_image->setColor(k, l, ofColor(h_prob[p_i] * 250.f));
            v_image->setColor(k, l, ofColor(v[p_i] * 250.f));
            h_image->setColor(k, l, ofColor(h[p_i] * 250.f));
        }
    }
}

void RBM::gibbsStep(int n) {

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

        for (int i = 0; i < 1000; i++) {

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
void testApp::draw() {

    // draw some images
    for (int i = 0; i < images.size(); i++) {
        images[i]->draw(28 * (i / 30), 28 * (i % 30));
    }


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

