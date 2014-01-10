#include "testApp.h"
#include <iostream>
#include "ofMain.h"

bool continuous_update = false;
bool update_step = false;
bool next_img = false;

//--------------------------------------------------------------
void testApp::draw() {

    ofSetColor(ofColor::black);
    ofDrawBitmapString(ofToString(n_images_read), ofGetWindowWidth() - 50, 320);
    ofSetColor(ofColor::white);

    // draw dataset images
    int image_i = 0;
    for (deque<ofImage *>::iterator i = images.begin(); i < images.end(); i++) {
        if (image_i > 100) {
            break;
        }
        (*i)->draw(ofGetWindowWidth() - 280 - 10 + 28 * (image_i % 10),
                10 + 28 * (image_i / 10));
        image_i++;
    }

    int img_size = 90;

    rbm->v_data_image->draw(10, 10, img_size, img_size);

    rbm->h_data_prob_image->draw(10, 10 + (img_size + 10) * 1, img_size,
            img_size);
    rbm->h_data_image->draw(20 + img_size, 10 + (img_size + 10) * 1, img_size,
            img_size);

    rbm->v_prob_image->draw(10, 10 + (img_size + 10) * 2, img_size, img_size);
    rbm->v_image->draw(20 + img_size, 10 + (img_size + 10) * 2, img_size,
            img_size);

    rbm->h_prob_image->draw(10, 10 + (img_size + 10) * 3, img_size, img_size);
    rbm->h_image->draw(20 + img_size, 10 + (img_size + 10) * 3, img_size,
            img_size);

    int fiter_size = 56;
    int side = 10;
    for (int i = 0; i < rbm->filters.size(); i++) {
        rbm->filters[i]->draw(30 + img_size * 2 + fiter_size * (i / side),
                10 + fiter_size * (i % side), fiter_size, fiter_size);
    }

    rbm->v_bias->draw(30 + img_size * 2, 590, img_size, img_size);
    rbm->h_bias->draw(30 + img_size * 2, 690, img_size, img_size);

    // get the next image
    if (images.empty()) {
        readBatch(100);
        return;
    }

    if (update_step || continuous_update) {

        update_step = false;

//        for (int i = 0; i < rbm->n_visible; i++) {
//            rbm->v_data[i] = 0;
//        }
//
//        int d = rbm->image_side;
//
//
//        if (ofRandom(1.0f) > 0.5f) {
//            for (int i = 0; i < d/2; i++) {
//                for (int j = 0; j < d/2; j++) {
//                    rbm->v_data[i * d + j] = 1.0f;
//                }
//            }
//        }
//        if (ofRandom(1.0f) > 0.5f) {
//            for (int i = d/2; i < d; i++) {
//                for (int j = d/2; j < d; j++) {
//                    rbm->v_data[i * d + j] = 1.0f;
//                }
//            }
//        }
//        if (ofRandom(1.0f) > 0.5f) {
//            for (int i = d/2; i < d; i++) {
//                for (int j = 0; j < d/2; j++) {
//                    rbm->v_data[i * d + j] = 1.0f;
//                }
//            }
//        }
//        if (ofRandom(1.0f) > 0.5f) {
//            for (int i = 0; i < d/2; i++) {
//                for (int j = d/2; j < d; j++) {
//                    rbm->v_data[i * d + j] = 1.0f;
//                }
//            }
//        }
//
//
//        if (ofRandom(1.0f) > 0.5f) {
//            for (int i = 0; i < d/2; i++) {
//                for (int j = d/3; j < 2*d/3; j++) {
//                    rbm->v_data[i * d + j] = 1.0f;
//                }
//            }
//        }
//        if (ofRandom(1.0f) > 0.5f) {
//            for (int i = d/2; i < d; i++) {
//                for (int j = d/3; j < 2*d/3; j++) {
//                    rbm->v_data[i * d + j] = 1.0f;
//                }
//            }
//        }
//        if (ofRandom(1.0f) > 0.5f) {
//            for (int i = d/3; i < 2*d/3; i++) {
//                for (int j = 0; j < d/2; j++) {
//                    rbm->v_data[i * d + j] = 1.0f;
//                }
//            }
//        }
//        if (ofRandom(1.0f) > 0.5f) {
//            for (int i = d/3; i < 2*d/3; i++) {
//                for (int j = d/2; j < d; j++) {
//                    rbm->v_data[i * d + j] = 1.0f;
//                }
//            }
//        }

        ofImage *img = images.back();
        images.pop_back();
        // put the image into the visible nodes
        unsigned char *px = img->getPixels();
        for (int i = 0; i < rbm->n_visible; i++) {
            rbm->v_data[i] = (float) (px[i] > 128);
        }
        img->clear();
        delete img;

        rbm->update();

        rbm->makeImages();

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

    n_images_read = 0;

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

    ofImage *img = images.back();
    images.pop_back();

    // put the image into the visible nodes
    unsigned char *px = img->getPixels();
    for (int i = 0; i < rbm->n_visible; i++) {
        rbm->v_data[i] = (float) (px[i] > 128);
    }

    img->clear();
    delete img;

}

bool testApp::readBatch(int n) {

    if (!data_file.is_open()) {
        return false;
    }

    for (int i = 0; i < n; i++) {

        // add an image
        images.push_back(new ofImage());
        n_images_read++;
        images.back()->allocate(28, 28, OF_IMAGE_GRAYSCALE);
        unsigned char *px = images.back()->getPixels();

        if (i % 100 == 0) {
            cout << "images: " << i << endl;
        }

        // fill with bytes
        for (int r = 0; r < n_rows; r++) {
            for (int c = 0; c < n_cols; c++) {

                if (data_file.eof()) {

                    data_file.close();
                    string path = ofToDataPath("train-images-idx3-ubyte", true);

                    data_file.open(path.c_str(), ios::binary);
                    data_file.read((char*) &magic_number, sizeof(magic_number));
                    magic_number = reverseInt(magic_number);
                    data_file.read((char*) &number_of_images,
                            sizeof(number_of_images));
                    number_of_images = reverseInt(number_of_images);
                    data_file.read((char*) &n_rows, sizeof(n_rows));
                    n_rows = reverseInt(n_rows);
                    data_file.read((char*) &n_cols, sizeof(n_cols));
                    n_cols = reverseInt(n_cols);

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
