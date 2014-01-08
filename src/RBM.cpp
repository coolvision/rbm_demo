/*
 * RBM.cpp
 *
 *  Created on: Jan 8, 2014
 *      Author: sk
 */

#include "RBM.h"

float sigmoid(float x) {
    return (1.0 / (1.0 + exp(-x)));
}

void RBM::update() {

    // (positive phase)
    // compute hidden nodes activations and probabilities
    for (int i = 0; i < n_hidden; i++) {
        h_prob[i] = 0.0f;
        for (int j = 0; j < n_visible; j++) {
            h_prob[i] += c[i] + v[j] * W[j * n_hidden + i];
        }
        h_prob[i] = sigmoid(h_prob[i]);
        h[i] = ofRandom(1.0f) > h_prob[i] ? 1.0f : 0.0f;
    }

    // positive phase associations
    int w_i;
    w_i = 0;
    for (int i = 0; i < n_visible; i++) {
        for (int j = 0; j < n_hidden; j++) {
            pos_weights[w_i] = v[i] * h_prob[j];
            w_i++;
        }
    }

    // (negative phase)
    // sample visible nodes
    for (int i = 0; i < n_visible; i++) {
        v_prob[i] = 0.0f;
        for (int j = 0; j < n_hidden; j++) {
            v_prob[i] += b[i] + h[j] * W[i * n_hidden + j];
        }
        v_prob[i] = sigmoid(v_prob[i]);
        v[i] = ofRandom(1.0f) > v_prob[i] ? 1.0f : 0.0f;
    }

    // and hidden nodes once again
    for (int i = 0; i < n_hidden; i++) {
        h_prob[i] = 0.0f;
        for (int j = 0; j < n_visible; j++) {
            h_prob[i] += c[i] + v[j] * W[j * n_hidden + i];
        }
        h_prob[i] = sigmoid(h_prob[i]);
        h[i] = ofRandom(1.0f) > h_prob[i] ? 1.0f : 0.0f;
    }

    // negative phase associations
    w_i = 0;
    for (int i = 0; i < n_visible; i++) {
        for (int j = 0; j < n_hidden; j++) {
            neg_weights[w_i] = v[i] * h_prob[j];
            w_i++;
        }
    }

    float learning_rate = 0.01;

    // update weights
    w_i = 0;
    for (int i = 0; i < n_visible; i++) {
        for (int j = 0; j < n_hidden; j++) {
            W[w_i] += learning_rate
                    * (pos_weights[w_i] - neg_weights[w_i]);
            w_i++;
        }
    }

    for (int i = 0; i < n_visible; i++) {


    }

    for (int i = 0; i < n_hidden; i++) {


    }


}

void RBM::CDUpdate() {

}

RBM::RBM(int image_side, int n_hidden) {

    this->image_side = image_side;
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
    pos_weights = new float[n_visible * n_hidden];
    neg_weights = new float[n_visible * n_hidden];

    filters.resize(n_hidden);
    for (int i = 0; i < n_hidden; i++) {
        filters[i] = new ofImage();
        filters[i]->allocate(image_side, image_side, OF_IMAGE_GRAYSCALE);
    }

    v_image = new ofImage();
    v_image->allocate(image_side, image_side, OF_IMAGE_GRAYSCALE);

    v_prob_image = new ofImage();
    v_prob_image->allocate(image_side, image_side, OF_IMAGE_GRAYSCALE);

    h_image_side = sqrt(n_hidden);

    h_image = new ofImage();
    h_image->allocate(h_image_side, h_image_side, OF_IMAGE_GRAYSCALE);

    h_prob_image = new ofImage();
    h_prob_image->allocate(h_image_side, h_image_side, OF_IMAGE_GRAYSCALE);
}

RBM::~RBM() {

    delete[] v;
    delete[] h;

    delete[] b;
    delete[] c;

    delete[] W;
    delete[] pos_weights;
    delete[] neg_weights;
}

void RBM::randomInit() {

    // set biases to 0
    // set units activations randomly
    memset(b, 0, n_visible * sizeof(float));
    memset(c, 0, n_hidden * sizeof(float));

    for (int i = 0; i < n_visible * n_hidden; i++) {
        W[i] = ofRandom(0.001f);
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
                int p_i = l * image_side + k;
                // weight index
                int w_i = i * n_visible + p_i;
                filters[i]->setColor(k, l, ofColor(1000 * W[w_i] * 255.0f));
            }
        }
        filters[i]->update();
    }

    // activation probabilities and sampled activations
    for (int k = 0; k < image_side; k++) {
        for (int l = 0; l < image_side; l++) {
            // pixel index
            int p_i = l * image_side + k;
            v_prob_image->setColor(k, l, ofColor(v_prob[p_i] * 255.0f));
            v_image->setColor(k, l, ofColor(v[p_i] * 255.0f));
        }
    }


    for (int k = 0; k < h_image_side; k++) {
        for (int l = 0; l < h_image_side; l++) {
            // pixel index
            int p_i = l * h_image_side + k;
            h_prob_image->setColor(k, l, ofColor(h_prob[p_i] * 255.0f));
            h_image->setColor(k, l, ofColor(h[p_i] * 255.0f));
        }
    }

    v_prob_image->update();
    v_image->update();

    h_prob_image->update();
    h_image->update();
}

