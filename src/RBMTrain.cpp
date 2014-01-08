/*
 * RBMTrain.cpp
 *
 *  Created on: Jan 8, 2014
 *      Author: sk
 */

#include "RBM.h"

float sigmoid(float x) {
    return (1.0f / (1.0f + exp(-x)));
}

void RBM::update() {

    float learning_rate = 0.001;

    // (positive phase)
    // compute hidden nodes activations and probabilities
    for (int i = 0; i < n_hidden; i++) {
        h_prob[i] = 0.0f;
        for (int j = 0; j < n_visible; j++) {
            h_prob[i] += c[i] + v_data[j] * W[j * n_hidden + i];
        }
        h_prob[i] = sigmoid(h_prob[i]);
        h[i] = ofRandom(1.0f) > h_prob[i] ? 1.0f : 0.0f;
    }

    // positive phase associations
    int w_i;

    w_i = 0;
    for (int i = 0; i < n_visible; i++) {
        for (int j = 0; j < n_hidden; j++) {
            pos_weights[w_i] = v_data[i] * h[j];
            w_i++;
        }
    }

    // (negative phase)
    // sample visible nodes
    for (int i = 0; i < n_visible; i++) {
        v_negative_prob[i] = 0.0f;
        for (int j = 0; j < n_hidden; j++) {
            v_negative_prob[i] += b[i] + h[j] * W[i * n_hidden + j];
        }
        v_negative_prob[i] = sigmoid(v_negative_prob[i]);
        v_negative[i] = ofRandom(1.0f) > v_negative_prob[i] ? 1.0f : 0.0f;
    }

    // and hidden nodes once again
    for (int i = 0; i < n_hidden; i++) {
        h_negative_prob[i] = 0.0f;
        for (int j = 0; j < n_visible; j++) {
            h_negative_prob[i] += c[i] + v_negative[j] * W[j * n_hidden + i];
        }
        h_negative_prob[i] = sigmoid(h_negative_prob[i]);
        h_negative[i] = ofRandom(1.0f) > h_negative_prob[i] ? 1.0f : 0.0f;
    }

    // negative phase associations
    w_i = 0;
    for (int i = 0; i < n_visible; i++) {
        for (int j = 0; j < n_hidden; j++) {
            neg_weights[w_i] = v_negative[i] * h_negative[j];
            w_i++;
        }
    }

    // update weights
    for (int i = 0; i < n_visible * n_hidden; i++) {
        W[i] += learning_rate * (pos_weights[i] - neg_weights[i]);
    }

    for (int i = 0; i < n_visible; i++) {
        b[i] += learning_rate * (v_data[i] - v_negative[i]);
    }

    for (int i = 0; i < n_hidden; i++) {
        c[i] += learning_rate * (h[i] - h_negative[i]);
    }
}

