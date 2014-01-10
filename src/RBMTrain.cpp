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

    float learning_rate = 0.1;
    float momentum = 0.5f;
    float weightcost = 0.0002;

    // (positive phase)
    // compute hidden nodes activations and probabilities
    for (int i = 0; i < n_hidden; i++) {
        h_data_prob[i] = 0.0f;
        for (int j = 0; j < n_visible; j++) {
            h_data_prob[i] += c[i] + v_data[j] * W[j * n_hidden + i];
        }
        h_data_prob[i] = sigmoid(h_data_prob[i]);
        h_data[i] = ofRandom(1.0f) > h_data_prob[i] ? 0.0f : 1.0f;
    }

    for (int i = 0; i < n_hidden; i++) {
        if (ofRandom(1.0f) > 0.5f) {
            h_data[i] = 0.0f;
        }
    }

    // positive phase associations
    for (int i = 0; i < n_visible * n_hidden; i++) {
        pos_weights[i] = v_data[i / n_hidden] * h_data[i % n_hidden];
    }

    for (int i = 0; i < n_hidden; i++) {
        h[i] = h_data[i];
    }

    for (int i = 0; i < 5; i++) {

    // run update for CD1 or persistent chain for PCD
    for (int i = 0; i < n_visible; i++) {
        v_prob[i] = 0.0f;
        for (int j = 0; j < n_hidden; j++) {
            v_prob[i] += b[i] + h[j] * W[i * n_hidden + j];
        }
        v_prob[i] = sigmoid(v_prob[i]);
        v[i] = ofRandom(1.0f) > v_prob[i] ? 0.0f : 1.0f;
    }

    // and hidden nodes
    for (int i = 0; i < n_hidden; i++) {
        h_prob[i] = 0.0f;
        for (int j = 0; j < n_visible; j++) {
            h_prob[i] += c[i] + v[j] * W[j * n_hidden + i];
        }
        h_prob[i] = sigmoid(h_prob[i]);
        h[i] = ofRandom(1.0f) > h_prob[i] ? 0.0f : 1.0f;
    }

//        for (int i = 0; i < n_hidden; i++) {
//            if (ofRandom(1.0f) > 0.5f) {
//                h[i] = 0.0f;
//            }
//        }

    }

    // negative phase associations
    for (int i = 0; i < n_visible * n_hidden; i++) {
        neg_weights[i] = v[i / n_hidden] * h[i % n_hidden];
    }

    for (int i = 0; i < n_visible * n_hidden; i++) {
        W_inc[i] = momentum * W_inc[i] +
                learning_rate * (pos_weights[i] - neg_weights[i]) -
                weightcost * W[i];
        W[i] += W_inc[i];
    }

    for (int i = 0; i < n_visible; i++) {
        b_inc[i] *= momentum;
        b_inc[i] += 0.01 * learning_rate * (v_data[i] - v[i]);
        b[i] += b_inc[i];
    }

    for (int i = 0; i < n_hidden; i++) {
        c_inc[i] *= momentum;
        c_inc[i] += 0.01 * learning_rate * (h_data[i] - h[i]);
        c[i] += c_inc[i];
    }
}

