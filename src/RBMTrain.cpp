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

float sign(float x) {
    if (x > 0.0f) {
        return 1.0f;
    } else {
        return -1.0f;
    }
}

// fall back to no mini-batch with batch_n == 1
void RBM::updateMiniBatch() {

    float learning_rate = 0.05;
    float momentum = 0.5f;
    float weightcost = 0.0002;

    // get the next data item, put it into the visible units
    // update with a training data batch
    if (sample_i >= n_training_samples) {
        sample_i = 0;
        sample_offset = training_data;
        epoch_i++;
    }

    sample_i++;
    for (int i = 0; i < n_visible; i++) {
        v_data[i] = (*sample_offset > 128.0f);
        sample_offset++;
    }

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

    // positive phase associations
    for (int i = 0; i < n_visible * n_hidden; i++) {
        pos_weights[i] = v_data[i / n_hidden] * h_data[i % n_hidden];
    }

    for (int i = 0; i < n_hidden; i++) {
        h[i] = h_data[i];
    }

    // increasing selectivity
    float activity_s = 0.99f;
    float total_active = 0.0f;
    for (int i = 0; i < n_hidden; i++) {
        mean_activity[i] = mean_activity[i] * activity_s + h[i]
            * (1.0f - activity_s);
        c[i] += (0.1f - mean_activity[i]) * 0.01f;
        total_active += h[i];
    }

    // increasing sparseness
//    q = activity_smoothing * q +
//    (1.0f - activity_smoothing) * (total_active / (float)n_hidden);
//    for (int i = 0; i < n_hidden; i++) {
//        c[i] += (0.1f - q) * 0.01f;
//    }

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
    }

    // negative phase associations
    for (int i = 0; i < n_visible * n_hidden; i++) {
        neg_weights[i] = v[i / n_hidden] * h[i % n_hidden];
    }

    for (int i = 0; i < n_visible * n_hidden; i++) {
        W_inc[i] = momentum * W_inc[i]
                + learning_rate * (pos_weights[i] - neg_weights[i])
                - weightcost * W[i];
        W[i] += W_inc[i];
    }

    for (int i = 0; i < n_visible; i++) {
        b_inc[i] *= momentum;
        b_inc[i] += learning_rate * (v_data[i] - v[i]);
        b[i] += b_inc[i];
    }

    for (int i = 0; i < n_hidden; i++) {
        c_inc[i] *= momentum;
        c_inc[i] += learning_rate * (h_data[i] - h[i]);
        c[i] += c_inc[i];
    }
}

