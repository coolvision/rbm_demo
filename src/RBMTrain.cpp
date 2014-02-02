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

void clear(float *p, size_t n) {
    memset(p, 0, n * sizeof(float));
}

void getState(float *p, float *s, size_t n) {
    for (int i = 0; i < n; i++) {
        p[i] = sigmoid(p[i]);
        s[i] = ofRandom(1.0f) > p[i] ? 0.0f : 1.0f;
    }
}

inline float dot(float *a, float *b, int len, int step) {
    float r = 0;
    while (len--) {
        r += *a++ * *b;
        b += step;
    }
    return r;
}

void multiply(float *m1, float *m2, float *m_res, size_t m1_rows,
        size_t m1_cols, size_t m2_cols) {
    float *p, *pa;
    int i, j;
    p = m_res;
    for (pa = m1, i = 0; i < m1_rows; i++, pa += m1_cols) {
        for (j = 0; j < m2_cols; j++) {
            *p++ = dot(pa, m2 + j, m1_cols, m2_cols);
        }
    }
}

void transpose(float *m, float *mt, size_t rows, size_t cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float *pm = m + i * cols + j;
            float *pmt = mt + j * rows + i;
            (*pmt) = (*pm);
        }
    }
}

// fall back to no mini-batch with batch_n == 1
void RBM::updateMiniBatch() {

    // get a data offset for the mini-batch worth of data
    if (batch_i * batch_size >= n_training_samples) {
        batch_i = 0;
        epoch_i++;
    }

    sample_offset = training_data + batch_i * batch_size * (n_visible - 1);
    for (int i = 0; i < batch_size; i++) {
        v_data[i * n_visible] = 1;
        for (int j = 1; j < n_visible; j++) {
            v_data[i * n_visible + j] = (*sample_offset > 128.0f);
            sample_offset++;
        }
    }
    batch_i++;
    sample_i = batch_i * batch_size;

    // (positive phase)
    clear(h_data_prob, n_hidden * batch_size);
    multiply(v_data, W, h_data_prob, batch_size, n_visible, n_hidden);
    getState(h_data_prob, h_data, n_hidden * batch_size);

    transpose(v_data, vt, batch_size, n_visible);
    multiply(vt, h_data, pos_weights, n_visible, batch_size, n_hidden);

    for (int i = 0; i < batch_size; i++) {
        h_data[i * batch_size] = 1;
    }

    for (int i = 0; i < n_hidden * batch_size; i++) {
        h[i] = h_data[i];
    }

    for (int cdk = 0; cdk < k; cdk++) {

        // run update for CD1 or persistent chain for PCD
        clear(v_prob, n_visible * batch_size);
        transpose(W, Wt, n_visible, n_hidden);
        multiply(h, Wt, v_prob, batch_size, n_hidden, n_visible);
        getState(v_prob, v, n_visible * batch_size);

        // and hidden nodes
        clear(h_prob, n_hidden * batch_size);
        multiply(v, W, h_prob, batch_size, n_visible, n_hidden);
        getState(h_prob, h, n_hidden * batch_size);
    }

    transpose(v, vt, batch_size, n_visible);
    multiply(vt, h, neg_weights, n_visible, batch_size, n_hidden);

    for (int i = 0; i < n_visible * n_hidden; i++) {
        W_inc[i] *= momentum;
        W_inc[i] += learning_rate * (pos_weights[i] - neg_weights[i])
                / (float) batch_size - weightcost * W[i];
        W[i] += W_inc[i];
    }

//    // visible units biases
//    for (int i = 0; i < n_visible; i++) {
//        // get the weight index
//        int wi = i * n_hidden;
//        b_inc[i] *= momentum;
//        b_inc[i] += learning_rate * (pos_weights[wi] - neg_weights[wi])
//                / (float) batch_size;
//        W[wi] += b_inc[i];
//    }
//
//    // hidden units biases
//    for (int i = 0; i < n_hidden; i++) {
//        int wi = i;
//        c_inc[i] *= momentum;
//        c_inc[i] += learning_rate * (pos_weights[wi] - neg_weights[wi])
//                / (float) batch_size;
//        W[wi] += c_inc[i];
//    }

    // increasing selectivity
    float activity_smoothing = 0.99f;
    float total_active = 0.0f;
    for (int i = 0; i < n_hidden; i++) {
        float activity = pos_weights[i] / (float)batch_size;
        mean_activity[i] = mean_activity[i] * activity_smoothing
                + activity * (1.0f - activity_smoothing);
        //W[i] += (0.01f - mean_activity[i]) * 0.01f;
        //total_active += activity;
    }

//    // increasing sparseness
//    q = activity_smoothing * q
//            + (1.0f - activity_smoothing) * (total_active / (float) n_hidden);
//    for (int i = 0; i < n_hidden; i++) {
//        W[i] += (0.1f - q) * 0.01f;
//    }

}
