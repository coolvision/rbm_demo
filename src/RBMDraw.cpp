/*
 * RBMDraw.cpp
 *
 *  Created on: Jan 8, 2014
 *      Author: sk
 */

#include "RBM.h"

void RBM::makeImages() {

    // put units values and weights into images

    // filters from weights
    float min_v = FLT_MAX;
    float max_v = -FLT_MAX;

    for (int i = 0; i < n_hidden - 1; i++) {

        for (int j = 0; j < n_visible - 1; j++) {
            int w_i = (j + 1) * n_hidden + (i + 1);
            float w = W[w_i];

            if (w > max_v)
                max_v = w;
            if (w < min_v)
                min_v = w;
        }

        for (int j = 0; j < n_visible; j++) {

            int w_i = (j + 1) * n_hidden + (i + 1);
            float w = W[w_i];

            if (w < 0) {
                filters[i]->setColor(j % image_side, j / image_side,
                        ofColor(((w - min_v) / (max_v - min_v)) * 255.0f, 0.0f,
                                0.0f));
            } else {
                filters[i]->setColor(j % image_side, j / image_side,
                        ofColor(((w - min_v) / (max_v - min_v)) * 255.0f));
            }
        }
    }

    for (int i = 0; i < n_hidden; i++) {
        filters[i]->update();
    }

    min_v = FLT_MAX;
    max_v = -FLT_MAX;
    for (int i = 0; i < n_visible - 1; i++) {
        float b = W[(i + 1) * n_hidden];
        if (b > max_v)
            max_v = b;
        if (b < min_v)
            min_v = b;
    }
    for (int i = 0; i < n_visible - 1; i++) {
        float b = W[(i + 1) * n_hidden];
        if (b < 0.0f) {
            v_bias->setColor(i % image_side, i / image_side,
                    ofColor(((b - min_v) / (max_v - min_v)) * 255.0f, 0.0f,
                            0.0f));
        } else {
            v_bias->setColor(i % image_side, i / image_side,
                    ofColor(0.0f, 0.0f,
                            ((b - min_v) / (max_v - min_v)) * 255.0f));
        }
    }

    min_v = FLT_MAX;
    max_v = -FLT_MAX;
    for (int i = 0; i < n_hidden - 1; i++) {
        float c = W[i + 1];
        if (c > max_v)
            max_v = c;
        if (c < min_v)
            min_v = c;
    }
    for (int i = 0; i < n_hidden - 1; i++) {
        float c = W[i + 1];
        if (c < 0.0f) {
            h_bias->setColor(i % h_image_side, i / h_image_side,
                    ofColor(((c - min_v) / (max_v - min_v)) * 255.0f, 0.0f,
                            0.0f));
        } else {
            h_bias->setColor(i % h_image_side, i / h_image_side,
                    ofColor(0.0f, 0.0f,
                            ((c - min_v) / (max_v - min_v)) * 255.0f));
        }
    }

    // activation probabilities and sampled activations
    for (int i = 0; i < n_visible - 1; i++) {
        v_data_image->setColor(i % image_side, i / image_side,
                ofColor(v_data[i + 1] * 255.0f));
        v_prob_image->setColor(i % image_side, i / image_side,
                ofColor(v_prob[i + 1] * 255.0f));
        v_image->setColor(i % image_side, i / image_side,
                ofColor(v[i + 1] * 255.0f));
    }

    for (int i = 0; i < n_hidden - 1; i++) {
        h_data_image->setColor(i % h_image_side, i / h_image_side,
                ofColor(h_data[i + 1] * 255.0f));
        h_image->setColor(i % h_image_side, i / h_image_side,
                ofColor(h[i + 1] * 255.0f));
    }

    max_v = -FLT_MAX;
    for (int i = 0; i < n_hidden; i++) {
        if (h_prob[i] > max_v) {
            max_v = h_prob[i];
        }
    }
    for (int i = 0; i < n_hidden - 1; i++) {
        h_data_prob_image->setColor(i % h_image_side, i / h_image_side,
                ofColor((h_prob[i + 1] / max_v) * 255.0f));
    }

    max_v = -FLT_MAX;
    for (int i = 0; i < n_hidden - 1; i++) {
        if (h_prob[i + 1] > max_v) {
            max_v = h_prob[i];
        }
    }
    for (int i = 0; i < n_hidden - 1; i++) {
        h_prob_image->setColor(i % h_image_side, i / h_image_side,
                ofColor((h_prob[i + 1] / max_v) * 255.0f));
    }

    v_bias->update();
    h_bias->update();

    v_data_image->update();
    v_prob_image->update();
    v_image->update();

    h_data_prob_image->update();
    h_data_image->update();
    h_prob_image->update();
    h_image->update();
}
