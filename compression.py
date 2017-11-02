#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compression Tools


Karen Ullrich, Oct 2017

References:

    [1] Michael T. Heath. 1996. Scientific Computing: An Introductory Survey (2nd ed.). Eric M. Munson (Ed.). McGraw-Hill Higher Education. Chapter 1
"""

import numpy as np

# -------------------------------------------------------
# General tools
# -------------------------------------------------------


def unit_round_off(t=23):
    """
    :param t:
        number significand bits
    :return:
        unit round off based on nearest interpolation, for reference see [1]
    """
    return 0.5 * 2. ** (1. - t)


SIGNIFICANT_BIT_PRECISION = [unit_round_off(t=i + 1) for i in range(23)]


def float_precision(x):

    out = np.sum([x < sbp for sbp in SIGNIFICANT_BIT_PRECISION])
    return out


def float_precisions(X, dist_fun, layer=1):

    X = X.flatten()
    out = [float_precision(2 * x) for x in X]
    out = np.ceil(dist_fun(out))
    return out


def special_round(input, significant_bit):
    delta = unit_round_off(t=significant_bit)
    rounded = np.floor(input / delta + 0.5)
    rounded = rounded * delta
    return rounded


def fast_infernce_weights(w, exponent_bit, significant_bit):

    return special_round(w, significant_bit)


def compress_matrix(x):

    if len(x.shape) != 2:
        A, B, C, D = x.shape
        x = x.reshape(A * B,  C * D)
        # remove non-necessary filters and rows
        x = x[:, (x != 0).any(axis=0)]
        x = x[(x != 0).any(axis=1), :]
    else:
        # remove unnecessary rows, columns
        x = x[(x != 0).any(axis=1), :]
        x = x[:, (x != 0).any(axis=0)]
    return x


def extract_pruned_params(layers, masks):

    post_weight_mus = []
    post_weight_vars = []

    for i, (layer, mask) in enumerate(zip(layers, masks)):
        # compute posteriors
        post_weight_mu, post_weight_var = layer.compute_posterior_params()
        post_weight_var = post_weight_var.cpu().data.numpy()
        post_weight_mu  = post_weight_mu.cpu().data.numpy()
        # apply mask to mus and variances
        post_weight_mu  = post_weight_mu * mask
        post_weight_var = post_weight_var * mask

        post_weight_mus.append(post_weight_mu)
        post_weight_vars.append(post_weight_var)

    return post_weight_mus, post_weight_vars


# -------------------------------------------------------
#  Compression rates (fast inference scenario)
# -------------------------------------------------------


def _compute_compression_rate(vars, in_precision=32., dist_fun=lambda x: np.max(x), overflow=10e38):
    # compute in  number of bits occupied by the original architecture
    sizes = [v.size for v in vars]
    nb_weights = float(np.sum(sizes))
    IN_BITS = in_precision * nb_weights
    # prune architecture
    vars = [compress_matrix(v) for v in vars]
    sizes = [v.size for v in vars]
    # compute
    significant_bits = [float_precisions(v, dist_fun, layer=k + 1) for k, v in enumerate(vars)]
    exponent_bit = np.ceil(np.log2(np.log2(overflow) + 1.) + 1.)
    total_bits = [1. + exponent_bit + sb for sb in significant_bits]
    OUT_BITS = np.sum(np.asarray(sizes) * np.asarray(total_bits))
    return nb_weights / np.sum(sizes), IN_BITS / OUT_BITS, significant_bits, exponent_bit


def compute_compression_rate(layers, masks):
    # reduce architecture
    weight_mus, weight_vars = extract_pruned_params(layers, masks)
    # compute overflow level based on maximum weight
    overflow = np.max([np.max(np.abs(w)) for w in weight_mus])
    # compute compression rate
    CR_architecture, CR_fast_inference, _, _ = _compute_compression_rate(weight_vars, dist_fun=lambda x: np.mean(x), overflow=overflow)
    print("Compressing the architecture will degrease the model by a factor of %.1f." % (CR_architecture))
    print("Making use of weight uncertainty can reduce the model by a factor of %.1f." % (CR_fast_inference))


def compute_reduced_weights(layers, masks):
    weight_mus, weight_vars = extract_pruned_params(layers, masks)
    overflow = np.max([np.max(np.abs(w)) for w in weight_mus])
    _, _, significant_bits, exponent_bits = _compute_compression_rate(weight_vars, dist_fun=lambda x: np.mean(x), overflow=overflow)
    weights = [fast_infernce_weights(weight_mu, exponent_bits, significant_bit) for weight_mu, significant_bit in
               zip(weight_mus, significant_bits)]
    return weights