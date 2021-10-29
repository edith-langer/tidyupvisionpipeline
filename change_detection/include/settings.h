#ifndef SETTINGS_H
#define SETTINGS_H

#include <PPFRecognizerParameter.h>

static const double ds_leaf_size_LV = 0.01;
static const double ds_leaf_size_ppf = 0.005;

static const int min_object_size_ds = 200;
static const int max_object_size_ds = 7000;
static const int min_object_volume = 0.000125; //0.05^3

const float max_dist_for_being_static = 0.2; //how much can the object be displaced to still count as static

static v4r::apps::PPFRecognizerParameter ppf_params;

#endif // SETTINGS_H
