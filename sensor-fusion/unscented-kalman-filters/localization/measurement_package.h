/*
 * measurement_package.h
 */

#ifndef MEASUREMENT_PACKAGE_H_
#define MEASUREMENT_PACKAGE_H_
#include <vector>

class MeasurementPackage {

public:
  struct control_s {
    float delta_x_f;
  };

  struct observation_s {
    std::vector <float> distance_f;
  };

  control_s control_s_;
  observation_s observation_s_;
};

#endif /* MEASUREMENT_PACKAGE_H_ */
