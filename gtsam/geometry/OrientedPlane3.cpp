/* ----------------------------------------------------------------------------

 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/*
 * @file OrientedPlane3.cpp
 * @date Dec 19, 2013
 * @author Alex Trevor
 * @author Zhaoyang Lv
 * @brief  A plane, represented by a normal direction and perpendicular distance
 */

#include <gtsam/geometry/OrientedPlane3.h>
#include <gtsam/geometry/Point2.h>
#include <iostream>

using namespace std;

namespace gtsam {

/* ************************************************************************* */
void OrientedPlane3::print(const string& s) const {
  Vector4 coeffs = planeCoefficients();
  cout << s << " : " << coeffs << endl;
}

/* ************************************************************************* */
OrientedPlane3 OrientedPlane3::Transform(const OrientedPlane3& plane,
    const Pose3& xr, OptionalJacobian<3, 6> Hr,
    OptionalJacobian<3, 3> Hp) {
  Matrix23 n_hr;
  Matrix22 n_hp;
  Unit3 n_rotated = xr.rotation().unrotate(plane.n_, n_hr, n_hp);

  Vector3 unit_vec = n_rotated.unitVector();
  double pred_d = plane.n_.unitVector().dot(xr.translation().vector()) + plane.d_;

  if (Hr) {
    *Hr = zeros(3, 6);
    Hr->block<2, 3>(0, 0) = n_hr;
    Hr->block<1, 3>(2, 3) = unit_vec;
  }
  if (Hp) {
    Vector2 hpp = plane.n_.basis().transpose() * xr.translation().vector();
    *Hp = zeros(3, 3);
    Hp->block<2, 2>(0, 0) = n_hp;
    Hp->block<1, 2>(2, 0) = hpp;
    (*Hp)(2, 2) = 1;
  }

  return OrientedPlane3(unit_vec(0), unit_vec(1), unit_vec(2), pred_d);
}

/* ************************************************************************* */
Vector3 OrientedPlane3::error(const OrientedPlane3& plane) const {
  Vector2 n_error = -n_.localCoordinates(plane.n_);
  double d_error = d_ - plane.d_;
  return Vector3(n_error(0), n_error(1), d_error);
}

/* ************************************************************************* */
OrientedPlane3 OrientedPlane3::retract(const Vector3& v) const {
  Vector2 n_v(v(0), v(1));
  Unit3 n_retracted = n_.retract(n_v);
  double d_retracted = d_ + v(2);
  return OrientedPlane3(n_retracted, d_retracted);
}

/* ************************************************************************* */
Vector3 OrientedPlane3::localCoordinates(const OrientedPlane3& y) const {
  Vector2 n_local = n_.localCoordinates(y.n_);
  double d_local = d_ - y.d_;
  Vector3 e;
  return Vector3(n_local(0), n_local(1), -d_local);
}

}
