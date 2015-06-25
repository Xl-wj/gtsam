/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/*
 * @file testOrientedPlane3.cpp
 * @date Dec 19, 2012
 * @author Alex Trevor
 * @author Zhaoyang Lv
 * @brief Tests the OrientedPlane3 class
 */

#include <gtsam/geometry/OrientedPlane3.h>
#include <gtsam/base/numericalDerivative.h>
#include <CppUnitLite/TestHarness.h>
#include <boost/assign/std/vector.hpp>

using namespace boost::assign;
using namespace gtsam;
using namespace std;
using boost::none;

GTSAM_CONCEPT_TESTABLE_INST(OrientedPlane3)
GTSAM_CONCEPT_MANIFOLD_INST(OrientedPlane3)

//*******************************************************************************
TEST (OrientedPlane3, getMethods) {
  Vector4 c;
  c << -1, 0, 0, 5;
  OrientedPlane3 plane1(c);
  OrientedPlane3 plane2(c[0], c[1], c[2], c[3]);
  Vector4 coefficient1 = plane1.planeCoefficients();
  double distance1 = plane1.distance();
  EXPECT(assert_equal(coefficient1, c, 1e-8));
  EXPECT(assert_equal(Unit3(-1,0,0).unitVector(), plane1.normal().unitVector()));
  EXPECT_DOUBLES_EQUAL(distance1, 5, 1e-8);
  Vector4 coefficient2 = plane2.planeCoefficients();
  double distance2 = plane2.distance();
  EXPECT(assert_equal(coefficient2, c, 1e-8));
  EXPECT_DOUBLES_EQUAL(distance2, 5, 1e-8);
  EXPECT(assert_equal(Unit3(-1,0,0).unitVector(), plane2.normal().unitVector()));
}

//*******************************************************************************
TEST (OrientedPlane3, transform) {
  // Test transforming a plane to a pose
  gtsam::Pose3 pose(gtsam::Rot3::ypr(-M_PI / 4.0, 0.0, 0.0),
      gtsam::Point3(2.0, 3.0, 4.0));
  OrientedPlane3 plane(-1, 0, 0, 5);
  OrientedPlane3 expected_meas(-sqrt(2.0) / 2.0, -sqrt(2.0) / 2.0, 0.0, 3);
  OrientedPlane3 transformed_plane = OrientedPlane3::Transform(plane, pose,
      none, none);
  EXPECT(assert_equal(expected_meas, transformed_plane, 1e-9));

  // Test the jacobians of transform
  Matrix actualH1, expectedH1, actualH2, expectedH2;
  {
    expectedH1 = numericalDerivative11<OrientedPlane3, Pose3>(
        boost::bind(&OrientedPlane3::Transform, plane, _1, none, none), pose);

    OrientedPlane3 tformed = OrientedPlane3::Transform(plane, pose, actualH1,
        none);
    EXPECT(assert_equal(expectedH1, actualH1, 1e-9));
  }
  {
    expectedH2 = numericalDerivative11<OrientedPlane3, OrientedPlane3>(
        boost::bind(&OrientedPlane3::Transform, _1, pose, none, none), plane);

    OrientedPlane3 tformed = OrientedPlane3::Transform(plane, pose, none,
        actualH2);
    EXPECT(assert_equal(expectedH2, actualH2, 1e-9));
  }

}

//*******************************************************************************
// Returns a random vector -- copied from testUnit3.cpp
inline static Vector randomVector(const Vector& minLimits,
    const Vector& maxLimits) {

  // Get the number of dimensions and create the return vector
  size_t numDims = dim(minLimits);
  Vector vector = zero(numDims);

  // Create the random vector
  for (size_t i = 0; i < numDims; i++) {
    double range = maxLimits(i) - minLimits(i);
    vector(i) = (((double) rand()) / RAND_MAX) * range + minLimits(i);
  }
  return vector;
}

//*******************************************************************************
TEST(OrientedPlane3, localCoordinates_retract) {

  size_t numIterations = 10000;
  Vector4 minPlaneLimit, maxPlaneLimit;
  minPlaneLimit << -1.0, -1.0, -1.0, 0.01;
  maxPlaneLimit << 1.0, 1.0, 1.0, 10.0;

  Vector3 minXiLimit, maxXiLimit;
  minXiLimit << -M_PI, -M_PI, -10.0;
  maxXiLimit << M_PI, M_PI, 10.0;
  for (size_t i = 0; i < numIterations; i++) {

    sleep(0);

    // Create a Plane
    OrientedPlane3 p1(randomVector(minPlaneLimit, maxPlaneLimit));
    Vector v12 = randomVector(minXiLimit, maxXiLimit);

    // Magnitude of the rotation can be at most pi
    if (v12.head<3>().norm() > M_PI)
      v12.head<3>() = v12.head<3>() / M_PI;
    OrientedPlane3 p2 = p1.retract(v12);

    // Check if the local coordinates and retract return the same results.
    Vector actual_v12 = p1.localCoordinates(p2);
    EXPECT(assert_equal(v12, actual_v12, 1e-6));
    OrientedPlane3 actual_p2 = p1.retract(actual_v12);
    EXPECT(assert_equal(p2, actual_p2, 1e-6));
  }
}

/* ************************************************************************* */
int main() {
  srand(time(NULL));
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
