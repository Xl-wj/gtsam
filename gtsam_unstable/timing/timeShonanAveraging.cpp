/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010-2019, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file   testShonanAveraging.cpp
 * @date   September 2019
 * @author Jing Wu
 * @author Frank Dellaert
 * @brief  Timing script for Shonan Averaging algorithm
 */

#include <gtsam/base/timing.h>
#include <gtsam_unstable/slam/ShonanAveraging.h>

#include <CppUnitLite/TestHarness.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>

using namespace std;
using namespace gtsam;

// string g2oFile = findExampleDataFile("toyExample.g2o");
// string g2oFile = "/Users/dellaert/git/SE-Sync/data/toy3D.g2o";
// string g2oFile =
// "/home/jingwu/catkin_workspace/gtsam/examples/Data/tinyGrid3D.g2o";

// save a single line of timing info to an output stream
void saveData(size_t p, double time1, double costP, double cost3, double time2,
              double min_eigenvalue, double suBound, std::ostream* os) {
    *os << (int)p << "\t" << time1 << "\t" << costP << "\t" << cost3 << "\t"
        << time2 << "\t" << min_eigenvalue << "\t" << suBound << endl;
}

int main(int argc, char* argv[]) {
    // primitive argument parsing:
    if (argc > 3) {
        throw runtime_error("Usage: timeShonanAveraging  [g2oFile]");
    }

    string g2oFile;
    try {
        if (argc > 1)
            g2oFile = argv[argc - 1];
        else
            //         g2oFile =
            //         "/home/jingwu/catkin_workspace/gtsam/examples/Data/toyExample.g2o";
            g2oFile = string(
                "/home/jingwu/Desktop/CS8903/SESync/data/SE3/tinyGrid3D.g2o");

    } catch (const exception& e) {
        cerr << e.what() << '\n';
        exit(1);
    }

    // Create a csv output file
    size_t pos1 = g2oFile.find("data/");
    size_t pos2 = g2oFile.find(".g2o");
    string name = g2oFile.substr(pos1 + 5, pos2 - pos1 - 5);
    cout << name << endl;
    ofstream csvFile("shonan_timing_of_" + name + ".csv");

    // Create Shonan averaging instance from the file.
    // ShonanAveragingParameters parameters;
    // double sigmaNoiseInRadians = 0 * M_PI / 180;
    // parameters.setNoiseSigma(sigmaNoiseInRadians);
    static const ShonanAveraging kShonan(g2oFile);

    // increase p value and try optimize using Shonan Algorithm. use chrono for
    // timing
    size_t pMin = 3;
    bool withDescent = true;
    Values Qstar;
    Vector minEigenVector;
    double CostP = 0, Cost3 = 0, lambdaMin = 0, suBound = 0;
    cout << "(int)p" << "\t" << "time1" << "\t" << "costP" << "\t" << "cost3" << "\t"
        << "time2" << "\t" << "MinEigenvalue" << "\t" << "SuBound" << endl;

    for (size_t p = pMin; p < 11; p++) {
        const Values initial = 
            (p > pMin && withDescent) ? kShonan.initializeWithDescent( p, Qstar, minEigenVector, lambdaMin) : kShonan.initializeRandomlyAt(p);
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        const Values result = kShonan.tryOptimizingAt(p, initial);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> timeUsed1 =
            chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        lambdaMin = kShonan.computeMinEigenValue(result, &minEigenVector);
        chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
        chrono::duration<double> timeUsed2 =
            chrono::duration_cast<chrono::duration<double>>(t3 - t1);
        Qstar = result;
        CostP = kShonan.costAt(p, result);
        const Values SO3Values = kShonan.roundSolution(result);
        Cost3 = kShonan.cost(SO3Values);
        suBound = (Cost3 - CostP) / CostP;

        saveData(p, timeUsed1.count(), CostP, Cost3, timeUsed2.count(),
                 lambdaMin, suBound, &cout);
        saveData(p, timeUsed1.count(), CostP, Cost3, timeUsed2.count(),
                 lambdaMin, suBound, &csvFile);
    }

    return 0;
}