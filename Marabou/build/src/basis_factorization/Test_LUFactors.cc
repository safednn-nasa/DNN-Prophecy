/* Generated file, do not edit */

#ifndef CXXTEST_RUNNING
#define CXXTEST_RUNNING
#endif

#define _CXXTEST_HAVE_STD
#define _CXXTEST_HAVE_EH
#include <cxxtest/TestListener.h>
#include <cxxtest/TestTracker.h>
#include <cxxtest/TestRunner.h>
#include <cxxtest/RealDescriptions.h>
#include <cxxtest/ErrorPrinter.h>

int main() {
 return CxxTest::ErrorPrinter().run();
}
#include "/content/Marabou/src/basis_factorization/tests/Test_LUFactors.h"

static LUFactorsTestSuite suite_LUFactorsTestSuite;

static CxxTest::List Tests_LUFactorsTestSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_LUFactorsTestSuite( "/content/Marabou/src/basis_factorization/tests/Test_LUFactors.h", 27, "LUFactorsTestSuite", suite_LUFactorsTestSuite, Tests_LUFactorsTestSuite );

static class TestDescription_LUFactorsTestSuite_test_f_forward_transformation : public CxxTest::RealTestDescription {
public:
 TestDescription_LUFactorsTestSuite_test_f_forward_transformation() : CxxTest::RealTestDescription( Tests_LUFactorsTestSuite, suiteDescription_LUFactorsTestSuite, 122, "test_f_forward_transformation" ) {}
 void runTest() { suite_LUFactorsTestSuite.test_f_forward_transformation(); }
} testDescription_LUFactorsTestSuite_test_f_forward_transformation;

static class TestDescription_LUFactorsTestSuite_test_f_backward_transformation : public CxxTest::RealTestDescription {
public:
 TestDescription_LUFactorsTestSuite_test_f_backward_transformation() : CxxTest::RealTestDescription( Tests_LUFactorsTestSuite, suiteDescription_LUFactorsTestSuite, 156, "test_f_backward_transformation" ) {}
 void runTest() { suite_LUFactorsTestSuite.test_f_backward_transformation(); }
} testDescription_LUFactorsTestSuite_test_f_backward_transformation;

static class TestDescription_LUFactorsTestSuite_test_v_forward_transformation : public CxxTest::RealTestDescription {
public:
 TestDescription_LUFactorsTestSuite_test_v_forward_transformation() : CxxTest::RealTestDescription( Tests_LUFactorsTestSuite, suiteDescription_LUFactorsTestSuite, 190, "test_v_forward_transformation" ) {}
 void runTest() { suite_LUFactorsTestSuite.test_v_forward_transformation(); }
} testDescription_LUFactorsTestSuite_test_v_forward_transformation;

static class TestDescription_LUFactorsTestSuite_test_v_backward_transformation : public CxxTest::RealTestDescription {
public:
 TestDescription_LUFactorsTestSuite_test_v_backward_transformation() : CxxTest::RealTestDescription( Tests_LUFactorsTestSuite, suiteDescription_LUFactorsTestSuite, 224, "test_v_backward_transformation" ) {}
 void runTest() { suite_LUFactorsTestSuite.test_v_backward_transformation(); }
} testDescription_LUFactorsTestSuite_test_v_backward_transformation;

static class TestDescription_LUFactorsTestSuite_test_forward_transformation : public CxxTest::RealTestDescription {
public:
 TestDescription_LUFactorsTestSuite_test_forward_transformation() : CxxTest::RealTestDescription( Tests_LUFactorsTestSuite, suiteDescription_LUFactorsTestSuite, 258, "test_forward_transformation" ) {}
 void runTest() { suite_LUFactorsTestSuite.test_forward_transformation(); }
} testDescription_LUFactorsTestSuite_test_forward_transformation;

static class TestDescription_LUFactorsTestSuite_test_backward_transformation : public CxxTest::RealTestDescription {
public:
 TestDescription_LUFactorsTestSuite_test_backward_transformation() : CxxTest::RealTestDescription( Tests_LUFactorsTestSuite, suiteDescription_LUFactorsTestSuite, 292, "test_backward_transformation" ) {}
 void runTest() { suite_LUFactorsTestSuite.test_backward_transformation(); }
} testDescription_LUFactorsTestSuite_test_backward_transformation;

static class TestDescription_LUFactorsTestSuite_test_invert_basis : public CxxTest::RealTestDescription {
public:
 TestDescription_LUFactorsTestSuite_test_invert_basis() : CxxTest::RealTestDescription( Tests_LUFactorsTestSuite, suiteDescription_LUFactorsTestSuite, 326, "test_invert_basis" ) {}
 void runTest() { suite_LUFactorsTestSuite.test_invert_basis(); }
} testDescription_LUFactorsTestSuite_test_invert_basis;

#include <cxxtest/Root.cpp>
