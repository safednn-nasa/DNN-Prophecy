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
#include "/content/Marabou/src/basis_factorization/tests/Test_SparseLUFactors.h"

static SparseLUFactorsTestSuite suite_SparseLUFactorsTestSuite;

static CxxTest::List Tests_SparseLUFactorsTestSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_SparseLUFactorsTestSuite( "/content/Marabou/src/basis_factorization/tests/Test_SparseLUFactors.h", 27, "SparseLUFactorsTestSuite", suite_SparseLUFactorsTestSuite, Tests_SparseLUFactorsTestSuite );

static class TestDescription_SparseLUFactorsTestSuite_test_f_forward_transformation : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseLUFactorsTestSuite_test_f_forward_transformation() : CxxTest::RealTestDescription( Tests_SparseLUFactorsTestSuite, suiteDescription_SparseLUFactorsTestSuite, 150, "test_f_forward_transformation" ) {}
 void runTest() { suite_SparseLUFactorsTestSuite.test_f_forward_transformation(); }
} testDescription_SparseLUFactorsTestSuite_test_f_forward_transformation;

static class TestDescription_SparseLUFactorsTestSuite_test_f_backward_transformation : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseLUFactorsTestSuite_test_f_backward_transformation() : CxxTest::RealTestDescription( Tests_SparseLUFactorsTestSuite, suiteDescription_SparseLUFactorsTestSuite, 185, "test_f_backward_transformation" ) {}
 void runTest() { suite_SparseLUFactorsTestSuite.test_f_backward_transformation(); }
} testDescription_SparseLUFactorsTestSuite_test_f_backward_transformation;

static class TestDescription_SparseLUFactorsTestSuite_test_v_forward_transformation : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseLUFactorsTestSuite_test_v_forward_transformation() : CxxTest::RealTestDescription( Tests_SparseLUFactorsTestSuite, suiteDescription_SparseLUFactorsTestSuite, 219, "test_v_forward_transformation" ) {}
 void runTest() { suite_SparseLUFactorsTestSuite.test_v_forward_transformation(); }
} testDescription_SparseLUFactorsTestSuite_test_v_forward_transformation;

static class TestDescription_SparseLUFactorsTestSuite_test_v_backward_transformation : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseLUFactorsTestSuite_test_v_backward_transformation() : CxxTest::RealTestDescription( Tests_SparseLUFactorsTestSuite, suiteDescription_SparseLUFactorsTestSuite, 253, "test_v_backward_transformation" ) {}
 void runTest() { suite_SparseLUFactorsTestSuite.test_v_backward_transformation(); }
} testDescription_SparseLUFactorsTestSuite_test_v_backward_transformation;

static class TestDescription_SparseLUFactorsTestSuite_test_forward_transformation : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseLUFactorsTestSuite_test_forward_transformation() : CxxTest::RealTestDescription( Tests_SparseLUFactorsTestSuite, suiteDescription_SparseLUFactorsTestSuite, 287, "test_forward_transformation" ) {}
 void runTest() { suite_SparseLUFactorsTestSuite.test_forward_transformation(); }
} testDescription_SparseLUFactorsTestSuite_test_forward_transformation;

static class TestDescription_SparseLUFactorsTestSuite_test_backward_transformation : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseLUFactorsTestSuite_test_backward_transformation() : CxxTest::RealTestDescription( Tests_SparseLUFactorsTestSuite, suiteDescription_SparseLUFactorsTestSuite, 321, "test_backward_transformation" ) {}
 void runTest() { suite_SparseLUFactorsTestSuite.test_backward_transformation(); }
} testDescription_SparseLUFactorsTestSuite_test_backward_transformation;

static class TestDescription_SparseLUFactorsTestSuite_test_invert_basis : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseLUFactorsTestSuite_test_invert_basis() : CxxTest::RealTestDescription( Tests_SparseLUFactorsTestSuite, suiteDescription_SparseLUFactorsTestSuite, 355, "test_invert_basis" ) {}
 void runTest() { suite_SparseLUFactorsTestSuite.test_invert_basis(); }
} testDescription_SparseLUFactorsTestSuite_test_invert_basis;

#include <cxxtest/Root.cpp>
