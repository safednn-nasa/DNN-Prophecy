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
#include "/content/Marabou/src/basis_factorization/tests/Test_SparseFTFactorization.h"

static SparseFTFactorizationTestSuite suite_SparseFTFactorizationTestSuite;

static CxxTest::List Tests_SparseFTFactorizationTestSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_SparseFTFactorizationTestSuite( "/content/Marabou/src/basis_factorization/tests/Test_SparseFTFactorization.h", 32, "SparseFTFactorizationTestSuite", suite_SparseFTFactorizationTestSuite, Tests_SparseFTFactorizationTestSuite );

static class TestDescription_SparseFTFactorizationTestSuite_test_forward_transformation : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseFTFactorizationTestSuite_test_forward_transformation() : CxxTest::RealTestDescription( Tests_SparseFTFactorizationTestSuite, suiteDescription_SparseFTFactorizationTestSuite, 50, "test_forward_transformation" ) {}
 void runTest() { suite_SparseFTFactorizationTestSuite.test_forward_transformation(); }
} testDescription_SparseFTFactorizationTestSuite_test_forward_transformation;

static class TestDescription_SparseFTFactorizationTestSuite_test_forward_transformation_with_B0 : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseFTFactorizationTestSuite_test_forward_transformation_with_B0() : CxxTest::RealTestDescription( Tests_SparseFTFactorizationTestSuite, suiteDescription_SparseFTFactorizationTestSuite, 107, "test_forward_transformation_with_B0" ) {}
 void runTest() { suite_SparseFTFactorizationTestSuite.test_forward_transformation_with_B0(); }
} testDescription_SparseFTFactorizationTestSuite_test_forward_transformation_with_B0;

static class TestDescription_SparseFTFactorizationTestSuite_test_backward_transformation : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseFTFactorizationTestSuite_test_backward_transformation() : CxxTest::RealTestDescription( Tests_SparseFTFactorizationTestSuite, suiteDescription_SparseFTFactorizationTestSuite, 146, "test_backward_transformation" ) {}
 void runTest() { suite_SparseFTFactorizationTestSuite.test_backward_transformation(); }
} testDescription_SparseFTFactorizationTestSuite_test_backward_transformation;

static class TestDescription_SparseFTFactorizationTestSuite_test_backward_transformation_2 : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseFTFactorizationTestSuite_test_backward_transformation_2() : CxxTest::RealTestDescription( Tests_SparseFTFactorizationTestSuite, suiteDescription_SparseFTFactorizationTestSuite, 226, "test_backward_transformation_2" ) {}
 void runTest() { suite_SparseFTFactorizationTestSuite.test_backward_transformation_2(); }
} testDescription_SparseFTFactorizationTestSuite_test_backward_transformation_2;

static class TestDescription_SparseFTFactorizationTestSuite_test_backward_transformation_with_B0 : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseFTFactorizationTestSuite_test_backward_transformation_with_B0() : CxxTest::RealTestDescription( Tests_SparseFTFactorizationTestSuite, suiteDescription_SparseFTFactorizationTestSuite, 260, "test_backward_transformation_with_B0" ) {}
 void runTest() { suite_SparseFTFactorizationTestSuite.test_backward_transformation_with_B0(); }
} testDescription_SparseFTFactorizationTestSuite_test_backward_transformation_with_B0;

static class TestDescription_SparseFTFactorizationTestSuite_test_store_and_restore : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseFTFactorizationTestSuite_test_store_and_restore() : CxxTest::RealTestDescription( Tests_SparseFTFactorizationTestSuite, suiteDescription_SparseFTFactorizationTestSuite, 304, "test_store_and_restore" ) {}
 void runTest() { suite_SparseFTFactorizationTestSuite.test_store_and_restore(); }
} testDescription_SparseFTFactorizationTestSuite_test_store_and_restore;

#include <cxxtest/Root.cpp>
