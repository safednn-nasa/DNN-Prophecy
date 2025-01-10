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
#include "/content/Marabou/src/basis_factorization/tests/Test_SparseLUFactorization.h"

static SparseLUFactorizationTestSuite suite_SparseLUFactorizationTestSuite;

static CxxTest::List Tests_SparseLUFactorizationTestSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_SparseLUFactorizationTestSuite( "/content/Marabou/src/basis_factorization/tests/Test_SparseLUFactorization.h", 32, "SparseLUFactorizationTestSuite", suite_SparseLUFactorizationTestSuite, Tests_SparseLUFactorizationTestSuite );

static class TestDescription_SparseLUFactorizationTestSuite_test_forward_transformation : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseLUFactorizationTestSuite_test_forward_transformation() : CxxTest::RealTestDescription( Tests_SparseLUFactorizationTestSuite, suiteDescription_SparseLUFactorizationTestSuite, 50, "test_forward_transformation" ) {}
 void runTest() { suite_SparseLUFactorizationTestSuite.test_forward_transformation(); }
} testDescription_SparseLUFactorizationTestSuite_test_forward_transformation;

static class TestDescription_SparseLUFactorizationTestSuite_test_forward_transformation_with_B0 : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseLUFactorizationTestSuite_test_forward_transformation_with_B0() : CxxTest::RealTestDescription( Tests_SparseLUFactorizationTestSuite, suiteDescription_SparseLUFactorizationTestSuite, 107, "test_forward_transformation_with_B0" ) {}
 void runTest() { suite_SparseLUFactorizationTestSuite.test_forward_transformation_with_B0(); }
} testDescription_SparseLUFactorizationTestSuite_test_forward_transformation_with_B0;

static class TestDescription_SparseLUFactorizationTestSuite_test_backward_transformation : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseLUFactorizationTestSuite_test_backward_transformation() : CxxTest::RealTestDescription( Tests_SparseLUFactorizationTestSuite, suiteDescription_SparseLUFactorizationTestSuite, 139, "test_backward_transformation" ) {}
 void runTest() { suite_SparseLUFactorizationTestSuite.test_backward_transformation(); }
} testDescription_SparseLUFactorizationTestSuite_test_backward_transformation;

static class TestDescription_SparseLUFactorizationTestSuite_test_backward_transformation_2 : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseLUFactorizationTestSuite_test_backward_transformation_2() : CxxTest::RealTestDescription( Tests_SparseLUFactorizationTestSuite, suiteDescription_SparseLUFactorizationTestSuite, 216, "test_backward_transformation_2" ) {}
 void runTest() { suite_SparseLUFactorizationTestSuite.test_backward_transformation_2(); }
} testDescription_SparseLUFactorizationTestSuite_test_backward_transformation_2;

static class TestDescription_SparseLUFactorizationTestSuite_test_backward_transformation_with_B0 : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseLUFactorizationTestSuite_test_backward_transformation_with_B0() : CxxTest::RealTestDescription( Tests_SparseLUFactorizationTestSuite, suiteDescription_SparseLUFactorizationTestSuite, 248, "test_backward_transformation_with_B0" ) {}
 void runTest() { suite_SparseLUFactorizationTestSuite.test_backward_transformation_with_B0(); }
} testDescription_SparseLUFactorizationTestSuite_test_backward_transformation_with_B0;

static class TestDescription_SparseLUFactorizationTestSuite_test_store_and_restore : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseLUFactorizationTestSuite_test_store_and_restore() : CxxTest::RealTestDescription( Tests_SparseLUFactorizationTestSuite, suiteDescription_SparseLUFactorizationTestSuite, 285, "test_store_and_restore" ) {}
 void runTest() { suite_SparseLUFactorizationTestSuite.test_store_and_restore(); }
} testDescription_SparseLUFactorizationTestSuite_test_store_and_restore;

#include <cxxtest/Root.cpp>
