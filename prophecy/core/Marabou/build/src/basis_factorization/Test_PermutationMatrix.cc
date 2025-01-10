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
#include "/content/Marabou/src/basis_factorization/tests/Test_PermutationMatrix.h"

static PermutationMatrixTestSuite suite_PermutationMatrixTestSuite;

static CxxTest::List Tests_PermutationMatrixTestSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_PermutationMatrixTestSuite( "/content/Marabou/src/basis_factorization/tests/Test_PermutationMatrix.h", 31, "PermutationMatrixTestSuite", suite_PermutationMatrixTestSuite, Tests_PermutationMatrixTestSuite );

static class TestDescription_PermutationMatrixTestSuite_test_reset : public CxxTest::RealTestDescription {
public:
 TestDescription_PermutationMatrixTestSuite_test_reset() : CxxTest::RealTestDescription( Tests_PermutationMatrixTestSuite, suiteDescription_PermutationMatrixTestSuite, 55, "test_reset" ) {}
 void runTest() { suite_PermutationMatrixTestSuite.test_reset(); }
} testDescription_PermutationMatrixTestSuite_test_reset;

static class TestDescription_PermutationMatrixTestSuite_test_assignment : public CxxTest::RealTestDescription {
public:
 TestDescription_PermutationMatrixTestSuite_test_assignment() : CxxTest::RealTestDescription( Tests_PermutationMatrixTestSuite, suiteDescription_PermutationMatrixTestSuite, 71, "test_assignment" ) {}
 void runTest() { suite_PermutationMatrixTestSuite.test_assignment(); }
} testDescription_PermutationMatrixTestSuite_test_assignment;

static class TestDescription_PermutationMatrixTestSuite_test_invert : public CxxTest::RealTestDescription {
public:
 TestDescription_PermutationMatrixTestSuite_test_invert() : CxxTest::RealTestDescription( Tests_PermutationMatrixTestSuite, suiteDescription_PermutationMatrixTestSuite, 92, "test_invert" ) {}
 void runTest() { suite_PermutationMatrixTestSuite.test_invert(); }
} testDescription_PermutationMatrixTestSuite_test_invert;

static class TestDescription_PermutationMatrixTestSuite_test_find_index_of_row : public CxxTest::RealTestDescription {
public:
 TestDescription_PermutationMatrixTestSuite_test_find_index_of_row() : CxxTest::RealTestDescription( Tests_PermutationMatrixTestSuite, suiteDescription_PermutationMatrixTestSuite, 171, "test_find_index_of_row" ) {}
 void runTest() { suite_PermutationMatrixTestSuite.test_find_index_of_row(); }
} testDescription_PermutationMatrixTestSuite_test_find_index_of_row;

#include <cxxtest/Root.cpp>
