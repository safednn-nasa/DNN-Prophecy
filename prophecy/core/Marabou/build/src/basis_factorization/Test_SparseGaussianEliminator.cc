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
#include "/content/Marabou/src/basis_factorization/tests/Test_SparseGaussianEliminator.h"

static SparseGaussianEliminatorTestSuite suite_SparseGaussianEliminatorTestSuite;

static CxxTest::List Tests_SparseGaussianEliminatorTestSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_SparseGaussianEliminatorTestSuite( "/content/Marabou/src/basis_factorization/tests/Test_SparseGaussianEliminator.h", 30, "SparseGaussianEliminatorTestSuite", suite_SparseGaussianEliminatorTestSuite, Tests_SparseGaussianEliminatorTestSuite );

static class TestDescription_SparseGaussianEliminatorTestSuite_test_sanity : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseGaussianEliminatorTestSuite_test_sanity() : CxxTest::RealTestDescription( Tests_SparseGaussianEliminatorTestSuite, suiteDescription_SparseGaussianEliminatorTestSuite, 143, "test_sanity" ) {}
 void runTest() { suite_SparseGaussianEliminatorTestSuite.test_sanity(); }
} testDescription_SparseGaussianEliminatorTestSuite_test_sanity;

#include <cxxtest/Root.cpp>
