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
#include "/content/Marabou/src/basis_factorization/tests/Test_SparseUnsortedArray.h"

static SparseUnsortedArrayTestSuite suite_SparseUnsortedArrayTestSuite;

static CxxTest::List Tests_SparseUnsortedArrayTestSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_SparseUnsortedArrayTestSuite( "/content/Marabou/src/basis_factorization/tests/Test_SparseUnsortedArray.h", 27, "SparseUnsortedArrayTestSuite", suite_SparseUnsortedArrayTestSuite, Tests_SparseUnsortedArrayTestSuite );

static class TestDescription_SparseUnsortedArrayTestSuite_test_empty_unsorted_array : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedArrayTestSuite_test_empty_unsorted_array() : CxxTest::RealTestDescription( Tests_SparseUnsortedArrayTestSuite, suiteDescription_SparseUnsortedArrayTestSuite, 42, "test_empty_unsorted_array" ) {}
 void runTest() { suite_SparseUnsortedArrayTestSuite.test_empty_unsorted_array(); }
} testDescription_SparseUnsortedArrayTestSuite_test_empty_unsorted_array;

static class TestDescription_SparseUnsortedArrayTestSuite_test_initialize_from_dense : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedArrayTestSuite_test_initialize_from_dense() : CxxTest::RealTestDescription( Tests_SparseUnsortedArrayTestSuite, suiteDescription_SparseUnsortedArrayTestSuite, 52, "test_initialize_from_dense" ) {}
 void runTest() { suite_SparseUnsortedArrayTestSuite.test_initialize_from_dense(); }
} testDescription_SparseUnsortedArrayTestSuite_test_initialize_from_dense;

static class TestDescription_SparseUnsortedArrayTestSuite_test_cloning : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedArrayTestSuite_test_cloning() : CxxTest::RealTestDescription( Tests_SparseUnsortedArrayTestSuite, suiteDescription_SparseUnsortedArrayTestSuite, 67, "test_cloning" ) {}
 void runTest() { suite_SparseUnsortedArrayTestSuite.test_cloning(); }
} testDescription_SparseUnsortedArrayTestSuite_test_cloning;

static class TestDescription_SparseUnsortedArrayTestSuite_test_iterate : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedArrayTestSuite_test_iterate() : CxxTest::RealTestDescription( Tests_SparseUnsortedArrayTestSuite, suiteDescription_SparseUnsortedArrayTestSuite, 89, "test_iterate" ) {}
 void runTest() { suite_SparseUnsortedArrayTestSuite.test_iterate(); }
} testDescription_SparseUnsortedArrayTestSuite_test_iterate;

static class TestDescription_SparseUnsortedArrayTestSuite_test_set : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedArrayTestSuite_test_set() : CxxTest::RealTestDescription( Tests_SparseUnsortedArrayTestSuite, suiteDescription_SparseUnsortedArrayTestSuite, 116, "test_set" ) {}
 void runTest() { suite_SparseUnsortedArrayTestSuite.test_set(); }
} testDescription_SparseUnsortedArrayTestSuite_test_set;

static class TestDescription_SparseUnsortedArrayTestSuite_test_merge_entries : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedArrayTestSuite_test_merge_entries() : CxxTest::RealTestDescription( Tests_SparseUnsortedArrayTestSuite, suiteDescription_SparseUnsortedArrayTestSuite, 152, "test_merge_entries" ) {}
 void runTest() { suite_SparseUnsortedArrayTestSuite.test_merge_entries(); }
} testDescription_SparseUnsortedArrayTestSuite_test_merge_entries;

#include <cxxtest/Root.cpp>
