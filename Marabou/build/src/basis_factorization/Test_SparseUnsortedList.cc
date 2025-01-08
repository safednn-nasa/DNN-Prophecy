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
#include "/content/Marabou/src/basis_factorization/tests/Test_SparseUnsortedList.h"

static SparseUnsortedListTestSuite suite_SparseUnsortedListTestSuite;

static CxxTest::List Tests_SparseUnsortedListTestSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_SparseUnsortedListTestSuite( "/content/Marabou/src/basis_factorization/tests/Test_SparseUnsortedList.h", 27, "SparseUnsortedListTestSuite", suite_SparseUnsortedListTestSuite, Tests_SparseUnsortedListTestSuite );

static class TestDescription_SparseUnsortedListTestSuite_test_empty_unsorted_list : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedListTestSuite_test_empty_unsorted_list() : CxxTest::RealTestDescription( Tests_SparseUnsortedListTestSuite, suiteDescription_SparseUnsortedListTestSuite, 42, "test_empty_unsorted_list" ) {}
 void runTest() { suite_SparseUnsortedListTestSuite.test_empty_unsorted_list(); }
} testDescription_SparseUnsortedListTestSuite_test_empty_unsorted_list;

static class TestDescription_SparseUnsortedListTestSuite_test_initialize_from_dense : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedListTestSuite_test_initialize_from_dense() : CxxTest::RealTestDescription( Tests_SparseUnsortedListTestSuite, suiteDescription_SparseUnsortedListTestSuite, 52, "test_initialize_from_dense" ) {}
 void runTest() { suite_SparseUnsortedListTestSuite.test_initialize_from_dense(); }
} testDescription_SparseUnsortedListTestSuite_test_initialize_from_dense;

static class TestDescription_SparseUnsortedListTestSuite_test_cloning : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedListTestSuite_test_cloning() : CxxTest::RealTestDescription( Tests_SparseUnsortedListTestSuite, suiteDescription_SparseUnsortedListTestSuite, 67, "test_cloning" ) {}
 void runTest() { suite_SparseUnsortedListTestSuite.test_cloning(); }
} testDescription_SparseUnsortedListTestSuite_test_cloning;

static class TestDescription_SparseUnsortedListTestSuite_test_iterate : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedListTestSuite_test_iterate() : CxxTest::RealTestDescription( Tests_SparseUnsortedListTestSuite, suiteDescription_SparseUnsortedListTestSuite, 89, "test_iterate" ) {}
 void runTest() { suite_SparseUnsortedListTestSuite.test_iterate(); }
} testDescription_SparseUnsortedListTestSuite_test_iterate;

static class TestDescription_SparseUnsortedListTestSuite_test_set : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedListTestSuite_test_set() : CxxTest::RealTestDescription( Tests_SparseUnsortedListTestSuite, suiteDescription_SparseUnsortedListTestSuite, 118, "test_set" ) {}
 void runTest() { suite_SparseUnsortedListTestSuite.test_set(); }
} testDescription_SparseUnsortedListTestSuite_test_set;

static class TestDescription_SparseUnsortedListTestSuite_test_merge_entries : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedListTestSuite_test_merge_entries() : CxxTest::RealTestDescription( Tests_SparseUnsortedListTestSuite, suiteDescription_SparseUnsortedListTestSuite, 154, "test_merge_entries" ) {}
 void runTest() { suite_SparseUnsortedListTestSuite.test_merge_entries(); }
} testDescription_SparseUnsortedListTestSuite_test_merge_entries;

#include <cxxtest/Root.cpp>
