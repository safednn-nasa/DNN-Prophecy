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
#include "/content/Marabou/src/basis_factorization/tests/Test_SparseUnsortedLists.h"

static SparseUnsortedListsTestSuite suite_SparseUnsortedListsTestSuite;

static CxxTest::List Tests_SparseUnsortedListsTestSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_SparseUnsortedListsTestSuite( "/content/Marabou/src/basis_factorization/tests/Test_SparseUnsortedLists.h", 27, "SparseUnsortedListsTestSuite", suite_SparseUnsortedListsTestSuite, Tests_SparseUnsortedListsTestSuite );

static class TestDescription_SparseUnsortedListsTestSuite_test_sanity : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedListsTestSuite_test_sanity() : CxxTest::RealTestDescription( Tests_SparseUnsortedListsTestSuite, suiteDescription_SparseUnsortedListsTestSuite, 42, "test_sanity" ) {}
 void runTest() { suite_SparseUnsortedListsTestSuite.test_sanity(); }
} testDescription_SparseUnsortedListsTestSuite_test_sanity;

static class TestDescription_SparseUnsortedListsTestSuite_test_store_restore : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedListsTestSuite_test_store_restore() : CxxTest::RealTestDescription( Tests_SparseUnsortedListsTestSuite, suiteDescription_SparseUnsortedListsTestSuite, 81, "test_store_restore" ) {}
 void runTest() { suite_SparseUnsortedListsTestSuite.test_store_restore(); }
} testDescription_SparseUnsortedListsTestSuite_test_store_restore;

static class TestDescription_SparseUnsortedListsTestSuite_test_add_last_row : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedListsTestSuite_test_add_last_row() : CxxTest::RealTestDescription( Tests_SparseUnsortedListsTestSuite, suiteDescription_SparseUnsortedListsTestSuite, 104, "test_add_last_row" ) {}
 void runTest() { suite_SparseUnsortedListsTestSuite.test_add_last_row(); }
} testDescription_SparseUnsortedListsTestSuite_test_add_last_row;

static class TestDescription_SparseUnsortedListsTestSuite_test_add_last_column : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedListsTestSuite_test_add_last_column() : CxxTest::RealTestDescription( Tests_SparseUnsortedListsTestSuite, suiteDescription_SparseUnsortedListsTestSuite, 139, "test_add_last_column" ) {}
 void runTest() { suite_SparseUnsortedListsTestSuite.test_add_last_column(); }
} testDescription_SparseUnsortedListsTestSuite_test_add_last_column;

static class TestDescription_SparseUnsortedListsTestSuite_test_get_row : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedListsTestSuite_test_get_row() : CxxTest::RealTestDescription( Tests_SparseUnsortedListsTestSuite, suiteDescription_SparseUnsortedListsTestSuite, 203, "test_get_row" ) {}
 void runTest() { suite_SparseUnsortedListsTestSuite.test_get_row(); }
} testDescription_SparseUnsortedListsTestSuite_test_get_row;

static class TestDescription_SparseUnsortedListsTestSuite_test_to_dense : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedListsTestSuite_test_to_dense() : CxxTest::RealTestDescription( Tests_SparseUnsortedListsTestSuite, suiteDescription_SparseUnsortedListsTestSuite, 229, "test_to_dense" ) {}
 void runTest() { suite_SparseUnsortedListsTestSuite.test_to_dense(); }
} testDescription_SparseUnsortedListsTestSuite_test_to_dense;

static class TestDescription_SparseUnsortedListsTestSuite_test_get_column : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedListsTestSuite_test_get_column() : CxxTest::RealTestDescription( Tests_SparseUnsortedListsTestSuite, suiteDescription_SparseUnsortedListsTestSuite, 269, "test_get_column" ) {}
 void runTest() { suite_SparseUnsortedListsTestSuite.test_get_column(); }
} testDescription_SparseUnsortedListsTestSuite_test_get_column;

static class TestDescription_SparseUnsortedListsTestSuite_test_deletions : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedListsTestSuite_test_deletions() : CxxTest::RealTestDescription( Tests_SparseUnsortedListsTestSuite, suiteDescription_SparseUnsortedListsTestSuite, 303, "test_deletions" ) {}
 void runTest() { suite_SparseUnsortedListsTestSuite.test_deletions(); }
} testDescription_SparseUnsortedListsTestSuite_test_deletions;

static class TestDescription_SparseUnsortedListsTestSuite_test_changes : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedListsTestSuite_test_changes() : CxxTest::RealTestDescription( Tests_SparseUnsortedListsTestSuite, suiteDescription_SparseUnsortedListsTestSuite, 341, "test_changes" ) {}
 void runTest() { suite_SparseUnsortedListsTestSuite.test_changes(); }
} testDescription_SparseUnsortedListsTestSuite_test_changes;

static class TestDescription_SparseUnsortedListsTestSuite_test_changes_and_deletions : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedListsTestSuite_test_changes_and_deletions() : CxxTest::RealTestDescription( Tests_SparseUnsortedListsTestSuite, suiteDescription_SparseUnsortedListsTestSuite, 383, "test_changes_and_deletions" ) {}
 void runTest() { suite_SparseUnsortedListsTestSuite.test_changes_and_deletions(); }
} testDescription_SparseUnsortedListsTestSuite_test_changes_and_deletions;

static class TestDescription_SparseUnsortedListsTestSuite_test_count_elements : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedListsTestSuite_test_count_elements() : CxxTest::RealTestDescription( Tests_SparseUnsortedListsTestSuite, suiteDescription_SparseUnsortedListsTestSuite, 432, "test_count_elements" ) {}
 void runTest() { suite_SparseUnsortedListsTestSuite.test_count_elements(); }
} testDescription_SparseUnsortedListsTestSuite_test_count_elements;

static class TestDescription_SparseUnsortedListsTestSuite_test_transpose : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedListsTestSuite_test_transpose() : CxxTest::RealTestDescription( Tests_SparseUnsortedListsTestSuite, suiteDescription_SparseUnsortedListsTestSuite, 461, "test_transpose" ) {}
 void runTest() { suite_SparseUnsortedListsTestSuite.test_transpose(); }
} testDescription_SparseUnsortedListsTestSuite_test_transpose;

static class TestDescription_SparseUnsortedListsTestSuite_test_clear : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedListsTestSuite_test_clear() : CxxTest::RealTestDescription( Tests_SparseUnsortedListsTestSuite, suiteDescription_SparseUnsortedListsTestSuite, 512, "test_clear" ) {}
 void runTest() { suite_SparseUnsortedListsTestSuite.test_clear(); }
} testDescription_SparseUnsortedListsTestSuite_test_clear;

#include <cxxtest/Root.cpp>
