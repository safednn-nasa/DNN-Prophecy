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
#include "/content/Marabou/src/basis_factorization/tests/Test_SparseUnsortedArrays.h"

static SparseUnsortedArraysTestSuite suite_SparseUnsortedArraysTestSuite;

static CxxTest::List Tests_SparseUnsortedArraysTestSuite = { 0, 0 };
CxxTest::StaticSuiteDescription suiteDescription_SparseUnsortedArraysTestSuite( "/content/Marabou/src/basis_factorization/tests/Test_SparseUnsortedArrays.h", 27, "SparseUnsortedArraysTestSuite", suite_SparseUnsortedArraysTestSuite, Tests_SparseUnsortedArraysTestSuite );

static class TestDescription_SparseUnsortedArraysTestSuite_test_sanity : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedArraysTestSuite_test_sanity() : CxxTest::RealTestDescription( Tests_SparseUnsortedArraysTestSuite, suiteDescription_SparseUnsortedArraysTestSuite, 42, "test_sanity" ) {}
 void runTest() { suite_SparseUnsortedArraysTestSuite.test_sanity(); }
} testDescription_SparseUnsortedArraysTestSuite_test_sanity;

static class TestDescription_SparseUnsortedArraysTestSuite_test_store_restore : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedArraysTestSuite_test_store_restore() : CxxTest::RealTestDescription( Tests_SparseUnsortedArraysTestSuite, suiteDescription_SparseUnsortedArraysTestSuite, 81, "test_store_restore" ) {}
 void runTest() { suite_SparseUnsortedArraysTestSuite.test_store_restore(); }
} testDescription_SparseUnsortedArraysTestSuite_test_store_restore;

static class TestDescription_SparseUnsortedArraysTestSuite_test_add_last_row : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedArraysTestSuite_test_add_last_row() : CxxTest::RealTestDescription( Tests_SparseUnsortedArraysTestSuite, suiteDescription_SparseUnsortedArraysTestSuite, 104, "test_add_last_row" ) {}
 void runTest() { suite_SparseUnsortedArraysTestSuite.test_add_last_row(); }
} testDescription_SparseUnsortedArraysTestSuite_test_add_last_row;

static class TestDescription_SparseUnsortedArraysTestSuite_test_add_last_column : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedArraysTestSuite_test_add_last_column() : CxxTest::RealTestDescription( Tests_SparseUnsortedArraysTestSuite, suiteDescription_SparseUnsortedArraysTestSuite, 139, "test_add_last_column" ) {}
 void runTest() { suite_SparseUnsortedArraysTestSuite.test_add_last_column(); }
} testDescription_SparseUnsortedArraysTestSuite_test_add_last_column;

static class TestDescription_SparseUnsortedArraysTestSuite_test_get_row : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedArraysTestSuite_test_get_row() : CxxTest::RealTestDescription( Tests_SparseUnsortedArraysTestSuite, suiteDescription_SparseUnsortedArraysTestSuite, 203, "test_get_row" ) {}
 void runTest() { suite_SparseUnsortedArraysTestSuite.test_get_row(); }
} testDescription_SparseUnsortedArraysTestSuite_test_get_row;

static class TestDescription_SparseUnsortedArraysTestSuite_test_to_dense : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedArraysTestSuite_test_to_dense() : CxxTest::RealTestDescription( Tests_SparseUnsortedArraysTestSuite, suiteDescription_SparseUnsortedArraysTestSuite, 229, "test_to_dense" ) {}
 void runTest() { suite_SparseUnsortedArraysTestSuite.test_to_dense(); }
} testDescription_SparseUnsortedArraysTestSuite_test_to_dense;

static class TestDescription_SparseUnsortedArraysTestSuite_test_get_column : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedArraysTestSuite_test_get_column() : CxxTest::RealTestDescription( Tests_SparseUnsortedArraysTestSuite, suiteDescription_SparseUnsortedArraysTestSuite, 269, "test_get_column" ) {}
 void runTest() { suite_SparseUnsortedArraysTestSuite.test_get_column(); }
} testDescription_SparseUnsortedArraysTestSuite_test_get_column;

static class TestDescription_SparseUnsortedArraysTestSuite_test_deletions : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedArraysTestSuite_test_deletions() : CxxTest::RealTestDescription( Tests_SparseUnsortedArraysTestSuite, suiteDescription_SparseUnsortedArraysTestSuite, 303, "test_deletions" ) {}
 void runTest() { suite_SparseUnsortedArraysTestSuite.test_deletions(); }
} testDescription_SparseUnsortedArraysTestSuite_test_deletions;

static class TestDescription_SparseUnsortedArraysTestSuite_test_changes : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedArraysTestSuite_test_changes() : CxxTest::RealTestDescription( Tests_SparseUnsortedArraysTestSuite, suiteDescription_SparseUnsortedArraysTestSuite, 341, "test_changes" ) {}
 void runTest() { suite_SparseUnsortedArraysTestSuite.test_changes(); }
} testDescription_SparseUnsortedArraysTestSuite_test_changes;

static class TestDescription_SparseUnsortedArraysTestSuite_test_changes_and_deletions : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedArraysTestSuite_test_changes_and_deletions() : CxxTest::RealTestDescription( Tests_SparseUnsortedArraysTestSuite, suiteDescription_SparseUnsortedArraysTestSuite, 383, "test_changes_and_deletions" ) {}
 void runTest() { suite_SparseUnsortedArraysTestSuite.test_changes_and_deletions(); }
} testDescription_SparseUnsortedArraysTestSuite_test_changes_and_deletions;

static class TestDescription_SparseUnsortedArraysTestSuite_test_count_elements : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedArraysTestSuite_test_count_elements() : CxxTest::RealTestDescription( Tests_SparseUnsortedArraysTestSuite, suiteDescription_SparseUnsortedArraysTestSuite, 432, "test_count_elements" ) {}
 void runTest() { suite_SparseUnsortedArraysTestSuite.test_count_elements(); }
} testDescription_SparseUnsortedArraysTestSuite_test_count_elements;

static class TestDescription_SparseUnsortedArraysTestSuite_test_transpose : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedArraysTestSuite_test_transpose() : CxxTest::RealTestDescription( Tests_SparseUnsortedArraysTestSuite, suiteDescription_SparseUnsortedArraysTestSuite, 461, "test_transpose" ) {}
 void runTest() { suite_SparseUnsortedArraysTestSuite.test_transpose(); }
} testDescription_SparseUnsortedArraysTestSuite_test_transpose;

static class TestDescription_SparseUnsortedArraysTestSuite_test_clear : public CxxTest::RealTestDescription {
public:
 TestDescription_SparseUnsortedArraysTestSuite_test_clear() : CxxTest::RealTestDescription( Tests_SparseUnsortedArraysTestSuite, suiteDescription_SparseUnsortedArraysTestSuite, 512, "test_clear" ) {}
 void runTest() { suite_SparseUnsortedArraysTestSuite.test_clear(); }
} testDescription_SparseUnsortedArraysTestSuite_test_clear;

#include <cxxtest/Root.cpp>
