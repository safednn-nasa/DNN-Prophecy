# CMake generated Testfile for 
# Source directory: /content/Marabou/regress/regress2
# Build directory: /content/Marabou/build/regress/regress2
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(ACASXU_experimental_v2a_3_7.nnet%acas_property_4.txt% "/usr/bin/python3" "/content/Marabou/regress/run_regression.py" "/content/Marabou/build/bin/Marabou" "/content/Marabou/resources/nnet/acasxu/ACASXU_experimental_v2a_3_7.nnet" "/content/Marabou/resources/properties/acas_property_4.txt" "unsat")
set_tests_properties(ACASXU_experimental_v2a_3_7.nnet%acas_property_4.txt% PROPERTIES  LABELS "regress2
    unsat acasxu" _BACKTRACE_TRIPLES "/content/Marabou/regress/CMakeLists.txt;9;add_test;/content/Marabou/regress/CMakeLists.txt;34;marabou_add_regress_test;/content/Marabou/regress/regress2/CMakeLists.txt;2;marabou_add_acasxu_test;/content/Marabou/regress/regress2/CMakeLists.txt;0;")
add_test(ACASXU_experimental_v2a_3_7.onnx%acas_property_4.txt% "/usr/bin/python3" "/content/Marabou/regress/run_regression.py" "/content/Marabou/build/bin/Marabou" "/content/Marabou/resources/onnx/acasxu/ACASXU_experimental_v2a_3_7.onnx" "/content/Marabou/resources/properties/acas_property_4.txt" "unsat")
set_tests_properties(ACASXU_experimental_v2a_3_7.onnx%acas_property_4.txt% PROPERTIES  LABELS "regress2
    unsat acasxu" _BACKTRACE_TRIPLES "/content/Marabou/regress/CMakeLists.txt;9;add_test;/content/Marabou/regress/CMakeLists.txt;37;marabou_add_regress_test;/content/Marabou/regress/regress2/CMakeLists.txt;2;marabou_add_acasxu_test;/content/Marabou/regress/regress2/CMakeLists.txt;0;")
add_test(reluBenchmark2.66962385178s_UNSAT.nnet%builtin_property.txt% "/usr/bin/python3" "/content/Marabou/regress/run_regression.py" "/content/Marabou/build/bin/Marabou" "/content/Marabou/resources/nnet/coav/reluBenchmark2.66962385178s_UNSAT.nnet" "/content/Marabou/resources/properties/builtin_property.txt" "unsat")
set_tests_properties(reluBenchmark2.66962385178s_UNSAT.nnet%builtin_property.txt% PROPERTIES  LABELS "regress2
    unsat coav" _BACKTRACE_TRIPLES "/content/Marabou/regress/CMakeLists.txt;9;add_test;/content/Marabou/regress/CMakeLists.txt;28;marabou_add_regress_test;/content/Marabou/regress/regress2/CMakeLists.txt;3;marabou_add_coav_test;/content/Marabou/regress/regress2/CMakeLists.txt;0;")
add_test(ACASXU_experimental_v2a_3_9.nnet%acas_property_3.txt%--num-workers=2+--snc+--initial-divides=2 "/usr/bin/python3" "/content/Marabou/regress/run_regression.py" "/content/Marabou/build/bin/Marabou" "/content/Marabou/resources/nnet/acasxu/ACASXU_experimental_v2a_3_9.nnet" "/content/Marabou/resources/properties/acas_property_3.txt" "unsat" "--num-workers=2+--snc+--initial-divides=2")
set_tests_properties(ACASXU_experimental_v2a_3_9.nnet%acas_property_3.txt%--num-workers=2+--snc+--initial-divides=2 PROPERTIES  LABELS "regress1
    unsat acasxu" _BACKTRACE_TRIPLES "/content/Marabou/regress/CMakeLists.txt;9;add_test;/content/Marabou/regress/CMakeLists.txt;43;marabou_add_regress_test;/content/Marabou/regress/regress2/CMakeLists.txt;4;marabou_add_acasxu_dnc_test;/content/Marabou/regress/regress2/CMakeLists.txt;0;")
add_test(reluBenchmark2.64361000061s_SAT.nnet%builtin_property.txt% "/usr/bin/python3" "/content/Marabou/regress/run_regression.py" "/content/Marabou/build/bin/Marabou" "/content/Marabou/resources/nnet/coav/reluBenchmark2.64361000061s_SAT.nnet" "/content/Marabou/resources/properties/builtin_property.txt" "sat")
set_tests_properties(reluBenchmark2.64361000061s_SAT.nnet%builtin_property.txt% PROPERTIES  LABELS "regress2
    sat coav" _BACKTRACE_TRIPLES "/content/Marabou/regress/CMakeLists.txt;9;add_test;/content/Marabou/regress/CMakeLists.txt;28;marabou_add_regress_test;/content/Marabou/regress/regress2/CMakeLists.txt;6;marabou_add_coav_test;/content/Marabou/regress/regress2/CMakeLists.txt;0;")
add_test(mnist10x10.nnet%image3_target9_epsilon0.005.txt% "/usr/bin/python3" "/content/Marabou/regress/run_regression.py" "/content/Marabou/build/bin/Marabou" "/content/Marabou/resources/nnet/mnist/mnist10x10.nnet" "/content/Marabou/resources/properties/mnist/image3_target9_epsilon0.005.txt" "unsat")
set_tests_properties(mnist10x10.nnet%image3_target9_epsilon0.005.txt% PROPERTIES  LABELS "regress2
    unsat mnist" _BACKTRACE_TRIPLES "/content/Marabou/regress/CMakeLists.txt;9;add_test;/content/Marabou/regress/CMakeLists.txt;49;marabou_add_regress_test;/content/Marabou/regress/regress2/CMakeLists.txt;14;marabou_add_mnist_test;/content/Marabou/regress/regress2/CMakeLists.txt;0;")
