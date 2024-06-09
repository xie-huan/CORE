#!/bin/bash
SCRIPT_DIR=$(pwd)
# echo "input PID BID"
# #while true
# #do
# read PID BID
PID=$1
BID=$2
#if [[ $PID == $'\x1b' ]]; then
#echo "Finish EXIT"
#break
#else
#PID=$1
#BID=$2
PROJECT_DIR=/mnt/project/${PID}_${BID}_buggy
export GZOLTAR_AGENT_JAR="/root/ase/gzoltar/com.gzoltar.agent.rt/target/com.gzoltar.agent.rt-1.7.4-SNAPSHOT-all.jar"
export GZOLTAR_CLI_JAR="/root/ase/gzoltar/com.gzoltar.cli/target/com.gzoltar.cli-1.7.4-SNAPSHOT-jar-with-dependencies.jar"
export GZOLTAR_ANT_JAR="/root/ase/gzoltar/com.gzoltar.ant/target/com.gzoltar.ant-1.7.4-SNAPSHOT-jar-with-dependencies.jar"

# Checkout defects4j project
#检出defects4j项目版本
defects4j checkout -p ${PID} -v ${BID}b -w ${PROJECT_DIR}
cd ${PROJECT_DIR}

#编译项目
defects4j compile

#获取源代码目录、测试代码目录、类文件目录
SRC_DIR=${PROJECT_DIR}/$(defects4j export -p dir.bin.classes)
TEST_DIR=${PROJECT_DIR}/$(defects4j export -p dir.bin.tests)
LIB_DIR=$(defects4j export -p cp.test)

RELEVANT_TESTS_FILE=${D4J_HOME}/framework/projects/${PID}/relevant_tests/${BID}
RELEVANT_TESTS=$(cat ${RELEVANT_TESTS_FILE} | sed 's/$/#*/' | sed ':a;N;$!ba;s/\n/:/g')

# List test methods
#列出测试方法
java -cp ${LIB_DIR}:${GZOLTAR_CLI_JAR} \
  com.gzoltar.cli.Main listTestMethods \
  ${TEST_DIR} \
  --outputFile ${PROJECT_DIR}/listTestMethods.txt \
  --includes ${RELEVANT_TESTS}

SER_FILE=${PROJECT_DIR}/gzoltar.ser
LOADED_CLASSES_FILE=${D4J_HOME}/framework/projects/${PID}/loaded_classes/${BID}.src
NORMAL_CLASSES=$(cat ${LOADED_CLASSES_FILE} | sed 's/$/:/' | sed ':a;N;$!ba;s/\n//g')
INNER_CLASSES=$(cat ${LOADED_CLASSES_FILE} | sed 's/$/$*:/' | sed ':a;N;$!ba;s/\n//g')
LOADED_CLASSES=${NORMAL_CLASSES}${INNER_CLASSES}

# Generate .ser file
#收集覆盖率
java -javaagent:${GZOLTAR_AGENT_JAR}=destfile=${SER_FILE},buildlocation=${SRC_DIR},includes=${LOADED_CLASSES},excludes="",inclnolocationclasses=false,output="file" \
  -cp ${GZOLTAR_CLI_JAR}:${LIB_DIR} \
  com.gzoltar.cli.Main runTestMethods \
  --testMethods ${PROJECT_DIR}/listTestMethods.txt \
  --collectCoverage 

# Generate report
#生成覆盖矩阵报告
java -cp ${GZOLTAR_CLI_JAR}:${LIB_DIR} \
  com.gzoltar.cli.Main faultLocalizationReport \
    --buildLocation ${SRC_DIR} \
    --granularity line \
    --inclPublicMethods \
    --inclStaticConstructors \
    --inclDeprecatedMethods \
    --dataFile ${SER_FILE} \
    --outputDirectory ${PROJECT_DIR} \
    --family sfl \
    --formula ochiai \
    --metric entropy \
    --formatter txt

# # Simple file processing
# #文件处理
# MATRIX_FILE=${PROJECT_DIR}/sfl/txt/matrix.txt
# SPECTRA_FILE=${PROJECT_DIR}/sfl/txt/spectra.csv
# TESTS_FILE=${PROJECT_DIR}/sfl/txt/tests.csv

# ARCHIVE_DIR=/root/ase/exec_info/${PID}/${BID}
# mkdir -p ${ARCHIVE_DIR}

# cp ${MATRIX_FILE} ${ARCHIVE_DIR}/matrix
# cp ${SPECTRA_FILE} ${ARCHIVE_DIR}/spectra
# cp ${TESTS_FILE} ${ARCHIVE_DIR}/tests
# tail -n +2 ${ARCHIVE_DIR}/matrix > ${ARCHIVE_DIR}/spectra
# tail -n +2 ${ARCHIVE_DIR}/tests > ${ARCHIVE_DIR}/tests

# # Copied code
# # Remove inner class(es) names (as there is not a .java file for each one)
# sed -i -E 's/(\$\w+)\$.*#/\1#/g' ${ARCHIVE_DIR}/spectra
# # Remove method name of each row in the spectra file
# sed -i 's/#.*:/#/g' ${ARCHIVE_DIR}/spectra
# # Replace class name symbol
# sed -i 's/\$/./g' ${ARCHIVE_DIR}/spectra 


# Simple file processing
#文件处理
MATRIX_FILE=${PROJECT_DIR}/sfl/txt/matrix.txt
SPECTRA_FILE=${PROJECT_DIR}/sfl/txt/spectra.csv
TESTS_FILE=${PROJECT_DIR}/sfl/txt/tests.csv

ARCHIVE_DIR=/mnt/exec_info/${PID}/${BID}
mkdir -p ${ARCHIVE_DIR}

mv ${MATRIX_FILE} ${ARCHIVE_DIR}/matrix
tail -n +2 ${SPECTRA_FILE} > ${ARCHIVE_DIR}/spectra
tail -n +2 ${TESTS_FILE} > ${ARCHIVE_DIR}/tests

# Copied code
# Remove inner class(es) names (as there is not a .java file for each one)
sed -i -E 's/(\$\w+)\$.*#/\1#/g' ${ARCHIVE_DIR}/spectra
# Remove method name of each row in the spectra file
sed -i 's/#.*:/#/g' ${ARCHIVE_DIR}/spectra
# Replace class name symbol
sed -i 's/\$/./g' ${ARCHIVE_DIR}/spectra
