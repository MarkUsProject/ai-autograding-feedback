#!/bin/bash

# Define test parameters
TEST_NAME="test_hw5.py"
TEST_SCRIPT="test_hw5.py"
TEST_OUTPUT_FILE="test_output1.txt"

# Step 1: Run test_hw5.py to get test errors file
# Note: name as test_output1.txt, since CSV file is too large to load tests will all fail
TEST_OUTPUT=$(/home/docker/.autotesting/scripts/defaultvenv/bin/pytest "$TEST_SCRIPT" 2>&1 | tee "$TEST_OUTPUT_FILE")
TEST_STATUS=${PIPESTATUS[0]}

if [ $TEST_STATUS -eq 0 ]; then
  STATUS="pass"
  MARKS_EARNED=1
else
  STATUS="fail"
  MARKS_EARNED=0
fi
MARKS_TOTAL=1

JSON_TEST_OUTPUT=$(echo "$TEST_OUTPUT" | jq -R -s .)

echo "{\"name\": \"$TEST_NAME\", \"output\": $JSON_TEST_OUTPUT, \"marks_earned\": $MARKS_EARNED, \"marks_total\": $MARKS_TOTAL, \"status\": \"$STATUS\", \"time\": null}"
