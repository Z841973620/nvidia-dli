#! /usr/bin/env bash

dir=`dirname $0`

cp /dli/task/assessment/histogram.py $dir/stubs

cat $dir/stubs/values.py          <(echo) \
    $dir/stubs/histogram.py       <(echo) \
    $dir/stubs/launch.py          <(echo) \
    $dir/stubs/solution.py        <(echo) \
    $dir/stubs/print_results.py | grep -v '%' > $dir/timed_assessment.py

python3 /dli/assessment/timed_assessment.py | grep XXX | awk '{ print $2 }'
