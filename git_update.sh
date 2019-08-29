#!/bin/bash
nowdate=`date`
echo ${nowdate}

echo "git marking"

git add *
git commit -m" alter some and automated update! "

branch_nam=`git symbolic-ref --short -q HEAD`
echo "will push to ${branch_name}"
git push origin ${branch_name}

