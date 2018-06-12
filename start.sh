#!/bin/bash
screen -dmS "learner" "python" "main_learner.py"
echo "Started learner thread."
sleep 3
for i in `seq 1 $1`;
do
    screen -dmS "actor-$i" "python" "main_actor.py"
    echo "Started actor $i"
    sleep 1
done 