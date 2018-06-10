#!/bin/bash
screen -dmS "learner" "python" "main_learner.py"
sleep 3
for i in `seq 1 $1`;
do
    screen -dmS "actor-$i" "python" "main_actor.py"
    sleep 1
done 