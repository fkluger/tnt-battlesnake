#!/bin/bash
for i in `seq 1 $1`;
do
    screen -dmS "actor-$i" "python" "main_actor.py" "--actor_index" "$i"
    echo "Started actor $i"
    sleep 1
done