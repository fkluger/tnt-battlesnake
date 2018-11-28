#!/bin/bash
for i in `seq $1 $2`;
do
    screen -dmS "actor-$i" "python" "main_actor.py" "--actor_index" "$i" "--starting_port" "$3"
    echo "Started actor $i"
    sleep 1
done