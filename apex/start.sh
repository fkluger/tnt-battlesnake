#!/bin/bash
for i in `seq 1 $1`;
do
    screen -dmS "actor-$i" "python" "actor.py"
done 