#!/bin/sh

until cd /gendock/gendock_project/
do
    echo "Waiting for server volume..."
done

# run a worker
celery -A gendock_project worker -l INFO -P threads
