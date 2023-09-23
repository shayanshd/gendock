#!/bin/sh

until cd /gendock/gendock_project/
do
    echo "Waiting for server volume..."
done


until python manage.py migrate
do
    echo "Waiting for db to be ready..."
    sleep 2
done


python manage.py collectstatic --noinput

# python manage.py createsuperuser --noinput

gunicorn gendock_project.wsgi --bind 0.0.0.0:8000